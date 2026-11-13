"""Train the next-token model.

    python train.py /workspace/tok /workspace/run --steps 60000
"""
import argparse, json, math, time, os
import numpy as np, torch, torch.nn.functional as F
from model import SketchGPT
from tokens import PAD

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--steps", type=int, default=60000)
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--drop", type=float, default=0.0)
ap.add_argument("--d", type=int, default=256)
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--eval-every", type=int, default=2000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(0)
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

train = torch.from_numpy(np.load(f"{a.data}/train.npy")).to(dev)
val = torch.from_numpy(np.load(f"{a.data}/validation.npy")).to(dev)[:50000]
print("train rows", len(train), "val rows", len(val), flush=True)

model = SketchGPT(a.d, a.layers, a.heads, a.drop).to(dev)
print("weights", sum(p.numel() for p in model.parameters()), flush=True)
decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
plain = [p for n, p in model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": plain, "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.95), fused=True)
fast = torch.compile(model)
warm = 1000


def lr_at(step):
    if step < warm:
        return a.lr * (step + 1) / warm
    return a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, a.steps - warm))))


def loss_on(rows):
    rows = rows.long()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = fast(rows[:, :-1])
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), rows[:, 1:].reshape(-1), ignore_index=PAD)


@torch.no_grad()
def val_loss():
    model.eval()
    tot = sum(loss_on(val[i:i + 2000]).item() for i in range(0, len(val), 2000))
    model.train()
    return tot / math.ceil(len(val) / 2000)


log, best, t0, running = [], 9e9, time.time(), 0.0
for step in range(a.steps + 1):
    if step % a.eval_every == 0:
        v = val_loss()
        entry = {"step": step, "train_loss": round(running / a.eval_every, 4) if step else None,
                 "val_loss": round(v, 4), "minutes": round((time.time() - t0) / 60, 1)}
        log.append(entry); print(json.dumps(entry), flush=True); running = 0.0
        json.dump(log, open(f"{a.out}/log.json", "w"))
        if v < best:
            best = v
            torch.save({"cfg": model.cfg, "state": model.state_dict(), "step": step, "val_loss": v}, f"{a.out}/best.pt")
    if step == a.steps:
        break
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    rows = train[torch.randint(len(train), (a.batch,), device=dev)]
    loss = loss_on(rows)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    running += loss.item()
print("best val loss", best)
