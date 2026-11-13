"""Train the next-token model.

    python train.py /workspace/tok /workspace/run --steps 60000
"""
import argparse, json, math, time, os
import numpy as np, torch, torch.nn.functional as F
from model import SketchGPT
from tokens import *

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
ap.add_argument("--move", type=float, default=0.8, help="share of training sketches that are randomly shrunk and shifted")
ap.add_argument("--min-scale", type=float, default=0.45)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(0)
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

train = torch.from_numpy(np.load(f"{a.data}/train.npy")).to(dev)
val = torch.from_numpy(np.load(f"{a.data}/validation.npy")).to(dev)[:50000]
print("train rows", len(train), "val rows", len(val), flush=True)



def coordinate_roles(rows):
    """For every token: 0 = not a coordinate, 1 = an x, 2 = a y, 3 = a circle's radius."""
    need = torch.zeros(VOCAB, dtype=torch.long, device=dev)
    for t in range(FIRST_TYPE, LAST_TYPE + 1):
        need[t] = n_coords(t)
    N, T = rows.shape
    role = torch.zeros((N, T), dtype=torch.uint8, device=dev)
    left = torch.zeros(N, dtype=torch.long, device=dev); idx = left.clone(); circle = torch.zeros(N, dtype=torch.bool, device=dev)
    for t in range(T):
        tok = rows[:, t].long()
        is_type = (tok >= FIRST_TYPE) & (tok <= LAST_TYPE)
        coord = (left > 0) & (tok >= COORD0)
        role[:, t] = torch.where(coord, torch.where(circle & (idx == 2), 3, 1 + idx % 2), 0).to(torch.uint8)
        circle = torch.where(is_type, (tok == CIRCLE) | (tok == CIRCLE + CONSTRUCTION), circle)
        idx = torch.where(is_type, 0, torch.where(coord, idx + 1, idx))
        left = torch.where(is_type, need[tok], torch.where(coord, left - 1, left))
    return role


def move(rows, role, gen=None):
    """Shrink each sketch by a random amount and slide it to a random spot on the grid.
    Without this every training sketch fills the whole grid, and the model learns to expect that."""
    B = rows.shape[0]
    u = lambda: torch.rand((B, 1), device=dev, generator=gen)
    s = a.min_scale + (1 - a.min_scale) * u()
    s = torch.where(u() < a.move, s, torch.ones_like(s))
    ox, oy = u() * (1 - s) * (BINS - 1), u() * (1 - s) * (BINS - 1)
    v = (rows - COORD0).float()
    new = torch.where(role == 1, v * s + ox, torch.where(role == 2, v * s + oy, (v * s).clamp(min=1)))
    return torch.where(role > 0, new.round().long().clamp(0, BINS - 1) + COORD0, rows)


train_role = coordinate_roles(train)
gen = torch.Generator(device=dev); gen.manual_seed(1)
val_moved = torch.cat([move(val[i:i + 5000].long(), coordinate_roles(val[i:i + 5000]), gen) for i in range(0, len(val), 5000)])

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
def val_loss(rows):
    model.eval()
    tot = sum(loss_on(rows[i:i + 2000]).item() for i in range(0, len(rows), 2000))
    model.train()
    return tot / math.ceil(len(rows) / 2000)


log, best, t0, running = [], 9e9, time.time(), 0.0
for step in range(a.steps + 1):
    if step % a.eval_every == 0:
        v, vm = val_loss(val), val_loss(val_moved)       # as stored (fills the grid), and shrunk-and-shifted
        entry = {"step": step, "train_loss": round(running / a.eval_every, 4) if step else None,
                 "val_loss": round(v, 4), "val_loss_moved": round(vm, 4), "minutes": round((time.time() - t0) / 60, 1)}
        log.append(entry); print(json.dumps(entry), flush=True); running = 0.0
        json.dump(log, open(f"{a.out}/log.json", "w"))
        if vm < best:
            best = vm
            torch.save({"cfg": model.cfg, "state": model.state_dict(), "step": step, "val_loss": v}, f"{a.out}/best.pt")
    if step == a.steps:
        break
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    pick = torch.randint(len(train), (a.batch,), device=dev)
    rows = move(train[pick].long(), train_role[pick]) if a.move > 0 else train[pick]
    loss = loss_on(rows)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    running += loss.item()
print("best val loss", best)
