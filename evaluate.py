"""Score the model on sketches it never saw.

    python evaluate.py /workspace/tok /workspace/run/best.pt /workspace/run/eval.json

For every test sketch and every point part-way through it, hide the next shape and ask the model
for its best guesses. Reported:
  exact        the top guess is the very shape the person drew (every grid coordinate equal)
  top3         the right shape is among the best three guesses
  kind         the top guess is at least the right kind of shape
  near         the top guess is within one grid step on every coordinate
and, for an autocomplete that only speaks up when confident (probability above a bar):
  precision    of the suggestions shown, the share that were right
  recall       of all the steps, the share that got a right suggestion
"""
import json, sys, collections
import numpy as np, torch
from model import SketchGPT
from suggest import next_shapes
from tokens import *

data, ckpt, out = sys.argv[1:4]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
dev = torch.device("cuda")
ck = torch.load(ckpt, map_location=dev)
model = SketchGPT(**ck["cfg"]).to(dev).eval()
model.load_state_dict(ck["state"])
rows = np.load(f"{data}/test.npy")
rng = np.random.default_rng(0)
rows = rows[rng.permutation(len(rows))[:limit]]

# one case per (sketch, number of shapes already drawn); group by prefix length so batches are rectangular
cases = collections.defaultdict(list)
for row in rows:
    ents = split_entities(row)
    pos = 1
    for k, (t, c) in enumerate(ents):
        if k >= 1:
            cases[pos].append((row[:pos], (t, c), k))
        pos += 1 + len(c)

records = []      # (probability of top guess, exact, top3, kind, near, shapes already drawn, true kind)
for T, group in sorted(cases.items()):
    for i in range(0, len(group), 512):
        part = group[i:i + 512]
        prefix = torch.from_numpy(np.stack([p for p, _, _ in part]).astype(np.int64)).to(dev)
        for (_, truth, k), guesses in zip(part, next_shapes(model, prefix, width=5)):
            want = canonical(truth)
            got = [(p, canonical((g[0], [x - COORD0 for x in g[1:]])) if g[0] != EOS else None) for p, g in guesses]
            top_p, top = got[0]
            exact = top == want
            top3 = any(g == want for _, g in got[:3])
            kind = top is not None and top[0] == want[0]
            near = kind and all(abs(x - y) <= 1 for x, y in zip(top[1], want[1]))
            records.append((top_p, exact, top3, kind, near, k, base_type(truth[0])))

r = np.array([(p, e, t3, kd, nr, k, bt) for p, e, t3, kd, nr, k, bt in records], dtype=np.float64)
n = len(r)
res = {"test_sketches": int(len(rows)), "steps_scored": n, "model_step": ck["step"], "val_loss": ck["val_loss"],
       "exact": r[:, 1].mean(), "top3": r[:, 2].mean(), "kind": r[:, 3].mean(), "near": r[:, 4].mean()}
res["by_shapes_drawn"] = {int(k): {"steps": int((r[:, 5] == k).sum()), "exact": r[r[:, 5] == k, 1].mean(),
                                   "top3": r[r[:, 5] == k, 2].mean()} for k in sorted(set(r[:, 5]))}
res["by_kind"] = {NAMES[int(b)]: {"steps": int((r[:, 6] == b).sum()), "exact": r[r[:, 6] == b, 1].mean(),
                                  "top3": r[r[:, 6] == b, 2].mean()} for b in sorted(set(r[:, 6]))}
res["confidence_bar"] = []
for bar in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    shown = r[:, 0] >= bar
    res["confidence_bar"].append({"bar": bar, "shown_share": shown.mean(),
                                  "precision": r[shown, 1].mean() if shown.any() else None,
                                  "recall": (r[:, 1] * shown).sum() / n})
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if not isinstance(v, (dict, list))}, indent=1))
for c in res["confidence_bar"]:
    print(c)
