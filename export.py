"""Save the trained model in a form the web page can load.

    python export.py /workspace/run/best.pt /workspace/tok web/

Writes weights.bin (16-bit floats), weights.json (where each tensor sits) and parity.json
(a test sketch with the PyTorch answer, so test/parity.mjs can prove the JavaScript model agrees).
"""
import json, sys
import numpy as np, torch
from model import SketchGPT
from suggest import next_shapes
from tokens import *

ckpt, data, out = sys.argv[1:4]
ck = torch.load(ckpt, map_location="cpu")
model = SketchGPT(**ck["cfg"]).eval()
model.load_state_dict(ck["state"])
model = model.half().float()          # round to what the page will load, so the parity check is exact

tensors, blob, offset = [], [], 0
for name, t in model.state_dict().items():
    arr = t.numpy().astype(np.float16)
    tensors.append({"name": name, "shape": list(arr.shape), "offset": offset})
    blob.append(arr.tobytes()); offset += arr.size
open(f"{out}/weights.bin", "wb").write(b"".join(blob))
json.dump({"cfg": ck["cfg"], "vocab": VOCAB, "block": BLOCK, "step": ck["step"], "tensors": tensors},
          open(f"{out}/weights.json", "w"))

rows = np.load(f"{data}/test.npy")
cases = []
for row in rows[:400:40]:
    ents = split_entities(row)
    k = max(1, len(ents) // 2)
    n = 1 + sum(1 + len(c) for _, c in ents[:k])
    prefix = torch.from_numpy(row[:n].astype(np.int64))[None]
    with torch.no_grad():
        logits = model(prefix)[0, -1]
    cases.append({"prefix": [int(t) for t in row[:n]], "logits": [round(float(x), 4) for x in logits],
                  "suggestions": [{"p": round(p, 5), "tokens": toks} for p, toks in next_shapes(model, prefix, 5)[0]]})
json.dump(cases, open(f"{out}/parity.json", "w"))
print("wrote", offset * 2 // 1024, "KB,", len(tensors), "tensors,", len(cases), "parity cases")
