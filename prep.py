"""Turn the SketchGraphs files into token rows.

    python prep.py /workspace/data /workspace/tok

Reads sg_t16_{train,validation,test}.npy, keeps the shapes (lines, circles, arcs, points) in the
order they were drawn, fits each sketch to a 64 x 64 grid, drops exact repeats, and removes from
validation/test any sketch that also appears in training (so the test is not leaked).
"""
import math, sys, os
from multiprocessing import Pool
import numpy as np
from tokens import *

SRC, DST = sys.argv[1], sys.argv[2]
CHUNK = 20000


def shape_points(label, p):
    """-> (type, defining points, extra points used only for the bounding box, radius)"""
    if label == 1:      # line
        a = (p["pntX"] + p["startParam"] * p["dirX"], p["pntY"] + p["startParam"] * p["dirY"])
        b = (p["pntX"] + p["endParam"] * p["dirX"], p["pntY"] + p["endParam"] * p["dirY"])
        return LINE, [a, b], [], 0.0
    if label == 2:      # circle
        cx, cy, r = p["xCenter"], p["yCenter"], p["radius"]
        return CIRCLE, [(cx, cy)], [(cx - r, cy - r), (cx + r, cy + r)], r
    if label == 6:      # arc
        cx, cy, r = p["xCenter"], p["yCenter"], p["radius"]
        base, sign = math.atan2(p["yDir"], p["xDir"]), (-1.0 if p["clockwise"] else 1.0)
        s, e = p["startParam"] % (2 * math.pi), p["endParam"] % (2 * math.pi)
        if s > e:
            e += 2 * math.pi
        at = lambda t: (cx + math.cos(base + sign * t) * r, cy + math.sin(base + sign * t) * r)
        pts = [at(s + (e - s) * k / 8) for k in range(9)]
        return ARC, [pts[0], pts[4], pts[8]], pts, r
    if label == 0:      # point
        return POINT, [(p["x"], p["y"])], [], 0.0
    return None


def tokenize(seq):
    shapes = []
    for op in seq:
        params = getattr(op, "parameters", None)
        if not hasattr(op, "references") and int(op.label) in (0, 1, 2, 6) and params:
            try:
                got = shape_points(int(op.label), params)
            except (KeyError, TypeError):
                return None
            shapes.append((got, bool(params.get("isConstruction", False))))
    if len(shapes) < 2:
        return None
    allp = np.array([q for (t, pts, extra, r), _ in shapes for q in pts + extra], dtype=np.float64)
    if not np.isfinite(allp).all():
        return None
    lo, hi = allp.min(0), allp.max(0)
    scale = float((hi - lo).max())
    if not scale > 1e-9:
        return None
    mid = (lo + hi) / 2
    q = lambda pt: [int(min(BINS - 1, max(0, round(((v - m) / scale + 0.5) * (BINS - 1))))) for v, m in zip(pt, mid)]
    row = [BOS]
    kept = 0
    for (t, pts, extra, r), construction in shapes:
        c = [v for pt in pts for v in q(pt)]
        if t == LINE and c[:2] == c[2:]:
            continue                                    # too small to see on the grid
        if t == ARC and (c[:2] == c[4:] or c[:2] == c[2:4] or c[2:4] == c[4:]):
            continue
        if t == CIRCLE:
            c.append(int(min(BINS - 1, max(1, round(r / scale * (BINS - 1))))))
        row.append(t + (CONSTRUCTION if construction else 0))
        row.extend(COORD0 + v for v in c)
        kept += 1
    if kept < 2:
        return None
    row.append(EOS)
    return row if len(row) <= BLOCK else None


def work(job):
    split, lo, hi = job
    from sketchgraphs.data import flat_array
    seqs = flat_array.load_dictionary_flat(np.load(f"{SRC}/sg_t16_{split}.npy", mmap_mode="r"))["sequences"]
    out = np.zeros((hi - lo, BLOCK), dtype=np.uint8)
    n = 0
    for i in range(lo, hi):
        row = tokenize(seqs[i])
        if row:
            out[n, :len(row)] = row
            n += 1
    return out[:n]


def unique_rows(a, skip=()):
    """Drop repeated rows (and any row in `skip`), keeping the first of each. Returns rows and their keys."""
    seen, keep = set(), []
    for i, r in enumerate(a):
        key = r.tobytes()
        if key not in seen and key not in skip:
            seen.add(key); keep.append(i)
    return a[keep], seen


if __name__ == "__main__":
    from sketchgraphs.data import flat_array
    os.makedirs(DST, exist_ok=True)
    report, train_keys = {}, None
    with Pool(16) as pool:      # the pod has ~18 processors to itself, whatever os.cpu_count() says
        for split in ("train", "validation", "test"):
            total = len(flat_array.load_dictionary_flat(np.load(f"{SRC}/sg_t16_{split}.npy", mmap_mode="r"))["sequences"])
            jobs = [(split, lo, min(total, lo + CHUNK)) for lo in range(0, total, CHUNK)]
            parts = []
            for n, part in enumerate(pool.imap(work, jobs, chunksize=1)):
                parts.append(part)
                if n % 50 == 0:
                    print(split, "chunk", n, "of", len(jobs), flush=True)
            rows = np.concatenate(parts)
            usable = len(rows)
            print(split, "usable", usable, "- removing repeats", flush=True)
            distinct = len(unique_rows(rows)[0]) if split != "train" else None
            rows, keys = unique_rows(rows, skip=train_keys or ())
            if split == "train":
                train_keys, distinct = keys, len(rows)
            np.save(f"{DST}/{split}.npy", rows)
            report[split] = {"sketches_in_file": total, "usable": usable, "distinct": distinct, "kept": len(rows)}
            print(split, report[split], flush=True)
    import json
    json.dump(report, open(f"{DST}/report.json", "w"), indent=1)
