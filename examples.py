"""Picture of example guesses for the page and the README.

    python examples.py TOKENS_DIR RUN_DIR/best.pt docs/examples.png [min_confidence=0.5] [count=9]

Takes test sketches in a random order, hides the next shape at a random point, and keeps the first `count` cases where the
model's top guess had at least `min_confidence` probability: the cases an autocomplete that only speaks up when sure would
show. Each example is two panels on the sketch so far (black): the guess (dashed blue), and beside it what the person
really drew (green when it is the same shape, red when it is not).
"""
import sys
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as CirclePatch
from model import SketchGPT
from suggest import next_shapes
from tokens import *

data, ckpt, out = sys.argv[1:4]
bar = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
count = int(sys.argv[5]) if len(sys.argv) > 5 else 9
INK, HELP, BLUE, GREEN, RED = "#15181d", "#9aa1ad", "#1769e0", "#1a9a55", "#d43d3d"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def circle_through(a, b, c):
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-6:
        return None
    A, B, C = a[0] ** 2 + a[1] ** 2, b[0] ** 2 + b[1] ** 2, c[0] ** 2 + c[1] ** 2
    x = (A * (b[1] - c[1]) + B * (c[1] - a[1]) + C * (a[1] - b[1])) / d
    y = (A * (c[0] - b[0]) + B * (a[0] - c[0]) + C * (b[0] - a[0])) / d
    return x, y, float(np.hypot(a[0] - x, a[1] - y))


def draw(ax, ent, color, lw=2.0, ghost=False):
    t, c = ent
    b = base_type(t)
    ls = (0, (4, 3)) if ghost else ((0, (1, 3)) if t > POINT else "-")
    if b == LINE:
        ax.plot([c[0], c[2]], [c[1], c[3]], color=color, lw=lw, ls=ls, solid_capstyle="round")
    elif b == CIRCLE:
        ax.add_patch(CirclePatch((c[0], c[1]), c[2], fill=False, color=color, lw=lw, ls=ls))
    elif b == POINT:
        ax.plot([c[0]], [c[1]], "o", color=color, ms=7 if ghost or lw > 2 else 4)
    else:
        circ = circle_through(c[0:2], c[2:4], c[4:6])
        if not circ:
            ax.plot(c[0::2], c[1::2], color=color, lw=lw, ls=ls); return
        x, y, r = circ
        a0, am, a1 = (np.arctan2(c[i + 1] - y, c[i] - x) for i in (0, 2, 4))
        tau = 2 * np.pi
        sweep = (a1 - a0) % tau if (am - a0) % tau <= (a1 - a0) % tau else -((a0 - a1) % tau)
        ang = a0 + sweep * np.linspace(0, 1, 40)
        ax.plot(x + r * np.cos(ang), y + r * np.sin(ang), color=color, lw=lw, ls=ls)


ck = torch.load(ckpt, map_location=dev)
model = SketchGPT(**ck["cfg"]).to(dev).eval(); model.load_state_dict(ck["state"])
rows = np.load(f"{data}/test.npy")
rng = np.random.default_rng(7)
picks, tried, right = [], 0, 0
for row in rows[rng.permutation(len(rows))]:
    ents = split_entities(row)
    if not 4 <= len(ents) <= 12:
        continue
    k = int(rng.integers(2, len(ents)))
    n = 1 + sum(1 + len(c) for _, c in ents[:k])
    p, guess = next_shapes(model, torch.from_numpy(row[:n].astype(np.int64))[None].to(dev), 5)[0][0]
    tried += 1
    if p < bar or guess[0] == EOS:
        continue
    got = (guess[0], [x - COORD0 for x in guess[1:]])
    ok = canonical(got) == canonical(ents[k])
    right += ok
    picks.append((ents, k, got, ok, p))
    if len(picks) == count:
        break
# Each example is a pair of panels: the model's guess on the left, what the person really drew on the right.
per_row = 3
n_rows = -(-len(picks) // per_row)
fig = plt.figure(figsize=(4.6 * per_row, 2.55 * n_rows))
grid = fig.add_gridspec(n_rows, per_row * 3 - 1, width_ratios=([1, 1, 0.22] * per_row)[:-1])


def panel(ax, ents, k, shape, color, ghost, title, title_color):
    ax.set_xlim(-3, 66); ax.set_ylim(-3, 66); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d9dde3")
    for e in ents[:k]:
        draw(ax, e, HELP if e[0] > POINT else INK)
    draw(ax, shape, color, lw=3.5, ghost=ghost)
    ax.set_title(title, fontsize=13, color=title_color, weight="bold")


for i, (ents, k, got, ok, p) in enumerate(picks):
    r, c = divmod(i, per_row)
    panel(fig.add_subplot(grid[r, c * 3]), ents, k, got, BLUE, True, "Model's guess", BLUE)
    panel(fig.add_subplot(grid[r, c * 3 + 1]), ents, k, ents[k], GREEN if ok else RED, False,
          "Person drew  ✓ same" if ok else "Person drew  ✗ different", GREEN if ok else RED)
fig.tight_layout()
fig.savefig(out, dpi=110); plt.close(fig)
print(f"{right} of {len(picks)} shown guesses matched; {len(picks)} kept out of {tried} cases tried (bar {bar})")
