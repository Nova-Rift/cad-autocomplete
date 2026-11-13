"""Pictures for the README: example guesses, the score bars, and the training curves.

    python viz.py /workspace/tok /workspace/run docs/
"""
import json, sys
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc as ArcPatch, Circle as CirclePatch
from model import SketchGPT
from suggest import next_shapes
from tokens import *

data, run, out = sys.argv[1:4]
INK, HELP, BLUE, GREEN, RED = "#15181d", "#9aa1ad", "#1769e0", "#1a9a55", "#d43d3d"


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
        ax.plot([c[0]], [c[1]], "o", color=color, ms=4)
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


def frame(ax):
    ax.set_xlim(-3, 66); ax.set_ylim(-3, 66); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d9dde3")


# ---- example guesses: see examples.py (python examples.py TOKENS RUN/best.pt docs/examples.png) ----

# ---- score bars ----
ev = json.load(open(f"{run}/eval.json"))
labels = ["Top guess is\nexactly right", "Right shape is in\nthe top 3 guesses", "Top guess is the\nright kind of shape"]
vals = [ev["exact"], ev["top3"], ev["kind"]]
fig, ax = plt.subplots(figsize=(5, 3.2))
bars = ax.bar(labels, [100 * v for v in vals], color=[BLUE, "#5aa2ff", "#a9cdff"], width=0.6)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{100 * v:.0f}%", ha="center", fontsize=20, weight="bold")
ax.set_ylim(0, 110); ax.set_yticks([]); ax.tick_params(axis="x", labelsize=11.5, length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{out}/scores.png", dpi=110); plt.close(fig)

# ---- precision / recall as the confidence bar rises ----
cb = [c for c in ev["confidence_bar"] if c["precision"] is not None]
fig, ax = plt.subplots(figsize=(5, 3.2))
x = [100 * c["bar"] for c in cb]
ax.plot(x, [100 * c["precision"] for c in cb], "-o", color=BLUE, lw=2.5)
ax.plot(x, [100 * c["recall"] for c in cb], "-o", color=GREEN, lw=2.5)
ax.text(x[-1] + 1.5, 100 * cb[-1]["precision"], "precision", color=BLUE, va="center", fontsize=12, weight="bold")
ax.text(x[-1] + 1.5, 100 * cb[-1]["recall"], "recall", color=GREEN, va="center", fontsize=12, weight="bold")
ax.set_xlabel("only suggest when the model is at least this sure (%)"); ax.set_ylim(0, 105); ax.set_xlim(-2, 112)
ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{out}/precision_recall.png", dpi=110); plt.close(fig)

# ---- training curves ----
# Both lines are measured on shrunk-and-shifted sketches, as the model is trained (val_loss_moved), so they compare like with like.
log = [e for e in json.load(open(f"{run}/log.json")) if e["train_loss"] is not None]
key = "val_loss_moved" if "val_loss_moved" in log[0] else "val_loss"
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.plot([e["step"] for e in log], [e["train_loss"] for e in log], color=INK, lw=2.5)
ax.plot([e["step"] for e in log], [e[key] for e in log], color=BLUE, lw=2.5)
ax.text(log[-1]["step"], log[-1]["train_loss"] - 0.02, "training sketches", color=INK, ha="right", va="top", fontsize=14, weight="bold")
ax.text(log[-1]["step"], log[-1][key] + 0.03, "unseen sketches", color=BLUE, ha="right", va="bottom", fontsize=14, weight="bold")
ax.set_xlabel("training steps", fontsize=13); ax.set_ylabel("error (lower is better)", fontsize=13); ax.tick_params(labelsize=11)
ax.set_xticks([0, 20000, 40000, 60000]); ax.set_xticklabels(["0", "20,000", "40,000", "60,000"])
lo = min(min(e["train_loss"], e[key]) for e in log)
ax.set_ylim(lo - 0.08, lo + 0.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{out}/training.png", dpi=110); plt.close(fig)
print("pictures written to", out)
