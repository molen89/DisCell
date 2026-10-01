"""Figure fig:sensitivity: how far each sensitivity arm moves each readout, in seed SDs.

A forest plot. One row per arm (the leakage rate's form, the fixed false-positive
floor, the weight of the response deviation, the adversary's capacity and weight),
grouped by section; one column per readout (held-out reconstruction, NMI, mirror R^2,
I(niche; w), the MLP probe's composition share, transport's fraction of the ceiling,
trusted tier). The point is the arm's move from the final configuration in units of
the final configuration's standard deviation over seeds; a zero line and a +/-2 SD
band. The shared axis is wide enough that the largest move, the fixed false-positive
floor's transport move on the primary section, is drawn, not clipped; moves beyond 5
SD carry their value.

Every value is a cell of tab:sensitivity: the figure calls that table's builder in
scripts/paper_tables.py and plots the moves it traced (its checks run first), so the
two cannot disagree. Arm labels are the table's, from the sensitivity files' flags.

Outputs ../fig_sensitivity.pdf (+ .png). Run from the repository root.
"""
from pathlib import Path
import json
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.transforms import blended_transform_factory as blend

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

PT = S.paper_tables()
OUT = Path(__file__).resolve().parent.parent / "fig_sensitivity"
# short titles, with the side of zero on which a move is an improvement
COLS = [("Recon.", "reconstruction", True), ("NMI", r"NMI of $\mathbf{z}$", True),
        ("Mirror $R^2$", r"mirror $R^2$", False),
        ("$\\I(\\text{niche}; \\vw)$", r"$I(\mathrm{niche};\mathbf{w})$", True),
        ("MLP comp.", "composition\nleakage", False), ("Transport", "transport", True)]
FAMILY = {"kappa_form": "form of the leakage rate", "fp_floor": "fixed false-positive floor",
          "alpha_w": "weight of the response deviation", "adversary": "adversary capacity, weight"}

_, tr = PT.table_sensitivity("figures/src/fig_sensitivity.py")
flags = {f: json.loads((PT.DATA / PT.OVARIAN / "experiments" / f"sensF_{f}.json").read_text())["arms"]
         for f in PT.SENS_FAMILIES}
short_to_ds = {v: k for k, v in PT.SHORT.items()}

# rows in the table's order: (kind, text, moves) with kind section | family | arm
rows, seen = [], set()
for c in tr.cells:
    parts = c["row"].split(" ")
    if c["row"].endswith("control") or c["row"].endswith("range"):
        continue
    # "<section> <family> <arm>": the family is a fixed token; arm keys may hold spaces
    n = next((i for i, p in enumerate(parts) if p in PT.SENS_FAMILIES), None)
    sec = " ".join(parts[:n]) if n else None
    if sec not in short_to_ds:
        raise SystemExit(f"unparsed row {c['row']!r}")
    f, arm = parts[n], " ".join(parts[n + 1:])
    key = (sec, f, arm)
    if key not in seen:
        seen.add(key)
        if not any(r[0] == "section" and r[1] == sec for r in rows):
            rows.append(("section", sec, None))
        if not any(r[0] == "family" and r[2] == (sec, f) for r in rows):
            rows.append(("family", FAMILY[f], (sec, f)))
        ds = short_to_ds[sec]
        rows.append(("arm", PT._arm_label(f, arm, flags[f][ds][arm]), {}))
    rows[-1][2][c["col"]] = c["values"][1]   # values = [arm mean, move in control sd]

S.apply()
plt.rcParams["text.latex.preamble"] += r"\newcommand{\alphaw}{\alpha_w}\newcommand{\lbar}{\bar{\ell}}"
H = {"section": 1.25, "family": 0.95, "arm": 1.0}
ys, y = [], 0.0
for kind, *_ in rows:
    y -= H[kind]
    ys.append(y)
XL = (-9.6, 7.6)
fig, axes = plt.subplots(1, len(COLS), figsize=(S.TEXTWIDTH, 3.9), sharey=True)
fig.subplots_adjust(left=0.205, right=0.995, bottom=0.085, top=0.935, wspace=0.10)
for ax, (col, title, up) in zip(axes, COLS):
    ax.axvspan(-2, 2, color=S.GHOST, lw=0, zorder=0)
    ax.axvline(0, color=S.MUTED, lw=0.6, zorder=1)
    for (kind, text, mv), yy in zip(rows, ys):
        if kind != "arm":
            ax.axhline(yy + 0.55 * H[kind], color="#e4e3dd", lw=0.4, zorder=0) if kind == "section" else None
            continue
        if col not in mv:
            ax.text(0, yy, "--", ha="center", va="center", fontsize=6, color=S.MUTED)
            continue
        v = mv[col]
        out = abs(v) > 2
        ax.plot([0, v], [yy, yy], color=S.BLUE_RAMP[1] if out else "#d6d5cf", lw=0.8, zorder=2)
        ax.scatter([v], [yy], s=11, zorder=3, lw=0.8,
                   facecolor=S.BLUE if out else "white", edgecolor=S.BLUE if out else S.INK2)
        if abs(v) > 5:
            ax.text(v + (0.5 if v < 0 else -0.5), yy + 0.12, rf"${v:.1f}$",
                    ha="left" if v < 0 else "right", va="bottom", fontsize=5.5, color=S.INK)
    # the band named in the panel, above the first row
    ax.text(0, ys[0] + 0.62, r"$\pm 2$ SD", ha="center", va="bottom", fontsize=5.6, color=S.INK2, zorder=5,
            bbox=dict(boxstyle="square,pad=0.15", facecolor=S.GHOST, edgecolor="none"))
    ax.set_xlim(*XL)
    ax.set_xticks([-8, -4, -2, 0, 2, 4])
    ax.set_xticklabels(["$-8$", "$-4$", "", "0", "", "4"])
    ax.tick_params(axis="x", labelsize=6)
    ax.set_title(title + (S.RIGHT if up else S.LEFT), fontsize=7, color=S.INK, pad=3, linespacing=1.05)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
ax0 = axes[0]
ax0.set_ylim(ys[-1] - 0.8, 0.2)
ax0.set_yticks([yy for (k, *_), yy in zip(rows, ys) if k == "arm"])
ax0.set_yticklabels([t for k, t, _ in rows if k == "arm"], fontsize=6.3)
for (kind, text, _), yy in zip(rows, ys):
    if kind == "section":
        fig.text(0.006, yy, rf"\textbf{{{PT.LONG[short_to_ds[text]]}}}", transform=blend(fig.transFigure, ax0.transData),
                 ha="left", va="center", fontsize=6.8, color=S.INK)
    elif kind == "family":
        fig.text(0.016, yy, rf"\emph{{{text}}}", transform=blend(fig.transFigure, ax0.transData),
                 ha="left", va="center", fontsize=6.3, color=S.INK2)
fig.text(0.605, 0.006, r"move from the final configuration, in its seed standard deviations (band: $\pm 2$)",
         ha="center", fontsize=7, color=S.INK)
S.save(fig, OUT)
