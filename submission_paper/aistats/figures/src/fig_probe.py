"""Figure: how the held-out probe grades the invariance of z (eq:probe), schematic.

(a) The protocol: the probe and the type-only baseline are fitted on training
tiles and scored on held-out tiles. (b) One component of one block for held-out
cells of one type: the baseline predicts the type mean, the probe predicts from
(mu_z, t); G_k is half the log ratio of their mean squared errors, and
exp(2 G_k) - 1 the share of within-type variance the probe explains. (c) Reading
G_b per block: its excess over the within-type permutation floor, and that excess
as a fraction of the uncontrolled (alpha_a = 0) fit's. All values are synthetic and
chosen for legibility. Outputs ../fig_probe.pdf (+ .png). Run from the repository root.
"""
from pathlib import Path
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "fig_probe"
S.apply()
fig = plt.figure(figsize=(S.TEXTWIDTH, 2.45))
axa = fig.add_axes([0.02, 0.06, 0.29, 0.82])
axb = fig.add_axes([0.40, 0.19, 0.25, 0.69])
axc = fig.add_axes([0.715, 0.19, 0.275, 0.69])

# ---- (a) protocol: tiles, fit on training tiles, score on held-out ones
held = {(1, 0), (3, 2), (0, 2)}
for gx in range(4):
    for gy in range(3):
        h = (gx, gy) in held
        axa.add_patch(Rectangle((gx * 0.19, 0.50 + gy * 0.13), 0.18, 0.12, fc="white" if h else S.GHOST,
                                ec=S.INK2, lw=0.6, hatch="//////" if h else None, zorder=2))
axa.text(0.38, 0.93, r"tiles of one section", fontsize=7, color=S.INK2, ha="center", va="bottom")
rows = [(0.36, S.GHOST, None, r"training tiles: fit the baseline and the probe"),
        (0.26, "white", "//////", r"held-out tiles: score both")]
for y, fc, hatch, text in rows:
    axa.add_patch(Rectangle((0.0, y - 0.035), 0.07, 0.07, fc=fc, ec=S.INK2, lw=0.6, hatch=hatch))
    axa.text(0.10, y, text, fontsize=7, color=S.INK, va="center")
axa.text(0.0, 0.14, r"baseline: $\bar v_k(t)$, the mean over cells of type $t$", fontsize=7, color=S.INK, va="center")
axa.text(0.0, 0.05, r"probe: $\hat v_k(\bm\mu_z, t)$, ridge or MLP", fontsize=7, color=S.INK, va="center")
axa.set_xlim(0, 1.0); axa.set_ylim(0, 1.0); axa.axis("off")
S.panel_label(axa, "a", x=0.0, y=1.0)

# ---- (b) one component, held-out cells of one type
rng = np.random.default_rng(3)
n = 220
signal = rng.normal(0, 1, n)
v = 0.30 + 0.035 * signal + 0.07 * rng.normal(0, 1, n)         # e.g. a composition component
pred = 0.30 + 0.035 * signal * 0.8 + 0.01 * rng.normal(0, 1, n) # what the probe reads from mu_z
mse_type = np.mean((v - v.mean()) ** 2)
mse_probe = np.mean((v - pred) ** 2)
g = 0.5 * np.log(mse_type / mse_probe)
axb.scatter(pred, v, s=5, color=S.INK, lw=0, alpha=0.7, zorder=3)
lo, hi = 0.16, 0.44
axb.plot([0.22, 0.38], [0.22, 0.38], color=S.BLUE, lw=1.1, zorder=2)
axb.axhline(v.mean(), color=S.MUTED, lw=1.0, ls=(0, (4, 2.5)), zorder=2)
axb.set_xlim(0.22, 0.38); axb.set_ylim(v.min() - 0.02, v.max() + 0.17)
axb.set_xlabel(r"probe prediction $\hat v_k$", labelpad=1.5)
axb.set_ylabel(r"component $v_k$ (held out)", labelpad=1.5)
axb.set_xticks([]); axb.set_yticks([])
axb.text(0.378, v.mean() - 0.01, r"baseline $\bar v_k(t)$", fontsize=6.5, color=S.INK2, ha="right", va="top",
         bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85), zorder=5)
axb.text(0.378, 0.372, r"probe $\hat v_k$", fontsize=6.5, color=S.INK, ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85), zorder=5)
axb.text(0.222, v.max() + 0.165,
         rf"$G_k = \tfrac12 \log \frac{{\mathrm{{MSE}}_{{\mathrm{{type}}}}}}{{\mathrm{{MSE}}_{{\mathrm{{probe}}}}}} = {g:.2f}$ nats"
         + "\n" + rf"$e^{{2G_k}} - 1 = {100 * (np.exp(2 * g) - 1):.0f}\,\%$ of within-type variance"
         + "\n" + r"illustrative values",
         fontsize=6.5, color=S.INK, ha="left", va="top", linespacing=1.3)
S.panel_label(axb, "b", x=-0.08, y=1.02)

# ---- (c) reading G_b: floor, this fit, uncontrolled fit
GUARD = 0.25                                   # the pass rule: excess at most this fraction of uncontrolled
blocks = [("composition", 0.004, 0.012, 0.030), ("image", 0.002, 0.005, 0.021)]
for r, (name, floor, fit, unc) in enumerate(blocks):
    y = 1 - r
    axc.plot([0, 0.034], [y, y], color=S.CELL_EDGE, lw=0.6, zorder=1)
    axc.plot([floor, fit], [y, y], color=S.BLUE, lw=3.0, solid_capstyle="butt", zorder=2, alpha=0.35)
    axc.plot([floor, unc], [y - 0.14, y - 0.14], color=S.INK2, lw=0.8, zorder=2)
    guard = floor + GUARD * (unc - floor)
    axc.plot([guard, guard], [y - 0.12, y + 0.12], color=S.INK, lw=1.0, ls=(0, (1.5, 1)), zorder=4)
    for x, mk, fc, ec in ((floor, "o", "white", S.INK2), (fit, "o", S.BLUE, S.BLUE), (unc, "s", S.INK2, S.INK2)):
        axc.scatter([x], [y], s=22, marker=mk, facecolor=fc, edgecolor=ec, lw=0.9, zorder=3)
    frac = (fit - floor) / (unc - floor)
    verdict = r"passes" if frac <= GUARD else r"fails"
    axc.text(0.0, y + 0.2, name, fontsize=7, color=S.INK, ha="left", va="bottom")
    axc.text(0.0, y - 0.3, rf"fraction of uncontrolled $= {frac:.2f}$: {verdict}", fontsize=6.5,
             color=S.INK2, ha="left", va="center")
axc.set_xlim(-0.001, 0.034); axc.set_ylim(-0.5, 1.65)
axc.set_yticks([]); axc.set_xticks([])
axc.spines["left"].set_visible(False)
axc.set_xlabel(r"$G_b$ (nats per component), illustrative", labelpad=1.5)
# key
kx, ky = 0.0, 1.58
for dx, mk, fc, ec, text in ((0.0, "o", "white", S.INK2, r"floor"), (0.0070, "o", S.BLUE, S.BLUE, r"this fit"),
                             (0.0155, "s", S.INK2, S.INK2, r"uncontrolled")):
    axc.scatter([kx + dx], [ky], s=18, marker=mk, facecolor=fc, edgecolor=ec, lw=0.9, clip_on=False)
    axc.text(kx + dx + 0.0012, ky, text, fontsize=6.5, color=S.INK, va="center")
axc.plot([0.0004, 0.0004], [ky - 0.2, ky - 0.08], color=S.INK, lw=1.0, ls=(0, (1.5, 1)), clip_on=False)
axc.text(0.0012, ky - 0.14, r"guard: $\tfrac14$ of uncontrolled", fontsize=6.5, color=S.INK, va="center")
S.panel_label(axc, "c", x=-0.04, y=1.02)

fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=250)
print("wrote", OUT.with_suffix(".pdf"), f"| G_k {g:.3f}, share {np.exp(2 * g) - 1:.2f}")
