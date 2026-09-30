"""Figure (candidate, real data): the held-out probe on an actual fit.

Same three panels as fig_probe.py, but (b) and (c) are measured, on the primary
section's reference fit at lineage labels (run finalL_s0, accepted checkpoint):
(b) one composition component -- the share of Tumour neighbours -- for held-out
fibroblasts: the ridge probe's prediction from (mu_z, t) against the value, with
the type mean as the baseline; G_k is that component's gain over all held-out
cells, rebuilt here with the probe's own code path, draws and seed, and checked
against the stored record. (c) Excess over the within-type permutation floor per
block and probe family, for this fit and for the uncontrolled fit (alpha_a = 0,
mean of its two seeds), from the stored re-grade record.

The assembly and the probe fit are cached in data/fig_probe_real.npz. Outputs
../fig_probe_real.pdf (+ .png). Run from the repository root.
"""
from pathlib import Path
import json
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

PT = S.paper_tables()

DS, RUN = "xenium_prime_ovarian_cancer_ffpe", "finalL_s0"
ROOT = Path(f"data/datasets/{DS}")
TARGET_NEIGHBOUR, SHOWN_TYPE = "Tumour", "Fibroblasts"
HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "fig_probe_real.npz"
OUT = HERE.parent / "fig_probe_real"

record = json.loads((ROOT / "runs" / RUN / "validation" / "probe_blocks.json").read_text())

if CACHE.exists():
    c = np.load(CACHE, allow_pickle=True)
    v_k, pred_k, base_k, t_test, names, k = (c["v_k"], c["pred_k"], c["base_k"], c["t_test"],
                                              list(c["names"]), int(c["k"]))
else:
    from discell.experiments.probe_regrade import Assemblies, run_config
    from discell.model import metrics as M
    cfg = run_config(DS, RUN)
    data = Assemblies().get(DS, cfg)
    mu_z = np.load(ROOT / "runs" / RUN / "validation" / "probe_latents.npz")["mu_z"]
    rows = np.concatenate(data.train_tiles + data.val_tiles)
    n_train = sum(len(tile) for tile in data.train_tiles)
    train = np.arange(len(rows)) < n_train
    z, t, v = mu_z[rows], data.t[rows], data.v_block[rows]
    # the probe's own draws, in its order: permutation 0, training subsample, test subsample
    rng = np.random.default_rng(cfg["seed"])
    M._permute_within_type(z, t, rng)
    train_rows = M._subsample_rows(np.flatnonzero(train), rng)
    test_rows = M._subsample_rows(np.flatnonzero(~train), rng)
    onehot = np.eye(int(t.max()) + 1, dtype=np.float64)
    design = lambda r: np.hstack([z[r], onehot[t[r]]])          # noqa: E731
    names = list(data.type_names)
    k = names.index(TARGET_NEIGHBOUR)
    assert k < data.n_comp, "the chosen neighbour type is the dropped composition column"
    pred_k = M._ridge_fit_predict(design(train_rows), v[train_rows][:, k], design(test_rows))
    v_k, base_k, t_test = v[test_rows][:, k], data.vbar_t[t[test_rows]][:, k], t[test_rows]
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, v_k=v_k, pred_k=pred_k, base_k=base_k, t_test=t_test, names=np.array(names, dtype=object), k=k)

g_k = 0.5 * np.log(np.mean((v_k - base_k) ** 2) / np.mean((v_k - pred_k) ** 2))
stored = record["ridge"]["comp"]["gain_per_col"][k]
# The rebuild reproduces the stored record to 0.4 % (0.003923 vs 0.003939 nats on 2026-09-28), not exactly;
# the cause is not located (the record dates from 2026-09-26). The shown value is the rebuilt one, which
# matches the plotted predictions; a larger gap would mean the rebuild is not the graded probe.
assert abs(g_k - stored) <= 0.01 * abs(stored), f"rebuilt gain {g_k} differs from the record's {stored}"

S.apply()
fig = plt.figure(figsize=(S.TEXTWIDTH, 2.45))
axa = fig.add_axes([0.02, 0.06, 0.28, 0.82])
axb = fig.add_axes([0.43, 0.19, 0.22, 0.69])
axc = fig.add_axes([0.72, 0.21, 0.265, 0.67])

# ---- (a) protocol, as in the schematic
held = {(1, 0), (3, 2), (0, 2)}
for gx in range(4):
    for gy in range(3):
        h = (gx, gy) in held
        axa.add_patch(Rectangle((gx * 0.19, 0.50 + gy * 0.13), 0.18, 0.12, fc="white" if h else S.GHOST,
                                ec=S.INK2, lw=0.6, hatch="//////" if h else None, zorder=2))
axa.text(0.38, 0.93, r"tiles of one section", fontsize=7, color=S.INK2, ha="center", va="bottom")
for y, fc, hatch, text in ((0.36, S.GHOST, None, r"training tiles: fit the baseline and the probe"),
                           (0.26, "white", "//////", r"held-out tiles: score both")):
    axa.add_patch(Rectangle((0.0, y - 0.035), 0.07, 0.07, fc=fc, ec=S.INK2, lw=0.6, hatch=hatch))
    axa.text(0.10, y, text, fontsize=7, color=S.INK, va="center")
axa.text(0.0, 0.14, r"baseline: $\bar v_k(t)$, the mean over cells of type $t$", fontsize=7, color=S.INK, va="center")
axa.text(0.0, 0.05, r"probe: $\hat v_k(\bm\mu_z, t)$, ridge or MLP", fontsize=7, color=S.INK, va="center")
axa.set_xlim(0, 1.0); axa.set_ylim(0, 1.0); axa.axis("off")
S.panel_label(axa, "a", x=0.0, y=1.0)

# ---- (b) measured: Tumour-neighbour share for held-out fibroblasts
shown = t_test == names.index(SHOWN_TYPE)
x, y = pred_k[shown], v_k[shown]
jitter = np.random.default_rng(0).uniform(-0.006, 0.006, y.size)   # shares are discrete; jitter for display only
axb.scatter(x, y + jitter, s=2.5, color=S.INK, lw=0, alpha=0.35, zorder=3, rasterized=True)
lo, hi = np.percentile(x, [0.5, 99.5])
axb.plot([lo, hi], [lo, hi], color=S.INK2, lw=0.7, ls=(0, (1, 1.2)), zorder=4)
axb.axhline(base_k[shown][0], color=S.MUTED, lw=1.0, ls=(0, (4, 2.5)), zorder=4)
edges = np.quantile(x, np.linspace(0, 1, 11))
idx = np.clip(np.digitize(x, edges[1:-1]), 0, 9)
axb.plot([x[idx == b].mean() for b in range(10)], [y[idx == b].mean() for b in range(10)], color=S.BLUE,
         lw=0.8, marker="o", ms=3, zorder=5)
axb.text(hi, 0.02, r"binned mean", fontsize=6.5, color=S.BLUE, ha="right", va="bottom", zorder=6,
         bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.9))
axb.set_xlim(lo, hi); axb.set_ylim(-0.03, 1.2)
axb.set_xlabel(r"probe prediction $\hat v_k$", labelpad=1.5)
axb.set_ylabel(rf"share of {TARGET_NEIGHBOUR} neighbours", labelpad=1.5)
axb.tick_params(labelsize=6.5)
axb.set_xticks([0.05, 0.10, 0.15])
axb.text(hi, base_k[shown][0] + 0.06, r"baseline", fontsize=6.5, color=S.INK2, ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85), zorder=5)
axb.text(lo + 0.002, 1.18, rf"held-out {SHOWN_TYPE.lower()}" + "\n"
         + rf"$G_k = {g_k:.4f}$ nats ({PT.probe_share(g_k):.1f}\,\%)",
         fontsize=6.5, color=S.INK, ha="left", va="top", linespacing=1.3)
S.panel_label(axb, "b", x=-0.1, y=1.02)

# ---- (c) measured: excess over the floor, this fit and the uncontrolled fit
rows_c = [("ridge", "comp", "composition, ridge"), ("ridge", "img", "image, ridge"),
          ("mlp", "comp", "composition, MLP"), ("mlp", "img", "image, MLP")]
xmax = 1.15 * max(record[f][b]["uncontrolled_excess"] for f, b, _ in rows_c)
for r, (fam, blk, label) in enumerate(rows_c):
    y = 3 - r
    ex, unc = record[fam][blk]["excess"], record[fam][blk]["uncontrolled_excess"]
    axc.plot([0, xmax], [y, y], color=S.CELL_EDGE, lw=0.6, zorder=1)
    axc.plot([0, ex], [y, y], color=S.BLUE, lw=3.0, solid_capstyle="butt", alpha=0.35, zorder=2)
    axc.scatter([0], [y], s=18, facecolor="white", edgecolor=S.INK2, lw=0.9, zorder=3)
    axc.scatter([ex], [y], s=18, color=S.BLUE, lw=0, zorder=3)
    axc.scatter([unc], [y], s=18, marker="s", color=S.INK2, lw=0, zorder=3)
    axc.text(0, y + 0.2, label, fontsize=6.5, color=S.INK, ha="left", va="bottom")
    axc.text(xmax, y - 0.12, rf"{ex / unc:.2f} of uncontrolled", fontsize=6.5, color=S.INK2, ha="right", va="top")
axc.set_xlim(-0.003, xmax); axc.set_ylim(-0.6, 4.2)
axc.set_yticks([]); axc.spines["left"].set_visible(False)
axc.tick_params(labelsize=6.5)
axc.set_xlabel(r"excess over the floor" + "\n" + r"(nats per component)", labelpad=1.5, linespacing=1.1)
for dx, mk, fc, ec, text in ((0.0, "o", "white", S.INK2, r"floor"), (0.30, "o", S.BLUE, S.BLUE, r"this fit"),
                             (0.60, "s", S.INK2, S.INK2, r"uncontrolled")):
    axc.scatter([dx * xmax], [4.05], s=16, marker=mk, facecolor=fc, edgecolor=ec, lw=0.9, clip_on=False)
    axc.text(dx * xmax + 0.03 * xmax, 4.05, text, fontsize=6.5, color=S.INK, va="center")
S.panel_label(axc, "c", x=-0.04, y=1.02)

fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=250)
print("wrote", OUT.with_suffix(".pdf"), f"| G_k {g_k:.5f} (record {stored:.5f}) | fibroblasts shown {shown.sum()}")
