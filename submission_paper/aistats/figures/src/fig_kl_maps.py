"""Figures: per-cell KL and posterior uncertainty of the final fits.

fig_kl_maps   rows KL_z, KL_w, mean posterior sd of z, of w; columns the four
              sections; seed 0. Colour is the cell's percentile within its
              section (target cells), so one scale serves every panel and the
              top decile is the darkest band; Unassigned cells grey. Cells are
              drawn in increasing order, so high values lie on top.
fig_kl_seeds  KL_w percentile maps of the three seeds, ovarian FFPE and the TMA
              core: which cells deviate is not reproducible between seeds.
fig_kl_hist   the magnitudes: histograms on a log axis, three seeds per section.

Reads data/datasets/<ds>/experiments/kl_maps_finalL_s{k}.npz
(discell.experiments.kl_maps). Run from the repository root.
"""
from pathlib import Path
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

SECTIONS = [("xenium_prime_ovarian_cancer_ffpe", "Ovarian cancer (FFPE)"),
            ("xenium_prime_human_lung_cancer_ffpe", "Lung cancer (FFPE)"),
            ("xenium_prime_human_ovary_ff", "Ovarian cancer (FF)"),
            ("gse315411_pdltma06_11_prime_solo", "Lung TMA core")]
QUANT = [("kl_z", r"KL of $\mathbf{z}$"), ("kl_w", r"KL of $\mathbf{w}$"),
         ("sigma_z", r"sd of $q(\mathbf{z})$"), ("nu_w", r"sd of $q(\mathbf{w})$")]
SECTION_COLOURS = [S.BLUE, S.ORANGE, S.AQUA, "#eda100"]   # validated categorical slots 1-4
UNASSIGNED = "#c9c7bf"
# pale for percentiles 0-90, dark for the top decile: the question is where the top 10% lie
CMAP = LinearSegmentedColormap.from_list(
    "decile", [(0.0, "#f1f5fb"), (0.899, "#9ec5f4"), (0.9, "#256abf"), (1.0, "#0d366b")])
OUT = Path(__file__).resolve().parent.parent
MPP = 0.2125


def load(ds, seed):
    return np.load(f"data/datasets/{ds}/experiments/kl_maps_finalL_s{seed}.npz")


def targets(d):
    names = [str(x) for x in d["type_names"]]
    return d["t"] != names.index("Unassigned") if "Unassigned" in names else np.ones(len(d["t"]), bool)


def percentile(values, keep):
    out = np.full(len(values), np.nan)
    out[keep] = (rankdata(values[keep]) - 0.5) / keep.sum() * 100
    return out


def size_for(n):
    return 0.012 if n > 500_000 else 0.03 if n > 150_000 else 0.12


def map_panel(ax, xy, pct, keep):
    ax.scatter(xy[~keep, 0], xy[~keep, 1], s=size_for(len(xy)), c=UNASSIGNED, lw=0, rasterized=True)
    order = np.where(keep)[0][np.argsort(pct[keep])]
    ax.scatter(xy[order, 0], xy[order, 1], s=size_for(len(xy)), c=pct[order], cmap=CMAP,
               vmin=0, vmax=100, lw=0, rasterized=True)
    x0, y0 = xy.min(0); x1, y1 = xy.max(0)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); S.tissue_axes(ax)


def grid(fig, aspects, n_rows, top, H, gap_y, left, gap_x=0.14):
    """Axes grid with one panel height and aspect-matched widths (inches)."""
    W, Hf = fig.get_size_inches()
    spare = W - left - sum(aspects) * H - gap_x * (len(aspects) - 1)
    axes = []
    for r in range(n_rows):
        x = left + spare / 2
        row = []
        for a in aspects:
            y = Hf - top - (r + 1) * H - r * gap_y
            row.append(fig.add_axes([x / W, y / Hf, a * H / W, H / Hf]))
            x += a * H + gap_x
        axes.append(row)
    return axes


def bar(ax, xy):
    """1 mm bar just under the panel, outside the tissue."""
    x0, y0 = xy.min(0); x1, y1 = xy.max(0)
    y = y1 + 0.07 * (y1 - y0)
    ax.plot([x0, x0 + 1000 / MPP], [y, y], color=S.INK, lw=1.1, solid_capstyle="butt", clip_on=False)
    ax.text(x0 + 1000 / MPP + 0.03 * (x1 - x0), y, r"1\,mm", fontsize=6.5, color=S.INK, ha="left",
            va="center", clip_on=False)


def colourbar(fig, rect):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 100)), cax=cax,
                      orientation="horizontal")
    cb.set_ticks([0, 50, 90, 100]); cb.outline.set_visible(False)
    cax.tick_params(labelsize=6.5, length=2)
    cax.set_title("percentile within the section (target cells)", fontsize=6.5, color=S.INK2, pad=2)
    cax.axvline(90, color="white", lw=0.8)


def fig_maps():
    data = [load(ds, 0) for ds, _ in SECTIONS]
    aspects = [np.ptp(d["xy_px"][:, 0]) / np.ptp(d["xy_px"][:, 1]) for d in data]
    H, TOP, GAPY = 1.10, 0.24, 0.06
    figh = TOP + 4 * H + 3 * GAPY + 0.75
    fig = plt.figure(figsize=(S.TEXTWIDTH, figh))
    axes = grid(fig, aspects, 4, TOP, H, GAPY, left=0.42)
    for c, ((ds, title), d) in enumerate(zip(SECTIONS, data)):
        keep = targets(d)
        for r, (key, label) in enumerate(QUANT):
            ax = axes[r][c]
            map_panel(ax, d["xy_px"], percentile(d[key], keep), keep)
            if r == 0:
                ax.set_title(title, fontsize=7.5, color=S.INK, pad=3)
            if c == 0:
                ax.text(-0.04, 0.5, label, transform=ax.transAxes, rotation=90, ha="right",
                        va="center", fontsize=7.5, color=S.INK)
        bar(axes[3][c], d["xy_px"])
    colourbar(fig, [0.30, 0.16 / figh, 0.40, 0.07 / figh])
    fig.savefig(OUT / "fig_kl_maps.pdf"); fig.savefig(OUT / "fig_kl_maps.png", dpi=250)
    plt.close(fig)


def fig_seeds():
    rows = [SECTIONS[0], SECTIONS[3]]
    data = [[load(ds, s) for s in range(3)] for ds, _ in rows]
    aspects = [np.ptp(d[0]["xy_px"][:, 0]) / np.ptp(d[0]["xy_px"][:, 1]) for d in data]
    left, gap = 0.42, 0.14
    Hs = [min(1.55, (S.TEXTWIDTH - left - 0.05 - 2 * gap) / (3 * a)) for a in aspects]
    figh = 0.22 + sum(Hs) + 0.45 + 0.60
    fig = plt.figure(figsize=(S.TEXTWIDTH, figh))
    for r, ((ds, title), seeds) in enumerate(zip(rows, data)):
        ax_row = grid(fig, [aspects[r]] * 3, 1, 0.22 + sum(Hs[:r]) + r * 0.45, Hs[r], 0, left=left)[0]
        for s, (ax, d) in enumerate(zip(ax_row, seeds)):
            keep = targets(d)
            map_panel(ax, d["xy_px"], percentile(d["kl_w"], keep), keep)
            ax.set_title(f"seed {s}", fontsize=7, color=S.INK2, pad=2)
        ax_row[0].text(-0.04, 0.5, title, transform=ax_row[0].transAxes, rotation=90, ha="right",
                       va="center", fontsize=7.5, color=S.INK)
        bar(ax_row[0], seeds[0]["xy_px"])
    colourbar(fig, [0.30, 0.18 / figh, 0.40, 0.07 / figh])
    fig.savefig(OUT / "fig_kl_seeds.pdf"); fig.savefig(OUT / "fig_kl_seeds.png", dpi=250)
    plt.close(fig)


def fig_hist():
    logs = {key: [] for key, _ in QUANT}
    for c, (ds, title) in enumerate(SECTIONS):
        for sd in range(3):
            d = load(ds, sd)
            keep = targets(d)
            for key, _ in QUANT:
                v = d[key][keep].astype(np.float64)
                # the sd of q(w) sits at the prior's 1: a linear axis shows it plainly
                logs[key].append((c, sd, v if key == "nu_w" else np.log10(np.clip(v, 1e-8, None))))
    fig, axes = plt.subplots(1, 4, figsize=(S.TEXTWIDTH, 1.75))
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.36, top=0.88, wspace=0.28)
    for ax, (key, label) in zip(axes, QUANT):
        pooled = np.concatenate([v for _, _, v in logs[key]])
        lo, hi = np.percentile(pooled, [0.1, 99.9])
        pad = 0.05 * (hi - lo)
        bins = np.linspace(lo - pad, hi + pad, 81)
        for c, sd, v in logs[key]:
            h, e = np.histogram(v, bins=bins, density=True)
            ax.plot(0.5 * (e[1:] + e[:-1]), h, color=SECTION_COLOURS[c], lw=0.9,
                    alpha=1.0 if sd == 0 else 0.55, ls="-" if sd == 0 else (0, (3, 1.5)))
        ax.set_title(label, fontsize=7.5, color=S.INK, pad=3)
        ax.set_xlabel({"kl_z": r"$\log_{10}$ nats", "kl_w": r"$\log_{10}$ nats",
                       "sigma_z": r"$\log_{10}$ sd", "nu_w": "sd"}[key], fontsize=7)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    axes[0].set_ylabel("density", fontsize=7)
    keys = [Line2D([], [], color=col, lw=1.2, label=t) for (_, t), col in zip(SECTIONS, SECTION_COLOURS)]
    keys += [Line2D([], [], color=S.INK2, lw=0.9, label="seed 0"),
             Line2D([], [], color=S.INK2, lw=0.9, ls=(0, (3, 1.5)), alpha=0.55, label="seeds 1, 2")]
    fig.legend(handles=keys, loc="lower center", ncol=6, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.8, columnspacing=1.0)
    fig.savefig(OUT / "fig_kl_hist.pdf"); fig.savefig(OUT / "fig_kl_hist.png", dpi=250)
    plt.close(fig)


S.apply()
fig_maps()
fig_seeds()
fig_hist()
print("wrote fig_kl_maps, fig_kl_seeds, fig_kl_hist")
