"""Figures: the latents of the final fits, as UMAP and as PCA.

Columns: the four sections (seed 0, accepted checkpoint, posterior means from
data/datasets/<ds>/experiments/kl_maps_finalL_s0.npz). Rows:
  a  mu_z of all cells, coloured by the cell's own lineage family;
  b  mu_w of all cells, coloured by the family dominating its neighbours;
  c  mu_z within one type, coloured by the family dominating its neighbours;
  d  mu_w within the same type, coloured the same way.
The type of rows c-d is chosen by rule, not by eye: among types with >= 3,000
cells, the one whose dominant-neighbour family is most mixed (highest entropy).
Families are the display grouping of fig_sections; Unassigned is grey, and a
cell's dominant neighbour family ignores Unassigned neighbours (grey if none).
UMAP/PCA of a random subsample (30,000 cells for a-b, 10,000 of the type for
c-d); embeddings cached in data/fig_latents.npz. Outputs ../fig_latents_umap.pdf (+ .png);
the PCA version (draw(..., "pca")) was dropped from the paper on 2026-09-29. Run from the repository root.
"""
from pathlib import Path
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402
from fig_sections import FAMILIES, COLOURS, UNASSIGNED, family  # noqa: E402

SECTIONS = [("xenium_prime_ovarian_cancer_ffpe", "Ovarian cancer (FFPE)"),
            ("xenium_prime_human_lung_cancer_ffpe", "Lung cancer (FFPE)"),
            ("xenium_prime_human_ovary_ff", "Ovarian cancer (FF)"),
            ("gse315411_pdltma06_11_prime_solo", "Lung TMA core")]
N_ALL, N_TYPE, MIN_TYPE = 30_000, 10_000, 3_000
HERE = Path(__file__).resolve().parent
OUT, CACHE = HERE.parent, HERE / "data" / "fig_latents.npz"


def colours(fam):
    return np.array([COLOURS[f] if f >= 0 else UNASSIGNED for f in fam])


def dominant_neighbour_family(fam, edge_i, edge_j):
    n, k = len(fam), len(FAMILIES)
    counts = np.zeros((n, k))
    for a, b in ((edge_i, edge_j), (edge_j, edge_i)):
        ok = fam[b] >= 0
        np.add.at(counts, (a[ok], fam[b][ok]), 1)
    out = counts.argmax(1)
    out[counts.sum(1) == 0] = -1
    return out


def section(ds, rng):
    d = np.load(f"data/datasets/{ds}/experiments/kl_maps_finalL_s0.npz")
    names = [str(x) for x in d["type_names"]]
    t = d["t"]
    fam_of_type = np.array([family(nm) for nm in names])
    fam = fam_of_type[t]
    dom = dominant_neighbour_family(fam, d["edge_i"], d["edge_j"])
    target = np.array([not nm.startswith("Unassigned") for nm in names])[t]
    # rows c-d: the large type with the most mixed neighbourhoods
    best, best_h = None, -1.0
    for g, nm in enumerate(names):
        m = (t == g) & target & (dom >= 0)
        if m.sum() < MIN_TYPE:
            continue
        p = np.bincount(dom[m], minlength=len(FAMILIES)) / m.sum()
        h = -(p[p > 0] * np.log(p[p > 0])).sum()
        if h > best_h:
            best, best_h = g, h
    all_idx = rng.choice(len(t), size=min(N_ALL, len(t)), replace=False)
    members = np.where(t == best)[0]
    type_idx = rng.choice(members, size=min(N_TYPE, len(members)), replace=False)
    return dict(mu_z=d["mu_z"], mu_w=d["mu_w"], fam=fam, dom=dom, all_idx=all_idx,
                type_idx=type_idx, type_name=names[best], entropy=best_h)


def embed(x, method, seed=0):
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    if method == "pca":
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        return u[:, :2] * s[:2]
    import umap
    return umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=seed).fit_transform(x)


def embeddings(data):
    cache = dict(np.load(CACHE)) if CACHE.exists() else {}
    for c, sec in enumerate(data):
        for row, (key, idx) in enumerate([("mu_z", "all_idx"), ("mu_w", "all_idx"),
                                          ("mu_z", "type_idx"), ("mu_w", "type_idx")]):
            for method in ("umap", "pca"):
                name = f"{c}_{row}_{method}"
                if name not in cache or len(cache[name]) != len(sec[idx]):
                    print("embedding", SECTIONS[c][0], row, method, flush=True)
                    cache[name] = embed(sec[key][sec[idx]], method)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, **cache)
    return cache


def draw(data, cache, method):
    fig, axes = plt.subplots(4, 4, figsize=(S.TEXTWIDTH, 7.6))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.915, bottom=0.08, wspace=0.08, hspace=0.30)
    row_labels = [r"intrinsic state $\boldsymbol{\mu}_z$, all cells \quad colour: the cell's own type",
                  r"response $\boldsymbol{\mu}_w$, all cells \quad colour: the type dominating the cell's neighbours",
                  r"intrinsic state $\boldsymbol{\mu}_z$, cells of one type (named) \quad colour: the type dominating the cell's neighbours",
                  r"response $\boldsymbol{\mu}_w$, the same cells \quad colour: the type dominating the cell's neighbours"]
    for c, (sec, (_, title)) in enumerate(zip(data, SECTIONS)):
        for row in range(4):
            ax = axes[row][c]
            idx = sec["all_idx"] if row < 2 else sec["type_idx"]
            colour_by = sec["fam"] if row == 0 else sec["dom"]
            xy = cache[f"{c}_{row}_{method}"]
            col = colours(colour_by[idx])
            order = np.random.default_rng(1).permutation(len(idx))
            ax.scatter(xy[order, 0], xy[order, 1], s=0.6, c=col[order], lw=0, rasterized=True)
            lo, hi = np.percentile(xy, [0.5, 99.5], axis=0)   # a stray point would squeeze the panel
            pad = 0.04 * (hi - lo)
            ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if row == 0:
                box0 = ax.get_position()
                fig.text((box0.x0 + box0.x1) / 2, 0.975, title, fontsize=7.5, color=S.INK,
                         ha="center", va="bottom")
            if row == 2:
                ax.set_title(sec["type_name"], fontsize=6.5, color=S.INK2, pad=2)
    for row, letter in enumerate("abcd"):
        box = axes[row][0].get_position()
        y = box.y1 + (0.008 if row != 2 else 0.026)
        fig.text(0.02, y, rf"\textbf{{{letter}}}\enspace " + row_labels[row], fontsize=7.5, color=S.INK,
                 ha="left", va="bottom")
    keys = [Line2D([], [], marker="o", ls="", ms=3.5, color=col, label=name)
            for (name, _), col in zip(FAMILIES, COLOURS)]
    keys.append(Line2D([], [], marker="o", ls="", ms=3.5, color=UNASSIGNED, label="Unassigned or none"))
    fig.legend(handles=keys, loc="lower center", ncol=4, frameon=False, fontsize=6.5,
               handletextpad=0.2, columnspacing=0.7, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(OUT / f"fig_latents_{method}.pdf"); fig.savefig(OUT / f"fig_latents_{method}.png", dpi=250)
    plt.close(fig)


def draw_main(data, cache):
    """The primary section only, as a 2 x 2 grid at column width (main text)."""
    sec, c = data[0], 0
    fig, axes = plt.subplots(2, 2, figsize=(S.COLWIDTH, 3.55))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.14, wspace=0.05, hspace=0.22)
    tname = sec["type_name"]
    heads = [r"\textbf{a}\enspace $\boldsymbol{\mu}_z$, all cells: own type",
             r"\textbf{b}\enspace $\boldsymbol{\mu}_w$, all cells: neighbours",
             rf"\textbf{{c}}\enspace $\boldsymbol{{\mu}}_z$, {tname}: neighbours",
             rf"\textbf{{d}}\enspace $\boldsymbol{{\mu}}_w$, {tname}: neighbours"]
    for row in range(4):
        ax = axes[row // 2][row % 2]
        idx = sec["all_idx"] if row < 2 else sec["type_idx"]
        colour_by = sec["fam"] if row == 0 else sec["dom"]
        xy = cache[f"{c}_{row}_umap"]
        col = colours(colour_by[idx])
        order = np.random.default_rng(1).permutation(len(idx))
        ax.scatter(xy[order, 0], xy[order, 1], s=0.35, c=col[order], lw=0, rasterized=True)
        lo, hi = np.percentile(xy, [0.5, 99.5], axis=0)
        pad = 0.04 * (hi - lo)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(heads[row], fontsize=7, color=S.INK, loc="left", pad=2)
    present = set(np.unique(sec["fam"])) | set(np.unique(sec["dom"]))   # only families on this section
    keys = [Line2D([], [], marker="o", ls="", ms=3, color=col, label=name)
            for k, ((name, _), col) in enumerate(zip(FAMILIES, COLOURS)) if k in present]
    keys.append(Line2D([], [], marker="o", ls="", ms=3, color=UNASSIGNED, label="Unassigned or none"))
    fig.legend(handles=keys, loc="lower center", ncol=3, frameon=False, fontsize=6,
               handletextpad=0.1, columnspacing=0.6, bbox_to_anchor=(0.5, -0.005))
    fig.savefig(OUT / "fig_latents_main.pdf"); fig.savefig(OUT / "fig_latents_main.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    S.apply()
    rng = np.random.default_rng(0)
    data = [section(ds, rng) for ds, _ in SECTIONS]
    for (ds, _), sec in zip(SECTIONS, data):
        print(ds, "rows c-d type:", sec["type_name"], "entropy %.2f" % sec["entropy"])
    cache = embeddings(data)
    draw(data, cache, "umap")
    draw_main(data, cache)
    print("wrote fig_latents_umap, fig_latents_main")
