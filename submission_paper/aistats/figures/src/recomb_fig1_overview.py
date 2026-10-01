"""RECOMB Figure 1 (fig:overview): DISCELL in four panels.

(a) Spill-over in real tissue: segmented polygons around the focal cell of
    fig_model_tissue (primary section, degree 6, median 6th-NN distance), its
    first ring (d <= 40 um) and arrows whose width is the kernel weight beta_ij
    (face length x exp(-d / tau), normalised). Inset: the four-channel morphology
    crop the niche encoder reads (128 um field), the cell's 25 um disc zeroed.
(b) The generative model as a block diagram: z (intrinsic) and w (response,
    prior set by niche c and type t) decode to rho; rho is mixed with the
    neighbours' influx at the fixed fraction kappa; the adversary acts on z.
(c) Why kappa is swept: the kappa-bound simplex (as fig_kappa_bound panel a).
(d) The breakdown point: one readout refitted over the kappa grid, its interval
    reaching zero at kappa*.

Panel a is data; b-d are schematic (compositions and readout chosen for
legibility; only the kappa grid is the model's). Output
../recomb_fig1_overview.pdf (+ .png). Run from anywhere.
"""
from pathlib import Path
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(ROOT))
import style as S  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "recomb_fig1_overview"
WIDTH = 6.5                     # RECOMB text width, letter paper with 1 in margins
HEIGHT = 2.15
TAU = 20.0                      # um, the model's kernel length
PRUNE_UM = 40.0                 # um, edges longer than this are pruned
HALF_A = 20.0                   # um, half-width of the tissue crop in (a)
DS = "xenium_prime_ovarian_cancer_ffpe"
BUNDLE = ROOT / f"data/datasets/{DS}/bundle"
EMB = ROOT / f"data/datasets/{DS}/embeddings/egomask_ego_v1.pt"
SAMPLE = Path("/home/rmolen/cellxgene_all_visium/highres_raw_with_images/10x_Xenium/extracted/Xenium_Prime_Ovarian_Cancer_FFPE")
GRID = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4])

S.apply()
fig = plt.figure(figsize=(WIDTH, HEIGHT))


# ---------------------------------------------------------------- (a) real tissue
def load_tissue():
    """The focal cell of fig_model_tissue (degree 6, median 6th-NN distance) on the primary
    section: segmented polygons, first ring, beta_ij, and the masked morphology crop."""
    import anndata as ad
    import pandas as pd
    import shapely
    import torch
    from scipy.spatial import cKDTree
    from discell.preprocess.crops import _ego_disk, crop_cell
    from discell.tiff import find_tissue_image

    meta = torch.load(EMB, map_location="cpu", weights_only=False)
    field, mask_um, out_px = float(meta["field_um"]), float(meta["mask_radius_um"]), int(meta["patch_px"])
    a = ad.read_h5ad(BUNDLE / "full.h5ad", backed="r")
    mpp = float(a.uns["microns_per_pixel"])
    cent = np.asarray(a.obsm["spatial"], dtype=np.float64)
    polys = np.asarray(shapely.from_wkb(pd.read_parquet(BUNDLE / "full_polygons.parquet")["wkb"].to_numpy()), dtype=object)
    deg = np.asarray(a.obs["voronoi_degree"])
    nn6 = cKDTree(cent).query(cent, k=7)[0][:, 6] * mpp
    ok = np.flatnonzero(deg == 6)
    focal = int(ok[np.argmin(np.abs(nn6[ok] - np.median(nn6)))])
    e = pd.read_parquet(BUNDLE / "full_edges_voronoi.parquet", columns=["i", "j", "shared_wall_um", "centroid_dist_um"])
    e = e[(e.centroid_dist_um <= PRUNE_UM) & ((e.i == focal) | (e.j == focal))]
    nbr = np.where(e.i.values == focal, e.j.values, e.i.values)
    w = e.shared_wall_um.values * np.exp(-e.centroid_dist_um.values / TAU)
    # morphology crop, exactly as the image model sees it: the ego disc set to zero
    tif = find_tissue_image(SAMPLE)
    crop = crop_cell(a, tif, focal, half_um=field / 2, channels=list(meta["channels"]), out_size=out_px)
    wide = crop_cell(a, tif, focal, half_um=256.0, channels=list(meta["channels"]),
                     out_size=int(out_px * 512 / field)).image.astype(np.float64)
    rgb = S.composite(crop.image.astype(np.float64), *S.stain_range(wide))
    rgb = rgb * _ego_disk(out_px, mask_um, field)[..., None]
    c0 = cent[focal]
    near = np.flatnonzero(np.linalg.norm(cent - c0, axis=1) * mpp < HALF_A * 1.9)
    to_um = lambda xy: (np.asarray(xy) - c0) * mpp  # noqa: E731
    return dict(focal=focal, nbr=nbr, beta=w / w.sum(), um=to_um(cent), near=near,
                poly={k: to_um(polys[k].exterior.coords) for k in set(near) | {focal}},
                rgb=rgb, field=field, mask_um=mask_um)


def panel_a(ax, axi, T):
    focal, nbr, beta, um = T["focal"], T["nbr"], T["beta"], T["um"]
    ring1 = set(nbr.tolist())
    for k in T["near"]:
        if k == focal:
            continue
        fc, ec, lw = ("#d6d4cd", "#8f8d86", 0.45) if k in ring1 else (S.GHOST, "#dedcd6", 0.3)
        ax.add_patch(Polygon(T["poly"][k], closed=True, fc=fc, ec=ec, lw=lw, zorder=2 if k in ring1 else 1))
    ax.add_patch(Polygon(T["poly"][focal], closed=True, fc=S.FOCAL_FILL, ec=S.BLUE, lw=0.8, zorder=4))
    for j, bj in zip(nbr, beta):
        v = -um[j]; u = v / np.linalg.norm(v)
        ax.add_patch(FancyArrowPatch(um[j] + u * 0.9, -u * 1.6, arrowstyle="-|>", mutation_scale=4 + 10 * bj,
                                     lw=0.4 + 5.0 * bj, color=S.BLUE, zorder=6, shrinkA=0, shrinkB=0))
    ax.scatter(*um[nbr].T, s=3, color=S.INK, zorder=7, lw=0)
    ax.text(0, 0, r"$i$", fontsize=7, color=S.INK, ha="center", va="center", zorder=8)
    ax.set_xlim(-HALF_A, HALF_A); ax.set_ylim(-HALF_A, HALF_A); S.tissue_axes(ax)
    ax.plot([-HALF_A + 2, -HALF_A + 12], [HALF_A - 2.2] * 2, color=S.INK, lw=1.2, solid_capstyle="butt", zorder=9)
    ax.text(-HALF_A + 7, HALF_A - 3.4, r"10\,$\mu$m", fontsize=6, color=S.INK, ha="center", va="bottom", zorder=9)
    ax.text(0.03, 0.965, r"arrow width: $\beta_{ij}$", transform=ax.transAxes, fontsize=6, color=S.BLUE,
            ha="left", va="top", zorder=10, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    # inset: the image context the niche encoder reads, the cell's 25 um disc zeroed
    ext = T["field"] / 2
    axi.imshow(T["rgb"], extent=(-ext, ext, ext, -ext), origin="upper", interpolation="lanczos")
    axi.add_patch(Circle((0, 0), T["mask_um"], fill=False, ec="#c3c2b7", lw=0.5, ls=(0, (2, 1.5))))
    axi.set_xlim(-ext, ext); axi.set_ylim(ext, -ext); S.tissue_axes(axi)
    axi.text(0.5, -0.04, r"$\boldsymbol{\Phi}_i$: image context" + "\n" + r"(cell masked)", transform=axi.transAxes,
             fontsize=6.5, color=S.INK2, ha="center", va="top", linespacing=1.1)


# ---------------------------------------------------------------- (b) block model
def box(ax, xy, text, w, h, fc="white", ec=S.INK2, fs=7, lw=0.7):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.025",
                                fc=fc, ec=ec, lw=lw, zorder=2, clip_on=False))
    ax.text(x, y, text, fontsize=fs, color=S.INK, ha="center", va="center", zorder=3, linespacing=1.1)


def arrow(ax, a, b, color=S.INK2, ls="-", lw=0.7, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=lw, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
                                 zorder=1))


def panel_b(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fs = 6.5
    box(ax, (0.085, 0.82), r"counts," + "\n" + r"type $t_i$", 0.15, 0.20, fs=fs)
    box(ax, (0.085, 0.30), r"niche" + "\n" + r"$\mathbf{c}_i$, $t_i$", 0.15, 0.20, fs=fs)
    box(ax, (0.345, 0.82), r"$\mathbf{z}_i$", 0.10, 0.15, fc=S.FOCAL_FILL, ec=S.BLUE, fs=7)
    box(ax, (0.345, 0.30), r"$\mathbf{w}_i$", 0.10, 0.15, fc=S.FOCAL_FILL, ec=S.BLUE, fs=7)
    ax.text(0.345, 0.925, r"intrinsic state", fontsize=6.5, color=S.BLUE, ha="center", va="bottom")
    ax.text(0.345, 0.205, r"response", fontsize=6.5, color=S.BLUE, ha="center", va="top")
    box(ax, (0.64, 0.62), r"cell's own $\boldsymbol{\rho}_i =$" + "\n" + r"$\mathrm{softmax}(a(\mathbf{z}_i) + \mathbf{B}\mathbf{w}_i)$", 0.40, 0.24, fs=fs)
    box(ax, (0.665, 0.14), r"neighbours' influx" + "\n" + r"$\bar{\boldsymbol{\rho}}_i = \sum_j \beta_{ij}\boldsymbol{\rho}_j$", 0.37, 0.24, fs=fs)
    box(ax, (0.945, 0.38), r"$\mathbf{x}_i$", 0.08, 0.15, fs=7)
    arrow(ax, (0.16, 0.82), (0.29, 0.82))
    arrow(ax, (0.16, 0.30), (0.29, 0.30))
    ax.text(0.225, 0.325, r"prior", fontsize=6.5, color=S.INK2, ha="center", va="bottom")
    arrow(ax, (0.405, 0.80), (0.435, 0.70))
    arrow(ax, (0.405, 0.32), (0.435, 0.54))
    arrow(ax, (0.85, 0.56), (0.925, 0.455))
    arrow(ax, (0.85, 0.20), (0.925, 0.305), color=S.BLUE)
    ax.text(0.915, 0.60, r"$1{-}\kappa$", fontsize=6.5, color=S.INK, ha="center", va="bottom")
    ax.text(0.905, 0.17, r"$\kappa$", fontsize=7, color=S.BLUE, ha="center", va="top")
    ax.text(0.90, -0.055, r"spill-over", fontsize=6.5, color=S.BLUE, ha="center", va="center")
    # adversary on z, given the type
    arrow(ax, (0.345, 0.385), (0.345, 0.735), color=S.ORANGE, ls=(0, (2, 1.5)), lw=0.8)
    ax.text(0.325, 0.56, r"adversary", fontsize=6.5, color=S.ORANGE, ha="right", va="center", rotation=90)


# ---------------------------------------------------------------- (c) kappa bound
CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
RHOBAR = np.array([0.20, 0.15, 0.65]); RHO = np.array([0.55, 0.35, 0.10]); KAPPA = 0.3


def xy(b):
    return np.asarray(b) @ CORNERS


def panel_c(ax):
    p = (1 - KAPPA) * RHO + KAPPA * RHOBAR
    kbar = float(np.min(p / RHOBAR))
    clean = lambda k: (p - k * RHOBAR) / (1 - k)  # noqa: E731
    ax.add_patch(Polygon(CORNERS, closed=True, fc=S.GHOST, ec=S.INK2, lw=0.7, zorder=1))
    path = np.array([xy(clean(k)) for k in np.linspace(0, kbar, 60)])
    ax.plot(*path.T, color=S.INK, lw=1.5, zorder=3, solid_capstyle="round")
    beyond = np.array([xy(clean(k)) for k in np.linspace(kbar, kbar + 0.06, 10)])
    ax.plot(*beyond.T, color=S.INK2, lw=0.9, ls=(0, (2.5, 1.5)), zorder=3)
    ax.plot(*np.stack([xy(p), xy(RHOBAR)]).T, color=S.MUTED, lw=0.6, ls=(0, (1.2, 1.6)), zorder=2)
    ax.scatter(*xy(RHOBAR), s=18, color=S.BLUE, zorder=5, lw=0)
    ax.scatter(*xy(p), s=16, color=S.INK, zorder=5, lw=0)
    ax.scatter(*xy(clean(kbar)), s=20, marker="D", facecolor="white", edgecolor=S.BLUE, lw=0.9, zorder=6)
    ax.text(*(xy(RHOBAR) + [0.05, 0.0]), r"$\bar{\boldsymbol{\rho}}_i$", fontsize=7, color=S.BLUE, va="center")
    ax.text(*(xy(p) + [0.04, 0.04]), r"$\mathbf{p}_i$", fontsize=7, color=S.INK, va="bottom")
    ax.text(*(xy(clean(kbar)) + [0.07, -0.04]), r"$\bar\kappa_i$", fontsize=7, color=S.INK, ha="left", va="top")
    ax.text(0.5, -0.24, r"all $\kappa \le \bar\kappa_i$ fit:" + "\n" + r"bounded above only",
            fontsize=6.5, color=S.INK2, ha="center", va="top", linespacing=1.35)
    ax.text(-0.05, -0.03, r"gene 1", fontsize=6.5, color=S.INK2, ha="left", va="top")
    ax.text(1.05, -0.03, r"gene 2", fontsize=6.5, color=S.INK2, ha="right", va="top")
    ax.text(0.5, CORNERS[2, 1] + 0.03, r"gene 3", fontsize=6.5, color=S.INK2, ha="center", va="bottom")
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.45, 0.98); ax.set_aspect("equal"); ax.axis("off")


# ---------------------------------------------------------------- (d) breakdown point
def panel_d(ax):
    m = 0.55 - 1.5 * GRID
    h = np.full(GRID.size, 0.17)
    b = int(np.argmax((m - h <= 0) | (m <= 0)))
    ok = np.arange(GRID.size) < b
    ax.axhline(0, color=S.MUTED, lw=0.7, ls=(0, (4, 2.5)), zorder=1)
    ax.vlines(GRID, m - h, m + h, color=S.INK2, lw=0.9, zorder=2)
    ax.scatter(GRID[ok], m[ok], s=13, color=S.INK, zorder=4, lw=0)
    ax.scatter(GRID[~ok], m[~ok], s=13, facecolor="white", edgecolor=S.INK2, lw=0.8, zorder=4)
    ax.axvline(GRID[b], color=S.BLUE, lw=1.0, ls=(0, (1.2, 1.4)), zorder=3)
    ax.text(GRID[b] - 0.012, 0.84, rf"$\kappa^\star = {GRID[b]:g}$", fontsize=7, color=S.INK, ha="right", va="center")
    ax.text(0.0, -0.04, r"null", fontsize=6.5, color=S.INK2, ha="left", va="top")
    ax.set_xlim(-0.025, 0.44); ax.set_ylim(-0.3, 0.9)
    ax.set_xticks(GRID, ["0", "", "", "0.2", "", "0.4"]); ax.set_yticks([])
    ax.set_xlabel(r"assumed $\kappa$", labelpad=1)
    ax.set_ylabel(r"a finding", labelpad=2)
    ax.spines["left"].set_visible(False)


# ---------------------------------------------------------------- layout (inches -> fractions)
def axes(x, w, y=0.20, h=1.62):
    return fig.add_axes([x / WIDTH, y / HEIGHT, w / WIDTH, h / HEIGHT])


def lbl(x, letter, title):
    y = (HEIGHT - 0.09) / HEIGHT                             # one common title baseline
    fig.text((x - 0.04) / WIDTH, y, rf"\textbf{{{letter}}}", ha="right", va="top", fontsize=9, color=S.INK)
    fig.text(x / WIDTH, y, title, ha="left", va="top", fontsize=7.2, color=S.INK)


ax_a = axes(0.20, 1.05, y=0.90, h=1.05)
ax_i = axes(0.20 + (1.05 - 0.66) / 2, 0.66, y=0.19, h=0.66)
panel_a(ax_a, ax_i, load_tissue())
# Phi_i feeds the niche descriptor: a light arrow from the image into panel b's niche box
fig.patches.append(FancyArrowPatch((1.08 / WIDTH, 0.55 / HEIGHT), (1.615 / WIDTH, 0.70 / HEIGHT),
                                   transform=fig.transFigure, arrowstyle="-|>", mutation_scale=6, lw=0.6,
                                   color=S.MUTED, shrinkA=0, shrinkB=0))
ax_b = axes(1.60, 2.66, y=0.24, h=1.66); panel_b(ax_b)
ax_c = axes(4.40, 1.04, y=0.24, h=1.66); panel_c(ax_c)
ax_d = axes(5.60, 0.82, y=0.46, h=1.38); panel_d(ax_d)
for x, letter, title in ((0.20, "a", "spill-over"), (1.60, "b", "the generative model"),
                         (4.40, "c", "counts bound $\\kappa$"), (5.60, "d", "breakdown point")):
    lbl(x, letter, title)
S.save(fig, OUT)
