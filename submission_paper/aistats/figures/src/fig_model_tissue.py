"""Figure 1 (a, b): what one cell's model sees in the tissue.

(a) The leakage neighbourhood: the focal cell, its first ring (model edges,
d <= 40 um) with arrows whose width is the kernel weight beta_ij, and its second
ring. (b) The image descriptor's input: the real four-channel morphology crop
the image model receives, a 128 um field at 0.5 um/px, with the 25 um disc
around the cell set to zero, all four channels shown (key below the panel); the
square labelled "a" is the extent of (a). Display contrast per channel comes from
a 512 um window around the cell, so a sparse stain is not stretched into noise.

The focal cell is the same "typical" cell as fig_voronoi_face (degree 6, median
6th-NN distance). Crop geometry is read from the pinned embedding file, never
assumed. Output ../fig_model_tissue.pdf (+ .png). Run from the repository root.
"""
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import shapely
import torch
from matplotlib import patheffects as pe
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

from discell.preprocess.crops import _ego_disk, crop_cell  # noqa: E402
from discell.tiff import find_tissue_image  # noqa: E402

DS = "xenium_prime_ovarian_cancer_ffpe"
BUNDLE = Path(f"data/datasets/{DS}/bundle"); NAME = "full"
EMB = Path(f"data/datasets/{DS}/embeddings/egomask_ego_v1.pt")
SAMPLE = Path("/home/rmolen/cellxgene_all_visium/highres_raw_with_images/10x_Xenium/extracted/Xenium_Prime_Ovarian_Cancer_FFPE")
PRUNE_UM, TAU_UM = 40.0, 20.0
OUT = Path(__file__).resolve().parent.parent / "fig_model_tissue"

S.apply()
meta = torch.load(EMB, map_location="cpu", weights_only=False)
FIELD_UM, MASK_UM, OUT_PX = float(meta["field_um"]), float(meta["mask_radius_um"]), int(meta["patch_px"])
CHANNELS = list(meta["channels"])

a = ad.read_h5ad(BUNDLE / f"{NAME}.h5ad", backed="r")
mpp = float(a.uns["microns_per_pixel"])
cent = np.asarray(a.obsm["spatial"], dtype=np.float64)
polys = np.asarray(shapely.from_wkb(pd.read_parquet(BUNDLE / f"{NAME}_polygons.parquet")["wkb"].to_numpy()), dtype=object)
deg = np.asarray(a.obs["voronoi_degree"])
nn6 = cKDTree(cent).query(cent, k=7)[0][:, 6] * mpp
ok = np.flatnonzero(deg == 6)
focal = int(ok[np.argmin(np.abs(nn6[ok] - np.median(nn6)))])      # same rule as fig_voronoi_face

e = pd.read_parquet(BUNDLE / f"{NAME}_edges_voronoi.parquet", columns=["i", "j", "shared_wall_um", "centroid_dist_um"])
e = e[e.centroid_dist_um <= PRUNE_UM]


def ring(cells):
    m = e.i.isin(cells) | e.j.isin(cells)
    return set(np.concatenate([e.i[m].values, e.j[m].values])) - set(cells)


mine = e[(e.i == focal) | (e.j == focal)]
nbr = np.where(mine.i.values == focal, mine.j.values, mine.i.values)
w = mine.shared_wall_um.values * np.exp(-mine.centroid_dist_um.values / TAU_UM)
beta = w / w.sum()
ring1 = set(nbr.tolist())
ring2 = ring(ring1 | {focal}) - ring1 - {focal}

# inches; the height matches the TikZ panel (c) beside it
FW, FH, P, LM, KEY, GAP = 1.37, 2.65, 1.12, 0.18, 0.20, 0.11
fig = plt.figure(figsize=(FW, FH))
axb = fig.add_axes([LM / FW, KEY / FH, P / FW, P / FH])
axa = fig.add_axes([LM / FW, (KEY + P + GAP) / FH, P / FW, P / FH])

# ---- (a) leakage neighbourhood
HALF_A = 21.0
c0 = cent[focal]
near = np.flatnonzero(np.linalg.norm(cent - c0, axis=1) * mpp < HALF_A * 1.9)
groups = {"other": [], "r2": [], "r1": []}
for k in near:
    key = "r1" if k in ring1 else "r2" if k in ring2 else "other"
    if k != focal:
        groups[key].append(np.asarray(polys[k].exterior.coords))
axa.add_collection(PolyCollection(groups["other"], facecolors=S.GHOST, edgecolors="#dedcd6", linewidths=0.3, zorder=1))
axa.add_collection(PolyCollection(groups["r2"], facecolors="#eceae4", edgecolors=S.CELL_EDGE, linewidths=0.35, zorder=2))
axa.add_collection(PolyCollection(groups["r1"], facecolors="#d6d4cd", edgecolors="#8f8d86", linewidths=0.45, zorder=3))
axa.add_collection(PolyCollection([np.asarray(polys[focal].exterior.coords)], facecolors=S.FOCAL_FILL,
                                  edgecolors=S.BLUE, linewidths=0.8, zorder=4))
for j, bj in zip(nbr, beta):
    src, dst = cent[j], c0
    v = dst - src; L = np.linalg.norm(v); u = v / L
    arr = FancyArrowPatch(src + u * 0.9 / mpp, dst - u * 1.6 / mpp, arrowstyle="-|>",
                          mutation_scale=4 + 10 * bj, lw=0.4 + 5.0 * bj, color=S.BLUE, zorder=6,
                          shrinkA=0, shrinkB=0)
    axa.add_patch(arr)
axa.scatter(*cent[list(ring1)].T, s=3, color=S.INK, zorder=7, lw=0)
# name the operator: these arrows are the leakage kernel, not the attention graph
axa.add_patch(FancyArrowPatch((0.05, 0.935), (0.17, 0.935), transform=axa.transAxes, arrowstyle="-|>",
                              mutation_scale=6, lw=1.4, color=S.BLUE, zorder=10))
axa.text(0.20, 0.935, r"leakage kernel $\beta_{ij}$", transform=axa.transAxes, fontsize=6.5, color=S.INK,
         va="center", zorder=10, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
h = HALF_A / mpp
axa.set_xlim(c0[0] - h, c0[0] + h); axa.set_ylim(c0[1] - h, c0[1] + h)
S.tissue_axes(axa); S.scalebar(axa, 10, mpp, pad=0.07, lw=1.2)
S.panel_label(axa, "a", x=-0.02, y=0.99)

# ---- (b) the image model's input, exactly: the ego disc is set to zero
crop = crop_cell(a, find_tissue_image(SAMPLE), focal, half_um=FIELD_UM / 2, channels=CHANNELS, out_size=OUT_PX)
img = crop.image.astype(np.float64)
cx, cy = crop.center_px


def stretch(p, lo=1, hi=99.6):
    a_, b_ = np.percentile(p, [lo, hi]); return np.clip((p - a_) / (b_ - a_ + 1e-9), 0, 1)


# per-channel display range from a 512 um window, not from the crop alone
wide = crop_cell(a, find_tissue_image(SAMPLE), focal, half_um=256.0, channels=CHANNELS,
                 out_size=int(OUT_PX * 512 / FIELD_UM)).image.astype(np.float64)
lo, hi = np.percentile(wide.reshape(-1, wide.shape[-1]), 1, axis=0), \
    np.percentile(wide.reshape(-1, wide.shape[-1]), 99.6, axis=0)
chan = np.clip((img - lo) / (hi - lo), 0, 1)
# additive composite of all four channels, in channel order; no orange (the "contact" role)
# named by the marker the image model reads each channel as; on the slide, channel 1 is an
# ATP1A1/CD45/E-cadherin mix and channel 3 an alphaSMA/vimentin mix (OME channel names)
STAINS = [(r"DAPI", "#3987e5"), (r"ATP1A1", "#3ccf4e"), (r"18S", "#8a8984"), (r"$\alpha$SMA", "#f03a2e")]
tint = np.array([[int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5)] for _, h in STAINS])
rgb = np.clip(np.einsum("hwc,ck->hwk", chan, tint), 0, 1)
keep = _ego_disk(OUT_PX, MASK_UM, FIELD_UM)                  # the pipeline's own mask
rgb = rgb * keep[..., None]
ext = FIELD_UM / 2
# local um, image orientation: row 0 of the crop is at y = -ext (top)
axb.imshow(rgb, extent=(-ext, ext, ext, -ext), origin="upper", interpolation="lanczos", zorder=1)
axb.add_patch(Circle((0, 0), MASK_UM, fill=False, edgecolor="#c3c2b7", lw=0.5, ls=(0, (2, 1.5)), zorder=3))
cell_xy = (np.asarray(polys[focal].exterior.coords) - [cx, cy]) * mpp
axb.plot(cell_xy[:, 0], cell_xy[:, 1], color=S.BLUE, lw=0.7, zorder=4)
# the window of (a), labelled, so the two panels read as overview and zoom
axb.add_patch(Rectangle((-HALF_A, -HALF_A), 2 * HALF_A, 2 * HALF_A, fill=False, ec="white", lw=0.55, zorder=4))
axb.text(-HALF_A - 1.5, -HALF_A - 1.0, r"\textbf{a}", color="white", fontsize=6.5, ha="right", va="bottom", zorder=6,
         path_effects=[pe.withStroke(linewidth=1.3, foreground="black")])
axb.set_xlim(-ext, ext); axb.set_ylim(ext, -ext)
S.tissue_axes(axb)
bx0, by0 = -ext + 0.07 * 2 * ext, ext - 0.07 * 2 * ext
axb.plot([bx0, bx0 + 25], [by0, by0], color="white", lw=1.2, solid_capstyle="butt", zorder=6)
axb.text(bx0 + 12.5, by0 - 0.035 * 2 * ext, r"25\,$\mu$m", color="white", fontsize=7, ha="center", va="bottom", zorder=6)
S.panel_label(axb, "b", x=-0.02, y=0.99)
# channel key: ink text beside a coloured sample, two by two under (b)
for k, (name, col) in enumerate(STAINS):
    x, y = LM / FW + (k % 2) * 0.50 * P / FW, (KEY - 0.07 - (k // 2) * 0.095) / FH
    fig.patches.append(Rectangle((x, y - 0.025 / FH), 0.065 / FW, 0.05 / FH, transform=fig.transFigure,
                                 fc=col, ec="none"))
    fig.text(x + 0.09 / FW, y, name, fontsize=6, color=S.INK, va="center")

fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=300)
print("wrote", OUT.with_suffix(".pdf"), "| focal", a.obs_names[focal], f"| field {FIELD_UM} um, mask {MASK_UM} um",
      f"| ring1 {len(ring1)}, ring2 {len(ring2)} | beta", np.round(np.sort(beta)[::-1], 2))
