"""Figure: why the image descriptor's mask is a 25 um disc.

(a) For every cell of the primary section, the covering radius -- the radius of
the smallest disc centred on the crop anchor (the centroid) that contains the
whole segmented polygon -- summarised as the number of cells that would stick out
of a disc of radius R, with the share of the 128 um field each radius masks.
(b) The largest cell that fits the 25 um disc; (c) the largest cell on the section,
which does not fit and is excluded. Both are drawn exactly as the image model
receives them (four channels, disc set to zero), with the cell's outline.

Crop geometry is read from the pinned embedding file. Outputs ../fig_mask_radius.pdf
(+ .png). Run from the repository root.
"""
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import shapely
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

from discell.preprocess.crops import _ego_disk, crop_cell  # noqa: E402
from discell.tiff import find_tissue_image  # noqa: E402

DS = "xenium_prime_ovarian_cancer_ffpe"
BUNDLE = Path(f"data/datasets/{DS}/bundle"); NAME = "full"
EMB = Path(f"data/datasets/{DS}/embeddings/egomask_ego_v1.pt")
SAMPLE = Path("/home/rmolen/cellxgene_all_visium/highres_raw_with_images/10x_Xenium/extracted/Xenium_Prime_Ovarian_Cancer_FFPE")
OUT = Path(__file__).resolve().parent.parent / "fig_mask_radius"
MARKS = [15.0, 22.0]                       # the radii considered before the choice

S.apply()
meta = torch.load(EMB, map_location="cpu", weights_only=False)
FIELD_UM, MASK_UM, OUT_PX = float(meta["field_um"]), float(meta["mask_radius_um"]), int(meta["patch_px"])
CHANNELS = list(meta["channels"])

a = ad.read_h5ad(BUNDLE / f"{NAME}.h5ad", backed="r")
mpp = float(a.uns["microns_per_pixel"])
cent = np.asarray(a.obsm["spatial"], dtype=np.float64)
polys = shapely.from_wkb(pd.read_parquet(BUNDLE / f"{NAME}_polygons.parquet")["wkb"].to_numpy())
coords, owner = shapely.get_coordinates(polys, return_index=True)
radius = np.zeros(len(polys))
np.maximum.at(radius, owner, np.linalg.norm(coords - cent[owner], axis=1) * mpp)
field_share = lambda r: np.pi * r ** 2 / FIELD_UM ** 2                     # noqa: E731
out_of = lambda r: int((radius > r).sum())                                 # noqa: E731

fig = plt.figure(figsize=(S.TEXTWIDTH, 2.4))
axa = fig.add_axes([0.075, 0.22, 0.37, 0.66])
P = 1.52
axb = fig.add_axes([0.50, 0.17, P / S.TEXTWIDTH, P / 2.4])
axc = fig.add_axes([0.50 + (P + 0.12) / S.TEXTWIDTH, 0.17, P / S.TEXTWIDTH, P / 2.4])

# (a) cells sticking out of a disc of radius R; drop lines at the radii considered
rmax = radius.max()
R = np.linspace(2, rmax, 400)
axa.plot(R, [max(out_of(r), 0.7) for r in R], color=S.INK, lw=1.3, zorder=3)
axa.set_yscale("log"); axa.set_ylim(0.7, 6e5); axa.set_xlim(2, 40)
axa.set_xlabel(r"disc radius $R$ ($\mu$m)", labelpad=1.5)
axa.set_ylabel(r"cells sticking out of the disc", labelpad=1.5)
for r in MARKS + [MASK_UM]:
    col = S.BLUE if r == MASK_UM else S.INK2
    n = max(out_of(r), 0.7)
    axa.plot([r, r], [0.7, n], color=col, lw=1.0 if r == MASK_UM else 0.7,
             ls="solid" if r == MASK_UM else (0, (3, 2)), zorder=2)
    axa.scatter([r], [n], s=14, color=col, zorder=4, lw=0)
# the trade-off, as a small table in the empty upper right
cols = [(0.53, r"disc radius" + "\n" + r"$R$ ($\mu$m)"), (0.76, r"cells" + "\n" + r"outside it"),
        (0.99, r"disc area," + "\n" + r"\% of field")]
for x, head in cols:
    axa.text(x, 0.955, head, transform=axa.transAxes, fontsize=6.5, color=S.INK2, ha="right", va="center",
             linespacing=1.1)
axa.plot([0.40, 0.99], [0.895, 0.895], transform=axa.transAxes, color=S.MUTED, lw=0.5)
rows = [(r, out_of(r)) for r in MARKS + [MASK_UM]] + [(rmax, 0)]
for k, (r, n) in enumerate(rows):
    y = 0.845 - 0.08 * k
    vals = [rf"{r:.1f}" if r == rmax else rf"{r:.0f}", rf"{n:,}", rf"{100 * field_share(r):.1f}\,\%"]
    for (x, _), v in zip(cols, vals):
        axa.text(x, y, v, transform=axa.transAxes, fontsize=6.5, color=S.INK, ha="right", va="center")
    if r == MASK_UM:
        axa.scatter([0.405], [y], s=12, color=S.BLUE, transform=axa.transAxes, lw=0, clip_on=False)
S.panel_label(axa, "a", x=-0.13, y=1.02)

# (b), (c) the two cells, as the image model receives them
image = find_tissue_image(SAMPLE)
keep = _ego_disk(OUT_PX, MASK_UM, FIELD_UM)
ext = FIELD_UM / 2
fits = int(np.flatnonzero(radius <= MASK_UM)[np.argmax(radius[radius <= MASK_UM])])
largest = int(np.argmax(radius))
for ax, cell, letter, title in ((axb, fits, "b", rf"largest cell that fits, {radius[fits]:.1f}\,$\mu$m"),
                                (axc, largest, "c", rf"largest cell, {radius[largest]:.1f}\,$\mu$m: does not fit")):
    crop = crop_cell(a, image, cell, half_um=ext, channels=CHANNELS, out_size=OUT_PX)
    wide = crop_cell(a, image, cell, half_um=256.0, channels=CHANNELS,
                     out_size=int(OUT_PX * 512 / FIELD_UM)).image.astype(np.float64)
    rgb = S.composite(crop.image.astype(np.float64), *S.stain_range(wide)) * keep[..., None]
    ax.imshow(rgb, extent=(-ext, ext, ext, -ext), origin="upper", interpolation="lanczos", zorder=1)
    ax.add_patch(Circle((0, 0), MASK_UM, fill=False, edgecolor="#c3c2b7", lw=0.6, ls=(0, (2, 1.5)), zorder=3))
    outline = (np.asarray(shapely.get_coordinates(polys[cell])) - np.asarray(crop.center_px)) * mpp
    ax.plot(outline[:, 0], outline[:, 1], color=S.BLUE, lw=0.8, zorder=4)
    ax.set_xlim(-ext, ext); ax.set_ylim(ext, -ext)
    S.tissue_axes(ax)
    ax.set_title(title, fontsize=7, color=S.INK, loc="left", pad=3)
    S.panel_label(ax, letter, x=-0.02, y=1.0)
bx0, by0 = -ext + 0.07 * 2 * ext, ext - 0.07 * 2 * ext
axb.plot([bx0, bx0 + 25], [by0, by0], color="white", lw=1.2, solid_capstyle="butt", zorder=6)
axb.text(bx0 + 12.5, by0 - 0.035 * 2 * ext, r"25\,$\mu$m", color="white", fontsize=7, ha="center", va="bottom", zorder=6)

# channel key under (b) and (c), ink text beside colour samples
for x, (name, col) in zip((0.50, 0.585, 0.695, 0.77), S.STAINS):
    fig.patches.append(Rectangle((x, 0.045), 0.012, 0.035, transform=fig.transFigure, fc=col, ec="none"))
    fig.text(x + 0.017, 0.062, name, fontsize=6.5, color=S.INK, va="center")

fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=300)
print("wrote", OUT.with_suffix(".pdf"), f"| fits {a.obs_names[fits]} r {radius[fits]:.2f} ({a.obs['cell_group'].iloc[fits]}), "
      f"largest {a.obs_names[largest]} r {radius[largest]:.2f} ({a.obs['cell_group'].iloc[largest]}) | out at 15/22/25: "
      f"{out_of(15)}/{out_of(22)}/{out_of(25)}")
