"""Figure: the four fitted sections, by lineage and by morphology.

Top row: every cell at its centroid, coloured by its lineage label grouped into
six broad families for display (the fitted labels are finer), Unassigned in
grey. Bottom row: the section's morphology image, the four stains the image
model reads in the colours of Figure 1b, from a coarse pyramid level of the
focus image, cropped to the cells' extent. Image orientation (y down). The TMA
panel is the fitted core only. Outputs ../fig_sections.pdf (+ .png). Run from
the repository root.
"""
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import tifffile
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

from discell.tiff import find_tissue_image  # noqa: E402

SECTIONS = [("xenium_prime_ovarian_cancer_ffpe", "full", r"Ovarian cancer (FFPE)"),
            ("xenium_prime_human_lung_cancer_ffpe", "full", r"Lung cancer (FFPE)"),
            ("xenium_prime_human_ovary_ff", "full", r"Ovarian cancer (FF)"),
            ("gse315411_pdltma06_11_prime_solo", "pdl018d", r"Lung TMA core")]
FAMILIES = [("Epithelial, tumour", ("tumour", "at1", "at2", "basal", "goblet", "multiciliated", "secretory",
                                      "distal epithelium", "neuroendocrine", "ciliated", "fallopian", "urothelial",
                                      "mesothelial")),
            ("Fibroblast, stroma", ("fibroblast", "stroma", "chondrocyte")),
            ("Muscle, pericyte", ("muscle", "pericyte")),
            ("Endothelial", ("ec ", "ec_", "endothelial", "lymphatic ec")),
            ("Immune, blood", ("macrophage", "t/nk", "b cells", "plasma", "neutrophil", "mast", "megakaryocyte",
                                  "erythroid")),
            ("Other", ("schwann",))]
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # validated categorical order, slots 1-6
UNASSIGNED = "#c9c7bf"
OUT = Path(__file__).resolve().parent.parent / "fig_sections"
MAX_PX = 1400


def family(name: str) -> int:
    low = name.lower() + " "
    if low.startswith("unassigned"):
        return -1
    if low.startswith("ec "):
        return 3
    for k, (_, keys) in enumerate(FAMILIES):
        if any(key in low for key in keys):
            return k
    return 5


def morphology(sample_dir, box_px):
    """Four-channel window over *box_px* from the coarsest pyramid level still >= MAX_PX wide."""
    import zarr
    path = find_tissue_image(sample_dir)
    x0, y0, x1, y1 = box_px
    with tifffile.TiffFile(path) as tif:
        levels = tif.series[0].levels                      # every section is CYX
        full_w = levels[0].shape[2]
        pick = max(k for k, lev in enumerate(levels) if (x1 - x0) * lev.shape[2] / full_w >= MAX_PX)
        scale = levels[pick].shape[2] / full_w
        store = tif.aszarr(series=0, level=pick)
        arr = zarr.open(store, mode="r")
        img = np.asarray(arr[:, int(y0 * scale):int(y1 * scale), int(x0 * scale):int(x1 * scale)])
        store.close()
    img = np.moveaxis(img, 0, -1).astype(np.float64)
    return S.composite(img, *S.stain_range(img))


def main():
    S.apply()
    data = []
    for ds, var, title in SECTIONS:
        a = ad.read_h5ad(f"data/datasets/{ds}/bundle/{var}.h5ad", backed="r")
        xy = np.asarray(a.obsm["spatial"], dtype=np.float64)
        fam = np.array([family(x) for x in a.obs["lineage"].astype(str)])
        data.append((title, xy, fam, float(a.uns["microns_per_pixel"]), a.uns["xenium_dir"]))

    # one panel height for all, each panel as wide as its section's aspect demands
    H, GAP, LEFT = 1.40, 0.09, 0.14                      # inches
    aspect = [np.ptp(xy[:, 0]) / np.ptp(xy[:, 1]) for _, xy, *_ in data]
    FIGH = 0.22 + 2 * H + 0.12 + 0.28 + 0.34             # title, rows, gap, bars, legend
    fig = plt.figure(figsize=(S.TEXTWIDTH, FIGH))
    spare = S.TEXTWIDTH - LEFT - sum(aspect) * H - GAP * (len(data) - 1)
    x = LEFT + spare / 2
    for c, (title, xy, fam, mpp, xdir) in enumerate(data):
        w = aspect[c] * H
        x0, y0 = xy.min(0); x1, y1 = xy.max(0)
        top = fig.add_axes([x / S.TEXTWIDTH, (FIGH - 0.22 - H) / FIGH, w / S.TEXTWIDTH, H / FIGH])
        bot = fig.add_axes([x / S.TEXTWIDTH, (FIGH - 0.22 - 2 * H - 0.12) / FIGH, w / S.TEXTWIDTH, H / FIGH])
        order = np.random.default_rng(0).permutation(len(xy))
        col = np.array([COLOURS[f] if f >= 0 else UNASSIGNED for f in fam])
        size = 0.012 if len(xy) > 500_000 else 0.03 if len(xy) > 150_000 else 0.12
        top.scatter(xy[order, 0], xy[order, 1], s=size, c=col[order], lw=0, rasterized=True)
        top.set_xlim(x0, x1); top.set_ylim(y0, y1); S.tissue_axes(top)
        top.set_title(title, fontsize=7.5, color=S.INK, pad=3)
        bot.imshow(morphology(xdir, (x0, y0, x1, y1)), extent=(x0, x1, y1, y0), interpolation="lanczos")
        bot.set_xlim(x0, x1); bot.set_ylim(y1, y0); S.tissue_axes(bot)
        # 1 mm bar under the image, outside the tissue
        L = 1000 / mpp
        bot.plot([x0, x0 + L], [y1 + 0.06 * (y1 - y0)] * 2, color=S.INK, lw=1.2, solid_capstyle="butt", clip_on=False)
        bot.text(x0 + L + 0.03 * (x1 - x0), y1 + 0.06 * (y1 - y0), r"1\,mm", fontsize=6.5, color=S.INK,
                 ha="left", va="center", clip_on=False)
        if c == 0:
            S.panel_label(top, "a", x=-0.01, y=1.0)
            S.panel_label(bot, "b", x=-0.01, y=1.0)
        x += w + GAP
    lin = [Line2D([], [], marker="o", ls="", ms=3.5, color=col, label=name) for (name, _), col in zip(FAMILIES, COLOURS)]
    lin.append(Line2D([], [], marker="o", ls="", ms=3.5, color=UNASSIGNED, label="Unassigned"))
    stain = [Line2D([], [], marker="s", ls="", ms=3.5, color=h, label=n) for n, h in S.STAINS]
    kw = dict(frameon=False, fontsize=6.5, handletextpad=0.2, columnspacing=1.0, loc="lower center")
    fig.legend(handles=lin, ncol=len(lin), bbox_to_anchor=(0.5, 0.155 / FIGH), title=None, **kw)
    fig.legend(handles=stain, ncol=len(stain), bbox_to_anchor=(0.5, -0.01), **kw)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=250)
    print("wrote", OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
