"""Shared style for the DISCELL manuscript figures.

Every figure is drawn at its final printed width (column 3.25 in, text 6.75 in)
so nothing is rescaled in LaTeX, with text set through LaTeX in Computer Modern
to match the body. Colours carry one role each, the same in every figure:

    BLUE    the Voronoi face and the kernel beta        (categorical slot 1)
    ORANGE  segmented-polygon contact / apposed wall    (categorical slot 2)
    AQUA    a third identity, only with a direct label  (slot 3, < 3:1 on white)
    INK     centroid distance, text and annotations
    greys   cells and chrome

The three categorical slots pass the palette validator with all pairs compared
(worst CVD dE 9.2, worst normal-vision dE 24.0 on white).
"""
import matplotlib as mpl
import numpy as np

TEXTWIDTH, COLWIDTH = 6.75, 3.25          # inches, from aistats2026.sty

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
CELL_FILL, CELL_EDGE = "#e4e3dd", "#a9a79f"
REGION_EDGE, GHOST = "#8f8d86", "#f1f0ec"
FOCAL_FILL = "#e3eefc"


def apply():
    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
        "font.family": "serif",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "savefig.dpi": 600,
        "figure.dpi": 150,
    })


def panel_label(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, rf"\textbf{{{letter}}}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color=INK)


def scalebar(ax, length_um, mpp, label=None, pad=0.06, lw=1.4):
    """A scale bar in the lower left of an equal-aspect tissue panel in pixels."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    w, h = x1 - x0, y1 - y0
    bx, by = x0 + pad * w, y0 + pad * h
    L = length_um / mpp
    ax.plot([bx, bx + L], [by, by], color=INK, lw=lw, solid_capstyle="butt", zorder=20)
    ax.text(bx + L / 2, by + 0.025 * h, label or rf"{length_um:g}\,$\mu$m",
            ha="center", va="bottom", fontsize=7, color=INK, zorder=20)


def tissue_axes(ax):
    """Equal-aspect tissue panel in IMAGE orientation: y increases downward, as in
    the slide's pixel grid and the microscopy, so a tissue panel and an image crop
    of the same place are never mirror images of each other."""
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    for s in ax.spines.values():
        s.set_visible(False)


# ---- the morphology composite shared by every figure that shows the image input
#: named by the marker the image model reads each channel as; on the slide, channel 1 is an
#: ATP1A1/CD45/E-cadherin mix and channel 3 an alphaSMA/vimentin mix (OME channel names).
#: Additive, in channel order; no orange (the "contact" role).
STAINS = [(r"DAPI", "#3987e5"), (r"ATP1A1", "#3ccf4e"), (r"18S", "#8a8984"), (r"$\alpha$SMA", "#f03a2e")]


def stain_range(wide):
    """Per-channel display range (1st, 99.6th percentile) from a wide window, so a
    sparse stain is not stretched into noise by a crop-only range."""
    flat = wide.reshape(-1, wide.shape[-1])
    return np.percentile(flat, 1, axis=0), np.percentile(flat, 99.6, axis=0)


def composite(img, lo, hi):
    """Four-channel crop -> RGB, each channel scaled to [lo, hi] and tinted by STAINS."""
    chan = np.clip((img - lo) / (hi - lo), 0, 1)
    tint = np.array([[int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5)] for _, h in STAINS])
    return np.clip(np.einsum("hwc,ck->hwk", chan, tint), 0, 1)



# ---- shared by the data figures (fig_kappa_sweep ... fig_synthetic_misspec)
BLUE_BAND = "#cde2fb"          # BLUE_RAMP[0]: a seed-range band under a BLUE line


def paper_tables():
    """scripts/paper_tables.py as a module: the loaders, section names and paths
    the generated tables use, so a figure reads exactly what its table reads."""
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location("paper_tables", repo / "scripts" / "paper_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save(fig, out):
    """Vector PDF for the paper and a PNG preview, side by side."""
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=250)
    print("wrote", out.with_suffix(".pdf"))


def light_grid(ax, axis="both"):
    ax.grid(axis=axis, color=GHOST, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


#: method identity in the comparison figures, as in fig_tradeoff.py: DISCELL blue,
#: the Cellina variants orange, the other methods grey; shape tells methods apart
METHOD_STYLE = {
    "DISCELL": dict(marker="o", color=BLUE, fill=True, size=34),
    "DISCELL, no adversary": dict(marker="o", color=BLUE, fill=False, size=30),
    "resolVI": dict(marker="s", color=INK2, fill=True, size=24),
    "SIMVI": dict(marker="v", color=INK2, fill=True, size=28),
    "MintFlow": dict(marker="D", color=INK2, fill=False, size=22),
    "Cellina": dict(marker="^", color=ORANGE, fill=True, size=30),
    "Cellina, niche domain": dict(marker="^", color=ORANGE, fill=False, size=30),
    "Cellina, own graph": dict(marker="P", color=ORANGE, fill=True, size=30),
}


def method_handles(names, scale=0.95):
    from matplotlib.lines import Line2D
    return [Line2D([], [], ls="none", marker=METHOD_STYLE[n]["marker"],
                   markersize=np.sqrt(METHOD_STYLE[n]["size"]) * scale,
                   markerfacecolor=METHOD_STYLE[n]["color"] if METHOD_STYLE[n]["fill"] else "white",
                   markeredgecolor=METHOD_STYLE[n]["color"], markeredgewidth=1.0, label=n)
            for n in names]


def method_point(ax, x, y, name, scale=1.0, z=4):
    st = METHOD_STYLE[name]
    ax.scatter([x], [y], s=st["size"] * scale, marker=st["marker"], linewidths=0.9, zorder=z,
               facecolor=st["color"] if st["fill"] else "white", edgecolor=st["color"])
