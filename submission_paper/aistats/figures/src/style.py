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
