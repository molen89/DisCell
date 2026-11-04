"""Make the figures of the DisCell documentation (docs/figures/*.png).

Every figure is either a schematic drawn from made-up numbers or a plot of a
fit to data simulated with ``discell.sim.simulate``. No real section is used.
Runs on the CPU in about five minutes::

    pip install ".[viz]"
    python docs/figures/make_figures.py                  # all figures
    python docs/figures/make_figures.py --only simplex   # one figure
    python docs/figures/make_figures.py --work fits/     # keep the fits

The simulated fits are saved under ``--work`` (a temporary folder by default);
a second run with the same folder reuses them.
"""

from __future__ import annotations

import argparse
import dataclasses
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

OUT = Path(__file__).resolve().parent

# Categorical slots, validated for all-pairs use on a white surface.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
TYPES = (BLUE, ORANGE, AQUA, VIOLET)
INK, MUTED, GRID = "#1f1f1f", "#6b6b66", "#d9d9d4"
DPI = 150

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})


def save(fig, name: str, dpi: int = DPI) -> None:
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"{path.name}: {path.stat().st_size / 1024:.0f} kB")


# -- schematics -----------------------------------------------------------------

def fig_sources() -> None:
    """The three sources of a cell's counts."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 3.6),
                                 gridspec_kw={"width_ratios": [1, 1.35]})
    ax.set_aspect("equal")
    ax.axis("off")
    center = np.array([0.0, 0.0])
    angles = np.deg2rad(np.arange(6) * 60 + 30)
    kinds = [ORANGE, ORANGE, ORANGE, AQUA, AQUA, AQUA]
    widths = [3.5, 2.2, 1.2, 2.8, 1.6, 1.0]
    for angle, color, width in zip(angles, kinds, widths):
        where = 1.9 * np.array([np.cos(angle), np.sin(angle)])
        ax.add_patch(Circle(where, 0.78, color=color, alpha=0.85, lw=0))
        ax.add_patch(FancyArrowPatch(where * 0.62, where * 0.36, arrowstyle="-|>",
                                     mutation_scale=9, lw=width, color=INK))
    ax.add_patch(Circle(center, 0.78, color=BLUE, lw=0))
    ax.text(0, 0, "cell $i$", ha="center", va="center", color="white", fontsize=9,
            weight="bold")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3.1, 3)
    ax.text(0, 2.95, "arrow width: kernel weight $\\beta_{ij}$", ha="center", fontsize=8,
            color=MUTED)
    ax.legend(handles=[Circle((0, 0), 1, color=ORANGE), Circle((0, 0), 1, color=AQUA)],
              labels=["tumor neighbors", "fibroblast neighbors"], loc="lower center",
              bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, handlelength=1)
    ax.set_title("a  a fibroblast and its neighbors", loc="left", fontsize=9, pad=16)

    genes = ["COL1A1", "DCN", "POSTN", "ACTA2", "EPCAM", "KRT19"]
    intrinsic = np.array([30, 22, 6, 4, 0, 0], dtype=float)
    response = np.array([0, 0, 12, 9, 0, 0], dtype=float)
    spill = np.array([2, 1, 1, 1, 7, 5], dtype=float)
    xs = np.arange(len(genes))
    bx.bar(xs, intrinsic, width=0.62, color=BLUE, label="intrinsic state  $a(z_i)$")
    bx.bar(xs, response, bottom=intrinsic + 0.4 * (response > 0), width=0.62, color=VIOLET,
           label="response to the niche  $Bw_i$")
    top = intrinsic + response + 0.4 * (response > 0)
    bx.bar(xs, spill, bottom=top + 0.4, width=0.62, color=ORANGE,
           label="spillover from neighbors  $\\kappa\\bar\\rho_i$")
    bx.set_xticks(xs, genes, fontsize=8)
    bx.set_ylabel("counts in cell $i$")
    bx.legend(fontsize=8, loc="upper right")
    bx.set_title("b  where its measured counts come from (schematic)", loc="left", fontsize=9)
    bx.grid(axis="y", color=GRID, lw=0.6)
    bx.set_axisbelow(True)
    save(fig, "sources")


def _bary(p: np.ndarray) -> np.ndarray:
    """Barycentric (3 shares) to 2-D triangle coordinates."""
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    return np.asarray(p) @ corners


def fig_simplex() -> None:
    """Why kappa has an upper bound but no lower bound."""
    rho_bar = np.array([0.20, 0.15, 0.65])     # the neighbors' average profile
    rho = np.array([0.50, 0.30, 0.20])        # the cell's own clean profile
    kappa = 0.15
    p = (1 - kappa) * rho + kappa * rho_bar
    kbar = min(p[g] / rho_bar[g] for g in range(3) if rho_bar[g] > 0)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.set_aspect("equal")
    ax.axis("off")
    tri = _bary(np.eye(3))
    ax.add_patch(Polygon(tri, closed=True, fill=False, ec=MUTED, lw=1))
    for corner, name, off in zip(tri, ("gene 1", "gene 2", "gene 3"),
                                 ((-0.06, -0.04), (0.06, -0.04), (0, 0.04))):
        ax.text(corner[0] + off[0], corner[1] + off[1], name, ha="center", fontsize=9)

    ks = np.linspace(0, kbar, 50)
    line = np.array([_bary((p - k * rho_bar) / (1 - k)) for k in ks])
    ax.plot(line[:, 0], line[:, 1], color=BLUE, lw=2.2, solid_capstyle="round")
    ax.plot(*np.array([_bary(rho_bar), _bary(p)]).T, color=MUTED, lw=0.8, ls=":")
    for k in (0.1, 0.2, 0.3):
        q = _bary((p - k * rho_bar) / (1 - k))
        ax.plot(*q, "o", ms=3.5, color=BLUE)
        ax.annotate(f"$\\kappa'$ = {k:g}", q, xytext=(8, 4), textcoords="offset points",
                    fontsize=7.5, color=MUTED)
    marks = [(rho_bar, "influx $\\bar\\rho_i$\n(neighbors' average)", ORANGE, (10, 0)),
             (p, "observed $p_i$  ($\\kappa'$ = 0)", INK, (-12, 6)),
             (rho, f"true clean $\\rho_i$  (true $\\kappa$ = {kappa:g})", VIOLET, (-12, -4)),
             ((p - kbar * rho_bar) / (1 - kbar),
              f"edge: $\\bar\\kappa_i$ = {kbar:.2f}\n(gene 3 would turn negative)", BLUE,
              (12, -16))]
    for point, text, color, off in marks:
        q = _bary(point)
        ax.plot(*q, "o", ms=8, color=color, mec="white", mew=1.5, zorder=4)
        ax.annotate(text, q, xytext=off, textcoords="offset points", fontsize=8,
                    ha="left" if off[0] >= 0 else "right", va="center")
    ax.set_title("Each assumed $\\kappa'$ gives a clean profile that fits the counts exactly.\n"
                 "Only the edge of the triangle stops the line: an upper bound, no lower one.",
                 fontsize=8.5)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 0.95)
    save(fig, "simplex")


def fig_toy_kappa_star() -> None:
    """A closed-form toy of what a breakdown point means."""
    kappa_true = 0.1
    # a fibroblast in niche A (few tumor neighbors) and B (many): clean levels and influx
    genes = {
        "gene 1: no response,\nonly spillover": (0.010, 0.010, 0.020, 0.100),
        "gene 2: real response,\nsame direction as spillover": (0.010, 0.018, 0.020, 0.100),
        "gene 3: real response,\nagainst spillover": (0.030, 0.040, 0.050, 0.010),
    }
    ks = np.linspace(0, 0.6, 241)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for (name, (clean_a, clean_b, influx_a, influx_b)), color in zip(genes.items(),
                                                                      (ORANGE, BLUE, AQUA)):
        p_a = (1 - kappa_true) * clean_a + kappa_true * influx_a
        p_b = (1 - kappa_true) * clean_b + kappa_true * influx_b
        diff = ((p_b - ks * influx_b) - (p_a - ks * influx_a)) / (1 - ks)
        ax.plot(ks, 100 * diff, color=color, lw=2, label=name)
        ratio = (p_b - p_a) / (influx_b - influx_a)
        if 0 < ratio < ks[-1]:
            ax.plot(ratio, 0, "o", ms=7, color=color, mec="white", mew=1.5, zorder=4)
            ax.annotate(f"$\\kappa^*$ = {ratio:.2f}", (ratio, 0),
                        xytext=(-50, -16) if ratio < 0.15 else (6, 8),
                        textcoords="offset points", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(kappa_true, color=MUTED, lw=0.8, ls="--")
    ax.text(kappa_true + 0.005, ax.get_ylim()[1] * 0.92, "true $\\kappa$", fontsize=8,
            color=MUTED)
    ax.set_xlabel("assumed spillover fraction $\\kappa'$")
    ax.set_ylabel("clean difference B - A\n(% of the cell's counts)")
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title("Toy: the same tissue, three findings, three breakdown points", fontsize=9)
    save(fig, "toy_kappa_star")


def fig_breakdown_schematic() -> None:
    """How a breakdown point is read off a sweep."""
    grid = [0, 0.05, 0.1, 0.2, 0.3, 0.4]
    xs = np.arange(len(grid))
    findings = [
        ("finding 1: holds at every $\\kappa$  ($\\kappa^*$ > 0.4)", BLUE,
         [0.60, 0.57, 0.55, 0.50, 0.46, 0.41], 0.12),
        ("finding 2: breaks  ($\\kappa^*$ = 0.2)", ORANGE,
         [0.40, 0.34, 0.27, 0.05, 0.0, -0.06], 0.11),
        ("finding 3: no finding (fails at $\\kappa$ = 0)", AQUA,
         [0.06, 0.05, 0.03, 0.0, -0.02, -0.04], 0.09),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    for offset, (name, color, est, half) in zip((-0.18, 0, 0.18), findings):
        est = np.asarray(est)
        lo, hi = est - half, est + half
        held = lo > 0
        broken = ~np.minimum.accumulate(held)          # once it fails, it stays failed
        ax.vlines(xs + offset, lo, hi, color=color, lw=2)
        ax.plot(xs[~broken] + offset, est[~broken], "o", ms=7, color=color, label=name)
        ax.plot(xs[broken] + offset, est[broken], "o", ms=7, mfc="white", mec=color, mew=1.6)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.annotate("first grid value whose interval\nreaches 0: $\\kappa^*$ = 0.2", (3, 0.01),
                xytext=(3.3, -0.33), fontsize=8, arrowprops={"arrowstyle": "->", "color": INK})
    ax.set_xticks(xs, [f"{k:g}" for k in grid])
    ax.set_xlabel("assumed spillover fraction $\\kappa$ (one refit per value, pooled over seeds)")
    ax.set_ylabel("finding (null = 0)")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_ylim(-0.42, 0.95)
    ax.set_title("Filled: the finding holds. Open: it does not (schematic)", fontsize=9)
    save(fig, "breakdown_schematic")


def _box(ax, xy, text, color, w=1.9, h=0.62, dashed=False, fill=True, fs=8.5):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color if fill else "white", ec=color if not fill else "none",
                                lw=1.4, ls="--" if dashed else "-", alpha=1.0))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color="white" if fill and color not in ("#e9e9e4",) else INK)


def _arrow(ax, a, b, color=INK, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, lw=1.1,
                                 color=color, ls="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def fig_model() -> None:
    """The generative model and its inference pieces."""
    obs = "#e9e9e4"
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    # observed inputs of the niche descriptor
    _box(ax, (1.4, 6.3), "neighbors' types\n$\\{t_j\\}$", obs, h=0.75)
    _box(ax, (3.6, 6.3), "image around the cell\n(cell masked) $\\Phi_i$", obs, w=2.3, h=0.75)
    _box(ax, (6.1, 6.3), "type $t_i$", obs, w=1.4)
    _box(ax, (2.5, 4.9),
         "niche descriptor $c_i$\n(attention over neighbor\ntypes $\\oplus\\ \\Phi_i$)",
         "#8a8984", w=2.7, h=0.95)
    _box(ax, (5.6, 4.9), "prior mean\n$m_\\psi(c_i, t_i)$", "#8a8984", w=1.9, h=0.8)
    _box(ax, (5.6, 3.5), "response $w_i$", VIOLET)
    _box(ax, (8.6, 4.9), "intrinsic state $z_i$\n$\\sim N(0, I)$", BLUE, h=0.8)
    _box(ax, (7.1, 2.2), "own clean profile\n$\\rho_i$ = softmax$(a(z_i) + Bw_i)$", "#8a8984",
         w=3.0, h=0.8)
    _box(ax, (10.5, 3.5), "neighbors' clean\nprofiles $\\rho_j$", obs, w=2.0, h=0.8)
    # the intrinsic path: the same cell decoded with w drawn from its prior
    _box(ax, (3.0, 3.3), "prior draw $\\breve w_i$\n$\\sim N(m_\\psi, I)$", VIOLET, w=1.9, h=0.8,
         fill=False, dashed=True)
    _box(ax, (10.5, 2.2), "influx $\\bar\\rho_i = \\sum_j \\beta_{ij}\\rho_j$", ORANGE, w=2.4)
    _box(ax, (8.8, 0.9), "$p_i = (1-\\kappa)\\rho_i + \\kappa\\bar\\rho_i$", "#8a8984", w=3.0)
    _box(ax, (4.6, 0.9), "counts $x_i \\sim$ Mult$(\\ell_i, p_i)$", obs, w=2.8)

    _arrow(ax, (1.4, 5.92), (2.0, 5.38))
    _arrow(ax, (3.6, 5.92), (3.1, 5.38))
    _arrow(ax, (3.85, 4.9), (4.65, 4.9))
    _arrow(ax, (6.1, 5.92), (5.9, 5.3))
    _arrow(ax, (5.6, 4.5), (5.6, 3.82))
    ax.text(5.72, 4.15, "$N(m_\\psi, I)$", fontsize=7.5, color=MUTED)
    _arrow(ax, (5.9, 3.18), (6.5, 2.62))
    _arrow(ax, (8.6, 4.5), (7.7, 2.62))
    _arrow(ax, (10.5, 3.1), (10.5, 2.52))
    _arrow(ax, (7.6, 1.8), (8.2, 1.22))
    _arrow(ax, (10.3, 1.88), (9.6, 1.22))
    _arrow(ax, (7.3, 0.9), (6.0, 0.9))
    _arrow(ax, (4.8, 4.55), (3.6, 3.72), color=VIOLET, dashed=True)
    _arrow(ax, (3.6, 2.9), (5.6, 2.3), color=VIOLET, dashed=True)
    ax.text(1.25, 2.15, "intrinsic path: $z_i$ must\nexplain the cell with $\\breve w_i$",
            fontsize=7.5, color=VIOLET)
    ax.text(10.62, 2.75, "stop-gradient", fontsize=7, color=MUTED)
    ax.text(10.15, 1.45, "fixed $\\kappa$", fontsize=7.5, color=MUTED)

    # adversary and probe (inference side)
    _box(ax, (10.6, 6.2), "adversary: predict the niche\n($y_i$, $\\Phi_i$) from $(z_i, t_i)$",
         "#c0392b", w=2.6, h=0.75, fill=False, dashed=True, fs=8)
    _arrow(ax, (9.2, 5.3), (9.8, 5.82), color="#c0392b", dashed=True)
    ax.text(8.0, 6.75, "z is trained to defeat it; the probe on held-out\ncells measures "
            "what is left (neighborhood information)", fontsize=7.5, color="#c0392b",
            ha="left")
    ax.text(0.15, 0.15, "gray: observed or fixed  ·  colored: latent or mixed  ·  "
            "the decoder never sees $t_i$; the $z$ encoder never sees the niche",
            fontsize=7.5, color=MUTED)
    save(fig, "model")


# -- simulated data -------------------------------------------------------------------

def _simulation():
    from discell.sim import simulate

    no_response = np.zeros(60, dtype=bool)
    no_response[:20] = True          # a third of the genes respond to nothing
    sim = simulate(n_cells=6000, n_genes=60, n_types=4, kappa=0.2, seed=0,
                   b_zero=no_response)
    return sim


def _config():
    import discell as dc

    # small networks and few epochs: minutes on a CPU
    return dc.Config(label_key="cell_type", image_key="X_image", kappa=0.2, device="cpu",
                     tile_cells=1024, epochs=100, eval_every=5, w_warmup_epochs=10,
                     patience=100, d_z=8, d_w=2, hidden=64, v_pcs=4)


KAPPAS = (0.0, 0.1, 0.2, 0.3, 0.4)
SEEDS = (0, 1, 2)


def _sweep(work: Path):
    import discell as dc

    sim = _simulation()
    adata = sim.to_anndata()
    dc.build_graph(adata, "cell_type")
    fits = dc.sensitivity.fit_sweep(adata, _config(), kappas=KAPPAS, seeds=SEEDS,
                                    out_dir=work / "sweep")
    return sim, adata, fits


def fig_sim_latents(work: Path) -> None:
    """z organizes by type, w by niche (simulated section)."""
    sim, adata, fits = _sweep(work)
    fits[(0.2, 0)].encode(adata)
    z = adata.obsm["X_discell_z"]
    w = adata.obsm["X_discell_w"]
    t = sim.t
    y = adata.obsm["discell_y"]
    # the niche, as one number: the share of type-1 neighbors, centered within type
    niche = y[:, 1].copy()
    for g in range(4):
        niche[t == g] -= niche[t == g].mean()
    rng = np.random.default_rng(0)
    show = rng.permutation(len(t))

    def pcs(m):
        m = m - m.mean(0)
        _, _, vt = np.linalg.svd(m, full_matrices=False)
        return m @ vt[:2].T

    fig, axes = plt.subplots(1, 5, figsize=(13.5, 2.9),
                             gridspec_kw={"width_ratios": [1.15, 1, 1, 1, 1]})
    pos = adata.obsm["spatial"]
    for g in range(4):
        rows = t == g
        axes[0].scatter(pos[rows, 0], pos[rows, 1], s=1.2, color=TYPES[g], lw=0,
                        label=f"type {g}", rasterized=True)
    axes[0].set_aspect("equal")
    axes[0].set_title("simulated section", fontsize=9)
    axes[0].legend(fontsize=7, markerscale=5, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                   ncol=4, handletextpad=0.1, columnspacing=0.6)
    lim = np.percentile(np.abs(niche), 98)
    panels = [(pcs(z), "type", "intrinsic state z, by type"),
              (pcs(z), "niche", "z, by niche (within type)"),
              (pcs(w), "type", "response w, by type"),
              (pcs(w), "niche", "w, by niche (within type)")]
    for ax, (xy, color_by, title) in zip(axes[1:], panels):
        if color_by == "type":
            ax.scatter(xy[show, 0], xy[show, 1], s=1.2, lw=0, rasterized=True,
                       c=np.array(TYPES)[t[show]])
        else:
            sc = ax.scatter(xy[show, 0], xy[show, 1], s=1.2, lw=0, rasterized=True,
                            c=niche[show], cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(title, fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(sc, ax=axes[1:], fraction=0.015, pad=0.01)
    cb.set_label("share of type-1 neighbors,\nminus the type's mean", fontsize=7.5)
    save(fig, "sim_latents", dpi=120)


def fig_sim_probe(work: Path) -> None:
    """Neighborhood information with and without the adversary (simulated section)."""
    import discell as dc

    sim, adata, fits = _sweep(work)
    with_adv = fits[(0.2, 0)]
    where = work / "no_adversary"
    if (where / "model.pt").exists():
        without = dc.Model.load(where, device="cpu")
    else:
        without = dc.fit(adata, dataclasses.replace(_config(), alpha_a=0.0), out_dir=where)
    reads = {name: dc.evaluate.heldout_probe(m, adata)
             for name, m in (("with the adversary", with_adv),
                             ("without ($\\alpha_a$ = 0)", without))}
    blocks = [("ridge", "comp"), ("mlp", "comp"), ("ridge", "img"), ("mlp", "img")]
    labels = ["linear probe\nneighbor types", "nonlinear probe\nneighbor types",
              "linear probe\nimage", "nonlinear probe\nimage"]
    xs = np.arange(len(blocks))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for offset, (name, rec), color in zip((-0.19, 0.19), reads.items(), (BLUE, MUTED)):
        values = [100 * rec[f][b]["var_fraction"] for f, b in blocks]
        bars = ax.bar(xs + offset, values, width=0.36, color=color, label=name)
        ax.bar_label(bars, fmt="%.0f", fontsize=7.5, padding=2, color=INK)
    ax.set_xticks(xs, labels, fontsize=8)
    ax.set_ylabel("neighborhood information in z\n(% of within-type variance)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Held-out probe on a simulated section ($\\kappa$ = 0.2, seed 0)", fontsize=9)
    save(fig, "sim_probe")


def fig_sim_sweep(work: Path) -> None:
    """Breakdown trajectories of a small sweep (simulated section)."""
    import discell as dc

    sim, adata, fits = _sweep(work)
    dirs = {}
    for (kappa, seed), model in fits.items():
        dirs[(kappa, seed)] = work / "sweep" / f"k{kappa:g}_s{seed}" / "draws"
        dc.sensitivity.fit_draws(model, adata,
                                 groups=("w_mi", "relocation_mean", "relocation_distribution"),
                                 n=200, min_cells=100, out_dir=dirs[(kappa, seed)])
    table = dc.sensitivity.finding_breakdown(dirs, grid=KAPPAS, seeds=SEEDS)
    members = [("niche_information_check", "niche information in w (check)"),
               ("relocation_finding", "relocation beyond its spillover part")]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for ax, (member, title) in zip(axes, members):
        dc.pl.breakdown_trajectory(table, member, ax=ax)
        ax.set_title(f"{title}: {table['members'][member]['status']}", fontsize=9)
        ax.set_xlabel("assumed spillover fraction $\\kappa$")
        for line in ax.get_lines()[:1]:
            line.set_color(BLUE)
        for coll in ax.collections[:1]:
            coll.set_color(BLUE)
    axes[0].set_ylabel("finding (null = 0)")
    fig.suptitle(f"A small sweep on a simulated section (true $\\kappa$ = 0.2; {len(SEEDS)} "
                 f"seeds; {100 * table['level']:.1f}% intervals)", fontsize=9, y=1.03)
    save(fig, "sim_sweep")


SCHEMATICS = {"sources": fig_sources, "simplex": fig_simplex, "toy_kappa_star": fig_toy_kappa_star,
              "breakdown_schematic": fig_breakdown_schematic, "model": fig_model}
DATA = {"sim_latents": fig_sim_latents, "sim_probe": fig_sim_probe, "sim_sweep": fig_sim_sweep}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="+", choices=sorted(SCHEMATICS) + sorted(DATA))
    parser.add_argument("--work", default=None, help="folder for the simulated fits")
    args = parser.parse_args()
    names = args.only or list(SCHEMATICS) + list(DATA)
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="discell_figs_"))
    import torch

    torch.set_num_threads(1)          # CPU fits are reproducible at one thread
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*CUDA.*")
    for name in names:
        if name in SCHEMATICS:
            SCHEMATICS[name]()
        else:
            DATA[name](work)


if __name__ == "__main__":
    main()
