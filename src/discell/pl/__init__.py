"""Plots of the readouts (matplotlib; the ``[viz]`` extra).

Each function draws from what :mod:`discell.tl` stored in the AnnData (or, for
the breakdown trajectory, from the table :func:`discell.sensitivity.breakdown`
returns, and for the per-gene split, from :func:`discell.tl.gene_parts` and
:func:`discell.tl.gene_assignments`), takes an optional ``ax`` and returns the axes.
"""

from __future__ import annotations

import numpy as np

_VIZ_MESSAGE = ("discell.pl needs matplotlib: pip install 'discell[viz]'")


def _axes(ax, figsize):
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:          # pragma: no cover - exercised without the extra
        raise ImportError(_VIZ_MESSAGE) from err
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def breakdown_trajectory(table: dict, member: str, *, ax=None):
    """One finding's seed-pooled estimate and interval across kappa, with its
    breakdown point; *table* is :func:`discell.sensitivity.breakdown`'s output."""
    ax = _axes(ax, (4.2, 3.2))
    entry = table["members"][member]
    points = [(float(k), v) for k, v in entry["trajectory"].items() if v is not None]
    kappas = np.array([k for k, _ in points])
    est = np.array([v["estimate"] for _, v in points])
    lo = np.array([v["ci"][0] for _, v in points])
    hi = np.array([v["ci"][1] for _, v in points])
    ax.fill_between(kappas, lo, hi, alpha=0.25, lw=0, label=f"{100 * table['level']:.1f}% CI")
    ax.plot(kappas, est, marker="o", ms=3, lw=1, label="pooled estimate")
    ax.axhline(0, color="0.3", lw=0.6)
    if entry.get("kappa_star") is not None:
        ax.axvline(entry["kappa_star"], color="#c9662a", ls="--", lw=0.8,
                   label=f"kappa* = {entry['kappa_star']:g}")
    ax.set_xlabel("assumed spillover fraction kappa")
    ax.set_title(f"{member}: {entry['status']}", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    return ax


def _nice_limit(x: float) -> float:
    for v in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0):
        if x <= v:
            return v
    return float(np.ceil(x))


def gene_map(parts: dict, gene: str, *, axes=None, shares=None, size: float | None = None,
             seed: int = 0):
    """One gene in one cell type across the section, from :func:`discell.tl.gene_parts`.

    Five panels: the observed share (per 1,000 counts of the cell), then the fold change
    of the gene's expected expression that the intrinsic state, the niche response and
    spillover each contribute against an average cell of the type (log2, red raises,
    blue lowers, one shared scale), then each cell's share of neighbors of
    ``parts["context_type"]`` (when given). Each dot is a cell of the type at its
    centroid; gray dots are all other cells. *shares*: the three parts' shares of the
    gene's variation (for example the seed means of :func:`discell.tl.gene_assignments`),
    printed on the part panels. Returns the axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:          # pragma: no cover - exercised without the extra
        raise ImportError(_VIZ_MESSAGE) from err
    has_context = parts.get("context") is not None
    n_panels = 5 if has_context else 4
    if axes is None:
        _, axes = plt.subplots(1, n_panels, figsize=(3.0 * n_panels, 3.2))
    c = list(parts["genes"]).index(gene)
    pos, nodes = np.asarray(parts["positions"]), np.asarray(parts["nodes"])
    other = np.ones(len(pos), dtype=bool)
    other[nodes] = False
    xy = pos[nodes]
    order = np.random.default_rng(seed).permutation(len(nodes))
    rate = 1e3 * np.asarray(parts["counts"])[:, c] / np.maximum(parts["totals"], 1.0)
    log2 = np.asarray(parts["phi"])[:, c, :] / np.log(2.0)
    lim = _nice_limit(float(np.percentile(np.abs(log2), 98)))
    s = size if size is not None else (0.5 if len(nodes) < 20_000 else 0.1)
    top = np.log1p(max(2.0, float(np.percentile(rate, 99.5))))
    panels = [("observed share", np.log1p(rate), dict(cmap="Purples", vmin=0, vmax=top))]
    for j, name in enumerate(("intrinsic state", "niche response", "spillover")):
        panels.append((name, log2[:, j], dict(cmap="RdBu_r", vmin=-lim, vmax=lim)))
    if has_context:
        panels.append((f"share of {parts['context_type']} neighbors",
                       np.nan_to_num(np.asarray(parts["context"])),
                       dict(cmap="Greens", vmin=0, vmax=1)))
    for j, (ax, (title, values, kw)) in enumerate(zip(np.ravel(axes), panels)):
        ax.scatter(pos[other, 0], pos[other, 1], s=0.02, c="#dedcd6", lw=0, rasterized=True)
        sc = ax.scatter(xy[order, 0], xy[order, 1], s=s, c=values[order], lw=0,
                        rasterized=True, **kw)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8)
        if j == 0:
            bar = ax.figure.colorbar(sc, ax=ax, fraction=0.046, orientation="horizontal")
            ticks = [v for v in (0, 1, 3, 10, 30, 100) if np.log1p(v) <= kw["vmax"]]
            bar.set_ticks(np.log1p(ticks), labels=[f"{v:g}" for v in ticks])
            bar.set_label("per 1,000 counts", fontsize=7)
        elif j <= 3:
            bar = ax.figure.colorbar(sc, ax=ax, fraction=0.046, orientation="horizontal")
            ticks = [-lim, 0.0, lim]
            bar.set_ticks(ticks, labels=[f"x1/{2 ** lim:g}", "x1", f"x{2 ** lim:g}"])
            bar.set_label("fold change vs. an average cell of the type", fontsize=7)
            if shares is not None:
                ax.text(0.03, 0.97, f"{100 * shares[j - 1]:.0f}%", transform=ax.transAxes,
                        ha="left", va="top", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        else:
            bar = ax.figure.colorbar(sc, ax=ax, fraction=0.046, orientation="horizontal")
            bar.set_label("share of neighbors (contact-weighted)", fontsize=7)
    np.ravel(axes)[0].figure.suptitle(
        f"{gene} in {parts['cell_type']} (assumed kappa = {parts['kappa']:g}); grey: other cells",
        fontsize=9)
    return axes


def gene_triangle(assignments: dict, cell_type: str, *, ax=None, size: float = 6.0):
    """The type's genes placed by their seed-mean shares of within-type variation
    (corners: intrinsic, response, spillover), from :func:`discell.tl.gene_assignments`.
    Color: the consensus assignment (gray, none). A filled point keeps its assignment at the
    first tested kappa at or above the kappa_bar given to ``gene_assignments`` (open when it
    does not, or when no kappa_bar was given)."""
    ax = _axes(ax, (4.2, 3.9))
    arr = assignments["arrays"]
    k = list(arr["type_names"]).index(cell_type)
    rows = np.flatnonzero(arr["population"][k])
    sh = np.clip(np.nan_to_num(arr["mean_shares"][k, rows]), 0.0, None)
    sh = sh / np.maximum(sh.sum(axis=1, keepdims=True), 1e-12)
    corners = np.array([[0.0, 0.0], [0.5, np.sqrt(3) / 2], [1.0, 0.0]])  # intr, resp, spill
    xy = sh @ corners
    assignment = arr["assignment"][k, rows]
    cut = assignments.get("first_tested_at_or_above")
    holds = (np.asarray(arr["kappa_star"][k, rows]) > cut) if cut is not None else \
        np.zeros(len(rows), dtype=bool)
    ax.plot(*np.vstack([corners, corners[:1]]).T, color="0.4", lw=0.8)
    gray = assignment < 0
    ax.scatter(xy[gray, 0], xy[gray, 1], s=size, color="#c9c8c2", lw=0, label="no assignment")
    colors = ("#2a78d6", "#4a3aa7", "#eb6834")
    for j, (name, color) in enumerate(zip(("intrinsic", "niche response", "spillover"),
                                           colors)):
        m = assignment == j
        ax.scatter(xy[m & holds, 0], xy[m & holds, 1], s=size * 2, color=color, lw=0,
                   label=f"{name}, holds beyond kappa_bar")
        ax.scatter(xy[m & ~holds, 0], xy[m & ~holds, 1], s=size * 2, facecolors="none",
                   edgecolors=color, lw=0.7, label=f"{name}, does not")
    for (x, y), text, ha, va in zip(corners, ("intrinsic", "response", "spillover"),
                                    ("right", "center", "left"), ("top", "bottom", "top")):
        ax.text(x, y, text, ha=ha, va=va, fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{cell_type}: {len(rows)} genes", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return ax


__all__ = ["breakdown_trajectory", "gene_map", "gene_triangle"]
