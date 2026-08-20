#!/usr/bin/env python3
"""Interactive pan/zoom view of a cell graph, for `ssh -X` or a forwarded port.

Backend selection is automatic, because the obvious choices each fail in a way
that is not catchable:

* **TkAgg is unusable under uv-managed CPython.** ``import tkinter`` succeeds,
  but ``_tkinter`` is statically linked and exposes no ``__file__``, which
  matplotlib's ``_tkagg`` needs to locate Tcl/Tk. Hence pyqt6 in the ``viz`` extra.
* **A dead ``$DISPLAY`` aborts Qt outright** rather than raising, so it cannot
  be caught. The display is probed with ``xdpyinfo`` first.

Order is QtAgg -> TkAgg -> WebAgg. Needs the extra::

    uv run --extra viz python -m discell.preprocess.plotting.show --sample <dir>
    uv run --extra viz python -m discell.preprocess.plotting.show --sample <dir> --backend WebAgg
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from discell.tiff import find_tissue_image
from discell.tiff import XENIUM_DEFAULT_CHANNELS, is_xenium_morphology, read_image_window
from discell.preprocess.plotting.cell_graph import (
    DEFAULT_GRAPH,
    plot_cell_graph,
)

log = logging.getLogger("discell.preprocess.plotting.show")


def display_reachable(timeout_s: float = 5.0) -> bool:
    """Whether ``$DISPLAY`` names an X server we can actually talk to.

    Worth probing rather than trusting: if ``$DISPLAY`` is set but the server is
    unreachable -- a stale ``ssh -X`` session, forwarding refused -- Qt does not
    raise, it prints "could not load the Qt platform plugin" and *aborts the
    process*. A try/except cannot recover from that, so the check has to happen
    before the backend is chosen.
    """
    import shutil

    display = os.environ.get("DISPLAY")
    if not display:
        return False
    probe = shutil.which("xdpyinfo")
    if probe is None:
        return True  # can't tell; let the toolkit try
    try:
        done = subprocess.run(
            [probe, "-display", display], capture_output=True, timeout=timeout_s
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return done.returncode == 0


def select_backend(preferred: str | None = None, webagg_port: int = 8988) -> str:
    """Pick an interactive matplotlib backend that actually loads here.

    Ordering matters on this setup. ``import tkinter`` succeeds under
    uv-managed CPython, but ``_tkinter`` is statically linked and exposes no
    ``__file__``, so matplotlib's ``_tkagg`` extension cannot locate Tcl/Tk and
    TkAgg raises ImportError at load time. Qt is the dependable X11 choice;
    WebAgg needs no display at all and is used whenever no X server is reachable.
    """

    import matplotlib

    display = os.environ.get("DISPLAY") if display_reachable() else None
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates += ["QtAgg", "TkAgg", "WebAgg"]
    seen: set[str] = set()
    ordered = [c for c in candidates if not (c in seen or seen.add(c))]

    rejected: dict[str, str] = {}
    for backend in ordered:
        if backend != "WebAgg" and not display:
            rejected[backend] = (
                "no reachable X server "
                f"(DISPLAY={os.environ.get('DISPLAY') or 'unset'})"
            )
            continue
        try:
            if backend == "WebAgg":
                matplotlib.rcParams["webagg.port"] = webagg_port
                matplotlib.rcParams["webagg.open_in_browser"] = False
            matplotlib.use(backend, force=True)
        except Exception as exc:  # missing toolkit, broken extension, no Tcl/Tk
            rejected[backend] = f"{type(exc).__name__}: {str(exc).splitlines()[-1][:110]}"
            continue
        for name, why in rejected.items():
            log.info("  backend %s unavailable (%s)", name, why)
        log.info("Using interactive backend: %s", backend)
        return backend

    raise RuntimeError(
        "No interactive backend available:\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in rejected.items())
        + "\nInstall the viz extra with: uv sync --extra viz"
    )


def show_interactive(
    adata,
    sample_dir: Path,
    graph: str = DEFAULT_GRAPH,
    label_key: str = "cluster",
    region: tuple[float, float, float, float] | None = None,
    overlay: bool = False,
    backend: str | None = None,
    webagg_port: int = 8988,
    edge_metric: str = "shared_wall_um",
    color_edges_by_metric: bool = False,
    edge_cmap: str = "viridis",
    max_gap_um: float | None = None,
    min_wall_um: float | None = None,
    max_centroid_um: float | None = None,
    min_apposed_um: float | None = None,
    channels: Sequence[int] | None = None,
    image_max_px: int = 6000,
    interpolation: str | None = None,
) -> None:
    """Open a pan/zoomable view, over ``ssh -X`` or in a browser.

    Vector layers are re-rasterised by the toolkit on every zoom, so this stays
    responsive where panning a 8000x10000 PNG in an image viewer does not.
    """

    chosen = select_backend(backend, webagg_port)
    import matplotlib.pyplot as plt

    cents = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if region is None:
        region = (
            float(cents[:, 0].min()), float(cents[:, 0].max()),
            float(cents[:, 1].min()), float(cents[:, 1].max()),
        )
    fig, ax = plt.subplots(figsize=(16, 12))
    if overlay:
        image_path = find_tissue_image(sample_dir)
        if image_path is not None:
            x0, x1, y0, y1 = region
            picked = channels
            if picked is None and is_xenium_morphology(image_path):
                picked = XENIUM_DEFAULT_CHANNELS
            image, _ = read_image_window(
                image_path, int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1)),
                max_px=image_max_px, channels=picked,
            )
            source_px = max(int(x1 - x0), int(y1 - y0))
            log.info("  %.0f um field: %d source px -> %d read (source is %.4f um/px)",
                     (x1 - x0) * float(adata.uns.get("microns_per_pixel", 1.0)),
                     source_px, max(image.shape[0], image.shape[1]),
                     float(adata.uns.get("microns_per_pixel", 1.0)))
            ax.imshow(image, extent=(x0, x1, y1, y0),
                      interpolation=interpolation or "nearest", zorder=1)

    drawn = plot_cell_graph(
        adata, ax, graph=graph, label_key=label_key, region=region,
        polygon_alpha=0.35 if overlay else 0.7,
        edge_color="#00e5ff" if overlay else "#111111",
        node_size=8.0,
        edge_metric=edge_metric, color_edges_by_metric=color_edges_by_metric,
        edge_cmap=edge_cmap, max_gap_um=max_gap_um, min_wall_um=min_wall_um,
        max_centroid_um=max_centroid_um, min_apposed_um=min_apposed_um,
    )
    short = edge_metric.replace("_um", "").replace("_", " ")
    kept = ""
    if drawn.get("edges_after_filter") != drawn.get("edges_before_filter"):
        kept = (f" — {drawn['edges_after_filter']:,}/{drawn['edges_before_filter']:,} "
                f"edges kept")
    ax.set_title(f"{adata.uns.get('sample', {}).get('sample_id', '')} — {graph} graph{kept}\n"
                 f"scroll/drag to zoom; edge width ∝ {short}")
    if color_edges_by_metric and drawn.get("edge_collection") is not None:
        bar = fig.colorbar(drawn["edge_collection"], ax=ax, fraction=0.03, pad=0.01)
        bar.set_label(f"{short} (µm)")
    plt.tight_layout()

    if chosen == "WebAgg":
        print(
            f"\nServing on http://localhost:{webagg_port}\n"
            f"  If this is a remote host, forward the port from your laptop:\n"
            f"    ssh -L {webagg_port}:localhost:{webagg_port} "
            f"{os.environ.get('USER', 'user')}@<this-host>\n"
            f"  then open http://localhost:{webagg_port}\n"
        )
    else:
        print(f"\nOpening {chosen} window on DISPLAY={os.environ.get('DISPLAY', '<unset>')}\n")
    plt.show()
