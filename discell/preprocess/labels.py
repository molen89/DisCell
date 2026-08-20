#!/usr/bin/env python3
"""10x's supplemental annotation files.

Cell-type and gene-panel groups ship as loose CSVs beside the outs directory,
and the clustering lives inside ``analysis.tar.gz``. All three are optional --
a sample without them still loads, just without those label columns.
"""

from __future__ import annotations

import io
import logging
import tarfile
from pathlib import Path

import pandas as pd

log = logging.getLogger("discell.preprocess.labels")


def read_cell_groups(xenium_dir: Path) -> pd.DataFrame:
    """Read the supplemental ``*_cell_groups.csv`` of curated cell-type labels.

    These are 10x's published annotations -- real cell types with a chosen
    palette -- rather than the unnamed graphclust clusters, so they are preferred
    as the default label when present. Not part of ``outs.zip``; download
    alongside it, e.g.::

        curl -O https://cf.10xgenomics.com/samples/xenium/3.0.0/<SAMPLE>/<SAMPLE>_cell_groups.csv

    Returns a frame indexed by ``cell_id`` with ``group`` and optionally
    ``color``; empty if no such file is present.
    """
    hits = sorted(xenium_dir.glob("*cell_groups.csv"))
    if not hits:
        return pd.DataFrame()
    frame = pd.read_csv(hits[0])
    if "cell_id" not in frame or "group" not in frame:
        log.warning("%s lacks cell_id/group columns -- ignored", hits[0].name)
        return pd.DataFrame()
    frame = frame.set_index("cell_id")
    log.info("  cell groups: %d cells, %d types (%s)", len(frame), frame["group"].nunique(),
             hits[0].name)
    return frame


def read_gene_groups(xenium_dir: Path) -> pd.DataFrame:
    """Read the supplemental ``*_gene_groups.csv`` of marker genes per cell type."""
    hits = sorted(xenium_dir.glob("*gene_groups.csv"))
    if not hits:
        return pd.DataFrame()
    frame = pd.read_csv(hits[0])
    if "gene" not in frame or "group" not in frame:
        log.warning("%s lacks gene/group columns -- ignored", hits[0].name)
        return pd.DataFrame()
    frame = frame.set_index("gene")
    log.info("  gene groups: %d genes, %d groups (%s)", len(frame), frame["group"].nunique(),
             hits[0].name)
    return frame


def read_clusters(xenium_dir: Path) -> pd.DataFrame:
    """Read every clustering from ``analysis.tar.gz`` or an ``analysis/`` dir.

    Returns a frame indexed by cell id with one column per clustering.
    """
    frames: dict[str, pd.Series] = {}

    tar_path = xenium_dir / "analysis.tar.gz"
    analysis_dir = xenium_dir / "analysis"
    if tar_path.exists():
        with tarfile.open(tar_path) as tar:
            for member in tar.getmembers():
                if not member.name.endswith("clusters.csv"):
                    continue
                name = Path(member.name).parent.name.replace("gene_expression_", "")
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                table = pd.read_csv(io.BytesIO(handle.read()))
                frames[name] = table.set_index("Barcode")["Cluster"]
    elif analysis_dir.is_dir():
        for csv in sorted(analysis_dir.glob("clustering/*/clusters.csv")):
            name = csv.parent.name.replace("gene_expression_", "")
            table = pd.read_csv(csv)
            frames[name] = table.set_index("Barcode")["Cluster"]
    else:
        log.info("No analysis outputs found -- no cluster labels")
        return pd.DataFrame()

    log.info("  %d clustering(s): %s", len(frames), ", ".join(sorted(frames)))
    return pd.DataFrame(frames)
