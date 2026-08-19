#!/usr/bin/env python3
"""Read 10x Visium HD binned outputs into an AnnData of raw counts.

The loader takes a sample directory (or a ``*_binned_outputs.tar.gz`` directly),
pulls out the count matrix for one bin size, attaches spatial coordinates,
Space Ranger cluster labels and sample-level metadata, then subsets to a set of
highly variable genes.

Counts stay raw. Normalisation is only ever done on a throwaway copy inside the
HVG step, so ``adata.X`` is always the integer UMI matrix.

The HVG panel is cached next to the sample as ``hvg_top_<n>_<bin_size>.json``.
Pass that file back via ``hvg_file=`` (or ``--hvg-file``) to skip HVG selection
entirely -- which is also how you pin one shared gene panel across many samples.

Layout this expects inside the tar (or a pre-extracted directory)::

    binned_outputs/square_008um/filtered_feature_bc_matrix.h5
    binned_outputs/square_008um/spatial/{scalefactors_json.json,tissue_positions.parquet}
    binned_outputs/square_008um/analysis/clustering/gene_expression_*/clusters.csv
    binned_outputs/square_008um/analysis/{umap,tsne,pca}/*/projection.csv

Note that ``square_002um`` ships without an ``analysis/`` directory, so no
cluster labels exist at that resolution.

Usage::

    python -m discell.data.visium_hd --self-test
    python -m discell.data.visium_hd --sample <dir-or-tar> --n-top-genes 1000 --out sample.h5ad
    python -m discell.data.visium_hd --sample <other> --hvg-file hvg_top_1000_square_008um.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tarfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("discell.data.visium_hd")

# --------------------------------------------------------------------------
# Defaults -- edit here rather than threading arguments through every call.
# --------------------------------------------------------------------------

#: Root holding one directory per Visium HD sample.
DEFAULT_DATA_ROOT = Path("/home/rmolen/cellxgene_all_visium/highres_raw_with_images/10x_VisHD")

#: Sample used by ``--self-test``. Smallest binned tar in the cohort (~0.77 GB,
#: 7356 bins at 8 um), so a full round-trip runs in well under a minute.
DEFAULT_TEST_SAMPLE = "Visium_HD_3prime_Arabidopsis"

#: Cohort-level metadata table written by the download pipeline.
DEFAULT_METADATA_CSV = "10x_visium_hd_merged.csv"

DEFAULT_BIN_SIZE = "square_008um"
BIN_SIZES = ("square_002um", "square_008um", "square_016um")

DEFAULT_MATRIX = "filtered"  # 'filtered' = tissue-covered bins only, 'raw' = every bin
DEFAULT_N_TOP_GENES = 1000
DEFAULT_HVG_FLAVOR = "seurat_v3"  # operates directly on raw counts
DEFAULT_HVG_FILENAME = "hvg_top_{n_top_genes}_{bin_size}.json"

#: Subdirectory (relative to the sample directory) holding files pulled out of
#: the tar. Extraction is skipped when these already exist.
DEFAULT_CACHE_DIRNAME = "binned_outputs_extracted"

#: ``s_008um_00262_00403-1`` -> bin size, array row, array col, gem group.
BARCODE_RE = re.compile(r"^s_(?P<um>\d+)um_(?P<row>\d+)_(?P<col>\d+)-(?P<gem>\d+)$")


@dataclass(frozen=True)
class ColumnNames:
    """Column names in the CSV/parquet files Space Ranger emits.

    Overriding these is how you adapt to a pipeline that renames things,
    without touching the reader functions themselves.
    """

    # analysis/clustering/*/clusters.csv and analysis/*/projection.csv
    barcode: str = "Barcode"
    cluster: str = "Cluster"

    # spatial/tissue_positions.parquet
    tp_barcode: str = "barcode"
    tp_in_tissue: str = "in_tissue"
    tp_array_row: str = "array_row"
    tp_array_col: str = "array_col"
    tp_pxl_row: str = "pxl_row_in_fullres"
    tp_pxl_col: str = "pxl_col_in_fullres"

    # 10x_visium_hd_merged.csv
    meta_sample_id: str = "sample_id"

    # obs/var columns this module writes
    gene_id: str = "gene_id"
    gene_name: str = "gene_name"
    array_row: str = "array_row"
    array_col: str = "array_col"


DEFAULT_COLUMNS = ColumnNames()


@dataclass
class HVGPanel:
    """A selected gene panel, as cached on disk."""

    gene_ids: list[str]
    gene_names: list[str]
    n_top_genes: int
    flavor: str
    bin_size: str
    sample_id: str
    source: str = ""
    n_genes_total: int = 0
    n_obs_used: int = 0
    stats: dict[str, list[float]] = field(default_factory=dict)
    created: str = ""
    format_version: int = 1

    def __len__(self) -> int:
        return len(self.gene_ids)

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame({"gene_id": self.gene_ids, "gene_name": self.gene_names})
        for key, values in self.stats.items():
            if len(values) == len(frame):
                frame[key] = values
        return frame


class VisiumHDError(RuntimeError):
    """Raised when a sample cannot be read as Visium HD binned output."""


# --------------------------------------------------------------------------
# Locating and unpacking a sample
# --------------------------------------------------------------------------


def resolve_sample(path: str | Path) -> tuple[Path, Path | None, str]:
    """Resolve *path* to ``(sample_dir, tar_path, sample_id)``.

    Accepts a sample directory, a ``*_binned_outputs.tar.gz`` file, or an
    already-extracted directory containing ``binned_outputs/``. ``tar_path`` is
    ``None`` when the data is already on disk unpacked.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise VisiumHDError(f"No such sample path: {path}")

    if path.is_file():
        if not path.name.endswith(".tar.gz"):
            raise VisiumHDError(f"Expected a directory or *.tar.gz, got {path}")
        sample_id = path.name.replace("_binned_outputs.tar.gz", "").replace(".tar.gz", "")
        return path.parent, path, sample_id

    if (path / "binned_outputs").is_dir():
        # Pointed straight at an extracted tree.
        return path, None, path.name

    tars = sorted(path.glob("*_binned_outputs.tar.gz"))
    if tars:
        if len(tars) > 1:
            log.warning("%d binned tars in %s, using %s", len(tars), path, tars[0].name)
        return path, tars[0], tars[0].name.replace("_binned_outputs.tar.gz", "")

    raise VisiumHDError(
        f"{path} contains neither binned_outputs/ nor a *_binned_outputs.tar.gz"
    )


def _wanted_member(rel_parts: Sequence[str], matrix: str, include_images: bool) -> bool:
    """Decide whether a tar member (path relative to the bin directory) is needed."""
    if not rel_parts:
        return False
    head = rel_parts[0]

    if head == f"{matrix}_feature_bc_matrix.h5":
        return True

    if head == "spatial":
        if len(rel_parts) < 2:
            return False
        name = rel_parts[-1]
        if name in ("scalefactors_json.json", "tissue_positions.parquet"):
            return True
        return include_images and name.startswith("tissue_") and name.endswith(".png")

    if head == "analysis":
        name = rel_parts[-1]
        if name == "clusters.csv":
            return True
        # PCA/UMAP/t-SNE embeddings, but not the bulky diffexp tables.
        return name == "projection.csv" and "diffexp" not in rel_parts

    return False


def _relative_to_bin(member_name: str, bin_size: str) -> list[str] | None:
    """Return the member path relative to ``.../<bin_size>/``, if it is under it."""
    parts = Path(member_name).parts
    try:
        idx = parts.index(bin_size)
    except ValueError:
        return None
    return list(parts[idx + 1 :])


def extract_bin_outputs(
    tar_path: Path,
    dest: Path,
    bin_size: str = DEFAULT_BIN_SIZE,
    matrix: str = DEFAULT_MATRIX,
    include_images: bool = False,
    force: bool = False,
) -> Path:
    """Extract the files needed for one bin size out of *tar_path* into *dest*.

    Returns the bin directory (``dest/<bin_size>``). Streams the tar in a single
    pass -- these archives are multi-GB, so seeking is not an option -- and stops
    early once the bin directory has been passed.
    """
    bin_dir = dest / bin_size
    matrix_h5 = bin_dir / f"{matrix}_feature_bc_matrix.h5"
    if matrix_h5.exists() and not force:
        log.info("Reusing extracted outputs: %s", bin_dir)
        return bin_dir

    log.info("Extracting %s from %s", bin_size, tar_path.name)
    started = time.time()
    dest.mkdir(parents=True, exist_ok=True)

    n_extracted = 0
    entered_bin_dir = False
    with tarfile.open(tar_path, "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            rel = _relative_to_bin(member.name, bin_size)
            if rel is None:
                # Left the bin subtree after having the essentials -- nothing
                # further in the archive can be relevant.
                if entered_bin_dir and matrix_h5.exists():
                    break
                continue
            entered_bin_dir = True
            if not _wanted_member(rel, matrix, include_images):
                continue

            target = bin_dir.joinpath(*rel)
            # Refuse anything that would escape the destination directory.
            if not str(target.resolve()).startswith(str(dest.resolve())):
                raise VisiumHDError(f"Unsafe path in archive: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            with open(target, "wb") as handle:
                while chunk := source.read(1 << 20):
                    handle.write(chunk)
            n_extracted += 1

    if not matrix_h5.exists():
        raise VisiumHDError(
            f"{matrix}_feature_bc_matrix.h5 not found for {bin_size} in {tar_path}"
        )
    log.info("Extracted %d files in %.1fs -> %s", n_extracted, time.time() - started, bin_dir)
    return bin_dir


def locate_bin_dir(
    sample_path: str | Path,
    bin_size: str = DEFAULT_BIN_SIZE,
    matrix: str = DEFAULT_MATRIX,
    cache_dir: str | Path | None = None,
    include_images: bool = False,
    force_extract: bool = False,
) -> tuple[Path, str, Path]:
    """Return ``(bin_dir, sample_id, sample_dir)``, extracting from tar if needed."""
    if bin_size not in BIN_SIZES:
        raise VisiumHDError(f"bin_size must be one of {BIN_SIZES}, got {bin_size!r}")

    sample_dir, tar_path, sample_id = resolve_sample(sample_path)

    # Prefer an already-extracted tree over unpacking the tar again.
    extracted = sample_dir / "binned_outputs" / bin_size
    if (extracted / f"{matrix}_feature_bc_matrix.h5").exists():
        log.info("Using pre-extracted %s", extracted)
        return extracted, sample_id, sample_dir

    if tar_path is None:
        raise VisiumHDError(f"No tar and no extracted {bin_size} under {sample_dir}")

    dest = Path(cache_dir) if cache_dir else sample_dir / DEFAULT_CACHE_DIRNAME
    bin_dir = extract_bin_outputs(
        tar_path, dest, bin_size, matrix, include_images, force_extract
    )
    return bin_dir, sample_id, sample_dir


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def read_counts(
    bin_dir: Path,
    matrix: str = DEFAULT_MATRIX,
    var_names: str = "gene_name",
    columns: ColumnNames = DEFAULT_COLUMNS,
):
    """Read the raw count matrix into an AnnData.

    ``adata.X`` holds unmodified integer UMI counts. Gene identifiers are kept in
    ``var[gene_id]`` and ``var[gene_name]`` regardless of which one indexes
    ``var_names``, so a panel selected on one sample can be matched on another.
    """
    import scanpy as sc

    h5 = bin_dir / f"{matrix}_feature_bc_matrix.h5"
    if not h5.exists():
        raise VisiumHDError(f"Missing count matrix: {h5}")

    log.info("Reading %s", h5)
    adata = sc.read_10x_h5(h5)

    ids = adata.var["gene_ids"] if "gene_ids" in adata.var else pd.Series(adata.var_names)
    adata.var[columns.gene_id] = np.asarray(ids, dtype=object)
    adata.var[columns.gene_name] = np.asarray(adata.var_names, dtype=object)
    adata.var.drop(columns=["gene_ids"], errors="ignore", inplace=True)

    if var_names == "gene_id":
        adata.var_names = pd.Index(adata.var[columns.gene_id].astype(str))
    elif var_names != "gene_name":
        raise VisiumHDError(f"var_names must be 'gene_name' or 'gene_id', got {var_names!r}")
    adata.var_names_make_unique()

    adata.obs_names = adata.obs_names.astype(str)
    log.info("  %d bins x %d genes", adata.n_obs, adata.n_vars)
    return adata


def parse_barcode_grid(barcodes: Iterable[str]) -> pd.DataFrame:
    """Recover ``array_row``/``array_col`` from Visium HD barcode strings.

    The bin grid position is encoded in the barcode itself
    (``s_008um_00262_00403-1``), so coordinates are available even without a
    parquet engine installed.
    """
    rows, cols = [], []
    for barcode in barcodes:
        match = BARCODE_RE.match(str(barcode))
        if match is None:
            rows.append(-1)
            cols.append(-1)
        else:
            rows.append(int(match.group("row")))
            cols.append(int(match.group("col")))
    return pd.DataFrame({"array_row": rows, "array_col": cols}, dtype=np.int32)


def read_scalefactors(bin_dir: Path) -> dict:
    path = bin_dir / "spatial" / "scalefactors_json.json"
    if not path.exists():
        log.warning("No scalefactors_json.json under %s", bin_dir)
        return {}
    return json.loads(path.read_text())


def attach_spatial(
    adata,
    bin_dir: Path,
    columns: ColumnNames = DEFAULT_COLUMNS,
) -> None:
    """Attach bin coordinates to ``obs`` and ``obsm['spatial']``.

    Grid coordinates always come from the barcode. Full-resolution pixel
    coordinates come from ``tissue_positions.parquet`` when a parquet engine is
    available; when it is not, ``obsm['spatial']`` falls back to micron positions
    derived from the grid and ``bin_size_um``, which is exact for a regular grid.
    """
    grid = parse_barcode_grid(adata.obs_names)
    unparsed = int((grid["array_row"] < 0).sum())
    if unparsed:
        log.warning("%d barcodes did not match the Visium HD pattern", unparsed)
    adata.obs[columns.array_row] = grid["array_row"].to_numpy()
    adata.obs[columns.array_col] = grid["array_col"].to_numpy()

    scalefactors = read_scalefactors(bin_dir)
    adata.uns.setdefault("spatial_scalefactors", scalefactors)

    positions = bin_dir / "spatial" / "tissue_positions.parquet"
    pixel = None
    if positions.exists():
        try:
            table = pd.read_parquet(positions)
        except ImportError:
            log.warning(
                "tissue_positions.parquet present but no parquet engine installed "
                "(pip install pyarrow) -- falling back to micron coordinates "
                "derived from the bin grid"
            )
        except Exception as exc:  # corrupt or unexpected schema
            log.warning("Could not read %s: %s", positions.name, exc)
        else:
            table = table.set_index(columns.tp_barcode)
            table = table.reindex(adata.obs_names)
            for src, dst in (
                (columns.tp_in_tissue, "in_tissue"),
                (columns.tp_pxl_row, columns.tp_pxl_row),
                (columns.tp_pxl_col, columns.tp_pxl_col),
            ):
                if src in table:
                    adata.obs[dst] = table[src].to_numpy()
            if columns.tp_pxl_row in table and columns.tp_pxl_col in table:
                pixel = table[[columns.tp_pxl_col, columns.tp_pxl_row]].to_numpy(dtype=np.float32)

    if pixel is not None and not np.isnan(pixel).all():
        adata.obsm["spatial"] = pixel
        adata.uns["spatial_units"] = "fullres_pixels"
    else:
        bin_um = float(scalefactors.get("bin_size_um", 0.0))
        scale = bin_um if bin_um > 0 else 1.0
        adata.obsm["spatial"] = np.column_stack(
            [grid["array_col"].to_numpy(np.float32), grid["array_row"].to_numpy(np.float32)]
        ) * scale
        adata.uns["spatial_units"] = "microns" if bin_um > 0 else "bins"


def attach_annotations(
    adata,
    bin_dir: Path,
    columns: ColumnNames = DEFAULT_COLUMNS,
    default_label: str = "graphclust",
) -> list[str]:
    """Attach Space Ranger cluster labels and embeddings.

    Every ``analysis/clustering/gene_expression_*/clusters.csv`` becomes a
    categorical ``obs`` column (``graphclust``, ``kmeans_2_clusters``, ...), and
    every ``projection.csv`` becomes an ``obsm`` entry. ``square_002um`` has no
    ``analysis/`` directory, so this is a no-op there.
    """
    analysis = bin_dir / "analysis"
    label_columns: list[str] = []
    if not analysis.is_dir():
        log.info("No analysis/ under %s -- no cluster labels at this bin size", bin_dir)
        adata.uns["label_columns"] = label_columns
        return label_columns

    for clusters_csv in sorted(analysis.glob("clustering/*/clusters.csv")):
        name = clusters_csv.parent.name.replace("gene_expression_", "")
        frame = pd.read_csv(clusters_csv).set_index(columns.barcode)
        values = frame[columns.cluster].reindex(adata.obs_names)
        # Plain object strings, not pandas StringDtype: anndata refuses to write
        # StringArray-backed categoricals by default.
        labels = values.map(lambda v: None if pd.isna(v) else str(v)).astype(object)
        adata.obs[name] = pd.Categorical(labels)
        label_columns.append(name)

    for projection_csv in sorted(analysis.glob("*/*/projection.csv")):
        kind = projection_csv.parents[1].name  # umap / tsne / pca
        frame = pd.read_csv(projection_csv).set_index(columns.barcode)
        frame = frame.reindex(adata.obs_names)
        key = f"X_{kind}_10x"
        adata.obsm[key] = frame.to_numpy(dtype=np.float32)

    log.info("Attached %d label column(s): %s", len(label_columns), ", ".join(label_columns))
    adata.uns["label_columns"] = label_columns
    if default_label in label_columns:
        adata.uns["default_label"] = default_label
    elif label_columns:
        adata.uns["default_label"] = label_columns[0]
    return label_columns


def find_metadata_csv(sample_dir: Path, filename: str = DEFAULT_METADATA_CSV) -> Path | None:
    """Look for the cohort metadata table in the sample dir and its ancestors."""
    for parent in (sample_dir, *sample_dir.parents[:3]):
        candidate = parent / filename
        if candidate.exists():
            return candidate
    return None


def attach_sample_metadata(
    adata,
    sample_id: str,
    sample_dir: Path,
    metadata_csv: str | Path | None = None,
    columns: ColumnNames = DEFAULT_COLUMNS,
) -> dict:
    """Record sample-level metadata in ``uns['sample']`` and ``obs['sample_id']``."""
    meta: dict = {"sample_id": sample_id}

    path = Path(metadata_csv) if metadata_csv else find_metadata_csv(sample_dir)
    if path and path.exists():
        table = pd.read_csv(path)
        if columns.meta_sample_id in table:
            hit = table[table[columns.meta_sample_id] == sample_id]
            if len(hit):
                row = hit.iloc[0].dropna().to_dict()
                meta.update({k: str(v) for k, v in row.items()})
            else:
                log.warning("%s not found in %s", sample_id, path.name)
    else:
        log.info("No cohort metadata CSV found for %s", sample_id)

    adata.obs["sample_id"] = pd.Categorical([sample_id] * adata.n_obs)
    adata.uns["sample"] = meta
    return meta


# --------------------------------------------------------------------------
# Highly variable genes
# --------------------------------------------------------------------------


def select_hvgs(
    adata,
    n_top_genes: int = DEFAULT_N_TOP_GENES,
    flavor: str = DEFAULT_HVG_FLAVOR,
    subsample: int | None = None,
    seed: int = 0,
    columns: ColumnNames = DEFAULT_COLUMNS,
    bin_size: str = DEFAULT_BIN_SIZE,
    sample_id: str = "",
) -> HVGPanel:
    """Rank genes by variability and return the top *n_top_genes* as a panel.

    ``seurat_v3`` consumes raw counts directly. The other flavors need
    log-normalised input, which is computed on a copy so *adata* is untouched --
    ``adata.X`` stays raw either way.

    *subsample* caps how many bins feed the ranking, which matters at
    ``square_002um`` where a sample can carry millions of near-empty bins.
    """
    import scanpy as sc

    n_top_genes = min(n_top_genes, adata.n_vars)
    if n_top_genes < 1:
        raise VisiumHDError("n_top_genes must be >= 1")

    source = adata
    if subsample is not None and subsample < adata.n_obs:
        rng = np.random.default_rng(seed)
        picked = rng.choice(adata.n_obs, size=subsample, replace=False)
        source = adata[np.sort(picked)]
        log.info("Ranking HVGs on %d of %d bins", subsample, adata.n_obs)

    work = source.copy()
    if flavor != "seurat_v3":
        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)

    log.info("Selecting top %d genes (flavor=%s)", n_top_genes, flavor)
    started = time.time()
    sc.pp.highly_variable_genes(work, n_top_genes=n_top_genes, flavor=flavor)
    log.info("  done in %.1fs", time.time() - started)

    hvg = work.var[work.var["highly_variable"]].copy()
    rank_col = next(
        (c for c in ("highly_variable_rank", "dispersions_norm", "variances_norm") if c in hvg),
        None,
    )
    if rank_col == "highly_variable_rank":
        hvg = hvg.sort_values(rank_col, ascending=True)
    elif rank_col is not None:
        hvg = hvg.sort_values(rank_col, ascending=False)

    stats = {
        col: hvg[col].astype(float).tolist()
        for col in ("means", "variances", "variances_norm", "dispersions", "dispersions_norm")
        if col in hvg
    }

    return HVGPanel(
        gene_ids=hvg[columns.gene_id].astype(str).tolist(),
        gene_names=hvg[columns.gene_name].astype(str).tolist(),
        n_top_genes=n_top_genes,
        flavor=flavor,
        bin_size=bin_size,
        sample_id=sample_id,
        n_genes_total=int(adata.n_vars),
        n_obs_used=int(source.n_obs),
        stats=stats,
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def hvg_panel_path(
    sample_dir: Path,
    n_top_genes: int = DEFAULT_N_TOP_GENES,
    bin_size: str = DEFAULT_BIN_SIZE,
    filename: str = DEFAULT_HVG_FILENAME,
) -> Path:
    return sample_dir / filename.format(n_top_genes=n_top_genes, bin_size=bin_size)


def save_hvg_panel(panel: HVGPanel, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(panel), indent=2))
    log.info("Wrote HVG panel (%d genes) -> %s", len(panel), path)
    return path


def load_hvg_panel(path: str | Path) -> HVGPanel:
    path = Path(path)
    payload = json.loads(path.read_text())
    known = {f for f in HVGPanel.__dataclass_fields__}
    panel = HVGPanel(**{k: v for k, v in payload.items() if k in known})
    panel.source = str(path)
    log.info("Loaded HVG panel (%d genes) from %s", len(panel), path)
    return panel


def _pad_missing_genes(subset, panel: HVGPanel, missing: Sequence[str], columns: ColumnNames):
    """Append all-zero columns for *missing* panel genes, then restore panel order."""
    import anndata as ad
    import scipy.sparse as sp

    name_by_id = dict(zip(panel.gene_ids, panel.gene_names))
    pad_names = [name_by_id.get(gene_id, gene_id) for gene_id in missing]
    # var_names may index either identifier; match whichever subset is using.
    indexes_by_name = subset.var.index.equals(
        pd.Index(subset.var[columns.gene_name].astype(str))
    )
    pad_var = pd.DataFrame(
        {columns.gene_id: list(missing), columns.gene_name: pad_names},
        index=pd.Index(pad_names if indexes_by_name else list(missing),
                       name=subset.var.index.name),
    )

    counts = sp.csr_matrix(subset.X) if not sp.issparse(subset.X) else subset.X.tocsr()
    zeros = sp.csr_matrix((subset.n_obs, len(missing)), dtype=counts.dtype)
    padded = ad.AnnData(
        X=sp.hstack([counts, zeros], format="csr"),
        obs=subset.obs.copy(),
        var=pd.concat([subset.var, pad_var]),
        obsm={k: v for k, v in subset.obsm.items()},
        uns=dict(subset.uns),
    )
    padded.var_names_make_unique()

    order = pd.Series(np.arange(padded.n_vars), index=padded.var[columns.gene_id].astype(str))
    order = order[~order.index.duplicated()].reindex(panel.gene_ids).dropna().astype(int)
    return padded[:, order.to_numpy()].copy()


def subset_to_panel(
    adata,
    panel: HVGPanel,
    on_missing: str = "warn",
    columns: ColumnNames = DEFAULT_COLUMNS,
):
    """Subset *adata* to the panel's genes, in the panel's order.

    Order is preserved so every sample presents the same feature axis to a
    model. Genes absent from this sample are handled per *on_missing*:
    ``'error'`` raises, ``'warn'`` drops them, ``'pad'`` inserts all-zero
    columns to keep the axis width fixed.
    """
    by_id = pd.Series(np.arange(adata.n_vars), index=adata.var[columns.gene_id].astype(str))
    by_id = by_id[~by_id.index.duplicated()]
    positions = by_id.reindex(panel.gene_ids)

    # Older panels, or matrices lacking Ensembl IDs, fall back to gene symbols.
    if positions.isna().all():
        by_name = pd.Series(np.arange(adata.n_vars), index=adata.var[columns.gene_name].astype(str))
        by_name = by_name[~by_name.index.duplicated()]
        positions = by_name.reindex(panel.gene_names)

    missing = [g for g, p in zip(panel.gene_ids, positions) if pd.isna(p)]
    if missing:
        message = f"{len(missing)}/{len(panel)} panel genes absent from {adata.n_vars} in sample"
        if on_missing == "error":
            raise VisiumHDError(f"{message}: {missing[:10]}")
        log.warning("%s (e.g. %s)", message, ", ".join(missing[:5]))

    found = positions.dropna().astype(int).to_numpy()
    subset = adata[:, found].copy()

    if missing and on_missing == "pad":
        subset = _pad_missing_genes(subset, panel, missing, columns)
        subset.var["in_hvg_panel"] = ~subset.var[columns.gene_id].astype(str).isin(set(missing))
    else:
        subset.var["in_hvg_panel"] = True

    subset.uns["hvg_panel"] = {
        "n_top_genes": panel.n_top_genes,
        "flavor": panel.flavor,
        "bin_size": panel.bin_size,
        "sample_id": panel.sample_id,
        "source": panel.source,
        "n_missing": len(missing),
    }
    log.info("Subset to %d genes (panel order preserved)", subset.n_vars)
    return subset


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def load_sample(
    sample_path: str | Path,
    bin_size: str = DEFAULT_BIN_SIZE,
    matrix: str = DEFAULT_MATRIX,
    n_top_genes: int = DEFAULT_N_TOP_GENES,
    hvg_file: str | Path | None = None,
    hvg_flavor: str = DEFAULT_HVG_FLAVOR,
    hvg_subsample: int | None = None,
    save_panel: bool = True,
    panel_filename: str = DEFAULT_HVG_FILENAME,
    on_missing: str = "warn",
    var_names: str = "gene_name",
    cache_dir: str | Path | None = None,
    metadata_csv: str | Path | None = None,
    columns: ColumnNames = DEFAULT_COLUMNS,
    default_label: str = "graphclust",
    include_all_genes: bool = False,
    seed: int = 0,
):
    """Load one Visium HD sample as raw counts subset to a HVG panel.

    Set *hvg_file* to reuse a cached panel and skip selection. Otherwise a panel
    is selected and, unless *save_panel* is off, written next to the sample as
    ``hvg_top_<n>_<bin_size>.json`` for exactly that purpose next time.

    With *include_all_genes*, the full gene set is returned with the panel
    flagged in ``var['highly_variable']`` rather than subset away.
    """
    bin_dir, sample_id, sample_dir = locate_bin_dir(
        sample_path, bin_size=bin_size, matrix=matrix, cache_dir=cache_dir
    )

    adata = read_counts(bin_dir, matrix=matrix, var_names=var_names, columns=columns)
    attach_spatial(adata, bin_dir, columns=columns)
    attach_annotations(adata, bin_dir, columns=columns, default_label=default_label)
    attach_sample_metadata(adata, sample_id, sample_dir, metadata_csv, columns=columns)
    adata.uns["bin_size"] = bin_size
    adata.uns["matrix"] = matrix
    adata.uns["counts_are_raw"] = True

    if hvg_file is not None:
        panel = load_hvg_panel(hvg_file)
        if panel.bin_size != bin_size:
            log.warning(
                "Panel was built on %s but loading %s", panel.bin_size, bin_size
            )
    else:
        panel = select_hvgs(
            adata,
            n_top_genes=n_top_genes,
            flavor=hvg_flavor,
            subsample=hvg_subsample,
            seed=seed,
            columns=columns,
            bin_size=bin_size,
            sample_id=sample_id,
        )
        if save_panel:
            path = hvg_panel_path(sample_dir, n_top_genes, bin_size, panel_filename)
            panel.source = str(path)  # set before writing so the file records its own path
            save_hvg_panel(panel, path)

    if include_all_genes:
        selected = set(panel.gene_ids)
        adata.var["highly_variable"] = adata.var[columns.gene_id].astype(str).isin(selected)
        adata.uns["hvg_panel"] = {"n_top_genes": panel.n_top_genes, "source": panel.source}
        return adata

    return subset_to_panel(adata, panel, on_missing=on_missing, columns=columns)


# --------------------------------------------------------------------------
# CLI and self-test
# --------------------------------------------------------------------------


def _assert_raw_counts(adata) -> None:
    import scipy.sparse as sp

    data = adata.X.data if sp.issparse(adata.X) else np.asarray(adata.X).ravel()
    sample = data[:100_000]
    assert sample.min() >= 0, "negative values in count matrix"
    assert np.allclose(sample, np.round(sample)), "X is not integer-valued -- counts were modified"


def self_test(sample_path: str | Path, bin_size: str, n_top_genes: int, keep: bool) -> int:
    """End-to-end check on one sample. Returns a process exit code."""
    import tempfile

    sample_dir, _, sample_id = resolve_sample(sample_path)
    panel_path = hvg_panel_path(sample_dir, n_top_genes, bin_size)
    created_panel = not panel_path.exists()

    print(f"\n=== self-test: {sample_id} / {bin_size} / top {n_top_genes} ===\n")

    print("[1/8] load + select HVGs")
    adata = load_sample(sample_path, bin_size=bin_size, n_top_genes=n_top_genes)
    assert adata.n_vars == n_top_genes, f"expected {n_top_genes} genes, got {adata.n_vars}"
    assert adata.n_obs > 0, "no bins loaded"
    print(f"      {adata.n_obs} bins x {adata.n_vars} genes")

    print("[2/8] counts are raw")
    _assert_raw_counts(adata)
    print(f"      total UMIs = {int(adata.X.sum()):,}")

    print("[3/8] gene identity preserved")
    assert adata.var["gene_id"].notna().all(), "missing gene_ids"
    assert adata.var["gene_name"].notna().all(), "missing gene_names"
    print(f"      first genes: {', '.join(adata.var_names[:5])}")

    print("[4/8] spatial + annotations")
    assert "spatial" in adata.obsm, "no spatial coordinates"
    assert adata.obs["array_row"].max() > 0, "barcode grid parse failed"
    labels = list(adata.uns.get("label_columns", []))
    print(f"      spatial units = {adata.uns.get('spatial_units')}; labels = {labels or 'none'}")
    if bin_size != "square_002um":
        assert labels, "expected cluster labels at this bin size"
        primary = adata.uns["default_label"]
        print(f"      {primary}: {adata.obs[primary].nunique()} clusters")

    print("[5/8] panel cached to disk")
    assert panel_path.exists(), f"panel not written to {panel_path}"
    panel = load_hvg_panel(panel_path)
    assert len(panel) == n_top_genes, "panel length mismatch"
    print(f"      {panel_path.name}")

    print("[6/8] reload from panel skips HVG step")
    started = time.time()
    reloaded = load_sample(sample_path, bin_size=bin_size, hvg_file=panel_path)
    assert list(reloaded.var["gene_id"]) == list(adata.var["gene_id"]), "gene order diverged"
    assert reloaded.n_obs == adata.n_obs, "bin count diverged"
    assert int(reloaded.X.sum()) == int(adata.X.sum()), "counts diverged on reload"
    print(f"      identical gene axis, {time.time() - started:.1f}s")

    print("[7/8] panel genes absent from the sample")
    # Stand in for reusing one cohort panel on a sample that lacks some genes.
    foreign = HVGPanel(
        gene_ids=list(panel.gene_ids[:50]) + ["NOT_A_GENE_1", "NOT_A_GENE_2"],
        gene_names=list(panel.gene_names[:50]) + ["NOT_A_GENE_1", "NOT_A_GENE_2"],
        n_top_genes=52,
        flavor=panel.flavor,
        bin_size=bin_size,
        sample_id=sample_id,
    )
    bin_dir, _, _ = locate_bin_dir(sample_path, bin_size=bin_size)
    raw = read_counts(bin_dir)
    dropped = subset_to_panel(raw, foreign, on_missing="warn")
    assert dropped.n_vars == 50, f"expected the 2 unknown genes dropped, got {dropped.n_vars}"
    padded = subset_to_panel(raw, foreign, on_missing="pad")
    assert padded.n_vars == 52, f"pad should hold the axis at 52, got {padded.n_vars}"
    assert list(padded.var["gene_id"]) == foreign.gene_ids, "pad broke panel order"
    assert padded.X[:, 50:].sum() == 0, "padded columns should be all-zero"
    assert padded.var["in_hvg_panel"].sum() == 50, "padded genes wrongly flagged as real"
    try:
        subset_to_panel(raw, foreign, on_missing="error")
    except VisiumHDError:
        pass
    else:
        raise AssertionError("on_missing='error' did not raise")
    print("      warn drops 2, pad zero-fills 2, error raises")

    print("[8/8] h5ad round-trip")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sample.h5ad"
        adata.write_h5ad(out)
        import anndata as ad

        back = ad.read_h5ad(out)
        assert back.shape == adata.shape, "shape changed through h5ad"
        assert int(back.X.sum()) == int(adata.X.sum()), "counts changed through h5ad"
        assert list(back.var["gene_name"]) == list(adata.var["gene_name"]), "genes changed"
        print(f"      {out.name} ({out.stat().st_size / 1e6:.1f} MB) reads back identical")

    if created_panel and not keep:
        panel_path.unlink()
        print(f"\n      (removed test panel {panel_path.name}; pass --keep to retain)")

    print("\nAll checks passed.\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample",
        default=str(DEFAULT_DATA_ROOT / DEFAULT_TEST_SAMPLE),
        help="sample directory, extracted binned_outputs tree, or *_binned_outputs.tar.gz",
    )
    parser.add_argument("--bin-size", default=DEFAULT_BIN_SIZE, choices=BIN_SIZES)
    parser.add_argument("--matrix", default=DEFAULT_MATRIX, choices=("filtered", "raw"))
    parser.add_argument("--n-top-genes", type=int, default=DEFAULT_N_TOP_GENES)
    parser.add_argument("--hvg-file", default=None, help="reuse a cached panel; skips HVG selection")
    parser.add_argument("--hvg-flavor", default=DEFAULT_HVG_FLAVOR,
                        choices=("seurat_v3", "seurat", "cell_ranger"))
    parser.add_argument("--hvg-subsample", type=int, default=None,
                        help="cap the bins used for HVG ranking (useful at square_002um)")
    parser.add_argument("--no-save-panel", action="store_true", help="do not cache the panel")
    parser.add_argument("--var-names", default="gene_name", choices=("gene_name", "gene_id"))
    parser.add_argument("--on-missing", default="warn", choices=("warn", "error", "pad"),
                        help="how to handle panel genes absent from this sample")
    parser.add_argument("--all-genes", action="store_true",
                        help="keep every gene, flagging the panel in var['highly_variable']")
    parser.add_argument("--cache-dir", default=None, help="where to extract tar members")
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--out", default=None, help="write the result to this .h5ad")
    parser.add_argument("--self-test", action="store_true", help="run end-to-end checks and exit")
    parser.add_argument("--keep", action="store_true", help="keep artifacts written by --self-test")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.self_test:
        return self_test(args.sample, args.bin_size, args.n_top_genes, args.keep)

    adata = load_sample(
        args.sample,
        bin_size=args.bin_size,
        matrix=args.matrix,
        n_top_genes=args.n_top_genes,
        hvg_file=args.hvg_file,
        hvg_flavor=args.hvg_flavor,
        hvg_subsample=args.hvg_subsample,
        save_panel=not args.no_save_panel,
        on_missing=args.on_missing,
        var_names=args.var_names,
        cache_dir=args.cache_dir,
        metadata_csv=args.metadata_csv,
        include_all_genes=args.all_genes,
    )
    print(adata)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(out)
        log.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
