#!/usr/bin/env python3
"""The fixed false-positive floor (todo 8.15b; devlog 2026-09-24 21:20, B).

``p_i = (1 - kappa_i - eta_i) rho_i + kappa_i rho_bar_i + eta_i u``: a share
``eta_i`` of each cell's counts is taken to be decoding / off-target noise,
spread evenly over the panel (``u = 1/G``; a stated approximation -- genomic
DNA binding is not gene-uniform). ``eta_i = min(lambda_i / l_i, ETA_CAP)``,
``l_i`` the cell's panel count total and ``lambda_i`` the expected
false-positive count of the whole panel in that cell.

``lambda`` comes from the Xenium cell table (``cells.parquet``; not the 10x
summary): per section, the mean per-cell count of each control kind --
negative-control probe, negative-control codeword and genomic control (DNA
counted) -- divided by the number of features of that kind in the panel and
multiplied by the number of panel genes G, then summed over kinds.

Per cell or per section is decided by data, across both sections: if the
per-cell control count correlates with segmented area within type (Spearman
>= AREA_RHO_MIN on every section), ``lambda_i`` is proportional to area with
the section total fixed; else one ``lambda`` per section.

Usage (the step-1 read, both sections)::

    python -m discell.model.fp_floor --section xenium_prime_ovarian_cancer_ffpe:full \\
        --section gse315411_pdltma06_11_prime_solo:pdl018d --out step1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

#: eta_i is capped here: a cell with very few counts is not declared mostly noise
ETA_CAP = 0.2
#: the pre-registered area threshold (within-type Spearman, every section)
AREA_RHO_MIN = 0.3
#: cells-table column -> the panel's feature_type of that control kind
CONTROL_KINDS = {"control_probe_counts": "Negative Control Probe",
                 "control_codeword_counts": "Negative Control Codeword",
                 "genomic_control_counts": "Genomic Control"}
#: types smaller than this are left out of the within-type Spearman
MIN_TYPE_CELLS = 200


def xenium_dir(dataset: str, variant: str) -> Path:
    from discell import paths

    return Path(paths.dataset(dataset).manifest()["bundles"][variant]["source"])


def panel_feature_counts(source: Path) -> dict[str, int]:
    """Features per ``feature_type`` in the run's ``cell_feature_matrix.h5``."""
    import collections

    import h5py

    with h5py.File(source / "cell_feature_matrix.h5", "r") as f:
        kinds = [k.decode() for k in f["matrix/features/feature_type"][:]]
    return dict(collections.Counter(kinds))


def read_cell_controls(dataset: str, variant: str) -> dict:
    """The control counts and segmented area of the model's cells, in the
    bundle's row order (the order ``assemble`` keeps)."""
    import anndata as ad
    import pandas as pd

    from discell import paths

    source = xenium_dir(dataset, variant)
    obs = ad.read_h5ad(paths.dataset(dataset).bundle(variant), backed="r").obs
    cells = pd.read_parquet(source / "cells.parquet").set_index("cell_id")
    cells = cells.reindex(obs.index.astype(str))
    if cells[list(CONTROL_KINDS)].isna().any().any():
        raise ValueError(f"{dataset}/{variant}: cells missing from cells.parquet")
    return {"counts": {k: cells[k].to_numpy(np.float64) for k in CONTROL_KINDS},
            "area": cells["cell_area"].to_numpy(np.float64),
            "n_features": panel_feature_counts(source),
            "source": str(source / "cells.parquet")}


def section_lambda(counts: dict, n_features: dict, n_genes: int) -> dict:
    """Each kind's mean per-cell count scaled to the panel, and their sum."""
    parts = {}
    for column, kind in CONTROL_KINDS.items():
        mean = float(np.mean(counts[column]))
        parts[column] = {"feature_type": kind, "n_features": int(n_features[kind]),
                         "mean_per_cell": mean,
                         "lambda": mean * n_genes / n_features[kind]}
    return {"n_genes": int(n_genes), "components": parts,
            "lambda": float(sum(p["lambda"] for p in parts.values()))}


def within_type_spearman(control, area, t, min_cells: int = MIN_TYPE_CELLS
                         ) -> dict:
    """Spearman(control count, area) per type; the cell-weighted mean over
    types with at least *min_cells* cells and a non-constant count."""
    from scipy.stats import spearmanr

    per_type, weights = {}, {}
    for g in np.unique(t):
        m = t == g
        if m.sum() < min_cells or np.ptp(control[m]) == 0:
            continue
        per_type[int(g)] = float(spearmanr(control[m], area[m]).statistic)
        weights[int(g)] = int(m.sum())
    w = np.array([weights[g] for g in per_type], dtype=np.float64)
    rho = np.array(list(per_type.values()))
    return {"weighted_mean": float((w * rho).sum() / w.sum()),
            "min": float(rho.min()), "max": float(rho.max()),
            "n_types": len(per_type), "per_type": per_type,
            "n_cells_per_type": weights}


def area_test(control, area, t, seed: int = 0) -> dict:
    """The pre-registered statistic plus two reads that say what it can see.

    ``spearman``: the rule's statistic. ``spearman_if_proportional``: the same
    statistic on counts drawn Poisson with each type's mean rate times
    ``A_i / mean_t(A)`` -- its value were lambda_i exactly proportional to
    area (control counts are mostly 0, which caps a rank correlation far
    below 1). ``area_elasticity``: the slope b of a Poisson fit
    ``log E[c_i] = a_t + b log A_i`` (b = 1: proportional, 0: no area).
    """
    from sklearn.linear_model import PoissonRegressor

    rng = np.random.default_rng(seed)
    rate = np.zeros_like(area)
    for g in np.unique(t):
        m = t == g
        rate[m] = control[m].mean() * area[m] / area[m].mean()
    types = np.unique(t)
    onehot = (t[:, None] == types[None, :]).astype(np.float64)
    design = np.hstack([onehot, np.log(area)[:, None]])
    fit = PoissonRegressor(alpha=0.0, fit_intercept=False, max_iter=1000
                           ).fit(design, control)
    return {"spearman": within_type_spearman(control, area, t),
            "spearman_if_proportional": within_type_spearman(
                rng.poisson(rate).astype(np.float64), area, t)["weighted_mean"],
            "area_elasticity": float(fit.coef_[-1]),
            "fraction_cells_with_a_control_count": float((control > 0).mean())}


def eta_from_lambda(lam, totals, cap: float = ETA_CAP) -> np.ndarray:
    """``eta_i = min(lambda_i / l_i, cap)``; a cell with no counts sits at the
    cap (its likelihood term is zero whatever eta is) unless its lambda is 0,
    so lambda = 0 is eta = 0 everywhere -- the pinned mixture."""
    totals = np.asarray(totals, dtype=np.float64)
    lam = np.broadcast_to(np.asarray(lam, dtype=np.float64), totals.shape)
    ratio = np.divide(lam, totals, out=np.full_like(totals, np.inf),
                      where=totals > 0)
    ratio[lam <= 0] = 0.0
    return np.minimum(ratio, cap)


def per_cell_lambda(lam: float, area, per_cell_area: bool) -> np.ndarray:
    """``lambda_i``: the section's lambda, or proportional to area with the
    section total (``n * lambda``) fixed."""
    area = np.asarray(area, dtype=np.float64)
    if not per_cell_area:
        return np.full(area.shape, float(lam))
    return lam * area / area.mean()


def fp_floor_eta(dataset: str, variant: str, totals, n_genes: int,
                 per_cell_area: bool = False, lam: float | None = None
                 ) -> tuple[np.ndarray, dict]:
    """``eta_i`` for every cell of one fit, in the bundle's row order, and the
    record of how: lambda by kind from the cell table, the lambda used (*lam*
    when given, e.g. a config's recorded value; else the table's), the area
    form, and the share of cells at the cap."""
    controls = read_cell_controls(dataset, variant)
    totals = np.asarray(totals, dtype=np.float64)
    if len(controls["area"]) != len(totals):
        raise ValueError(f"cell table has {len(controls['area'])} cells, the "
                         f"model {len(totals)}")
    record = section_lambda(controls["counts"], controls["n_features"], n_genes)
    used = record["lambda"] if lam is None else float(lam)
    eta = eta_from_lambda(per_cell_lambda(used, controls["area"],
                                          per_cell_area), totals)
    median = float(np.median(totals))
    record.update(cells_table=controls["source"], lambda_used=used,
                  per_cell_area=bool(per_cell_area), eta_cap=ETA_CAP,
                  cap_share=float((eta >= ETA_CAP).mean()),
                  eta_mean=float(eta.mean()), eta_median=float(np.median(eta)),
                  median_counts=median, share_of_median=used / median)
    return eta, record


def section_read(dataset: str, variant: str, totals=None, t=None,
                 n_genes: int | None = None, seed: int = 0) -> dict:
    """Step 1 on one section: lambda by kind, its share of the median count,
    the area test and the cap share under both lambda forms. *totals*, *t*
    and *n_genes* default to the bundle's (panel count totals, curated
    type with the loader's Unassigned fill, gene count)."""
    import anndata as ad

    from discell import paths

    if totals is None or t is None or n_genes is None:
        import pandas as pd

        from discell.data.loader import UNASSIGNED

        bundle = ad.read_h5ad(paths.dataset(dataset).bundle(variant))
        totals = np.asarray(bundle.X.sum(axis=1)).ravel()
        label = bundle.uns.get("default_label") or "cell_group"
        series = pd.Series(bundle.obs[label]).astype(object)
        series = series.where(series.notna(), UNASSIGNED).astype(str)
        t = pd.factorize(series, sort=True)[0]
        n_genes = bundle.n_vars
    controls = read_cell_controls(dataset, variant)
    lam = section_lambda(controls["counts"], controls["n_features"], n_genes)
    control = sum(controls["counts"].values())
    median = float(np.median(totals))
    reads = {"dataset": dataset, "variant": variant, "n_cells": int(len(totals)),
             "cells_table": controls["source"], "median_counts": median,
             **lam, "share_of_median": lam["lambda"] / median,
             "fp_share_of_all_counts": lam["lambda"] * len(totals)
             / float(np.sum(totals)),
             "area_test": area_test(control, controls["area"], np.asarray(t),
                                    seed=seed)}
    for form, per_cell in (("section", False), ("area", True)):
        eta = eta_from_lambda(per_cell_lambda(lam["lambda"], controls["area"],
                                              per_cell), totals)
        reads[f"eta_{form}"] = {"cap_share": float((eta >= ETA_CAP).mean()),
                                "mean": float(eta.mean()),
                                "median": float(np.median(eta))}
    return reads


def area_decision(sections: list[dict]) -> dict:
    """The rule across sections: lambda_i proportional to area only if every
    section's within-type Spearman reaches AREA_RHO_MIN."""
    rhos = {f"{s['dataset']}:{s['variant']}":
            s["area_test"]["spearman"]["weighted_mean"] for s in sections}
    per_cell = all(r >= AREA_RHO_MIN for r in rhos.values())
    return {"threshold": AREA_RHO_MIN, "spearman": rhos,
            "per_cell_area": per_cell,
            "decision": ("lambda_i proportional to area, section total fixed"
                         if per_cell else "one lambda per section")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--section", action="append", required=True,
                        help="dataset:variant (repeat)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    sections = [section_read(*s.split(":")) for s in args.section]
    record = {"sections": sections, "area_rule": area_decision(sections),
              "eta_cap": ETA_CAP}
    text = json.dumps(record, indent=1)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
