#!/usr/bin/env python3
"""GO cellular-component localisation of the response decoder B (todo 6b.10).

Celcomen's takeaway, turned into a read on our own object. If w is a
*response* to the neighbourhood, the genes it moves should sit where a
response lives: secreted and plasma-membrane products, rather than the
cytoplasmic/nuclear housekeeping bulk. If instead the loadings are
indifferent to localisation -- or, worse, grow more extracellular as the
leak coefficient kappa is raised -- then what B carries is the neighbour's
transcripts, not the cell's answer to them.

The statistic is the atlas's, unchanged (``atlas.hallmark_labels``): a
two-sided Mann-Whitney U on the ranks of ``|loading|`` over the *expressed*
panel (prevalence >= 1%), set members against the rest of the panel, no
top-k cut. Only the gene sets differ -- four GO cellular-component
localisations instead of the MSigDB hallmarks -- and BH is applied across
sets x columns within a run rather than per column, because the four sets
are read as one table. The reported effect is the rank-biserial
correlation ``2 * AUC - 1``: positive means the set's loadings are larger
than the panel's.

Two kinds of column are scored, both of which every finished run has:
the atlas programmes (``atlas/programs.npy``: the effective-rank,
varimax-rotated, effect-scale loadings) and the raw columns of B
(``best.pt``). Neither needs the model or the slide re-run; the panel and
its prevalence are read straight off the bundle.

Usage::

    python -m discell.experiments.go_localisation --dataset <id> \\
        --run best_s0 --run best_s1 --run best_s2 [--kappa-sweep]

Outputs land in ``data/datasets/<ds>/experiments/go_localisation.{json,md,png}``.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np

from discell import paths

log = logging.getLogger("discell.experiments.go_localisation")

EXTERNAL = Path("data/external")
GO_OBO = EXTERNAL / "go-basic.obo"
GOA_GAF = EXTERNAL / "goa_human.gaf.gz"
GO_OBO_URL = "https://current.geneontology.org/ontology/go-basic.obo"
GOA_GAF_URL = "https://current.geneontology.org/annotations/goa_human.gaf.gz"

#: The four localisations, as GO CC roots. A gene belongs to a set when any
#: of its CC annotations is that root or a descendant of it (is_a / part_of),
#: which is what makes "nuclear chromatin" count as nucleus.
GO_SETS = {"extracellular": ("GO:0005576", "GO:0005615"),
           "plasma_membrane": ("GO:0005886",),
           "cytoplasm": ("GO:0005737",),
           "nucleus": ("GO:0005634",)}

MIN_SET = 5               #: untestable below this many panel genes (atlas rule)
PREVALENCE = 0.01         #: expressed-panel threshold, as in the atlas
Q_THRESHOLD = 0.05
KAPPA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4)
KAPPA_SEEDS = (0, 1, 2)


# ---------------------------------------------------------------- annotation

def go_closure(obo: Path = GO_OBO) -> tuple[dict[str, set[str]], str]:
    """``{set name: every GO id at or below its roots}`` and the obo version.

    Descent follows ``is_a`` and ``part_of``, the two relations under which
    a component is contained in another; nothing else is traversed.
    """
    parent = collections.defaultdict(set)
    version, term = "", None
    for line in obo.read_text().splitlines():
        if line.startswith("data-version:") and not version:
            version = line.split(": ", 1)[1]
        elif line == "[Term]":
            term = None
        elif line.startswith("["):
            term = "__other__"
        elif term != "__other__" and line:
            if line.startswith("id: GO:"):
                term = line[4:]
            elif term and line.startswith("is_a: "):
                parent[term].add(line[6:16])
            elif term and line.startswith("relationship: part_of GO:"):
                parent[term].add(line[22:32])
    child = collections.defaultdict(set)
    for node, parents in parent.items():
        for p in parents:
            child[p].add(node)

    def below(root: str) -> set[str]:
        seen, stack = {root}, [root]
        while stack:
            for c in child[stack.pop()]:
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return seen

    return ({name: set().union(*(below(r) for r in roots))
             for name, roots in GO_SETS.items()}, version)


def read_go_sets(panel: set[str], obo: Path = GO_OBO,
                 gaf: Path = GOA_GAF) -> tuple[dict[str, set[str]], dict]:
    """The four CC sets intersected with *panel*, plus the provenance record.

    Annotations are the GO consortium's human GAF: aspect C only, NOT
    qualifiers dropped, every evidence code kept (including IEA -- excluding
    it would bias the panel towards well-studied genes, which is exactly the
    axis a loading vector is read on).
    """
    closure, obo_version = go_closure(obo)
    symbol_go = collections.defaultdict(set)
    gaf_date, n_annotations = "", 0
    with gzip.open(gaf, "rt") as handle:
        for line in handle:
            if line.startswith("!"):
                if line.startswith("!date-generated:") and not gaf_date:
                    gaf_date = line.split(": ", 1)[1].strip()
                continue
            field = line.split("\t")
            if field[8] != "C" or "NOT" in field[3]:
                continue
            symbol_go[field[2]].add(field[4])
            n_annotations += 1
    sets = {name: {g for g in panel if symbol_go.get(g, ()) and
                   symbol_go[g] & terms}
            for name, terms in closure.items()}
    provenance = {
        "gaf_url": GOA_GAF_URL, "gaf_date_generated": gaf_date,
        "gaf_annotations_cc": n_annotations,
        "obo_url": GO_OBO_URL, "obo_data_version": obo_version,
        "downloaded": date.fromtimestamp(gaf.stat().st_mtime).isoformat(),
        "read_on": date.today().isoformat(),
        "roots": {n: list(r) for n, r in GO_SETS.items()},
        "closure_terms": {n: len(t) for n, t in closure.items()},
        "panel_annotated": sum(1 for g in panel if g in symbol_go)}
    return sets, provenance


# ------------------------------------------------------------------ the read

def rank_enrichment(loading: np.ndarray, gene_names: np.ndarray,
                    expressed: np.ndarray,
                    sets: dict[str, set[str]]) -> list[dict]:
    """``atlas.hallmark_labels``'s statistic, one row per set, no BH yet.

    Two-sided Mann-Whitney U on ``|loading|`` over the expressed panel,
    members against the rest; ``auc`` in [0, 1] is the U statistic divided
    by the cell count, ``effect`` its rank-biserial form ``2 auc - 1``, and
    ``direction`` the mean *signed* loading of the members (the column's
    sign is arbitrary, so it orients the row, it does not gate it).
    """
    from scipy.stats import mannwhitneyu

    panel = gene_names[expressed]
    values = np.abs(loading[expressed])
    signed = loading[expressed]
    index = {g: i for i, g in enumerate(panel)}
    rows = []
    for name, members in sets.items():
        idx = np.array([index[g] for g in members if g in index], dtype=int)
        if len(idx) < MIN_SET or len(idx) >= len(panel):
            continue
        mask = np.zeros(len(panel), dtype=bool)
        mask[idx] = True
        stat = mannwhitneyu(values[mask], values[~mask],
                            alternative="two-sided", method="asymptotic")
        auc = float(stat.statistic) / (mask.sum() * (~mask).sum())
        rows.append({"set": name, "n_panel": int(mask.sum()), "auc": auc,
                     "effect": 2.0 * auc - 1.0, "p": float(stat.pvalue),
                     "direction": float(signed[mask].mean())})
    return rows


def benjamini_hochberg(rows: list[dict]) -> None:
    """BH step-up over one family of rows, writing ``q`` in place."""
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    m = len(rows)
    for rank, i in enumerate(order):
        rows[i]["q"] = min(rows[i]["p"] * m / (rank + 1), 1.0)
    for rank in range(m - 2, -1, -1):
        rows[order[rank]]["q"] = min(rows[order[rank]]["q"],
                                     rows[order[rank + 1]]["q"])
    for row in rows:
        row["significant"] = bool(row["q"] <= Q_THRESHOLD)


# ----------------------------------------------------------------- run input

def panel_prevalence(dataset: str, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """``(gene_names, expressed)`` read straight off the bundle's CSR X.

    Per-gene prevalence is the column non-zero count, which the CSR index
    array gives without materialising the matrix -- the model is not needed
    to know which genes the panel expresses.
    """
    import h5py

    with h5py.File(paths.dataset(dataset).bundle(variant), "r") as handle:
        names = handle["var/gene_name"]
        names = names["values"] if isinstance(names, h5py.Group) else names
        gene_names = np.asarray([n.decode() if isinstance(n, bytes) else str(n)
                                 for n in names[:]])
        n_cells = int(handle["X"].attrs["shape"][0])
        nnz = np.bincount(handle["X/indices"][:], minlength=len(gene_names))
    return gene_names, (nnz / n_cells) >= PREVALENCE


def run_columns(dataset: str, run: str) -> dict:
    """``{"variant", "kappa", "seed", "programs" (G, r) | None, "b" (G, d_w)}``."""
    import torch

    run_dir = paths.dataset(dataset).root / "runs" / run
    payload = torch.load(run_dir / "best.pt", map_location="cpu",
                         weights_only=False)
    programs = run_dir / "atlas" / "programs.npy"
    return {"variant": payload["config"]["variant"],
            "kappa": float(payload["config"].get("kappa", float("nan"))),
            "seed": int(payload["config"].get("seed", -1)),
            "programs": np.load(programs) if programs.exists() else None,
            "b": payload["model"]["B.weight"].numpy()}


def enrich_run(columns: dict, gene_names: np.ndarray, expressed: np.ndarray,
               sets: dict[str, set[str]]) -> dict:
    """The full table for one run: one family of tests per kind of column.

    BH runs over sets x columns within a kind, so the programme table and
    the raw-B table are each corrected as the table they are read as.
    """
    out: dict = {"kappa": columns["kappa"], "seed": columns["seed"]}
    for kind, matrix in (("programmes", columns["programs"]),
                         ("b_columns", columns["b"])):
        if matrix is None:
            out[kind] = None
            continue
        family, table = [], []
        for k in range(matrix.shape[1]):
            rows = rank_enrichment(matrix[:, k], gene_names, expressed, sets)
            for row in rows:
                row["column"] = k
            family.extend(rows)
            table.append(rows)
        benjamini_hochberg(family)
        out[kind] = table
    return out


# -------------------------------------------------------------------- output

def effect_of(entry: dict, kind: str, column: int, name: str) -> float | None:
    table = entry.get(kind)
    if not table or column >= len(table):
        return None
    return next((r["effect"] for r in table[column] if r["set"] == name), None)


def markdown(result: dict) -> str:
    lines = [f"# GO cellular-component localisation of B -- {result['dataset']}",
             "",
             "Two-sided Mann-Whitney on |loading| over the expressed panel "
             f"({result['n_expressed']} of {result['n_genes']} genes), set "
             "members vs the rest; effect = rank-biserial 2·AUC−1, BH across "
             "sets × columns within each run and kind.", ""]
    ann = result["annotation"]
    lines += [f"Annotation: GO consortium human GAF ({ann['gaf_url']}, "
              f"generated {ann['gaf_date_generated']}) over "
              f"{ann['obo_url']} {ann['obo_data_version']}; downloaded "
              f"{ann['downloaded']}. Panel genes with any CC annotation: "
              f"{ann['panel_annotated']}.", ""]
    lines += ["| set | panel genes |", "| --- | --- |"]
    lines += [f"| {n} | {c} |" for n, c in result["set_sizes"].items()]
    for run in result["pinned_runs"]:
        entry = result["runs"][run]
        for kind in ("programmes", "b_columns"):
            table = entry.get(kind)
            if not table:
                continue
            lines += ["", f"## {run} — {kind}", "",
                      "| column | " + " | ".join(
                          f"{n} effect (q)" for n in result["set_sizes"]) + " |",
                      "| --- |" + " --- |" * len(result["set_sizes"])]
            for k, rows in enumerate(table):
                by = {r["set"]: r for r in rows}
                cells = []
                for name in result["set_sizes"]:
                    r = by.get(name)
                    cells.append("n/a" if r is None else
                                 f"{r['effect']:+.3f} ({r['q']:.2g})"
                                 + ("*" if r["significant"] else ""))
                lines.append(f"| {k} | " + " | ".join(cells) + " |")
    if result.get("kappa_sweep"):
        lines += ["", "## kappa sensitivity (programme 0, effect per seed)", "",
                  "| kappa | " + " | ".join(result["set_sizes"]) + " |",
                  "| --- |" + " --- |" * len(result["set_sizes"])]
        for kappa, per_set in result["kappa_sweep"]["programme_0"].items():
            cells = [", ".join(f"{v:+.3f}" for v in per_set[n] if v is not None)
                     or "n/a" for n in result["set_sizes"]]
            lines.append(f"| {kappa} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def figure(result: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(result["set_sizes"])
    rows, labels = [], []
    # the heatmap is the pinned runs only: with the kappa sweep on, its 18
    # runs belong on the kappa axes, not as 126 more heatmap rows
    for run in result["pinned_runs"]:
        entry = result["runs"][run]
        for kind in ("programmes", "b_columns"):
            table = entry.get(kind)
            if not table:
                continue
            for k, cells in enumerate(table):
                by = {r["set"]: r for r in cells}
                rows.append([by[n]["effect"] if n in by else np.nan
                             for n in names])
                labels.append(f"{run} {kind[:4]}{k}")
    sweep = result.get("kappa_sweep")
    fig, axes = plt.subplots(1, 2 if sweep else 1,
                             figsize=(13 if sweep else 7,
                                      min(14.0, max(3.0, 0.28 * len(rows) + 1.6))),
                             squeeze=False)
    grid = np.asarray(rows)
    ax = axes[0][0]
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-0.2, vmax=0.2, aspect="auto")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    for i, run_row in enumerate(rows):
        for j, value in enumerate(run_row):
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=5)
    ax.set_title("rank-biserial effect (|loading| vs panel)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.04)
    if sweep:
        ax = axes[0][1]
        for name in names:
            xs, ys = [], []
            for kappa, per_set in sweep["programme_0"].items():
                for value in per_set[name]:
                    if value is not None:
                        xs.append(float(kappa))
                        ys.append(value)
            ax.scatter(xs, ys, s=16, label=name)
            means = [(float(k), float(np.mean([v for v in p[name]
                                               if v is not None])))
                     for k, p in sweep["programme_0"].items()
                     if any(v is not None for v in p[name])]
            ax.plot([m for m, _ in means], [v for _, v in means], lw=1)
        ax.axhline(0, color="0.4", lw=0.6)
        ax.set_xlabel("kappa"); ax.set_ylabel("effect (programme 0)")
        ax.set_title("kappa sensitivity, 3 seeds per grid point", fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------- CLI

def build(args: argparse.Namespace) -> dict:
    kappa_runs = ([f"sweep3_k{k:g}_s{s}" for k in KAPPA_GRID
                   for s in KAPPA_SEEDS] if args.kappa_sweep else [])
    wanted = list(dict.fromkeys(list(args.run) + kappa_runs))
    root = paths.dataset(args.dataset).root / "runs"
    wanted = [r for r in wanted if (root / r / "best.pt").exists()]
    if not wanted:
        raise SystemExit("no finished run of those named exists")

    columns = {run: run_columns(args.dataset, run) for run in wanted}
    variants = {c["variant"] for c in columns.values()}
    if len(variants) > 1:
        raise SystemExit(f"runs span bundles {variants}; panels differ")
    gene_names, expressed = panel_prevalence(args.dataset, variants.pop())
    sets, provenance = read_go_sets(set(gene_names[expressed]))
    log.info("panel %d genes, %d expressed; sets %s", len(gene_names),
             int(expressed.sum()), {n: len(s) for n, s in sets.items()})

    result = {"dataset": args.dataset, "n_genes": int(len(gene_names)),
              "n_expressed": int(expressed.sum()),
              "prevalence_threshold": PREVALENCE,
              "q_threshold": Q_THRESHOLD, "annotation": provenance,
              "set_sizes": {n: len(s) for n, s in sets.items()},
              "pinned_runs": [r for r in args.run if r in wanted] or wanted,
              "runs": {}}
    for run in wanted:
        result["runs"][run] = enrich_run(columns[run], gene_names, expressed,
                                         sets)
        log.info("%s: programme 0 %s", run,
                 {n: round(effect_of(result["runs"][run], "programmes", 0, n)
                           or float("nan"), 3) for n in sets})

    if args.kappa_sweep:
        sweep = {"runs": kappa_runs, "programme_0": {}, "b_column_mean": {}}
        for kappa in KAPPA_GRID:
            key = f"{kappa:g}"
            sweep["programme_0"][key] = {n: [] for n in sets}
            sweep["b_column_mean"][key] = {n: [] for n in sets}
            for seed in KAPPA_SEEDS:
                entry = result["runs"].get(f"sweep3_k{kappa:g}_s{seed}")
                if entry is None:
                    continue
                for name in sets:
                    sweep["programme_0"][key][name].append(
                        effect_of(entry, "programmes", 0, name))
                    per_column = [effect_of(entry, "b_columns", k, name)
                                  for k in range(len(entry["b_columns"]))]
                    sweep["b_column_mean"][key][name].append(
                        float(np.mean([v for v in per_column if v is not None])))
        result["kappa_sweep"] = sweep

    out = paths.dataset(args.dataset).root / "experiments"
    stem = getattr(args, "out_stem", None) or "go_localisation"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(
        json.dumps(result, indent=2, default=float))
    (out / f"{stem}.md").write_text(markdown(result))
    figure(result, out / f"{stem}.png")
    log.info("wrote %s", out / f"{stem}.json")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", action="append", default=[],
                        help="a finished run; repeatable")
    parser.add_argument("--kappa-sweep", action="store_true",
                        help=f"also read sweep3_k{{{','.join(f'{k:g}' for k in KAPPA_GRID)}}}"
                             f"_s{{{','.join(map(str, KAPPA_SEEDS))}}}")
    parser.add_argument("--out-stem", default=None,
                        help="file stem under experiments/ (default "
                             "go_localisation), so a side read does not "
                             "overwrite the pinned runs' table")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    build(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
