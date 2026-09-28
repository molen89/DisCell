#!/usr/bin/env python3
"""Inventory of the runs per dataset with a proposed keep / archive action.

Devlog policy for the lineage re-pin (2026-09-25): keep ``finalL_*``,
``uncontrolledL_*``, ``sweepL_*`` (and the queue's projection arms
``projL*``), the ``runs/best`` pointer and the baseline latents; archive the
superseded families -- ``final_s*``, ``best_*``, ``sweep3_*``, ``ladder_az_*``,
``wfix*``, ``qp_*``, ``r12_*``, ``aw_*``, ``advc_*``, ``adv_*``, ``fp_*``,
``uncontrolled*``, ``ablation_*``, ``abl_*`` -- **once their read-out
tables exist in experiments/**. The final-repair brief (2026-09-27) also
keeps ``dw*``, ``reference*``, ``phiproj*``, ``best_pre_az0.5`` and the whole
``synthetic_smoke`` dataset.

A run counts as read out when one of its family's read-out tables names it:
its own name (or, for a symlinked alias, the alias or its target), the sweep
key a sweep report stores (``k0.1_s0`` for ``sweep3_k0.1_s0``), or a glob a
table writes (``uncontrolled500_s*``). A table in another dataset's
``experiments/`` counts only if it names this dataset's id (the combined
tables live on ovarian or FF). A run of an archive family that no table names
is kept and flagged; a family with no table at all is flagged MISSING. Names
no rule knows are listed as UNCLASSIFIED and kept.

``--dry-run`` prints the inventory and sizes and moves nothing. ``--apply``
MOVES (never deletes) every run the dry run marks ``archive`` into
``runs/_archive/`` of its dataset: the read-out tables stay in experiments/
(not touched), a symlink whose target moves is moved with it (both land in
``_archive/``, so the relative link still resolves), and each dataset's
``runs/_archive/ARCHIVE_2026-09-27.tsv`` records run, size, family and the
table(s) naming it. It refuses before moving anything if a name already
exists in ``_archive/``, a kept run would have to move, or ``runs/best``
would not resolve to a kept run.

    python scripts/archive_2026-09-25.py --dry-run
    python scripts/archive_2026-09-25.py --dry-run --runs     # + one line per run
    python scripts/archive_2026-09-25.py --apply
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "datasets"
BASELINES = Path("/home/rmolen/github/DisCell-baselines/results")

#: (family, run-name glob), checked in order; the first match decides
KEEP = (("best (pointer)", "best"), ("finalL_*", "finalL_*"),
        ("uncontrolledL_*", "uncontrolledL_*"), ("sweepL_*", "sweepL_*"),
        ("projL* (lineage projection test)", "projL*"),
        # the final-repair brief's keep list (2026-09-27)
        ("best_pre_az0.5", "best_pre_az0.5"), ("dw*", "dw*"),
        ("reference*", "reference*"), ("phiproj*", "phiproj*"))
#: datasets kept whole
KEEP_DATASETS = ("synthetic_smoke",)
#: (family, run-name glob, read-out table globs under experiments/)
ARCHIVE = (
    ("final_s*", "final_s*", ("envelope_table_at_best.md", "envelope_table_ci.md",
                              "probe_regrade_final.*", "baseline_battery.*")),
    ("best_*", "best_*", ("envelope_table.md", "probe_regrade_final.*",
                          "alpha_z_decision*.json", "baseline_battery.*")),
    ("sweep3_*", "sweep3_*", ("kappa_sweep_sweep3.json", "alpha_w_sweep_sweep3.json",
                              "d_w_sweep_sweep3.json", "probe_regrade_sweep3_*")),
    ("ladder_az_*", "ladder_az_*", ("alpha_z_sweep_ladder_az.json",
                                    "alpha_z_decision*.json", "probe_regrade_ladder_az.*")),
    ("wfix*", "wfix*", ("wcollapse*.json", "wcollapse*.md", "probe_regrade_wfix*")),
    ("qp_*", "qp_*", ("queryprior.*", "probe_regrade_qp.*")),
    ("r12_*", "r12_*", ("r12_arms.*", "r12_depth_test.*", "probe_regrade_r12.*")),
    ("aw_*", "aw_*", ("awladder.*", "probe_regrade_awladder.*")),
    ("advc_*", "advc_*", ("adv_confirm.*",)),
    ("adv_*", "adv_*", ("adv_ladder.*", "adv_confirm.*")),
    ("fp_*", "fp_*", ("fp_floor.*", "probe_regrade_fp.*")),
    ("uncontrolled*", "uncontrolled*", ("probe_regrade_*.md", "adv_ladder.json",
                                        "adv_confirm.json", "convergence_uncontrolled.json")),
    ("ablation_*", "ablation_*", ("objective_ablations.json", "cycle_target_ablation*.json",
                                  "neighbour_dose_ablation*.json", "external_*ablation*.json")),
    ("abl_*", "abl_*", ("objective_ablations.json",)),
)
#: sweep families whose reports key runs without the tag prefix
SWEEP_PREFIX = {"sweep3_*": "sweep3_", "ladder_az_*": "ladder_az_"}
TOKEN = r"(?<![A-Za-z0-9_.+-]){}(?![A-Za-z0-9_.+-])"


def size_of(path: Path) -> int:
    """Bytes under *path*, symlinks not followed (an alias costs nothing)."""
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            f = Path(root) / name
            if not f.is_symlink():
                total += f.stat().st_size
    return total


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def classify(name: str, dataset: str = "") -> tuple[str, str, tuple]:
    """(action class, family, table globs) for one run name."""
    if dataset in KEEP_DATASETS:
        return "keep", f"{dataset}/* (kept whole)", ()
    for family, pattern in KEEP:
        if fnmatch.fnmatch(name, pattern):
            return "keep", family, ()
    for family, pattern, tables in ARCHIVE:
        if fnmatch.fnmatch(name, pattern):
            return "archive", family, tables
    return "unclassified", "UNCLASSIFIED", ()


class Tables:
    """Read-out tables and their text, cached; a run is covered when a table
    of its family names it."""

    def __init__(self) -> None:
        self.text: dict[Path, str] = {}

    def read(self, path: Path) -> str:
        if path not in self.text:
            try:
                self.text[path] = path.read_text(errors="replace")
            except OSError:
                self.text[path] = ""
        return self.text[path]

    def candidates(self, dataset: str, globs: tuple) -> list[Path]:
        """The family's tables: this dataset's, plus other datasets' that
        name this dataset."""
        out = []
        for ds_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
            exp = ds_dir / "experiments"
            if not exp.is_dir():
                continue
            for pattern in globs:
                for path in sorted(exp.glob(pattern)):
                    if not path.is_file() or path.suffix not in (".json", ".md"):
                        continue
                    if ds_dir.name == dataset or dataset in self.read(path):
                        out.append(path)
        return list(dict.fromkeys(out))

    def names(self, path: Path, keys: list[str]) -> bool:
        text = self.read(path)
        if any(re.search(TOKEN.format(re.escape(k)), text) for k in keys):
            return True
        globs = set(re.findall(r"[A-Za-z0-9_.+-]*\*[A-Za-z0-9_.+*-]*", text))
        return any(fnmatch.fnmatch(k, g) for g in globs if len(g) > 3 for k in keys)


def inventory(tables: Tables) -> list[dict]:
    rows = []
    for ds_dir in sorted(p for p in DATA.iterdir() if (p / "runs").is_dir()):
        dataset, runs_dir = ds_dir.name, ds_dir / "runs"
        entries = sorted(runs_dir.iterdir(), key=lambda p: p.name)
        aliases = defaultdict(list)            # target name -> alias names
        for p in entries:
            if p.is_symlink():
                aliases[Path(os.readlink(p)).name].append(p.name)
        for p in entries:
            name = p.name
            if name == "_archive":
                rows.append(dict(dataset=dataset, run=name, family="(existing _archive)",
                                 cls="existing", size=size_of(p), tables=[], covered=None,
                                 alias_of=None))
                continue
            cls, family, globs = classify(name, dataset)
            target = Path(os.readlink(p)).name if p.is_symlink() else None
            keys = [name] + ([target] if target else []) + aliases.get(name, [])
            if family in SWEEP_PREFIX:
                keys += [k[len(SWEEP_PREFIX[family]):] for k in keys
                         if k.startswith(SWEEP_PREFIX[family])]
            found = tables.candidates(dataset, globs) if cls == "archive" else []
            covering = [t for t in found if tables.names(t, keys)]
            finished = (p / "metrics.json").exists()   # follows an alias to its target
            rows.append(dict(dataset=dataset, run=name, family=family, cls=cls,
                             size=size_of(p), tables=found, covering=covering,
                             covered=bool(covering) if cls == "archive" else None,
                             alias_of=target, finished=finished))
    return rows


def action(row: dict) -> str:
    if row["cls"] == "existing":
        return "(already archived)"
    if row["cls"] == "keep":
        return "keep"
    if row["cls"] == "unclassified":
        return "keep (UNCLASSIFIED: no rule)"
    if not row["finished"]:
        return "keep (FLAG: unfinished or running, no metrics.json)"
    if row["covered"]:
        return "archive" + (" (alias)" if row["alias_of"] else "")
    return "keep (FLAG: no read-out table names it)"


def rel(path: Path) -> str:
    return str(path.relative_to(DATA.parent.parent))


ARCHIVE_TSV = "ARCHIVE_2026-09-27.tsv"


def plan_moves(rows: list[dict]) -> dict[str, list[dict]]:
    """Per dataset, the rows to move: the dry run's ``archive`` rows plus any
    symlink whose target moves (it would dangle otherwise). Exits before
    anything moves if a kept link (``runs/best`` among them) would dangle or a
    name is already taken in ``_archive/``."""
    by_ds = defaultdict(dict)
    for r in rows:
        by_ds[r["dataset"]][r["run"]] = r
    plan = {}
    for dataset, runs in by_ds.items():
        moving = {n for n, r in runs.items() if action(r).startswith("archive")}
        grew = True
        while grew:                        # links to links: until nothing new
            grew = False
            for n, r in runs.items():
                if r["alias_of"] in moving and n not in moving:
                    if r["cls"] == "keep":
                        sys.exit(f"REFUSED: {dataset}: kept {n} -> {r['alias_of']} "
                                 "would dangle; nothing moved")
                    r["family"] += " (link to an archived target)"
                    moving.add(n)
                    grew = True
        archive = DATA / dataset / "runs" / "_archive"
        taken = [n for n in moving if (archive / n).exists() or (archive / n).is_symlink()]
        if taken:
            sys.exit(f"REFUSED: {dataset}: already in _archive/: {', '.join(sorted(taken))}; "
                     "nothing moved")
        if moving:
            plan[dataset] = [runs[n] for n in sorted(moving)]
    return plan


def apply(rows: list[dict]) -> int:
    """Move the plan into ``runs/_archive/``, log it, verify what stayed."""
    plan = plan_moves(rows)
    kept = defaultdict(set)
    for r in rows:
        if r["cls"] != "existing" and not action(r).startswith("archive"):
            kept[r["dataset"]].add(r["family"])
    print("# Archive applied (moved, never deleted)\n")
    for dataset, items in plan.items():
        runs_dir = DATA / dataset / "runs"
        archive = runs_dir / "_archive"
        archive.mkdir(exist_ok=True)
        moving = {r["run"] for r in items}
        tsv = archive / ARCHIVE_TSV
        new = not tsv.exists()
        with tsv.open("a") as out:
            if new:
                out.write("run\tsize_bytes\tfamily\ttables\n")
            for r in items:
                src = runs_dir / r["run"]
                src.rename(archive / r["run"])
                if r["alias_of"] and r["alias_of"] not in moving:   # target stays
                    link = archive / r["run"]
                    link.unlink()
                    link.symlink_to(Path("..") / r["alias_of"])
                tables = r.get("covering") or []
                out.write(f"{r['run']}\t{r['size']}\t{r['family']}\t"
                          f"{';'.join(rel(t) for t in tables)}\n")
        size = sum(r["size"] for r in items)
        print(f"* {dataset}: moved {len(items)} runs, {size / 1024 ** 3:.2f} GB "
              f"-> {rel(archive)}; log {rel(tsv)}")
        print(f"  kept families: {', '.join(sorted(kept[dataset]))}")
    return verify(plan)


def verify(plan: dict[str, list[dict]]) -> int:
    """runs/best resolves, no kept family moved, no symlink dangles."""
    problems = []
    for ds_dir in sorted(p for p in DATA.iterdir() if (p / "runs").is_dir()):
        runs_dir = ds_dir / "runs"
        best = runs_dir / "best"
        if best.is_symlink() or best.exists():
            ok = (best.resolve() / "best.pt").exists()
            print(f"* {ds_dir.name}: runs/best -> {os.readlink(best) if best.is_symlink() else '(dir)'}"
                  f" {'resolves' if ok else 'DOES NOT RESOLVE'}")
            if not ok:
                problems.append(f"{ds_dir.name}: runs/best does not resolve")
        for r in plan.get(ds_dir.name, []):
            if classify(r["run"], ds_dir.name)[0] == "keep":
                problems.append(f"{ds_dir.name}: kept {r['run']} moved")
        for d in (runs_dir, runs_dir / "_archive"):
            if d.is_dir():
                problems += [f"dangling {rel(p)}" for p in d.iterdir()
                             if p.is_symlink() and not p.exists()]
    print("\n" + ("\n".join(f"* PROBLEM: {p}" for p in problems) if problems
                  else "Verified: every runs/best resolves, no kept family moved, "
                       "no dangling symlink in runs/ or runs/_archive/."))
    return 1 if problems else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="report only")
    mode.add_argument("--apply", action="store_true",
                      help="move the proposed runs into runs/_archive/ (never deletes)")
    parser.add_argument("--runs", action="store_true", help="also one line per run")
    args = parser.parse_args(argv)

    tables = Tables()
    rows = inventory(tables)
    if args.apply:
        return apply(rows)
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)

    print("# Run inventory, proposed actions (dry run -- nothing moved)\n")
    grand = defaultdict(int)
    flags = []
    for dataset, items in by_ds.items():
        best = DATA / dataset / "runs" / "best"
        print(f"## {dataset}\n")
        if best.is_symlink():
            print(f"runs/best -> {os.readlink(best)}\n")
        print("| family | runs | size | proposed | read-out table(s) naming them | not named |")
        print("|---|---|---|---|---|---|")
        fam = defaultdict(list)
        for r in items:
            fam[r["family"]].append(r)
        for family, rs in fam.items():
            acts = defaultdict(int)
            for r in rs:
                acts[action(r)] += 1
                grand[action(r).split(" ")[0]] += r["size"]
            size = sum(r["size"] for r in rs)
            covering = sorted({rel(t) for r in rs for t in r.get("covering") or []})
            found = sorted({rel(t) for r in rs for t in r.get("tables") or []})
            uncovered = [r["run"] for r in rs if r["cls"] == "archive" and not r["covered"]]
            unfinished = [r["run"] for r in rs if r["cls"] == "archive" and not r["finished"]]
            if unfinished:
                flags.append(f"{dataset} {family}: unfinished / running (no metrics.json), kept: "
                             + ", ".join(unfinished))
            if rs[0]["cls"] == "archive" and not found:
                flags.append(f"{dataset} {family}: MISSING -- no read-out table in experiments/ "
                             f"({', '.join(rs[0].get('tables') or []) or 'none of the candidate names exist'})")
            elif uncovered:
                flags.append(f"{dataset} {family}: {len(uncovered)} of {len(rs)} runs named by no table: "
                             + ", ".join(uncovered))
            if rs[0]["cls"] == "unclassified":
                flags.append(f"{dataset}: UNCLASSIFIED " + ", ".join(r["run"] for r in rs))
            table_cell = ("<br>".join(covering) if covering else
                          ("MISSING" if rs[0]["cls"] == "archive" else ""))
            print(f"| {family} | {len(rs)} | {human(size)} | "
                  + ", ".join(f"{k} x{v}" for k, v in acts.items())
                  + f" | {table_cell} | {', '.join(uncovered) if uncovered else ''} |")
        total = sum(r["size"] for r in items)
        arch = sum(r["size"] for r in items if action(r).startswith("archive"))
        print(f"\n{dataset}: {human(total)} in runs/, {human(arch)} proposed for archive.\n")
        if args.runs:
            for r in items:
                alias = f" -> {r['alias_of']}" if r["alias_of"] else ""
                print(f"    {r['run']}{alias}  [{r['family']}]  {human(r['size'])}  {action(r)}")
            print()

    print("## Baseline latents (keep)\n")
    base_total = 0
    if BASELINES.is_dir():
        for tool in sorted(p for p in BASELINES.iterdir() if p.is_dir()):
            for run in sorted(p for p in tool.iterdir() if p.is_dir()):
                n = size_of(run)
                base_total += n
                has = "latents.h5ad" if (run / "latents.h5ad").exists() else "no latents.h5ad"
                print(f"* {tool.name}/{run.name}: {human(n)} ({has}) -- keep")
    print(f"\nBaseline results: {human(base_total)}.\n")

    print("## Totals\n")
    for k in ("archive", "keep", "(already"):
        print(f"* {k.strip('(')}: {human(grand.get(k, 0))}")
    print("\n## Flags\n")
    print("\n".join(f"* {f}" for f in flags) if flags else "(none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
