#!/usr/bin/env python3
"""LaTeX tables for the AISTATS paper, generated from the frozen result files.

The results froze on 2026-09-29 at 15:07:21 (the last line of the Unassigned
re-read queue). Every number in a generated table is read from a file named in
``docs/results_manifest.md``; nothing is typed by hand. Each output file starts
with LaTeX comments (never rendered) that give the command, the time, every
source path with its modification time, the provenance of every row and
column, the reading convention and every pending cell with its reason.

Tables (one file each under ``submission_paper/aistats/tables/generated/``):

* ``headline``       -- tab:headline, section 4.2 (manifest row 1)
* ``probe``          -- tab:probe, section 3.3 (rows 2, 3; replaces the
                        hand-built table of the same label)
* ``battery``        -- tab:battery, section 4.6 (row 4)
* ``sensitivity``    -- tab:sensitivity, section 4.5 / appendix (rows 11, 21-23)
* ``kappa_sweep``    -- tab:kappa-sweep, section 4.5 appendix (row 9)

Usage::

    python scripts/paper_tables.py              # all five
    python scripts/paper_tables.py --only headline battery
    python scripts/paper_tables.py --out /some/dir

Checks built in (a failed check turns the affected cells into ``\\pending{}``
and says why in the comments, rather than printing a number that disagrees
with its own source):

* every source carries the Unassigned mask where the file records one;
* the held-out column of the headline agrees with the envelope's held-out mean;
* each probe fraction equals its excess over the seed-mean uncontrolled
  excess of the same file, and each pass flag equals ``fraction <= 0.25``;
* battery and probe entries of sections the Cellina extra queue rewrote after
  the freeze equal the snapshot that queue took before writing;
* the sensitivity tables were rendered after the masked re-read of their
  stage (queue log) and every run they read carries the mask;
* the kappa sweep's reads at kappa = 0.1 equal the final fits' own reads.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "datasets"
OUT = REPO / "submission_paper" / "aistats" / "tables" / "generated"
FREEZE = datetime(2026, 9, 29, 15, 7, 21)
FREEZE_LOG = REPO / "scripts" / "logs" / "unassigned_2026-09-28" / "queue.log"
# taken by the Cellina extra queue at 15:11:25, after the freeze and before it
# added its own columns: the frozen state of the files it rewrites
SNAPSHOT = REPO / "scripts" / "logs" / "cellina_extra_2026-09-29" / "backup"

OVARIAN = "xenium_prime_ovarian_cancer_ffpe"
LUNG = "xenium_prime_human_lung_cancer_ffpe"
FF = "xenium_prime_human_ovary_ff"
GSE = "gse315411_pdltma06_11_prime_solo"
DUAL = "gse315411_pdltma06_10_prime_dual"
TRAINED = [OVARIAN, LUNG, FF, GSE]
SECTIONS = [OVARIAN, LUNG, FF, GSE, DUAL]
SHORT = {OVARIAN: "Ovarian FFPE", LUNG: "Lung FFPE", FF: "Ovarian FF",
         GSE: "TMA core", DUAL: "TMA serial"}
LONG = {OVARIAN: "Ovarian cancer (FFPE)", LUNG: "Lung cancer (FFPE)",
        FF: "Ovarian cancer (FF)", GSE: "Lung TMA core",
        DUAL: "Lung TMA, serial section"}
SEEDS = (0, 1, 2)
FINAL = "finalL_s{s}"
GUARD = 0.25            # the probe guard: a block passes at <= 0.25 u
MASK = "Unassigned"

CONVENTION = ("accepted checkpoint; 3-seed mean with [min, max] over "
              "finalL_s0-s2; tile-bootstrap 95 % CI where the source has one "
              "(envelope of the per-seed intervals); Unassigned excluded as a "
              "target of every read; cycle reads on the label-free top-decile "
              "cycling set (q90); rounding half-up from the value in the file")


# ---------------------------------------------------------------- formatting

def rnd(x: float, nd: int) -> str:
    """Half-up rounding of the value as stored (its repr), to *nd* places."""
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd),
                                         rounding=ROUND_HALF_UP)
    return f"{q:.{nd}f}"


def num(x: float, nd: int) -> str:
    s = rnd(x, nd)
    return f"${s}$" if s.startswith("-") else s


def span(lo: float, hi: float, nd: int) -> str:
    a, b = rnd(lo, nd), rnd(hi, nd)
    if a.startswith("-") or b.startswith("-"):
        return f"{num(lo, nd)} to {num(hi, nd)}"
    return f"{a}--{b}"


def interval(lo: float, hi: float, nd: int) -> str:
    return f"[{num(lo, nd)}, {num(hi, nd)}]"


def pend(why: str) -> str:
    return f"\\pending{{{why}}}"


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


# ------------------------------------------------------------------- tracing

class Trace:
    """Sources read, provenance lines and pending cells of one table."""

    def __init__(self, name: str):
        self.name = name
        self.sources: dict[Path, float] = {}
        self.prov: list[str] = []
        self.pending: list[str] = []
        self.notes: list[str] = []
        self.cells: list[dict] = []     # for the round-trip test

    def read_json(self, path: Path):
        self.sources[path] = path.stat().st_mtime
        return json.loads(path.read_text())

    def read_text(self, path: Path) -> str:
        self.sources[path] = path.stat().st_mtime
        return path.read_text()

    def cell(self, row: str, col: str, text: str, values, nd: int,
             src: Path, key: str) -> str:
        self.cells.append({"row": row, "col": col, "text": text,
                           "values": values, "nd": nd,
                           "src": rel(src), "key": key})
        return text

    def pend(self, where: str, why: str, short: str) -> str:
        self.pending.append(f"{where}: {why}")
        return pend(short)

    def header(self, command: str, convention: str) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        L = [f"% Generated table: {self.name}. DO NOT EDIT BY HAND;"
             " regenerate instead.",
             f"% Command: {command}",
             f"% Generated: {now} (results frozen {FREEZE:%Y-%m-%d %H:%M:%S},"
             f" last line of {rel(FREEZE_LOG)})",
             "%", "% Sources read (modification time):"]
        for p, t in sorted(self.sources.items(), key=lambda kv: str(kv[0])):
            stamp = datetime.fromtimestamp(t)
            flag = "" if stamp <= FREEZE else "  [after the freeze; see checks]"
            L.append(f"%   {rel(p)}  ({stamp:%Y-%m-%d %H:%M:%S}){flag}")
        L += ["%", "% Reading convention:"]
        L += [f"%   {line}" for line in _wrap(convention)]
        L += ["%", "% Provenance (row / column -> file : key):"]
        L += [f"%   {line}" for line in self.prov]
        if self.notes:
            L += ["%", "% Checks and notes:"]
            for n in self.notes:
                L += [f"%   - {line}" if i == 0 else f"%     {line}"
                      for i, line in enumerate(_wrap(n))]
        L += ["%", "% Pending cells:"]
        if not self.pending:
            L.append("%   none")
        for p in self.pending:
            L += [f"%   - {line}" if i == 0 else f"%     {line}"
                  for i, line in enumerate(_wrap(p))]
        L.append("%")
        return "\n".join(L) + "\n"


def _wrap(text: str, width: int = 90) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    return lines + ([cur] if cur else [])


def mtime(p: Path) -> datetime:
    return datetime.fromtimestamp(p.stat().st_mtime)


# ------------------------------------------------------------ envelope parse

def parse_envelope(text: str) -> dict[str, dict]:
    """Rows of an ``envelope_table_ci_at_best.md`` by their metric label."""
    rows, head = {}, None
    for line in text.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "metric":
            head = cells
            continue
        if head is None:
            continue
        rec = dict(zip(head, cells))
        out = {}
        for k in ("min", "mean", "max", "held-out mean"):
            v = rec.get(k, "")
            out[k] = float(v) if v not in ("", None) else None
        out["n"] = int(rec["n"]) if rec.get("n") else 0
        ci_key = next((h for h in head if h.startswith("tile 95 % CI")), None)
        out["ci"] = None
        if ci_key and rec.get(ci_key):
            m = re.match(r"\[([-+0-9.]+), ([-+0-9.]+)\] \((\d+)\)( †)?",
                         rec[ci_key])
            if m:
                out["ci"] = (float(m.group(1)), float(m.group(2)),
                             int(m.group(3)), bool(m.group(4)))
        rows[cells[0]] = out
    return rows


# ------------------------------------------------------------ table 1: headline

HEADLINE_ROWS = [
    # (label, envelope label, cross-slide key or None, digits)
    ("Reconstruction",
     "held-out reconstruction (nats/count)", "recon_val_targets", 3),
    ("NMI of $\\vz$ with the type", "NMI vs the type labels", "nmi_targets", 3),
    ("Mirror $R^2$", "mirror R²", "mirror.r2", 3),
    ("Cycle $R^2$ of $\\vz$", "cycle R² (z, top-decile set)",
     "cycle_r2_z_q90", 3),
    ("Cycle $R^2$ of $\\vw$", "cycle R² (w, top-decile set)",
     "cycle_r2_w_q90", 3),
    ("$\\I(\\text{niche}; \\vw)$ excess",
     "I(niche; w) excess over within-type floor (nats)", None, 2),
    ("Transport",
     "transport, mean read: fraction of ceiling (trusted)", None, 2),
    ("Atlas cross-seed cosine", "atlas cross-seed axis-1 cosine", None, 2),
]
TRANSPORT_LABEL = "transport, mean read: fraction of ceiling (trusted)"


def _dig(d: dict, dotted: str):
    for part in dotted.split("."):
        d = (d or {}).get(part)
    return d


def table_headline(command: str) -> tuple[str, Trace]:
    tr = Trace("tab:headline")
    env = {}
    for ds in TRAINED:
        p = DATA / ds / "experiments" / "envelope_table_ci_at_best.md"
        text = tr.read_text(p)
        env[ds] = parse_envelope(text)
        if "Unassigned excluded as a metric target" not in text:
            raise SystemExit(f"{rel(p)}: not the masked envelope")
        m = re.search(r"Some atlas cross-seed axis_cosine entries were null "
                      r"and were skipped for: ([^.]*)\.", text)
        if m:
            tr.notes.append(f"{SHORT[ds]}: the envelope skipped null atlas "
                            f"cross-seed cosines for {m.group(1)}; the cell "
                            "is over the seed pairs that exist.")
    # held-out section: per-seed cross-slide reads of the core's fits
    cross = {}
    for s in SEEDS:
        p = (DATA / GSE / "runs" / FINAL.format(s=s) / "crossslide"
             / f"{DUAL}.json")
        sec = tr.read_json(p)["held_out_section"]
        if MASK not in (sec.get("eval_mask") or {}).get("excluded_types", []):
            raise SystemExit(f"{rel(p)}: not read under the Unassigned mask")
        cross[s] = (p, sec)
    env_gse = rel(DATA / GSE / "experiments" / "envelope_table_ci_at_best.md")

    cols = SECTIONS
    head = ("Read & " + " & ".join(SHORT[c] for c in cols) + " \\\\")
    body = []
    for label, elabel, ckey, nd in HEADLINE_ROWS:
        mean_cells, range_cells, ci_cells = [], [], []
        any_ci = False
        for ds in cols:
            where = f"{label} / {SHORT[ds]}"
            if elabel == TRANSPORT_LABEL and ds != DUAL:
                why = ("transport rows are held until the held-out-tiles "
                       "sensitivity check (manifest row 37, queue "
                       "transport_heldout_2026-09-29, expected ~2026-09-30); "
                       "the pre-set rule can switch the headline read")
                mean_cells.append(tr.pend(where, why, "held-out-tile check"))
                range_cells.append("")
                ci_cells.append("")
                src = DATA / ds / "experiments" / "envelope_table_ci_at_best.md"
                tr.prov.append(f"{where}: withheld (would be {rel(src)} : "
                               f"row '{elabel}')")
                continue
            if ds == DUAL:
                if ckey is None:
                    mean_cells.append("--")
                    range_cells.append("")
                    ci_cells.append("")
                    tr.prov.append(f"{where}: not evaluated on the held-out "
                                   "section (envelope held-out n = 0)")
                    continue
                vals = [_dig(cross[s][1], ckey) for s in SEEDS]
                mean = float(np.mean(vals))
                env_mean = env[GSE][elabel]["held-out mean"]
                src = cross[0][0]
                keytxt = (f"held_out_section.{ckey} over seeds 0-2 "
                          f"(runs/finalL_s{{0,1,2}}/crossslide/{DUAL}.json)")
                if env_mean is None or abs(mean - env_mean) > 0.5e-4 + 1e-12:
                    why = (f"per-seed cross-slide mean {mean!r} does not "
                           f"reproduce the envelope's held-out mean {env_mean}")
                    mean_cells.append(tr.pend(where, why, "check"))
                    range_cells.append("")
                    ci_cells.append("")
                    continue
                mean_cells.append(tr.cell(label, SHORT[ds], num(mean, nd),
                                          [mean], nd, src, keytxt + " mean"))
                range_cells.append(tr.cell(label + " range", SHORT[ds],
                                           span(min(vals), max(vals), nd),
                                           [min(vals), max(vals)], nd, src,
                                           keytxt + " min, max"))
                ci_cells.append("--")
                tr.prov.append(f"{where}: {keytxt}; mean checked against "
                               f"{env_gse} : '{elabel}' held-out mean")
                continue
            row = env[ds].get(elabel)
            src = DATA / ds / "experiments" / "envelope_table_ci_at_best.md"
            if row is None or row["n"] == 0:
                mean_cells.append(tr.pend(where, "row missing from the "
                                          "envelope", "missing"))
                range_cells.append("")
                ci_cells.append("")
                continue
            if row["n"] != 3:
                tr.notes.append(f"{where}: envelope n = {row['n']} seeds.")
            mean_cells.append(tr.cell(label, SHORT[ds], num(row["mean"], nd),
                                      [row["mean"]], nd, src,
                                      f"'{elabel}' mean"))
            range_cells.append(tr.cell(label + " range", SHORT[ds],
                                       span(row["min"], row["max"], nd),
                                       [row["min"], row["max"]], nd, src,
                                       f"'{elabel}' min, max"))
            if row["ci"]:
                any_ci = True
                lo, hi, n, dagger = row["ci"]
                if dagger:
                    tr.notes.append(f"{where}: the envelope marks the CI "
                                    "with a dagger (a recomputed point did "
                                    "not reproduce the stored one).")
                ci_cells.append(tr.cell(label + " CI", SHORT[ds],
                                        interval(lo, hi, nd), [lo, hi], nd,
                                        src, f"'{elabel}' CI column"))
            else:
                ci_cells.append("--")
            tr.prov.append(f"{where}: {rel(src)} : row '{elabel}' "
                           "(mean; min, max; CI column)")
        if elabel == TRANSPORT_LABEL:
            # the four withheld cells as one spanning marker (width)
            npend = sum(c.startswith("\\pending") for c in mean_cells)
            rest = mean_cells[npend:]
            mean_cells = [f"\\multicolumn{{{npend}}}{{c}}{{{mean_cells[0]}}}"]
            mean_cells += rest
        body.append(f"{label} & " + " & ".join(mean_cells) + " \\\\")
        if any(range_cells):
            body.append("\\quad range over seeds & "
                        + " & ".join(range_cells) + " \\\\")
        if any_ci:
            body.append("\\quad $95\\%$ interval & "
                        + " & ".join(ci_cells) + " \\\\")
        body.append("\\addlinespace")
    body = body[:-1]
    tr.notes.append("The held-out section (TMA serial) is read through the "
                    "models fitted on the core; the envelope gives it a mean "
                    "only, so the range is recomputed from the three per-seed "
                    "cross-slide files and the mean is checked against the "
                    "envelope. It has no tile-bootstrap interval, and "
                    "I(niche; w), transport and atlas are not read on it "
                    "('--', not pending: no queue computes them).")
    tr.notes.append("Values are rounded from the envelope's printed digits "
                    "(4 decimals; transport and atlas 3), so a cell can in "
                    "rare cases differ in its last digit from rounding the "
                    "unrounded per-run value.")
    caption = (
        "Model quality on the four sections and the held-out serial section "
        "of the TMA core: mean over three seeds, the range over seeds and, "
        "where available, the $200\\um$ tile-bootstrap $95\\%$ interval "
        "(lowest lower and highest upper bound over the seeds). "
        "Reconstruction is the held-out log-likelihood in nats per count. "
        "Transport is the fraction of the noise ceiling recovered by the "
        "mean transport read. Cycle "
        "$R^2$ is read on the top decile of the S and G2M score among "
        "held-out cells. $\\I(\\text{niche}; \\vw)$ is the excess over a "
        "within-type permutation floor, in nats. The atlas cosine compares the "
        "first programme axis between seeds. The serial section is read "
        "through the models fitted on the core (--: not read there). "
        "Unassigned cells are not targets of any read.")
    tex = (tr.header(command, CONVENTION + "; the serial section's values "
                     "come from the core's three fits read on it")
           + "\\begin{table*}[t]\n\\centering\n"
           f"\\caption{{{caption}}}\n\\label{{tab:headline}}\n\\small\n"
           "\\setlength{\\tabcolsep}{3pt}\n"
           "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
           f"{head}\n\\midrule\n" + "\n".join(body)
           + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    return tex, tr


# --------------------------------------------------------------- table 2: probe


def _probe_file(tr: Trace, ds: str) -> tuple[Path, dict]:
    p = DATA / ds / "experiments" / "probe_regrade_lineage_final.json"
    return p, tr.read_json(p)


def _uncontrolled_mean(d: dict, block: str) -> float:
    return float(np.mean([d[f"DisCell/uncontrolledL_s{s}"][block]["excess"]
                          for s in (0, 1)]))


def _probe_consistent(d: dict, entry: str, block: str) -> str | None:
    """None if the entry's fraction and pass flag follow from its excess."""
    rec = d[entry][block]
    u = rec.get("fraction_of_uncontrolled")
    if u is None:
        return "no fraction of the uncontrolled reference"
    expect = rec["excess"] / _uncontrolled_mean(d, block)
    if abs(u - expect) > 1e-6:
        return (f"fraction {u:.4f} is not its excess over the file's "
                f"uncontrolled seed mean ({expect:.4f})")
    if bool(rec["pass"]) != (u <= GUARD):
        return f"pass flag {rec['pass']} disagrees with fraction {u:.4f}"
    return None


def _snapshot_check(tr: Trace, ds: str, fname: str, live: dict,
                    entries: list[str]) -> set[str]:
    """Entries that differ from the frozen snapshot (empty if none/no file)."""
    snap = SNAPSHOT / ds / fname
    if not snap.exists():
        return set()
    frozen = tr.read_json(snap)
    bad = {e for e in entries if e in frozen and
           json.dumps(frozen[e], sort_keys=True)
           != json.dumps(live.get(e), sort_keys=True)}
    live_p = DATA / ds / "experiments" / fname
    tr.notes.append(
        f"{SHORT[ds]}: {rel(live_p)} was compared with the frozen snapshot "
        f"{rel(snap)} for the entries used ({', '.join(entries)}): "
        + ("identical." if not bad else f"DIFFERENT for {sorted(bad)}."))
    return bad


def pct(v: float, nd: int = 1) -> str:
    """A variance fraction as a percentage, rounded half-up exactly."""
    q = (Decimal(repr(float(v))) * 100).quantize(Decimal(1).scaleb(-nd),
                                                  rounding=ROUND_HALF_UP)
    s = f"{q:.{nd}f}"
    return f"${s}$" if s.startswith("-") else s


def pct_span(vals: list[float], nd: int = 1) -> str:
    return f"{pct(min(vals), nd)}--{pct(max(vals), nd)}"


# comparison methods: (row label, [(method, section -> probe entry)])
PROBE_METHOD_ROWS = [
    ("resolVI, Cellina", [
        ("resolVI", {ds: "resolVI" for ds in SECTIONS}),
        # the serial section's Cellina: the 'Cellina (lineage)' entry, same
        # model and excess as 'Cellina (lineage, transfer)', whose fraction
        # is against a stale reference (checked below)
        ("Cellina", {ds: "Cellina (lineage)" for ds in SECTIONS})]),
    ("MintFlow, SIMVI", [
        ("MintFlow", {GSE: "MintFlow (lineage)",
                      DUAL: "MintFlow (lineage, transfer)"}),
        ("SIMVI", {GSE: "SIMVI (lineage)",
                   DUAL: "SIMVI (lineage, fit on this section)"})]),
]
PROBE_BLOCK_ORDER = [("mlp_comp", "Composition, MLP probe"),
                     ("mlp_img", "Image, MLP probe"),
                     ("ridge_comp", "Composition, ridge probe"),
                     ("ridge_img", "Image, ridge probe")]


def table_probe(command: str) -> tuple[str, Trace]:
    tr = Trace("tab:probe")
    files = {}
    for ds in SECTIONS:
        p, d = _probe_file(tr, ds)
        files[ds] = (p, d)
        used = ([f"DisCell/finalL_s{s}" for s in SEEDS]
                + ["DisCell/uncontrolledL_s0", "DisCell/uncontrolledL_s1"]
                + [m[ds] for _, ms in PROBE_METHOD_ROWS for _, m in ms
                   if ds in m])
        bad = _snapshot_check(tr, ds, p.name, d, used)
        if bad:
            raise SystemExit(f"{SHORT[ds]}: {sorted(bad)} differ from the "
                             "frozen snapshot")
        # every variance fraction is exp(2 excess) - 1 of its own excess
        for e in used:
            for b, _ in PROBE_BLOCK_ORDER:
                rec = d[e][b]
                if abs(rec["var_fraction"]
                       - (np.exp(2 * rec["excess"]) - 1)) > 1e-9:
                    raise SystemExit(f"{rel(p)} {e}.{b}: var_fraction is not "
                                     "exp(2 excess) - 1")
    x = _probe_consistent(files[DUAL][1], "Cellina (lineage, transfer)",
                          "ridge_comp")
    if x:
        tr.notes.append("TMA serial: the probe entry 'Cellina (lineage, "
                        f"transfer)' has {x}; the entry 'Cellina (lineage)' "
                        "(same transferred model, same excess, consistent "
                        "fraction) is used.")
    head = "& " + " & ".join(SHORT[c] for c in SECTIONS) + " \\\\"
    body = []
    finals = [f"DisCell/finalL_s{s}" for s in SEEDS]
    uncs = ["DisCell/uncontrolledL_s0", "DisCell/uncontrolledL_s1"]
    for block, label in PROBE_BLOCK_ORDER:
        body.append(f"\\multicolumn{{{len(SECTIONS) + 1}}}{{@{{}}l}}"
                    f"{{\\emph{{{label}}}}} \\\\")
        rows = {"without": [], "with": [], "left": []}
        for ds in SECTIONS:
            p, d = files[ds]
            vu = [d[e][block]["var_fraction"] for e in uncs]
            vf = [d[e][block]["var_fraction"] for e in finals]
            probs = [_probe_consistent(d, e, block) for e in finals]
            rows["without"].append(tr.cell(
                f"{label} without", SHORT[ds], pct_span(vu), [100 * min(vu), 100 * max(vu)], 1, p,
                f"DisCell/uncontrolledL_s{{0,1}}.{block}.var_fraction x100"))
            rows["with"].append(tr.cell(
                f"{label} with", SHORT[ds], pct_span(vf), [100 * min(vf), 100 * max(vf)], 1, p,
                f"DisCell/finalL_s{{0,1,2}}.{block}.var_fraction x100"))
            if any(probs):
                rows["left"].append(tr.pend(
                    f"{label} fraction left / {SHORT[ds]}",
                    "; ".join(q for q in probs if q), "check"))
            else:
                us = [d[e][block]["fraction_of_uncontrolled"] for e in finals]
                rows["left"].append(tr.cell(
                    f"{label} left", SHORT[ds], span(min(us), max(us), 2),
                    [min(us), max(us)], 2, p,
                    f"DisCell/finalL_s{{0,1,2}}.{block}."
                    "fraction_of_uncontrolled"))
            tr.prov.append(f"{label} / {SHORT[ds]}: {rel(p)} : "
                           f"uncontrolledL_s{{0,1}} and finalL_s{{0,1,2}} "
                           f".{block}.var_fraction (x100, min-max); finalL "
                           f".{block}.fraction_of_uncontrolled (min-max)")
        mlp = block.startswith("mlp")
        left = [f"\\textbf{{{c}}}" if mlp else c for c in rows["left"]]
        body.append("\\quad without the adversary (\\%) & "
                    + " & ".join(rows["without"]) + " \\\\")
        body.append("\\quad with the adversary (\\%) & "
                    + " & ".join(rows["with"]) + " \\\\")
        fl = "\\textbf{fraction left}" if mlp else "fraction left"
        body.append(f"\\quad {fl} & " + " & ".join(left) + " \\\\")
        for rlabel, methods in PROBE_METHOD_ROWS:
            cells = []
            for ds in SECTIONS:
                p, d = files[ds]
                parts = []
                for meth, entries in methods:
                    if ds not in entries:
                        parts.append("--")
                        tr.prov.append(f"{label} / {meth} / {SHORT[ds]}: not "
                                       "run (whole-section fit queued)")
                        continue
                    e = entries[ds]
                    v = d[e][block]["var_fraction"]
                    parts.append(tr.cell(f"{label} {meth}", SHORT[ds],
                                         pct(v), [100 * v], 1, p,
                                         f"'{e}'.{block}.var_fraction x100"))
                    tr.prov.append(f"{label} / {meth} / {SHORT[ds]}: {rel(p)}"
                                   f" : '{e}'.{block}.var_fraction x100")
                cells.append(", ".join(parts) if parts != ["--", "--"]
                             else "--")
            body.append(f"\\quad {rlabel} (\\%) & " + " & ".join(cells)
                        + " \\\\")
        body.append("\\addlinespace")
    body = body[:-1]
    tr.notes.append("The fraction left is the file's fraction_of_uncontrolled "
                    "(each seed's excess over the seed-mean excess of the two "
                    "fits without the adversary), checked against the excesses "
                    "in the same file. Percentages are var_fraction = "
                    "exp(2 excess) - 1, checked. No pass/fail verdicts are "
                    "shown (author, 2026-09-29: the probe is reported "
                    "descriptively; the development threshold is disclosed "
                    "once in the method). Bold on the MLP 'fraction left' rows "
                    "follows the hand-built table this replaces; it marks no "
                    "verdict.")
    tr.notes.append("MintFlow and SIMVI on ovarian FFPE, lung FFPE and "
                    "ovarian FF: '--' (whole-section fits queued, "
                    "baselines_complete); fill from the probe files when they "
                    "land.")
    caption = (
        "The held-out probe on the final fits, per section. For each block "
        "and probe, the share of the block's within-type variance that the "
        "probe explains from the intrinsic latent beyond the permutation "
        "floor ($e^{2\\,\\mathrm{excess}} - 1$, in \\%): for the model "
        "trained without the adversary ($\\alphaa = 0$, two seeds), with it "
        "(the final fits, three seeds), and for the comparison methods on "
        "the same section. The \\emph{fraction left} is the excess with the "
        "adversary as a fraction of the mean excess without it: $0$ when the "
        "adversary removes everything the probe can find, $1$ when it "
        "removes nothing. Ranges are over seeds; Unassigned cells are not "
        "targets. The serial section of the TMA core is graded with our "
        "models fitted on the core; there, Cellina and MintFlow are the "
        "core's models transferred, and resolVI and SIMVI are fitted on the "
        "serial section itself. MintFlow and SIMVI have so far been run on "
        "the TMA core only (--: not yet run).")
    tex = (tr.header(command, "range [min-max] over seeds (finalL_s0-s2; "
                     "uncontrolledL_s0-s1) of the per-block probe excess "
                     "expressed as exp(2 excess) - 1 in %, and of the excess "
                     "as a fraction of the seed-mean uncontrolled excess; "
                     "comparison methods single fit; accepted checkpoint; "
                     "Unassigned excluded as a target")
           + "\\begin{table*}[t]\n\\centering\n"
           f"\\caption{{{caption}}}\n\\label{{tab:probe}}\n\\small\n"
           "\\setlength{\\tabcolsep}{4pt}\n"
           "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
           f"{head}\n\\midrule\n" + "\n".join(body)
           + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    return tex, tr


# ------------------------------------------------------------ table 3: battery

# method -> section -> (battery entry, probe entry) or a reason it is absent
BATTERY_METHODS = {
    "resolVI": {ds: ("resolVI", "resolVI") for ds in SECTIONS},
    "SIMVI": {GSE: ("SIMVI (lineage)", "SIMVI (lineage)"),
              DUAL: ("SIMVI (lineage, fit on this section)",
                     "SIMVI (lineage, fit on this section)")},
    "MintFlow": {GSE: ("MintFlow (lineage)", "MintFlow (lineage)"),
                 DUAL: ("MintFlow (lineage, transfer)",
                        "MintFlow (lineage, transfer)")},
    # the serial section's Cellina probe is read from the 'Cellina (lineage)'
    # entry: same model and same excess as 'Cellina (lineage, transfer)',
    # whose fraction is against a stale reference (see the checks)
    "Cellina": {OVARIAN: ("Cellina (lineage)", "Cellina (lineage)"),
                LUNG: ("Cellina (lineage)", "Cellina (lineage)"),
                FF: ("Cellina (lineage)", "Cellina (lineage)"),
                GSE: ("Cellina (lineage)", "Cellina (lineage)"),
                DUAL: ("Cellina (lineage, transfer)", "Cellina (lineage)")},
}
QUEUED = {("SIMVI", OVARIAN), ("SIMVI", LUNG), ("SIMVI", FF),
          ("MintFlow", OVARIAN), ("MintFlow", LUNG), ("MintFlow", FF)}
BATTERY_COLS = [
    # (header, key, digits, source)
    ("NMI", "nmi", 3, "battery"),
    ("Comp., ridge", "ridge_comp", 2, "probe"),
    ("Comp., MLP", "mlp_comp", 2, "probe"),
    ("Mirror $R^2$", "mirror.r2", 3, "battery"),
    ("Cycle $R^2$", "cycle_q90.z.r2_pooled", 3, "battery"),
    ("Recon.", "reconstruction.recon", 3, "battery"),
]


def table_battery(command: str) -> tuple[str, Trace]:
    tr = Trace("tab:battery")
    bat, prb = {}, {}
    for ds in SECTIONS:
        bp = DATA / ds / "experiments" / "baseline_battery_lineage.json"
        bat[ds] = (bp, tr.read_json(bp))
        pp, pd = _probe_file(tr, ds)
        prb[ds] = (pp, pd)
    bad = {}
    for ds in SECTIONS:
        used_b = [f"DisCell/finalL_s{s}" for s in SEEDS] + [
            v[ds][0] for v in BATTERY_METHODS.values() if ds in v]
        used_p = [f"DisCell/finalL_s{s}" for s in SEEDS] + [
            "DisCell/uncontrolledL_s0", "DisCell/uncontrolledL_s1"] + [
            v[ds][1] for v in BATTERY_METHODS.values() if ds in v]
        bad[ds] = (_snapshot_check(tr, ds, bat[ds][0].name, bat[ds][1], used_b)
                   | _snapshot_check(tr, ds, prb[ds][0].name, prb[ds][1],
                                     used_p))
        for e in used_b:
            em = (bat[ds][1].get(e) or {}).get("eval_mask") or {}
            if MASK not in em.get("excluded_types", []):
                bad[ds].add(e)
                tr.notes.append(f"{SHORT[ds]}/{e}: no Unassigned mask record")
    # the stale-reference entry on the serial section, reported
    d = prb[DUAL][1]
    for block in ("ridge_comp", "mlp_comp"):
        x = _probe_consistent(d, "Cellina (lineage, transfer)", block)
        if x:
            tr.notes.append(
                f"TMA serial, probe entry 'Cellina (lineage, transfer)', "
                f"{block}: {x}. Its excess equals that of the entry "
                "'Cellina (lineage)' (same fitted model, transferred), whose "
                "fraction is consistent, so that entry is used.")

    def value(ds, entry, key, src):
        if src == "battery":
            return _dig(bat[ds][1][entry], key)
        return prb[ds][1][entry][key]["fraction_of_uncontrolled"]

    head = ("Section & Method & " + " & ".join(h for h, *_ in BATTERY_COLS)
            + " \\\\")
    body = []
    for ds in SECTIONS:
        first = True
        # DisCell: mean row and range row
        mean_cells, rng_cells = [], []
        for h, key, nd, src in BATTERY_COLS:
            entries = [f"DisCell/finalL_s{s}" for s in SEEDS]
            where = f"{SHORT[ds]} / DisCell / {h}"
            path = bat[ds][0] if src == "battery" else prb[ds][0]
            probs = ([_probe_consistent(prb[ds][1], e, key) for e in entries]
                     if src == "probe" else [])
            if any(e in bad[ds] for e in entries) or any(probs):
                mean_cells.append(tr.pend(where, "failed a consistency "
                                          "check (see notes)", "check"))
                rng_cells.append("")
                continue
            vals = [value(ds, e, key, src) for e in entries]
            m = float(np.mean(vals))
            kk = (key if src == "battery"
                  else f"{key}.fraction_of_uncontrolled")
            mean_cells.append(tr.cell(f"{SHORT[ds]} DisCell", h, num(m, nd),
                                      [m], nd, path,
                                      f"DisCell/finalL_s{{0,1,2}}.{kk} mean"))
            rng_cells.append(tr.cell(f"{SHORT[ds]} DisCell range", h,
                                     span(min(vals), max(vals), nd),
                                     [min(vals), max(vals)], nd, path,
                                     f"DisCell/finalL_s{{0,1,2}}.{kk} "
                                     "min, max"))
            tr.prov.append(f"{where}: {rel(path)} : DisCell/finalL_s{{0,1,2}}"
                           f".{kk} (mean; min, max)")
        lab = f"\\multirow{{6}}{{*}}{{{SHORT[ds]}}}" if first else ""
        first = False
        body.append(f"{lab} & DISCELL & " + " & ".join(mean_cells) + " \\\\")
        body.append(" & \\quad range & " + " & ".join(rng_cells) + " \\\\")
        for meth, where_map in BATTERY_METHODS.items():
            cells = []
            if (meth, ds) in QUEUED:
                why = ("whole-section fit queued (baselines_complete, GPU 1, "
                       "~5 days from the freeze); the paper will state the "
                       "measured reason if it cannot run")
                cells.append(f"\\multicolumn{{{len(BATTERY_COLS)}}}{{c}}{{"
                             + tr.pend(f"{SHORT[ds]} / {meth} / all columns",
                                       why, "queued") + "$^{a}$}")
                body.append(f" & {meth} & " + " & ".join(cells) + " \\\\")
                continue
            for h, key, nd, src in BATTERY_COLS:
                where = f"{SHORT[ds]} / {meth} / {h}"
                if ds not in where_map:
                    cells.append("--")
                    continue
                be, pe = where_map[ds]
                entry = be if src == "battery" else pe
                path = bat[ds][0] if src == "battery" else prb[ds][0]
                if entry in bad[ds]:
                    cells.append(tr.pend(where, "entry differs from the "
                                         "frozen snapshot", "check"))
                    continue
                if meth == "MintFlow" and key == "reconstruction.recon":
                    cells.append("n/r$^{b}$")
                    tr.prov.append(f"{where}: not reported (issue B-mf1, "
                                   "manifest 'What must NOT be quoted')")
                    continue
                v = value(ds, entry, key, src)
                if v is None:
                    cells.append("n/a$^{c}$")
                    tr.prov.append(f"{where}: {rel(path)} : {entry}.{key} "
                                   "absent (no decoder)")
                    continue
                if src == "probe":
                    x = _probe_consistent(prb[ds][1], entry, key)
                    if x:
                        cells.append(tr.pend(where, x, "check"))
                        continue
                kk = (key if src == "battery"
                      else f"{key}.fraction_of_uncontrolled")
                cells.append(tr.cell(f"{SHORT[ds]} {meth}", h, num(v, nd),
                                     [v], nd, path, f"{entry}.{kk}"))
                tr.prov.append(f"{where}: {rel(path)} : '{entry}'.{kk}")
            body.append(f" & {meth} & " + " & ".join(cells) + " \\\\")
        body.append("\\midrule")
    body = body[:-1]
    tr.notes.append("Cellina's niche-adversary and own-graph rows (manifest "
                    "rows 34-35, queue cellina_extra) are not included; the "
                    "live GSE battery files already carry a "
                    "'cellina_nicheadv' column written after the freeze, "
                    "which this table ignores.")
    tr.notes.append("NMI and reconstruction here are the battery's own reads, "
                    "computed the same way for every method; for DISCELL they "
                    "differ slightly from the headline table's (the trainer's "
                    "reads), so the two tables' DISCELL rows are not "
                    "interchangeable.")
    for ds in SECTIONS:
        counts = {e: (bat[ds][1][e].get("reconstruction") or {}).get("n_cells")
                  for e in [f"DisCell/finalL_s{s}" for s in SEEDS]
                  + [v[ds][0] for v in BATTERY_METHODS.values() if ds in v]}
        tr.notes.append(f"{SHORT[ds]}: target cells scored per entry "
                        f"(reconstruction.n_cells): {counts}. The scored sets "
                        "differ slightly between entries (each DISCELL seed "
                        "has its own split).")
    tr.notes.append(
        "Caption claims verified 2026-09-29 against DisCell-baselines: "
        "common.py filter_cells (>= 5 counts, the only filter); every "
        "result's config.json n_cells equals the export's cells with >= 5 "
        "counts (e.g. TMA core 69,206 of 69,422, the export's held-out split "
        "included); resolVI trains with train_size 1.0 (scvi-tools RESOLVI "
        "requires it); MintFlow has no split; Cellina (config train_size "
        "0.9) and SIMVI (SimVI.train default train_size 0.9) hold out a "
        "random 10 % for early stopping. Serial section: resolvi_lineage "
        "config transfer_mode = fit_on_target (the transfer failed and it "
        "was refitted on the section); SIMVI lineage dual config h5ad = the "
        "dual export (fit on the section); Cellina transfer_mode = transfer; "
        "MintFlow transfer = true, fitted_on = the core export.")
    tr.notes.append("On the serial section resolVI and SIMVI are fitted on "
                    "the serial section itself; MintFlow and Cellina are the "
                    "core's models transferred.")
    caption = (
        "One battery for every method, at lineage labels, on the held-out "
        "cells of each section, with Unassigned cells not targets. "
        "NMI of the intrinsic latent with the type; \\emph{Comp.}: the "
        "held-out excess of the ridge and MLP probes of neighbour "
        "composition as a fraction of that of DISCELL without the adversary "
        "(the fraction left of \\cref{tab:probe}); mirror $R^2$; cycle $R^2$ "
        "of the intrinsic latent on the top-decile cycling set; held-out "
        "reconstruction in nats per count. For DISCELL, the mean over "
        "three seeds and the range. DISCELL's held-out tiles are excluded "
        "from its training loss. The comparison methods are fitted as their software is "
        "designed, on every cell of the section with at least five counts, "
        "held-out tiles included, and are scored on our held-out cells: "
        "resolVI and MintFlow train on all of them, and Cellina and SIMVI "
        "on a random nine tenths of the cells (the rest serve their own "
        "early stopping), so their reads, reconstruction above all, are "
        "not held-out reads. On the serial section, resolVI and SIMVI are "
        "fitted on that section itself, so theirs are not held-out reads "
        "either; Cellina, MintFlow and DISCELL are the core's models "
        "transferred. "
        "$^{a}$Whole-section fits of SIMVI and MintFlow on the three "
        "Xenium sections are queued; if a method cannot be run on the "
        "resources available, the table will say so with the measured "
        "reason. $^{b}$Not reported: MintFlow's reconstruction is under "
        "inspection. $^{c}$SIMVI has no count decoder. --: not applicable.")
    tex = (tr.header(command, "lineage labels; accepted checkpoint; one "
                     "Unassigned mask for every method; DisCell mean and "
                     "[min, max] over finalL_s0-s2, baselines single fit; "
                     "cycle on the label-free top-decile set; probe fractions "
                     "against the seed-mean excess of uncontrolledL_s0-s1")
           + "\\begin{table*}[t]\n\\centering\n"
           f"\\caption{{{caption}}}\n\\label{{tab:battery}}\n\\small\n"
           "\\setlength{\\tabcolsep}{4pt}\n"
           "\\begin{tabular}{@{}llcccccc@{}}\n\\toprule\n"
           f" & & & \\multicolumn{{2}}{{c}}{{Probe fraction}} & & & \\\\\n"
           "\\cmidrule(lr){4-5}\n"
           f"{head}\n\\midrule\n" + "\n".join(body)
           + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    return tex, tr


# -------------------------------------------------------- table 4: sensitivity

SENS_FAMILIES = ["kappa_form", "fp_floor", "alpha_w", "adversary"]
SENS_TITLE = {"kappa_form": "Form of the leakage rate",
              "fp_floor": "Fixed false-positive floor",
              "alpha_w": "Weight of the response deviation",
              "adversary": "Adversary capacity and weight"}
# read -> key per family (the families' records name them differently)
SENS_READS = [
    ("Recon.", 3, {"kappa_form": "recon", "fp_floor": "recon",
                   "alpha_w": "recon", "adversary": "recon"}),
    ("NMI", 3, {"kappa_form": "nmi", "fp_floor": "nmi", "alpha_w": "nmi",
                "adversary": "nmi"}),
    ("Mirror $R^2$", 3, {"kappa_form": "mirror_r2", "fp_floor": "mirror_r2",
                         "alpha_w": "mirror", "adversary": "mirror_r2"}),
    ("$\\I(\\text{niche}; \\vw)$", 2,
     {"kappa_form": "w_niche_mi_excess", "fp_floor": "w_niche_mi_excess",
      "alpha_w": "niche_excess", "adversary": "w_niche_mi_excess"}),
]


def _arm_label(family: str, arm: str, flag: str) -> str:
    """A paper label for an arm, its constants parsed from the arm's flag."""
    val = (re.findall(r"--[a-z-]+ ([0-9.]+)", flag) or [None])[0]
    if family == "kappa_form":
        return {"depth": "per cell, by depth ratio",
                "gene": "per gene",
                "density": "per cell, by density ratio"}[arm]
    if family == "fp_floor":
        return ("area-scaled floor per cell" if "area" in flag
                else "one floor per section")
    if family == "alpha_w":
        m = re.fullmatch(r"m(\d+)", arm).group(1)
        return f"$\\alphaw = {m}/\\lbar$"
    if family == "adversary":
        if "head-steps" in flag:
            return f"head steps {val}"
        if "head-width" in flag:
            return f"head width {val}"
        if "ensemble" in flag:
            return f"ensemble of {val}"
        if "comp-weight" in flag:
            return f"composition weight {val}"
    raise ValueError(f"{family}/{arm}: unknown arm")


def _queue_step_time(pattern: str) -> datetime | None:
    last = None
    for line in FREEZE_LOG.read_text().splitlines():
        if pattern in line:
            last = datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S")
    return last


# where a family's records carry the MLP composition excess, it must equal
# the per-run probe file the column is read from
MLP_CHECK = {"adversary": "mlp_comp_excess",
             "alpha_w": "block_excess:mlp/comp"}


def table_sensitivity(command: str) -> tuple[str, Trace]:
    tr = Trace("tab:sensitivity")

    def mlp_comp(ds: str, run: str) -> float:
        return tr.read_json(DATA / ds / "runs" / run / "validation"
                            / "probe_blocks.json")["mlp"]["comp"]["var_fraction"]

    tr.sources[FREEZE_LOG] = FREEZE_LOG.stat().st_mtime
    start = _queue_step_time("CPU  start sensF_tables_excl")
    done = _queue_step_time("CPU  done  sensF_tables_excl")
    if start is None or done is None:
        raise SystemExit("the masked sensF re-render is not in the queue log")
    fam = {}
    for f in SENS_FAMILIES:
        p = DATA / OVARIAN / "experiments" / f"sensF_{f}.json"
        md = p.with_suffix(".md")
        fam[f] = (p, tr.read_json(p))
        tr.sources[md] = md.stat().st_mtime
        for q in (p, md):
            if mtime(q) < start:
                raise SystemExit(f"{rel(q)} predates the masked re-render "
                                 f"({start})")
        tr.notes.append(f"{rel(p)} and .md written {mtime(p):%H:%M:%S}, "
                        f"inside the masked re-render step sensF_tables_excl "
                        f"({start:%H:%M:%S}-{done:%H:%M:%S}) of the "
                        "Unassigned queue.")
    # every run the tables read carries the mask on its probe and guard reads
    unmasked = []
    for f, (p, d) in fam.items():
        for ds, sec in d["sections"].items():
            runs = list(sec["control"]["names"])
            for a in sec["arms"].values():
                runs += a["names"]
            for run in runs:
                for sub in ("validation/probe_blocks.json", "degeneracy.json"):
                    q = DATA / ds / "runs" / run / sub
                    j = tr.read_json(q)
                    em = (j.get("eval_mask")
                          or (j.get("battery") or {}).get("eval_mask") or {})
                    if MASK not in em.get("excluded_types", []):
                        unmasked.append(f"{ds}/{run}/{sub}")
    tr.notes.append("Mask check: the probe and guard reads of every control "
                    "and arm run carry the Unassigned mask"
                    + ("." if not unmasked else f", EXCEPT {unmasked}."))
    order = [OVARIAN, GSE, FF]
    head = ("Arm & Seeds & " + " & ".join(h for h, *_ in SENS_READS)
            + " & MLP comp. (\\%) \\\\")
    body = []
    for ds in order:
        blocks = [(f, d) for f, (p, d) in fam.items() if ds in d["sections"]]
        if not blocks:
            continue
        # the control, once per section; identical in every family (checked)
        f0, d0 = blocks[0]
        ctrl = d0["sections"][ds]["control"]
        cells, rng = [], []
        for h, nd, keys in SENS_READS:
            vals = [r[keys[f0]] for r in ctrl["records"]]
            for f, d in blocks[1:]:
                other = [r[keys[f]] for r in d["sections"][ds]["control"]
                         ["records"]]
                if not np.allclose(vals, other, rtol=0, atol=1e-9):
                    tr.notes.append(f"{SHORT[ds]} control, {h}: family {f} "
                                    f"reads {other}, family {f0} {vals}.")
            m = float(np.mean(vals))
            cells.append(tr.cell(f"{SHORT[ds]} control", h, num(m, nd), [m],
                                 nd, fam[f0][0], f"sections.{ds}.control."
                                 f"records[*].{keys[f0]} mean"))
            rng.append(tr.cell(f"{SHORT[ds]} control range", h,
                               span(min(vals), max(vals), nd),
                               [min(vals), max(vals)], nd, fam[f0][0],
                               f"sections.{ds}.control.records[*]."
                               f"{keys[f0]} min, max"))
        cv = [mlp_comp(ds, run) for run in ctrl["names"]]
        c_mean, c_sd = float(np.mean(cv)), float(np.std(cv, ddof=1))
        src = DATA / ds / "runs" / "<run>" / "validation" / "probe_blocks.json"
        cells.append(tr.cell(f"{SHORT[ds]} control", "MLP comp.",
                             pct(c_mean), [100 * c_mean], 1, src,
                             "mlp.comp.var_fraction x100 mean"))
        rng.append(tr.cell(f"{SHORT[ds]} control range", "MLP comp.",
                           pct_span(cv), [100 * min(cv), 100 * max(cv)], 1, src,
                           "mlp.comp.var_fraction x100 min, max"))
        tr.prov.append(f"{SHORT[ds]} control: {rel(fam[f0][0])} : sections."
                       f"{ds}.control.records[*] (mean; min, max), checked "
                       "equal in " + ", ".join(f for f, _ in blocks)
                       + f"; MLP comp.: data/datasets/{ds}/runs/finalL_s"
                       "{0,1,2}/validation/probe_blocks.json : "
                       "mlp.comp.var_fraction")
        body.append(f"\\multicolumn{{{len(SENS_READS) + 3}}}{{@{{}}l}}"
                    f"{{\\emph{{{LONG[ds]}}}}} \\\\")
        body.append(f"\\quad final configuration & {len(ctrl['records'])} & "
                    + " & ".join(cells) + " \\\\")
        body.append("\\quad\\quad range over seeds & & " + " & ".join(rng)
                    + " \\\\")
        for f, d in blocks:
            sec = d["sections"][ds]
            body.append(f"\\quad \\emph{{{SENS_TITLE[f]}}}"
                        + " &" * (len(SENS_READS) + 2) + " \\\\")
            for arm, a in sec["arms"].items():
                flag = d["arms"][ds][arm]
                label = _arm_label(f, arm, flag)
                cells = []
                for h, nd, keys in SENS_READS:
                    key = keys[f]
                    vals = [r[key] for r in a["records"]
                            if r.get(key) is not None]
                    mv = sec["moves"][arm].get(key)
                    where = f"{SHORT[ds]} / {f} / {arm} / {h}"
                    if not vals or mv is None:
                        cells.append(tr.pend(where, "read or move missing",
                                             "missing"))
                        continue
                    m = float(np.mean(vals))
                    txt = f"{num(m, nd)} ({num(mv['in_sd'], 1)})"
                    cells.append(tr.cell(f"{SHORT[ds]} {f} {arm}", h, txt,
                                         [m, mv["in_sd"]], [nd, 1], fam[f][0],
                                         f"sections.{ds}.arms.{arm}.records"
                                         f"[*].{key} mean; moves.{arm}.{key}"
                                         ".in_sd"))
                av = [mlp_comp(ds, run) for run in a["names"]]
                check = MLP_CHECK.get(f)
                if check:
                    for rec in a["records"]:
                        exc = _dig(tr.read_json(
                            DATA / ds / "runs" / rec["run"] / "validation"
                            / "probe_blocks.json"), "mlp.comp.excess")
                        if abs(rec[check] - exc) > 1e-9:
                            raise SystemExit(f"{ds}/{rec['run']}: the "
                                             f"sensitivity file's {check} "
                                             "differs from probe_blocks.json")
                a_mean = float(np.mean(av))
                move = (a_mean - c_mean) / c_sd
                cells.append(tr.cell(f"{SHORT[ds]} {f} {arm}", "MLP comp.",
                                     f"{pct(a_mean)} ({num(move, 1)})",
                                     [a_mean * 100, move], [1, 1], src,
                                     "mlp.comp.var_fraction x100 mean; move "
                                     "in control sd (ddof 1)"))
                tr.prov.append(f"{SHORT[ds]} {f} {arm} ({flag}): "
                               f"{rel(fam[f][0])} : sections.{ds}.arms.{arm}"
                               f".records[*] (mean), sections.{ds}.moves."
                               f"{arm}.<key>.in_sd; MLP comp.: data/datasets/"
                               f"{ds}/runs/{{{','.join(a['names'])}}}/"
                               "validation/probe_blocks.json : "
                               "mlp.comp.var_fraction")
                body.append(f"\\quad\\quad {label} & {len(a['records'])} & "
                            + " & ".join(cells) + " \\\\")
        body.append("\\addlinespace")
    tr.pend("Transport row (all arms and controls)",
            "the sensitivity files' trusted fraction-of-ceiling rows are held "
            "until the held-out-tiles check (manifest row 37); the table "
            "carries one pending row instead of a column", "")
    body.append(f"\\multicolumn{{{len(SENS_READS) + 3}}}{{@{{}}l}}"
                "{Transport, fraction of ceiling, every arm: "
                "\\pending{held-out-tile check}} \\\\")
    tr.notes.append("Reads left out on purpose: the sensitivity files' cycle "
                    "rows and their 'cycle_w <= 0.02' guard are on the "
                    "retired label-derived cycling set (manifest: must not be "
                    "quoted); the pooled legacy probe rows; the tile-split "
                    "and all-panel transport rows.")
    caption = (
        "Sensitivity of the final configuration to the assumptions of the "
        "model, each arm refitted with one setting changed. For the final "
        "configuration, the mean over its seeds with the range; for an arm, "
        "its mean and, in brackets, its move from the final configuration "
        "in units of the final configuration's standard deviation over "
        "seeds; \\emph{Seeds} gives the number of seeds fitted. "
        "Reconstruction is held-out, in nats per count; "
        "$\\I(\\text{niche}; \\vw)$ is the excess over the within-type floor, "
        "in nats. \\emph{MLP comp.}: the residual neighbour composition "
        "in the intrinsic latent, as the share of its within-type variance "
        "that the nonlinear probe recovers beyond the permutation floor "
        "(as in \\cref{tab:probe}), in \\%. The leakage "
        "rate is set per cell from its neighbours' depth or density "
        "relative to its own, or per gene, at the same mean rate. "
        "Unassigned cells are not targets.")
    tex = (tr.header(command, "control: mean (min-max) over finalL_s0-s2; "
                     "arm: mean over its seeds (2 or 3) and, in brackets, "
                     "(arm mean - control mean) / control sd (ddof 1) as "
                     "stored by the sensitivity tables; accepted checkpoint; "
                     "Unassigned excluded as a target; MLP comp. = "
                     "mlp.comp.var_fraction x100 from each run's own "
                     "validation/probe_blocks.json, its move computed here "
                     "with the same rule (control sd, ddof 1)")
           + "\\begin{table}[t]\n\\centering\n"
           f"\\caption{{{caption}}}\n\\label{{tab:sensitivity}}\n\\small\n"
           "\\setlength{\\tabcolsep}{3pt}\n"
           "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
           f"{head}\n\\midrule\n" + "\n".join(body)
           + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return tex, tr


# -------------------------------------------------------- table 5: kappa sweep

SWEEP_READS = [("Recon.", "recon", 3), ("NMI", "nmi", 3),
               ("Mirror $R^2$", "mirror_r2", 3),
               ("Cycle $R^2$, $\\vz$", "cycle_r2_z_q90", 3),
               ("Cycle $R^2$, $\\vw$", "cycle_r2_w_q90", 3)]


# the guard's permutation floor is recomputed on every read; re-reads of the
# same fit agree to ~1e-5, far below the 2 decimals printed
MI_TOL = 5e-4


def table_kappa_sweep(command: str) -> tuple[str, Trace]:
    tr = Trace("tab:kappa-sweep")
    head = ("Section & $\\kappa$ & " + " & ".join(h for h, *_ in SWEEP_READS)
            + " & $\\I(\\text{niche}; \\vw)$ \\\\")
    body = []
    for ds in TRAINED:
        p = DATA / ds / "experiments" / "kappa_sweep_sweepL.json"
        d = tr.read_json(p)
        if MASK not in d["eval_mask"]["excluded_types"]:
            raise SystemExit(f"{rel(p)}: not masked")
        grid = d["config"]["values"]
        runs = d["runs"]
        # kappa = 0.1 is the final fits: its reads must equal theirs
        mism = []
        for r in runs:
            if r["kappa"] != d["config"]["kappa"]:
                continue
            q = DATA / ds / "runs" / FINAL.format(s=r["seed"]) / "degeneracy.json"
            own = tr.read_json(q)
            b = own.get("battery") or {}
            pairs = {"recon": b.get("recon_val_targets"),
                     "nmi": b.get("nmi_targets"),
                     "mirror_r2": (b.get("mirror") or {}).get("r2"),
                     "cycle_r2_z_q90": b.get("cycle_r2_z_q90"),
                     "cycle_r2_w_q90": b.get("cycle_r2_w_q90"),
                     "w_niche_mi_excess": (own.get("w_channel") or {})
                     .get("w_niche_mi_excess")}
            for k, v in pairs.items():
                tol = MI_TOL if k == "w_niche_mi_excess" else 1e-6
                if v is None or abs(v - r[k]) > tol:
                    mism.append((r["seed"], k, r[k], v))
        bad_keys = {k for _, k, *_ in mism}
        for s, k, a, b in mism:
            tr.notes.append(f"{SHORT[ds]}: at kappa = 0.1, seed {s}, the "
                            f"sweep's {k} = {a:.4f} but the final fit's own "
                            f"read is {b if b is None else round(b, 4)}.")
        # I(niche; w): the sweep file recomputes the guard; compare it with
        # every fit's own guard read (degeneracy.json w_channel)
        agree = {s: 0 for s in SEEDS}
        total = {s: 0 for s in SEEDS}
        for r in runs:
            run = (FINAL.format(s=r["seed"]) if r["kappa"] == d["config"]["kappa"]
                   else f"sweepL_k{r['kappa']:g}_s{r['seed']}")
            own = tr.read_json(DATA / ds / "runs" / run / "degeneracy.json")
            v = (own.get("w_channel") or {}).get("w_niche_mi_excess")
            sw = d["w_channel_guard"][f"k{r['kappa']:g}_s{r['seed']}"]
            total[r["seed"]] += 1
            agree[r["seed"]] += int(v is not None and abs(
                v - sw["w_niche_mi_excess"]) <= MI_TOL)
        mi_ok = all(agree[s] == total[s] for s in SEEDS)
        mi_why = ("the sweep file's I(niche; w) excess agrees with each "
                  "fit's own guard read on "
                  + ", ".join(f"seed {s}: {agree[s]}/{total[s]} kappa points"
                              for s in SEEDS)
                  + "; at kappa = 0.1 it should equal the final fits' read "
                  "(the sweep re-read appears to grade every seed on one "
                  "data split); needs a re-read")
        tr.notes.append(f"{SHORT[ds]}: I(niche; w) check against the fits' "
                        "own degeneracy.json: " + ("all agree." if mi_ok
                                                   else mi_why))
        for i, kap in enumerate(grid):
            rows = [r for r in runs if r["kappa"] == kap]
            cells = []
            for h, key, nd in SWEEP_READS:
                where = f"{SHORT[ds]} / kappa {kap} / {h}"
                vals = [r[key] for r in rows if r.get(key) is not None]
                if len(vals) != len(SEEDS) or key in bad_keys:
                    cells.append(tr.pend(where, f"{len(vals)} seeds, or the "
                                         "kappa = 0.1 check failed", "check"))
                    continue
                cells.append(tr.cell(f"{SHORT[ds]} {kap:g}", h,
                                     span(min(vals), max(vals), nd),
                                     [min(vals), max(vals)], nd, p,
                                     f"runs[kappa={kap:g}].{key} min, max"))
            where = f"{SHORT[ds]} / kappa {kap:g} / I(niche; w)"
            if mi_ok:
                vals = [d["w_channel_guard"][f"k{kap:g}_s{r['seed']}"]
                        ["w_niche_mi_excess"] for r in rows]
                cells.append(tr.cell(f"{SHORT[ds]} {kap:g}", "I(niche; w)",
                                     span(min(vals), max(vals), 2),
                                     [min(vals), max(vals)], 2, p,
                                     f"w_channel_guard[k{kap:g}_s*]"
                                     ".w_niche_mi_excess min, max"))
            else:
                cells.append(tr.pend(where, mi_why, "re-read"))
            lab = f"\\multirow{{{len(grid)}}}{{*}}{{{SHORT[ds]}}}" if i == 0 else ""
            ktxt = f"{kap:g}"
            if kap == d["config"]["kappa"]:
                ktxt = f"\\textbf{{{ktxt}}}"
            body.append((lab, ktxt, cells))
        tr.prov.append(f"{SHORT[ds]}: {rel(p)} : runs[*] grouped by 'kappa' "
                       f"over seeds {sorted({r['seed'] for r in runs})}, keys "
                       + ", ".join(k for _, k, _ in SWEEP_READS)
                       + "; I(niche; w) from w_channel_guard['k<kappa>_s<seed>']"
                       ".w_niche_mi_excess only if it agrees with every fit's "
                       "own runs/<run>/degeneracy.json w_channel read")
        tr.notes.append(f"{SHORT[ds]}: the kappa = 0.1 rows were checked "
                        f"against runs/finalL_s{{0,1,2}}/degeneracy.json "
                        "(masked battery and w_channel): "
                        + ("all equal." if not mism else
                           f"mismatches in {sorted(bad_keys)}."))
        body.append(None)
    body = body[:-1]
    # the I(niche; w) column appears only when every cell of it is a number;
    # otherwise one spanning pending row replaces it (width)
    mi_pending = any(r[2][-1].startswith("\\pending") for r in body if r)
    ncol = 2 + len(SWEEP_READS) + (0 if mi_pending else 1)
    if mi_pending:
        head = head.rsplit(" & ", 1)[0] + " \\\\"
        # sections whose I(niche; w) passed its check are withheld too, so
        # the column appears whole or not at all
        tr.cells = [c for c in tr.cells if c["col"] != "I(niche; w)"]
        tr.notes.append("I(niche; w) is withheld for every section while any "
                        "section fails its check (column shown whole or not "
                        "at all).")
    lines = []
    for r in body:
        if r is None:
            lines.append("\\midrule")
            continue
        lab, ktxt, cells = r
        cells = cells[:-1] if mi_pending else cells
        lines.append(f"{lab} & {ktxt} & " + " & ".join(cells) + " \\\\")
    if mi_pending:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{$\\I(\\text{{niche}}; "
                     "\\vw)$ excess across $\\kappa$: "
                     "\\pending{re-read}} \\\\")
    body = lines
    caption = (
        "The leakage sweep: every setting refitted at each rate $\\kappa$ "
        "with three seeds, as the range over seeds; the final configuration "
        f"is $\\kappa = {d['config']['kappa']:g}$ (bold), whose fits are those of "
        "\\cref{tab:headline}. Held-out reconstruction in nats per count; "
        "NMI of $\\vz$ with the type; mirror $R^2$; cycle $R^2$ on the "
        "top-decile cycling set; $\\I(\\text{niche}; \\vw)$, the response's "
        "information about the niche as its excess over a within-type "
        "permutation floor, in nats. Unassigned cells are not targets.")
    tex = (tr.header(command, "range [min-max] over seeds 0-2 at each kappa; "
                     "kappa = 0.1 is finalL_s0-s2; accepted checkpoint; "
                     "Unassigned excluded as a target; cycle reads on the "
                     "top-decile (q90) set")
           + "\\begin{table}[t]\n\\centering\n"
           f"\\caption{{{caption}}}\n\\label{{tab:kappa-sweep}}\n\\small\n"
           "\\setlength{\\tabcolsep}{4pt}\n"
           f"\\begin{{tabular}}{{@{{}}ll{'c' * (ncol - 2)}@{{}}}}\n\\toprule\n"
           f"{head}\n\\midrule\n" + "\n".join(body)
           + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return tex, tr


# ----------------------------------------------------------------------- main

TABLES = {"headline": table_headline,
          "probe": table_probe,
          "battery": table_battery,
          "sensitivity": table_sensitivity,
          "kappa_sweep": table_kappa_sweep}


def build(names: list[str], out: Path, command: str) -> dict[str, Trace]:
    out.mkdir(parents=True, exist_ok=True)
    traces = {}
    for name in names:
        tex, tr = TABLES[name](command)
        (out / f"{name}.tex").write_text(tex)
        traces[name] = tr
        print(f"wrote {rel(out / (name + '.tex'))}: {len(tr.cells)} cells "
              f"traced, {len(tr.pending)} pending")
    return traces


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--only", nargs="+", choices=list(TABLES),
                   default=list(TABLES))
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    rest = sys.argv[1:] if argv is None else argv
    command = " ".join(["python scripts/paper_tables.py", *rest])
    build(args.only, args.out, command)


if __name__ == "__main__":
    main()
