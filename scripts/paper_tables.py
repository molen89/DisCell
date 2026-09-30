#!/usr/bin/env python3
"""LaTeX tables for the AISTATS paper, generated from the frozen result files.

The results froze on 2026-09-29 at 15:07:21 (the last line of the Unassigned
re-read queue). Every number in a generated table is read from a file named in
``docs/results_manifest.md``; nothing is typed by hand. Each output file starts
with LaTeX comments (never rendered) that give the command, the time, every
source path with its modification time, the provenance of every row and
column, the reading convention and every pending cell with its reason.

Tables (one file each under ``submission_paper/aistats/tables/generated/``):

* ``headline``       -- tab:headline, section 4.2 (manifest row 1): mean
                        (min-max) per cell
* ``headline_full``  -- tab:headline-full, app:readouts: the same reads with
                        the range and the 95 % intervals on rows of their own
* ``probe``          -- tab:probe, section 3.3 (rows 2, 3; replaces the
                        hand-built table of the same label)
* ``battery``        -- tab:battery, section 4.6 (row 4)
* ``sensitivity``    -- tab:sensitivity, section 4.5 / appendix (rows 11, 21-23)
* ``kappa_sweep``    -- tab:kappa-sweep, section 4.5 appendix (row 9)

Added 2026-09-30 (after the author's transport decision of that day):

* ``timing``            -- tab:timing, app:timing (rows 5, 28; replaces the
                           hand-built table of the same label)
* ``breakdown``         -- tab:breakdown, sec:results-sweep / app:sweep (row 38)
* ``breakdown_traj``    -- tab:breakdown-traj, app:sweep (row 38, trajectories)
* ``transport_heldout`` -- tab:transport-heldout, sec:results-w or appendix
                           (row 37, sensitivity row)
* ``cellina_cf``        -- tab:cellina-cf, sec:results-w / sec:results-baselines
                           (row 36)
* ``synthetic``         -- tab:synthetic, app:planted / appendix synthetic
* ``planted_percell``   -- tab:planted-percell, app:planted / app:kl-maps
* ``context``           -- tab:context, app:baselines: what each method's context
                           latent carries (context_grade.json, devlog 2026-09-30)

Post-freeze sources (the queues of 2026-09-29/30) are flagged in each header
and named in its notes. Whole-section SIMVI / MintFlow cells follow
DisCell-baselines/results/feasibility.tsv, else the baselines_complete queue
log: running or queued -> pending; failed / capped / not attempted -> "could
not be run on the resources currently available" with the measured reason.

Usage::

    python scripts/paper_tables.py              # all thirteen
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
import csv
import io
import json
import math
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
INLINE = {OVARIAN: "ovarian FFPE", LUNG: "lung FFPE", FF: "ovarian FF",
          GSE: "the TMA core", DUAL: "the TMA serial section"}
SEEDS = (0, 1, 2)
FINAL = "finalL_s{s}"
GUARD = 0.25            # the probe guard: a block passes at <= 0.25 u
MASK = "Unassigned"

# comparison-method results and the whole-section baseline queue
BRES = REPO.parent / "DisCell-baselines" / "results"
FEAS = BRES / "feasibility.tsv"
BQ_LOG = REPO / "scripts" / "logs" / "baselines_complete_2026-09-28" / "queue.log"
# post-freeze analyses (2026-09-29 / 30)
HELDOUT = (REPO / "scripts" / "logs" / "transport_heldout_2026-09-29"
           / "transport_heldout_comparison")
BREAKDOWN = REPO / "scripts" / "logs" / "breakdown_2026-09-29"
SIDE_BY_SIDE = (REPO / "scripts" / "logs" / "cellina_extra_2026-09-29"
                / "transport_side_by_side_all.md")
SYNTH = DATA / "synthetic_smoke" / "experiments"
TIMING_ALL = (REPO / "scripts" / "logs" / "final_lineage_2026-09-25"
              / "timing_all_lineage.md")
WHOLE_SECTION = (OVARIAN, LUNG, FF)        # the queue's whole-section targets
CELLINA_EXTRA = [
    # (row label, section -> battery/probe entry)
    ("Cellina, niche domain", {OVARIAN: "cellina_nicheadv",
                               LUNG: "cellina_nicheadv",
                               FF: "cellina_nicheadv",
                               GSE: "cellina_nicheadv",
                               DUAL: "cellina_nicheadv_transfer"}),
    ("Cellina, own graph", {GSE: "cellina_owngraph",
                            DUAL: "cellina_owngraph_transfer"}),
]

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


def span_pending(cells: list[str]) -> list[str]:
    """Adjacent identical pending cells as one spanning cell (a pending
    marker is wider than a number; this keeps pending tables in width)."""
    out: list[str] = []
    i = 0
    while i < len(cells):
        j = i
        while ("\\pending" in cells[i] and j + 1 < len(cells)
               and cells[j + 1] == cells[i]):
            j += 1
        out.append(cells[i] if j == i
                   else f"\\multicolumn{{{j - i + 1}}}{{c}}{{{cells[i]}}}")
        i = j + 1
    return out


def tex_escape(s: str) -> str:
    """Plain text (e.g. a measured failure reason) made safe for LaTeX."""
    rep = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$",
           "#": "\\#", "_": "\\_", "{": "\\{", "}": "\\}",
           "~": "\\textasciitilde{}", "^": "\\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in s)


def finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def vrange(vals: list[float], nd: int) -> str:
    """A seed range; one value when both ends print the same."""
    a, b = span(min(vals), max(vals), nd), num(min(vals), nd)
    return b if rnd(min(vals), nd) == rnd(max(vals), nd) else a


#: the preferred direction of each readout (author, 2026-09-30): "up", higher
#: is better; "down", lower is better; "zero", its target is zero (no better
#: direction). A readout absent here has no preferred direction and no mark.
DIRECTION = {
    "recon": "up", "nmi": "up", "cycle_z": "up", "mi": "up",
    "transport": "up", "reada": "up", "twin": "up", "atlas": "up",
    "syn_nmi": "up", "syn_cca": "up", "syn_cosine": "up", "kl_auc": "up",
    "probe": "down", "fraction_left": "down", "mirror": "down",
    "comp_r2": "down", "time": "down",
    "cycle_w": "zero",
    # a context latent graded in the positive direction (tab:context)
    "ctx_probe": "up",
}
ARROWS = {"up": "$\\uparrow$", "down": "$\\downarrow$",
          "zero": "($\\approx 0$)"}
ARROW_NOTE = "Arrows give the preferred direction"
BOLD_NOTE = "bold marks the best value per section"
ZERO_NOTE = "$\\approx 0$: expected to be near zero"


def arrow(readout: str) -> str:
    """' <mark>' for a readout's label, '' when it has no direction."""
    d = DIRECTION.get(readout)
    return f" {ARROWS[d]}" if d else ""


def best(values: list, readout: str, texts: list[str] | None = None
         ) -> set[int]:
    """Indices of the best value by the readout's direction (max for "up",
    min for "down"), ignoring None and NaN; with *texts*, every entry that
    prints the same as the best is best too (a tie at the printed digits)."""
    d = DIRECTION[readout]
    if d not in ("up", "down"):
        raise ValueError(f"{readout}: no better direction to bold by")
    ok = [i for i, v in enumerate(values) if finite(v)]
    if not ok:
        return set()
    pick = (max if d == "up" else min)(ok, key=lambda i: values[i])
    if texts is None:
        return {i for i in ok if values[i] == values[pick]}
    return {i for i in ok if texts[i] == texts[pick]}


def bold(s: str) -> str:
    """A cell in bold, its math too."""
    return f"\\textbf{{\\boldmath {s}}}" if "$" in s else f"\\textbf{{{s}}}"


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
    # (label, envelope label, cross-slide key or None, digits, readout)
    ("Reconstruction",
     "held-out reconstruction (nats/count)", "recon_val_targets", 3, "recon"),
    ("NMI of $\\vz$ with the type", "NMI vs the type labels", "nmi_targets", 3,
     "nmi"),
    ("Mirror $R^2$", "mirror R²", "mirror.r2", 3, "mirror"),
    ("Cycle $R^2$ of $\\vz$", "cycle R² (z, top-decile set)",
     "cycle_r2_z_q90", 3, "cycle_z"),
    ("Cycle $R^2$ of $\\vw$", "cycle R² (w, top-decile set)",
     "cycle_r2_w_q90", 3, "cycle_w"),
    ("$\\I(\\text{niche}; \\vw)$ excess",
     "I(niche; w) excess over within-type floor (nats)", None, 2, "mi"),
    ("Transport",
     "transport, mean read: fraction of ceiling (trusted)", None, 2,
     "transport"),
    ("Atlas cross-seed cosine", "atlas cross-seed axis-1 cosine", None, 2,
     "atlas"),
]
TRANSPORT_LABEL = "transport, mean read: fraction of ceiling (trusted)"


def _dig(d: dict, dotted: str):
    for part in dotted.split("."):
        d = (d or {}).get(part)
    return d


#: per-run files behind the envelope's rows (for the sources list)
RAW_FILES = ("degeneracy.json", "recon_modes.json", "bootstrap_ci.json",
             "transport/transport.json", "atlas/atlas.json")


def raw_envelope(tr: Trace, ds: str) -> dict[str, dict]:
    """The envelope's rows recomputed from the unrounded per-run values:
    the records ``scripts/envelope_tables.py`` renders its markdown from
    (the same reader, so the same files and mask), before its printing
    rounds them. Keyed by the envelope's row label, in the shape of
    ``parse_envelope``; the markdown is then only a check, so every cell is
    rounded once, from the raw value."""
    import contextlib
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import envelope_tables as E
        from discell.model import eval_mask as EM
    finally:
        sys.path.pop(0)
    if MASK not in EM.exclusions():
        raise SystemExit("raw envelope: the Unassigned mask is not in force")
    with contextlib.redirect_stdout(io.StringIO()):
        recs = [E.run_record(ds, FINAL.format(s=s), "best") for s in SEEDS]
    for s in SEEDS:
        for f in RAW_FILES:
            q = DATA / ds / "runs" / FINAL.format(s=s) / f
            if q.exists():
                tr.sources[q] = q.stat().st_mtime
    out = {}
    for label, key, _fmt in E.ROWS:
        lo, mean, hi, n = E.envelope(recs, key)
        bounds = [r["ci"][key]["ci95"] for r in recs
                  if key in (r.get("ci") or {}) and r["ci"][key]["ci95"]
                  and all(finite(b) for b in r["ci"][key]["ci95"])]
        rep = all(r["ci"][key]["reproduces"] for r in recs
                  if key in (r.get("ci") or {}))
        out[label] = {"min": lo, "mean": mean, "max": hi, "n": n,
                      "ci": ((min(b[0] for b in bounds),
                              max(b[1] for b in bounds), len(bounds),
                              not rep) if bounds else None)}
    return out


def _printed_agrees(raw: float | None, printed: float | None) -> bool:
    """The markdown's printed value is the raw one at its printed digits."""
    if raw is None or printed is None:
        return raw is None and printed is None
    return abs(raw - printed) <= 0.5e-3 + 1e-9    # 3 or 4 printed decimals


def _headline_data(tr: Trace) -> tuple[list, dict[str, int]]:
    """The headline's cells, read and checked once for both renderings: the
    slim main-text table (tab:headline) and the full one with the intervals
    (tab:headline-full). Returns one (label, digits, readout, cells) per row,
    each cell a dict of kind "num" (mean, min, max, ci, mark, src, key),
    "pend" (where, why, short) or "na" (not read on that section), and the
    sections whose transport is over fewer than three seeds."""
    env, raw = {}, {}
    for ds in TRAINED:
        p = DATA / ds / "experiments" / "envelope_table_ci_at_best.md"
        text = tr.read_text(p)
        env[ds] = parse_envelope(text)
        raw[ds] = raw_envelope(tr, ds)
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

    rows = []
    seeds_n: dict[str, int] = {}      # transport cells over fewer seeds
    for label, elabel, ckey, nd, readout in HEADLINE_ROWS:
        cells = []
        for ds in SECTIONS:
            where = f"{label} / {SHORT[ds]}"
            if ds == DUAL:
                if ckey is None:
                    cells.append({"kind": "na"})
                    tr.prov.append(f"{where}: not evaluated on the held-out "
                                   "section (envelope held-out n = 0)")
                    continue
                vals = [_dig(cross[s][1], ckey) for s in SEEDS]
                mean = float(np.mean(vals))
                env_mean = env[GSE][elabel]["held-out mean"]
                keytxt = (f"held_out_section.{ckey} over seeds 0-2 "
                          f"(runs/finalL_s{{0,1,2}}/crossslide/{DUAL}.json)")
                if env_mean is None or abs(mean - env_mean) > 0.5e-4 + 1e-12:
                    cells.append({"kind": "pend", "where": where, "short":
                                  "check", "why": (
                                      f"per-seed cross-slide mean {mean!r} "
                                      "does not reproduce the envelope's "
                                      f"held-out mean {env_mean}")})
                    continue
                cells.append({"kind": "num", "mean": mean, "min": min(vals),
                              "max": max(vals), "ci": None, "mark": "",
                              "src": cross[0][0], "key": keytxt})
                tr.prov.append(f"{where}: {keytxt}; mean checked against "
                               f"{env_gse} : '{elabel}' held-out mean")
                continue
            row = raw[ds].get(elabel)
            printed = env[ds].get(elabel)
            src = DATA / ds / "experiments" / "envelope_table_ci_at_best.md"
            if row is None or row["n"] == 0:
                cells.append({"kind": "pend", "where": where, "short":
                              "missing", "why": "row missing from the "
                              "envelope"})
                continue
            agree = (printed is not None and printed["n"] == row["n"]
                     and all(_printed_agrees(row[k], printed[k])
                             for k in ("min", "mean", "max"))
                     and (row["ci"] is None) == (printed["ci"] is None)
                     and (row["ci"] is None or all(
                         _printed_agrees(row["ci"][i], printed["ci"][i])
                         for i in (0, 1))))
            if not agree:
                cells.append({"kind": "pend", "where": where, "short":
                              "check", "why": "the per-run values do not "
                              "reproduce the envelope's printed row"})
                continue
            mark = ""
            if row["n"] != 3:
                tr.notes.append(f"{where}: envelope n = {row['n']} seeds.")
                if elabel == TRANSPORT_LABEL:
                    mark = "$^{\\dagger}$"
                    seeds_n[ds] = row["n"]
            ci = None
            if row["ci"]:
                lo, hi, n, dagger = row["ci"]
                ci = (lo, hi)
                if dagger:
                    tr.notes.append(f"{where}: the envelope marks the CI "
                                    "with a dagger (a recomputed point did "
                                    "not reproduce the stored one).")
            cells.append({"kind": "num", "mean": row["mean"], "min": row["min"],
                          "max": row["max"], "ci": ci, "mark": mark,
                          "src": src, "key": f"'{elabel}'"})
            tr.prov.append(f"{where}: the per-run values behind "
                           f"{rel(src)} : row '{elabel}' (runs/finalL_s0-s2; "
                           "mean; min, max; CI envelope), rounded once from "
                           "the raw values; the printed row is a check")
        rows.append((label, nd, readout, cells))
    tr.notes.append("The held-out section (TMA serial) is read through the "
                    "models fitted on the core; the envelope gives it a mean "
                    "only, so the range is recomputed from the three per-seed "
                    "cross-slide files and the mean is checked against the "
                    "envelope. It has no tile-bootstrap interval, and "
                    "I(niche; w), transport and atlas are not read on it "
                    "('--', not pending: no queue computes them).")
    tr.notes.append("Values are rounded once, half-up, from the unrounded "
                    "per-run values that the envelope markdown is printed "
                    "from (scripts/envelope_tables.py run_record, same files "
                    "and mask); the markdown's printed digits (4 decimals; "
                    "transport and atlas 3) are only checked against them. "
                    "Rounding the printed digits again had shown the TMA "
                    "core's transport maximum 0.7547 as 0.76 (audit "
                    "2026-09-30, item 5).")
    tr.notes.append("Transport released 2026-09-30 (devlog 'Transport rule "
                    "triggered; the published read stays the headline', "
                    "author): the published read (fold 0 scored, cell-split "
                    "ceiling, trusted tier) is the headline; the held-out-"
                    "tiles read is tab:transport-heldout.")
    tr.notes.append("Arrows: the preferred direction of each read (author, "
                    "2026-09-30); none is bold, since no method competes here.")
    return rows, seeds_n


def _headline_caption_parts(seeds_n: dict[str, int]) -> tuple[str, str]:
    """The transport sentence (without its interval) and the shared tail."""
    few = ", ".join(f"{SHORT[ds]}, {n} of 3" for ds, n in seeds_n.items())
    transport = (
        "Transport is the fraction of the noise ceiling (cell-split "
        "reliability) recovered by the mean transport read, over the panels "
        "in the trusted tier"
        + (f" ($^{{\\dagger}}$seeds with a trusted panel: {few})"
           if seeds_n else ""))
    tail = (
        ". The same read scored on held-out tiles only is a sensitivity "
        "row (\\cref{tab:transport-heldout}). Cycle "
        "$R^2$ is read on the top decile of the S and G2M score among "
        "held-out cells. $\\I(\\text{niche}; \\vw)$ is the excess over a "
        "within-type permutation floor, in nats. The atlas cosine compares the "
        "first programme axis between seeds. The serial section is read "
        "through the models fitted on the core (--: not read there). "
        f"{ARROW_NOTE}; {ZERO_NOTE}. "
        "Unassigned cells are not targets of any read.")
    return transport, tail


def table_headline(command: str) -> tuple[str, Trace]:
    """tab:headline, the main text's table: one row per read, each cell the
    mean over seeds with the range; the intervals are in tab:headline-full."""
    tr = Trace("tab:headline")
    rows, seeds_n = _headline_data(tr)
    body = []
    for label, nd, readout, cells in rows:
        out = []
        # the range in a smaller size beside the mean; a row with a negative
        # range ("-7.232 to -7.148") puts it under the mean, to fit the page
        stack = any(c["kind"] == "num" and " to " in span(c["min"], c["max"],
                                                          nd) for c in cells)
        for ds, c in zip(SECTIONS, cells):
            if c["kind"] == "na":
                out.append("--")
            elif c["kind"] == "pend":
                out.append(tr.pend(c["where"], c["why"], c["short"]))
            else:
                mean = f"{num(c['mean'], nd)}{c['mark']}"
                rng = f"{{\\scriptsize ({span(c['min'], c['max'], nd)})}}"
                text = (f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{mean}\\\\{rng}"
                        "\\end{tabular}" if stack else f"{mean} {rng}")
                out.append(tr.cell(label, SHORT[ds], text,
                                   [c["mean"], c["min"], c["max"]], nd,
                                   c["src"], f"{c['key']} mean (min, max)"))
        body.append(f"{label}{arrow(readout)} & " + " & ".join(out) + " \\\\")
    transport, tail = _headline_caption_parts(seeds_n)
    caption = (
        "Model quality on the four sections and the held-out serial section "
        "of the TMA core: the mean over three seeds, with the range over "
        "seeds in brackets; the $95\\%$ intervals are in "
        "\\cref{tab:headline-full}. "
        "Reconstruction is the held-out log-likelihood in nats per count. "
        + transport + tail)
    tex = (tr.header(command, CONVENTION + "; the serial section's values "
                     "come from the core's three fits read on it; mean "
                     "(min-max) per cell, the intervals in tab:headline-full")
           + "\\begin{table*}[t]\n\\centering\n"
           f"\\caption{{{caption}}}\n\\label{{tab:headline}}\n\\footnotesize\n"
           "\\setlength{\\tabcolsep}{3pt}\n"
           "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
           "Read & " + " & ".join(SHORT[c] for c in SECTIONS) + " \\\\\n"
           "\\midrule\n" + "\n".join(body)
           + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    return tex, tr


def table_headline_full(command: str) -> tuple[str, Trace]:
    """tab:headline-full (appendix): the reads of tab:headline with the range
    over seeds and the tile-bootstrap intervals on rows of their own."""
    tr = Trace("tab:headline-full")
    rows, seeds_n = _headline_data(tr)
    body = []
    for label, nd, readout, cells in rows:
        mean_cells, range_cells, ci_cells = [], [], []
        for ds, c in zip(SECTIONS, cells):
            if c["kind"] == "na":
                mean_cells.append("--")
                range_cells.append("")
                ci_cells.append("")
                continue
            if c["kind"] == "pend":
                mean_cells.append(tr.pend(c["where"], c["why"], c["short"]))
                range_cells.append("")
                ci_cells.append("")
                continue
            mean_cells.append(tr.cell(label, SHORT[ds],
                                      num(c["mean"], nd) + c["mark"],
                                      [c["mean"]], nd, c["src"],
                                      f"{c['key']} mean"))
            range_cells.append(tr.cell(label + " range", SHORT[ds],
                                       span(c["min"], c["max"], nd),
                                       [c["min"], c["max"]], nd, c["src"],
                                       f"{c['key']} min, max"))
            if c["ci"]:
                lo, hi = c["ci"]
                ci_cells.append(tr.cell(label + " CI", SHORT[ds],
                                        interval(lo, hi, nd), [lo, hi], nd,
                                        c["src"], f"{c['key']} CI column"))
            else:
                ci_cells.append("--")
        body.append(f"{label}{arrow(readout)} & " + " & ".join(mean_cells)
                    + " \\\\")
        if any(range_cells):
            body.append("\\quad range over seeds & "
                        + " & ".join(range_cells) + " \\\\")
        if any(c.get("ci") for c in cells):
            body.append("\\quad $95\\%$ interval & "
                        + " & ".join(ci_cells) + " \\\\")
        body.append("\\addlinespace")
    body = body[:-1]
    transport, tail = _headline_caption_parts(seeds_n)
    caption = (
        "The reads of \\cref{tab:headline} with their intervals, on the four "
        "sections and the held-out serial section of the TMA core: mean "
        "over three seeds, the range over seeds and, where available, the "
        "$200\\um$ tile-bootstrap $95\\%$ interval (lowest lower and highest "
        "upper bound over the seeds). Reconstruction is the held-out "
        "log-likelihood in nats per count. " + transport
        + "; its interval is from half-tile subsampling" + tail)
    tex = (tr.header(command, CONVENTION + "; the serial section's values "
                     "come from the core's three fits read on it")
           + "\\begin{table*}[t]\n\\centering\n"
           f"\\caption{{{caption}}}\n\\label{{tab:headline-full}}\n\\small\n"
           "\\setlength{\\tabcolsep}{3pt}\n"
           "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
           "Read & " + " & ".join(SHORT[c] for c in SECTIONS) + " \\\\\n"
           "\\midrule\n" + "\n".join(body)
           + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    return tex, tr


# --------------------------------------------------------------- table 2: probe


def _probe_file(tr: Trace, ds: str) -> tuple[Path, dict]:
    p = DATA / ds / "experiments" / "probe_regrade_lineage_final.json"
    return p, tr.read_json(p)


BLOCK_MAP = {"ridge_comp": ("ridge", "comp"), "ridge_img": ("ridge", "img"),
             "mlp_comp": ("mlp", "comp"), "mlp_img": ("mlp", "img")}


def _method_probe_path(ds: str, entry: str) -> Path:
    name = re.sub(r"[^A-Za-z0-9]+", "_", entry).strip("_")
    return DATA / ds / "experiments" / "probe_regrade_lineage" / f"{name}.json"


def _convert_method_probe(tr: Trace, ds: str, p: Path, final: dict) -> dict:
    """A per-method probe record in the final table's per-block layout; its
    uncontrolled reference must be the final table's seed mean."""
    raw = tr.read_json(p)
    rec = {}
    for b, (fam, blk) in BLOCK_MAP.items():
        r = raw[fam][blk]
        unc = _uncontrolled_mean(final, b)
        if abs(r["uncontrolled_excess"] - unc) > 1e-9:
            raise SystemExit(f"{rel(p)} {b}: uncontrolled reference "
                             f"{r['uncontrolled_excess']} is not the final "
                             f"table's seed mean {unc}")
        rec[b] = {k: r[k] for k in ("excess", "var_fraction",
                                     "fraction_of_uncontrolled", "pass")}
    return rec


def _probe_with_methods(tr: Trace, ds: str, wanted: list[str]
                        ) -> tuple[Path, dict, dict[str, Path]]:
    """The final probe table of a section, with any wanted entry it does not
    hold yet taken from that method's own probe record (written by the
    baseline queue's scoring step; the final table is re-rendered only at the
    end of that queue). Returns (final path, entries, entry -> source path)."""
    p, d = _probe_file(tr, ds)
    d = dict(d)
    src = {e: p for e in d}
    # the converter is checked once per section on an entry held in both
    checked = tr.__dict__.setdefault("converter_checked", set())
    if ds not in checked:
        for e in list(d):
            q = _method_probe_path(ds, e)
            if e.startswith("DisCell") or not q.exists():
                continue
            conv = _convert_method_probe(tr, ds, q, d)
            for b in BLOCK_MAP:
                for k in ("excess", "var_fraction", "fraction_of_uncontrolled"):
                    if abs(conv[b][k] - d[e][b][k]) > 1e-12:
                        raise SystemExit(f"{rel(q)}: {b}.{k} differs from the "
                                         f"final table's '{e}'")
            tr.notes.append(f"{SHORT[ds]}: per-method probe records convert "
                            f"exactly to the final table's layout (checked on "
                            f"'{e}', {rel(q)}).")
            checked.add(ds)
            break
    for e in wanted:
        if e in d:
            continue
        q = _method_probe_path(ds, e)
        if q.exists():
            d[e] = _convert_method_probe(tr, ds, q, d)
            src[e] = q
            tr.notes.append(f"{SHORT[ds]}: '{e}' is not yet in {p.name} (the "
                            "queue re-renders it at its end); read from its "
                            f"own probe record {rel(q)}, whose uncontrolled "
                            "reference equals the final table's seed mean "
                            "(checked).")
    return p, d, src


# ---------------------------------------------------- whole-section baselines

def baseline_state(tr: Trace, tool: str, ds: str) -> tuple[str, dict | None]:
    """The whole-section attempt of SIMVI or MintFlow on a section: the last
    feasibility.tsv row if there is one (ok | truncated | capped | failed |
    not_attempted), else 'running' or 'queued' from the queue's log."""
    rows = list(csv.DictReader(io.StringIO(tr.read_text(FEAS)),
                               delimiter="\t"))
    mine = [r for r in rows if r["tool"] == tool and r["dataset"] == ds]
    if mine:
        return mine[-1]["outcome"], mine[-1]
    log = tr.read_text(BQ_LOG)
    key = f"fit_{tool.lower()}_{ds}"
    started = f" start {key} (" in log
    done = f" done  {key} (" in log
    return ("running" if started and not done else "queued"), None


def section_methods(tr: Trace, ds: str, bat: dict) -> list[dict]:
    """The comparison methods of one section, in table order: each with its
    battery and probe entries, or the state of its whole-section attempt."""
    out = [{"label": "resolVI", "battery": "resolVI", "probe": "resolVI"}]
    fixed = {("SIMVI", GSE): "SIMVI (lineage)",
             ("SIMVI", DUAL): "SIMVI (lineage, fit on this section)",
             ("MintFlow", GSE): "MintFlow (lineage)",
             ("MintFlow", DUAL): "MintFlow (lineage, transfer)"}
    for tool in ("SIMVI", "MintFlow"):
        m = {"label": tool, "tool": tool}
        if (tool, ds) in fixed:
            m["battery"] = m["probe"] = fixed[(tool, ds)]
        else:
            state, row = baseline_state(tr, tool, ds)
            m["state"], m["row"] = state, row
            e = whole_entry(bat, tool)
            if state in RUNNABLE and e:
                m["battery"] = m["probe"] = e
                m["truncated"] = "truncated" in e
            elif state in RUNNABLE:
                m["state"] = "scoring"
        out.append(m)
    out.append({"label": "Cellina",
                "battery": ("Cellina (lineage, transfer)" if ds == DUAL
                            else "Cellina (lineage)"),
                # the serial section's probe: 'Cellina (lineage)' (same model
                # and excess as the transfer entry, whose fraction is stale)
                "probe": "Cellina (lineage)"})
    for label, where in CELLINA_EXTRA:
        if ds in where:
            out.append({"label": label, "battery": where[ds],
                        "probe": where[ds], "extra": True})
    return out


NOT_RUN = "could not be run on the resources currently available"
RUNNABLE = ("ok", "truncated")


def whole_entry(bat: dict, tool: str) -> str | None:
    """The battery column of a whole-section fit, if scored."""
    if tool == "SIMVI":
        return "SIMVI (lineage)" if "SIMVI (lineage)" in bat else None
    for k in bat:
        if k == "MintFlow (lineage)" or k.startswith("MintFlow (lineage, trunc"):
            return k
    return None


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


def probe_share(excess: float) -> float:
    """A probe excess (nats per component) as the share of the block's
    within-type variance that the probe explains beyond the permutation
    floor, 1 - exp(-2 excess), in %. Every probe percentage of the paper
    (tables and figures) goes through this. The records' var_fraction,
    exp(2 excess) - 1, is the error ratio minus one, not a share, and is
    unbounded (above 100 % for a latent that carries the niche); it is
    checked against the excess but never printed (author, 2026-09-30)."""
    return float(-100.0 * np.expm1(-2.0 * excess))


def pct(v: float, nd: int = 1) -> str:
    """A share already in %, rounded half-up exactly."""
    return num(v, nd)


def pct_span(vals: list[float], nd: int = 1) -> str:
    return f"{pct(min(vals), nd)}--{pct(max(vals), nd)}"


# comparison-method rows of the probe table: (row label, [method labels])
PROBE_ROWS = [("resolVI, Cellina", ["resolVI", "Cellina"]),
              ("MintFlow, SIMVI", ["MintFlow", "SIMVI"]),
              ("Cellina, niche domain", ["Cellina, niche domain"]),
              ("Cellina, own graph", ["Cellina, own graph"])]
PROBE_BLOCK_ORDER = [("mlp_comp", "Composition, MLP probe"),
                     ("mlp_img", "Image, MLP probe"),
                     ("ridge_comp", "Composition, ridge probe"),
                     ("ridge_img", "Image, ridge probe")]


def _state_phrase(pending: dict[str, list[str]]) -> str:
    """'MintFlow on ovarian FFPE and SIMVI on ...' from state -> [items]."""
    parts = []
    for state, items in pending.items():
        if items:
            verb = {"running": "is being run", "queued": "are queued",
                    "scoring": "are being scored"}.get(state, state)
            if state == "running" and len(items) > 1:
                verb = "are being run"
            if state == "queued" and len(items) == 1:
                verb = "is queued"
            parts.append(f"{' and '.join(items)} {verb}")
    return "; ".join(parts)


def _pending_items(methods_by_ds: dict[str, list[dict]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for tool in ("MintFlow", "SIMVI"):
        for state in ("running", "scoring", "queued"):
            secs = [INLINE[ds] for ds, ms in methods_by_ds.items() for m in ms
                    if m.get("tool") == tool and m.get("state") == state
                    and "battery" not in m]
            if secs:
                out.setdefault(state, []).append(
                    f"{tool} on {', '.join(secs)}")
    return out


def table_probe(command: str) -> tuple[str, Trace]:
    tr = Trace("tab:probe")
    files, methods = {}, {}
    for ds in SECTIONS:
        bp = DATA / ds / "experiments" / "baseline_battery_lineage.json"
        methods[ds] = section_methods(tr, ds, tr.read_json(bp))
        wanted = [m["probe"] for m in methods[ds] if m.get("probe")]
        p, d, src = _probe_with_methods(tr, ds, wanted)
        files[ds] = (p, d, src)
        used = ([f"DisCell/finalL_s{s}" for s in SEEDS]
                + ["DisCell/uncontrolledL_s0", "DisCell/uncontrolledL_s1"]
                + [e for e in wanted if src.get(e) == p])
        bad = _snapshot_check(tr, ds, p.name, tr.read_json(p), used)
        if bad:
            raise SystemExit(f"{SHORT[ds]}: {sorted(bad)} differ from the "
                             "frozen snapshot")
        # the files' var_fraction is exp(2 excess) - 1 of its own excess (a
        # check on the source; the tables print probe_share of the excess)
        for e in used + [e for e in wanted if e in d and e not in used]:
            for b, _ in PROBE_BLOCK_ORDER:
                rec = d[e][b]
                if abs(rec["var_fraction"]
                       - (np.exp(2 * rec["excess"]) - 1)) > 1e-9:
                    raise SystemExit(f"{rel(src[e])} {e}.{b}: var_fraction "
                                     "is not exp(2 excess) - 1")
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
                    f"{{\\emph{{{label}}}{arrow('probe')}}} \\\\")
        rows = {"without": [], "with": [], "left": []}
        with_mean = []                   # the final fits' mean, per section
        for ds in SECTIONS:
            p, d, _ = files[ds]
            vu = [probe_share(d[e][block]["excess"]) for e in uncs]
            vf = [probe_share(d[e][block]["excess"]) for e in finals]
            with_mean.append(float(np.mean(vf)))
            probs = [_probe_consistent(d, e, block) for e in finals]
            rows["without"].append(tr.cell(
                f"{label} without", SHORT[ds], pct_span(vu), [min(vu), max(vu)], 1, p,
                f"DisCell/uncontrolledL_s{{0,1}}.{block}.excess: 1 - exp(-2 excess), x100"))
            rows["with"].append(tr.cell(
                f"{label} with", SHORT[ds], pct_span(vf), [min(vf), max(vf)], 1, p,
                f"DisCell/finalL_s{{0,1,2}}.{block}.excess: 1 - exp(-2 excess), x100"))
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
                           f".{block}.excess as 1 - exp(-2 excess) (x100, "
                           "min-max); finalL "
                           f".{block}.fraction_of_uncontrolled (min-max)")
        # the comparison methods: (text, value or None) per part of a cell
        parts: dict[tuple[str, str], list] = {}
        for rlabel, labels in PROBE_ROWS:
            for ds in SECTIONS:
                p, d, src = files[ds]
                ps = parts[(rlabel, ds)] = []
                for meth in labels:
                    m = next((x for x in methods[ds] if x["label"] == meth),
                             None)
                    if m is None or m.get("probe") not in d:
                        ps.append(("--", None))
                        why = ("not run on this section" if m is None
                               else f"whole-section fit: {m.get('state')}")
                        tr.prov.append(f"{label} / {meth} / {SHORT[ds]}: "
                                       f"-- ({why})")
                        continue
                    e = m["probe"]
                    v = probe_share(d[e][block]["excess"])
                    ps.append((tr.cell(f"{label} {meth}", SHORT[ds],
                                       pct(v), [v], 1, src[e],
                                       f"'{e}'.{block}.excess: 1 - exp(-2 "
                                       "excess), x100"), v))
                    tr.prov.append(f"{label} / {meth} / {SHORT[ds]}: "
                                   f"{rel(src[e])} : '{e}'.{block}"
                                   ".excess as 1 - exp(-2 excess), x100")
        # bold: the lowest share per section among the methods, DISCELL by
        # the mean of its final fits (the fits without the adversary are an
        # ablation, not a contestant); ties at the printed digits all bold
        for i, ds in enumerate(SECTIONS):
            cands = [("with", 0, with_mean[i])] + [
                (rl, j, v) for rl, _ in PROBE_ROWS
                for j, (_, v) in enumerate(parts[(rl, ds)])]
            vals = [c[2] for c in cands]
            win = best(vals, "probe", [pct(v) if finite(v) else "" for v in vals])
            for k in win:
                rl, j, _ = cands[k]
                if rl == "with":
                    rows["with"][i] = bold(rows["with"][i])
                else:
                    t, v = parts[(rl, ds)][j]
                    parts[(rl, ds)][j] = (bold(t), v)
            tr.notes.append(f"{label} / {SHORT[ds]}: bold "
                            + ", ".join("DISCELL (mean of the final fits)"
                                        if cands[k][0] == "with" else
                                        f"{cands[k][0]} part {cands[k][1] + 1}"
                                        for k in sorted(win)) + ".")
        body.append("\\quad without the adversary (\\%) & "
                    + " & ".join(rows["without"]) + " \\\\")
        body.append("\\quad with the adversary (\\%) & "
                    + " & ".join(rows["with"]) + " \\\\")
        body.append(f"\\quad fraction left{arrow('fraction_left')} & "
                    + " & ".join(rows["left"]) + " \\\\")
        for rlabel, _ in PROBE_ROWS:
            cells = []
            for ds in SECTIONS:
                ps = [t for t, _ in parts[(rlabel, ds)]]
                cells.append("--" if all(q == "--" for q in ps)
                             else ", ".join(ps))
            body.append(f"\\quad {rlabel} (\\%) & " + " & ".join(cells)
                        + " \\\\")
        body.append("\\addlinespace")
    body = body[:-1]
    tr.notes.append("The fraction left is the file's fraction_of_uncontrolled "
                    "(each seed's excess over the seed-mean excess of the two "
                    "fits without the adversary), checked against the excesses "
                    "in the same file. Percentages are 1 - exp(-2 excess) "
                    "(author, 2026-09-30: the share of within-type variance "
                    "explained; the files' var_fraction, exp(2 excess) - 1, "
                    "is checked against the excess but not printed). No "
                    "pass/fail verdicts are "
                    "shown (author, 2026-09-29: the probe is reported "
                    "descriptively; the development threshold is disclosed "
                    "once in the method). Bold (author, 2026-09-30) marks the "
                    "lowest share per block and section among DISCELL with "
                    "the adversary (its mean over seeds) and the comparison "
                    "methods; the old bold on the MLP 'fraction left' rows is "
                    "gone, so bold has one meaning.")
    pending = _pending_items(methods)
    for state, items in pending.items():
        for it in items:
            tr.pending.append(f"{it}: whole-section fit {state} (queue "
                              "baselines_complete_2026-09-28); '--' until "
                              "scored")
    failed = [(m["label"], ds, m["row"]) for ds, ms in methods.items()
              for m in ms if m.get("state") not in (None, "running", "queued",
                                                    "scoring", *RUNNABLE)]
    still = _state_phrase(pending)
    caption = (
        "The held-out probe on the final fits, per section. For each block "
        "and probe, the share of the block's within-type variance that the "
        "probe explains from the intrinsic latent beyond the permutation "
        "floor ($1 - e^{-2\\,\\mathrm{excess}}$, in \\%): for the model "
        "trained without the adversary ($\\alphaa = 0$, two seeds), with it "
        "(the final fits, three seeds), and for the comparison methods on "
        "the same section. The \\emph{fraction left} is the excess with the "
        "adversary as a fraction of the mean excess without it: $0$ when the "
        "adversary removes everything the probe can find, $1$ when it "
        "removes nothing. Ranges are over seeds; Unassigned cells are not "
        f"targets. {ARROW_NOTE}; {BOLD_NOTE} and probe block, among DISCELL "
        "with the adversary (by its mean over seeds) and the comparison "
        "methods. The serial section of the TMA core is graded with our "
        "models fitted on the core; there, the Cellina rows and MintFlow are "
        "the core's models transferred, and resolVI and SIMVI are fitted on "
        "the serial section itself. \\emph{Cellina, niche domain}: Cellina "
        "with its domain adversary given our niche label (clusters of "
        "neighbour composition), which matches what this probe grades, so "
        "it is Cellina's best case on the probe, not its published setting. "
        "\\emph{Cellina, own graph}: Cellina on its own neighbour graph "
        "instead of ours."
        + (f" Whole-section fits still in progress: {still} (--)." if still
           else "")
        + (" " + "; ".join(f"{lab} on {INLINE[ds]} {NOT_RUN}"
                           for lab, ds, _ in failed) + " (--)." if failed
           else "")
        + " --: not run on that section.")
    tex = (tr.header(command, "range [min-max] over seeds (finalL_s0-s2; "
                     "uncontrolledL_s0-s1) of the per-block probe excess "
                     "expressed as 1 - exp(-2 excess) in %, and of the excess "
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

BATTERY_COLS = [
    # (header, key, digits, source, readout)
    ("NMI", "nmi", 3, "battery", "nmi"),
    ("Comp., ridge", "ridge_comp", 2, "probe", "fraction_left"),
    ("Comp., MLP", "mlp_comp", 2, "probe", "fraction_left"),
    ("Mirror $R^2$", "mirror.r2", 3, "battery", "mirror"),
    ("Cycle $R^2$", "cycle_q90.z.r2_pooled", 3, "battery", "cycle_z"),
    ("Recon.", "reconstruction.recon", 3, "battery", "recon"),
]
BATTERY_LABEL = {"Cellina, niche domain": "Cellina, niche$^{d}$",
                 "Cellina, own graph": "Cellina, own graph"}


def table_battery(command: str) -> tuple[str, Trace]:
    tr = Trace("tab:battery")
    bat, prb, methods = {}, {}, {}
    for ds in SECTIONS:
        bp = DATA / ds / "experiments" / "baseline_battery_lineage.json"
        bat[ds] = (bp, tr.read_json(bp))
        methods[ds] = section_methods(tr, ds, bat[ds][1])
        wanted = [m["probe"] for m in methods[ds] if m.get("probe")]
        prb[ds] = _probe_with_methods(tr, ds, wanted)
    bad = {}
    for ds in SECTIONS:
        pp, pd, psrc = prb[ds]
        used_b = [f"DisCell/finalL_s{s}" for s in SEEDS] + [
            m["battery"] for m in methods[ds] if m.get("battery")]
        used_p = [f"DisCell/finalL_s{s}" for s in SEEDS] + [
            "DisCell/uncontrolledL_s0", "DisCell/uncontrolledL_s1"] + [
            m["probe"] for m in methods[ds]
            if m.get("probe") and psrc.get(m["probe"]) == pp]
        bad[ds] = (_snapshot_check(tr, ds, bat[ds][0].name, bat[ds][1], used_b)
                   | _snapshot_check(tr, ds, pp.name, tr.read_json(pp),
                                     used_p))
        for e in used_b:
            if e not in bat[ds][1]:
                bad[ds].add(e)
                tr.notes.append(f"{SHORT[ds]}/{e}: not in the battery file")
                continue
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

    def path_of(ds, entry, src):
        return bat[ds][0] if src == "battery" else prb[ds][2][entry]

    # short heads for the two probe columns (with the arrows the long ones
    # overran the text width; a spanning head over them overran the page)
    shown_head = {"Comp., ridge": "Ridge", "Comp., MLP": "MLP"}
    head = ("Section & Method & " + " & ".join(shown_head.get(h, h) + arrow(rd)
                                              for h, *_, rd in BATTERY_COLS)
            + " \\\\")
    body = []
    ncol = len(BATTERY_COLS)
    notrun: list[str] = []
    any_pending = any_trunc = False
    for ds in SECTIONS:
        nrows = 2 + len(methods[ds])
        # DisCell: mean row and range row
        mean_cells, rng_cells, mean_vals = [], [], []
        # rows of the section: a scored row is [prefix, cells, values] (its
        # cells can be bold), anything else a finished line
        sec_rows: list = []
        for h, key, nd, src, _ in BATTERY_COLS:
            entries = [f"DisCell/finalL_s{s}" for s in SEEDS]
            where = f"{SHORT[ds]} / DisCell / {h}"
            path = bat[ds][0] if src == "battery" else prb[ds][0]
            probs = ([_probe_consistent(prb[ds][1], e, key) for e in entries]
                     if src == "probe" else [])
            if any(e in bad[ds] for e in entries) or any(probs):
                mean_cells.append(tr.pend(where, "failed a consistency "
                                          "check (see notes)", "check"))
                rng_cells.append("")
                mean_vals.append(None)
                continue
            vals = [value(ds, e, key, src) for e in entries]
            m = float(np.mean(vals))
            mean_vals.append(m)
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
        lab = f"\\multirow{{{nrows}}}{{*}}{{{SHORT[ds]}}}"
        sec_rows.append([f"{lab} & DISCELL", mean_cells, mean_vals])
        sec_rows.append(" & \\quad range & " + " & ".join(rng_cells) + " \\\\")
        for mt in methods[ds]:
            meth = mt["label"]
            shown = BATTERY_LABEL.get(meth, meth)
            if "battery" not in mt:
                state, row = mt["state"], mt.get("row")
                where = f"{SHORT[ds]} / {meth} / all columns"
                if state in ("running", "queued", "scoring"):
                    any_pending = True
                    why = {"running": "whole-section fit running",
                           "queued": "whole-section fit queued",
                           "scoring": "fit done, battery not yet written"
                           }[state] + " (queue baselines_complete_2026-09-28)"
                    cell = tr.pend(where, why, state) + "$^{a}$"
                else:
                    notrun.append(f"{meth}, {INLINE[ds]}: "
                                  + tex_escape(row["reason"] if row else state))
                    cell = f"{NOT_RUN}$^{{e}}$"
                    tr.prov.append(f"{where}: feasibility.tsv outcome "
                                   f"'{state}': {row}")
                sec_rows.append(f" & {shown} & \\multicolumn{{{ncol}}}{{c}}"
                                f"{{{cell}}} \\\\")
                continue
            if mt.get("truncated"):
                any_trunc = True
                shown += "$^{f}$"
            cells, cvals = [], []
            for h, key, nd, src, _ in BATTERY_COLS:
                where = f"{SHORT[ds]} / {meth} / {h}"
                cvals.append(None)          # set below when a number shows
                entry = mt["battery"] if src == "battery" else mt["probe"]
                if entry in bad[ds] or (src == "probe"
                                        and entry not in prb[ds][1]):
                    cells.append(tr.pend(where, "entry missing or differs "
                                         "from the frozen snapshot", "check"))
                    continue
                path = path_of(ds, entry, src)
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
                cvals[-1] = v
                tr.prov.append(f"{where}: {rel(path)} : '{entry}'.{kk}")
            sec_rows.append([f" & {shown}", cells, cvals])
        # bold: the best value per column in the section, DISCELL by its
        # mean; ties at the printed digits all bold
        scored = [r for r in sec_rows if isinstance(r, list)]
        for j, (h, key, nd, src, rd) in enumerate(BATTERY_COLS):
            vals = [r[2][j] for r in scored]
            for k in best(vals, rd, [num(v, nd) if finite(v) else ""
                                     for v in vals]):
                scored[k][1][j] = bold(scored[k][1][j])
                tr.notes.append(f"{SHORT[ds]} / {h}: bold on "
                                f"{scored[k][0].split('& ')[-1]}.")
        for r in sec_rows:
            body.append(r if isinstance(r, str)
                        else f"{r[0]} & " + " & ".join(r[1]) + " \\\\")
        body.append("\\midrule")
    body = body[:-1]
    tr.notes.append("NMI and reconstruction here are the battery's own reads, "
                    "computed the same way for every method; for DISCELL they "
                    "differ slightly from the headline table's (the trainer's "
                    "reads), so the two tables' DISCELL rows are not "
                    "interchangeable.")
    for ds in SECTIONS:
        counts = {e: (bat[ds][1][e].get("reconstruction") or {}).get("n_cells")
                  for e in [f"DisCell/finalL_s{s}" for s in SEEDS]
                  + [m["battery"] for m in methods[ds] if m.get("battery")]}
        tr.notes.append(f"{SHORT[ds]}: target cells scored per entry "
                        f"(reconstruction.n_cells): {counts}. The scored sets "
                        "differ slightly between entries (each DISCELL seed "
                        "has its own split).")
    feas_rows = [(m["label"], ds, m.get("row")) for ds in WHOLE_SECTION
                 for m in methods[ds] if m.get("tool")]
    tr.notes.append("Whole-section attempts (DisCell-baselines/results/"
                    "feasibility.tsv, else the baselines_complete queue log): "
                    + "; ".join(f"{lab} {SHORT[ds]}: "
                                + next(m.get("state", "scored")
                                       if "battery" not in m else
                                       f"scored as '{m['battery']}'"
                                       + (f" (feasibility: {r['outcome']}, "
                                          f"{r['wall_h']} h)" if r else "")
                                       for m in methods[ds]
                                       if m["label"] == lab)
                                for lab, ds, r in feas_rows) + ".")
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
        "MintFlow transfer = true, fitted_on = the core export. The Cellina "
        "niche-domain and own-graph fits (cellina_extra_2026-09-29) are "
        "Cellina runs with the same data and split rule; on the serial "
        "section they are the core's fits transferred.")
    # the whole-section fits' state, from feasibility.tsv and the queue log
    whole = {ds: methods[ds] for ds in WHOLE_SECTION}
    done = [f"{m['label']} on {INLINE[ds]}" for ds, ms in whole.items()
            for m in ms if m.get("tool") and "battery" in m]
    still = _state_phrase(_pending_items(whole))
    foot_a = ("$^{a}$Whole-section fits of SIMVI and MintFlow: "
              + (f"{' and '.join(done)} {'is' if len(done) == 1 else 'are'} "
                 "done and reported; " if done else "")
              + still + "; a method that cannot be run on the resources "
              "available will be reported as such, with the measured "
              "reason. " if any_pending else "")
    foot_e = (f"$^{{e}}$Measured reason: {'; '.join(notrun)}. "
              if notrun else "")
    foot_f = ("$^{f}$Stopped by its time cap before 50 epochs. "
              if any_trunc else "")
    caption = (
        "One battery for every method, at lineage labels, on the held-out "
        "cells of each section, with Unassigned cells not targets. "
        "NMI of the intrinsic latent with the type; \\emph{Ridge}, "
        "\\emph{MLP}: the held-out excess of the ridge and MLP probes of "
        "neighbour "
        "composition as a fraction of that of DISCELL without the adversary "
        "(the fraction left of \\cref{tab:probe}); mirror $R^2$; cycle $R^2$ "
        "of the intrinsic latent on the top-decile cycling set; held-out "
        "reconstruction in nats per count. For DISCELL, the mean over "
        f"three seeds and the range. {ARROW_NOTE}; {BOLD_NOTE}, for DISCELL "
        "by its mean. DISCELL's held-out tiles are excluded "
        "from its training loss. The comparison methods are fitted as their software is "
        "designed, on every cell of the section with at least five counts, "
        "held-out tiles included, and are scored on our held-out cells: "
        "resolVI and MintFlow train on all of them, and Cellina and SIMVI "
        "on a random nine tenths of the cells (the rest serve their own "
        "early stopping), so their reads, reconstruction above all, are "
        "not held-out reads. On the serial section, resolVI and SIMVI are "
        "fitted on that section itself, so theirs are not held-out reads "
        "either; the Cellina rows, MintFlow and DISCELL are the core's "
        "models transferred. \\emph{Cellina, own graph}: Cellina on its own "
        "neighbour graph instead of ours. DISCELL's rows are the battery's "
        "own reads, on the cell set shared with the comparison methods, and "
        "so differ slightly from \\cref{tab:headline}. "
        + foot_a +
        "$^{b}$Not reported: MintFlow's reconstruction is under "
        "inspection. $^{c}$SIMVI has no count decoder. $^{d}$Cellina, niche domain: Cellina with "
        "its domain adversary given our niche label (clusters of neighbour "
        "composition, fitted on training cells) in place of the tissue "
        "regions of its own paper, which our sections do not have. The "
        "label matches what the composition probe grades, so this row is "
        "Cellina's best case on the probe, not its published setting. "
        + foot_e + foot_f + "--: not applicable.")
    tex = (tr.header(command, "lineage labels; accepted checkpoint; one "
                     "Unassigned mask for every method; DisCell mean and "
                     "[min, max] over finalL_s0-s2, baselines single fit; "
                     "cycle on the label-free top-decile set; probe fractions "
                     "against the seed-mean excess of uncontrolledL_s0-s1")
           + "\\begin{table*}[t]\n\\centering\n"
           f"\\caption{{{caption}}}\n\\label{{tab:battery}}\n\\small\n"
           "\\setlength{\\tabcolsep}{2pt}\n"
           # rows a little tighter: the table and its caption fill a page
           "\\renewcommand{\\arraystretch}{0.95}\n"
           "\\begin{tabular}{@{}llcccccc@{}}\n\\toprule\n"
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
        return {"depth": "per cell, depth ratio",
                "gene": "per gene",
                "density": "per cell, density ratio"}[arm]
    if family == "fp_floor":
        return ("area-scaled, per cell" if "area" in flag
                else "one per section")
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
        return probe_share(tr.read_json(
            DATA / ds / "runs" / run / "validation"
            / "probe_blocks.json")["mlp"]["comp"]["excess"])

    def transport(ds: str, run: str, tier: str = "extrapolation_trusted"):
        """Fraction of ceiling of the published transport read, trusted tier
        by default (None when the fit has no trusted panel)."""
        q = DATA / ds / "runs" / run / "transport" / "transport.json"
        t = tr.read_json(q)
        if MASK not in ((t.get("eval_mask") or {}).get("excluded_types")
                        or []):
            raise SystemExit(f"{rel(q)}: not read under the Unassigned mask")
        v = (t["summary"].get(tier) or {}).get("counterfactual_of_ceiling")
        return v if finite(v) else None

    # where a family's records carry a fraction of ceiling, it must equal the
    # per-run transport file at the tier it names: kappa_form's is the
    # trusted tier, alpha_w's 'readA_of_ceiling' is the ALL-PANEL read
    T_CHECK = {"kappa_form": ("read_a_of_ceiling_trusted",
                              "extrapolation_trusted"),
               "alpha_w": ("readA_of_ceiling", "extrapolation")}
    t_checked = []
    few_t: list[str] = []

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
    sens_dir = {"Recon.": "recon", "NMI": "nmi", "Mirror $R^2$": "mirror",
                "$\\I(\\text{niche}; \\vw)$": "mi"}
    head = ("Arm & Seeds & " + " & ".join(h + arrow(sens_dir[h])
                                          for h, *_ in SENS_READS)
            + f" & MLP (\\%){arrow('probe')} & Transport{arrow('transport')}"
            " \\\\")
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
                             pct(c_mean), [c_mean], 1, src,
                             "mlp.comp.excess: 1 - exp(-2 excess), x100 mean"))
        rng.append(tr.cell(f"{SHORT[ds]} control range", "MLP comp.",
                           pct_span(cv), [min(cv), max(cv)], 1, src,
                           "mlp.comp.excess: 1 - exp(-2 excess), x100 min, max"))
        ct = [transport(ds, run) for run in ctrl["names"]]
        ctv = [v for v in ct if v is not None]
        tsrc = DATA / ds / "runs" / "<run>" / "transport" / "transport.json"
        t_mean = float(np.mean(ctv))
        t_sd = float(np.std(ctv, ddof=1)) if len(ctv) > 1 else float("nan")
        tmark = ""
        if len(ctv) < len(ct):
            tmark = "$^{\\dagger}$"
            few_t.append(f"{SHORT[ds]} final configuration: {len(ctv)} of "
                         f"{len(ct)}")
        cells.append(tr.cell(f"{SHORT[ds]} control", "Transport",
                             num(t_mean, 2) + tmark, [t_mean], 2, tsrc,
                             "summary.extrapolation_trusted."
                             "counterfactual_of_ceiling mean"))
        rng.append(tr.cell(f"{SHORT[ds]} control range", "Transport",
                           vrange(ctv, 2), [min(ctv), max(ctv)], 2, tsrc,
                           "summary.extrapolation_trusted."
                           "counterfactual_of_ceiling min, max"))
        tr.prov.append(f"{SHORT[ds]} control: {rel(fam[f0][0])} : sections."
                       f"{ds}.control.records[*] (mean; min, max), checked "
                       "equal in " + ", ".join(f for f, _ in blocks)
                       + f"; MLP comp.: data/datasets/{ds}/runs/finalL_s"
                       "{0,1,2}/validation/probe_blocks.json : "
                       "mlp.comp.excess as 1 - exp(-2 excess); Transport: "
                       "data/datasets/"
                       f"{ds}/runs/finalL_s{{0,1,2}}/transport/transport.json"
                       " : summary.extrapolation_trusted."
                       "counterfactual_of_ceiling")
        body.append(f"\\multicolumn{{{len(SENS_READS) + 4}}}{{@{{}}l}}"
                    f"{{\\emph{{{LONG[ds]}}}}} \\\\")
        body.append(f"\\quad final configuration & {len(ctrl['records'])} & "
                    + " & ".join(cells) + " \\\\")
        body.append("\\quad\\quad range & & " + " & ".join(rng)
                    + " \\\\")
        for f, d in blocks:
            sec = d["sections"][ds]
            body.append(f"\\multicolumn{{{len(SENS_READS) + 4}}}{{@{{}}l}}"
                        f"{{\\quad \\emph{{{SENS_TITLE[f]}}}}} \\\\")
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
                                     [a_mean, move], [1, 1], src,
                                     "mlp.comp.excess: 1 - exp(-2 excess), "
                                     "x100 mean; move "
                                     "in control sd (ddof 1)"))
                at = [transport(ds, run) for run in a["names"]]
                chk, tier = T_CHECK.get(f, (None, None))
                if chk:
                    for rec in a["records"]:
                        v = transport(ds, rec["run"], tier)
                        w = rec.get(chk)
                        if (v is None) != (not finite(w)) or (
                                v is not None and abs(v - w) > 1e-9):
                            raise SystemExit(f"{ds}/{rec['run']}: {chk} "
                                             f"{w} differs from transport.json "
                                             f"{v}")
                        t_checked.append(rec["run"])
                atv = [v for v in at if v is not None]
                where = f"{SHORT[ds]} / {f} / {arm} / Transport"
                if not atv:
                    cells.append(tr.pend(where, "no seed has a trusted panel",
                                         "no panel"))
                else:
                    am = float(np.mean(atv))
                    mark = ""
                    if len(atv) < len(at):
                        mark = "$^{\\dagger}$"
                        few_t.append(f"{SHORT[ds]} {label}: {len(atv)} of "
                                     f"{len(at)}")
                    tmove = (am - t_mean) / t_sd
                    cells.append(tr.cell(f"{SHORT[ds]} {f} {arm}", "Transport",
                                         f"{num(am, 2)} ({num(tmove, 1)})"
                                         + mark, [am, tmove], [2, 1], tsrc,
                                         "summary.extrapolation_trusted."
                                         "counterfactual_of_ceiling mean; move "
                                         "in control sd (ddof 1)"))
                tr.prov.append(f"{SHORT[ds]} {f} {arm} ({flag}): "
                               f"{rel(fam[f][0])} : sections.{ds}.arms.{arm}"
                               f".records[*] (mean), sections.{ds}.moves."
                               f"{arm}.<key>.in_sd; MLP comp.: data/datasets/"
                               f"{ds}/runs/{{{','.join(a['names'])}}}/"
                               "validation/probe_blocks.json : "
                               "mlp.comp.excess as 1 - exp(-2 excess)")
                body.append(f"\\quad\\quad {label} & {len(a['records'])} & "
                            + " & ".join(cells) + " \\\\")
        body.append("\\addlinespace")
    body = body[:-1]
    tr.notes.append("Transport (released 2026-09-30, devlog 'Transport rule "
                    "triggered; the published read stays the headline'): the "
                    "published read's trusted-tier fraction of ceiling, read "
                    "per run from runs/<run>/transport/transport.json (masked, "
                    "checked), because the fp_floor and adversary files carry "
                    "no point value. Equal to the sensitivity file's own record "
                    f"on the {len(t_checked)} arm runs where one exists "
                    "(kappa_form read_a_of_ceiling_trusted against the trusted "
                    "tier; alpha_w readA_of_ceiling against the ALL-PANEL tier "
                    "summary.extrapolation, which is what that key holds, "
                    "despite manifest row 21). Fewer seeds with a trusted panel: "
                    + ("; ".join(few_t) if few_t else "none") + ".")
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
        "in nats. \\emph{MLP}: the residual neighbour composition "
        "in the intrinsic latent, as the share of the block's within-type "
        "variance that the nonlinear probe explains beyond the permutation "
        "floor, $1 - e^{-2\\,\\mathrm{excess}}$ (as in \\cref{tab:probe}), "
        "in \\%. \\emph{Transport}: the "
        "fraction of the noise ceiling recovered by the mean transport read, "
        "trusted tier (as in \\cref{tab:headline})"
        + ("; $^{\\dagger}$over the seeds with a trusted panel" if few_t
           else "") + ". The leakage "
        "rate is set per cell from its neighbours' depth or density "
        "relative to its own, or per gene, at the same mean rate; the "
        "false-positive floor is fixed once per section, or per cell scaled "
        f"by its area. {ARROW_NOTE}, so a move in the direction of a column's arrow "
        "is an improvement. Unassigned cells are not targets.")
    tex = (tr.header(command, "control: mean (min-max) over finalL_s0-s2; "
                     "arm: mean over its seeds (2 or 3) and, in brackets, "
                     "(arm mean - control mean) / control sd (ddof 1) as "
                     "stored by the sensitivity tables; accepted checkpoint; "
                     "Unassigned excluded as a target; MLP comp. = "
                     "100 (1 - exp(-2 mlp.comp.excess)) from each run's own "
                     "validation/probe_blocks.json, its move computed here "
                     "with the same rule (control sd, ddof 1)")
           + "\\begin{table}[t]\n\\centering\n"
           f"\\caption{{{caption}}}\n\\label{{tab:sensitivity}}\n\\footnotesize\n"
           "\\setlength{\\tabcolsep}{2pt}\n"
           "\\begin{tabular}{@{}lccccccc@{}}\n\\toprule\n"
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
    sweep_dir = {"recon": "recon", "nmi": "nmi", "mirror_r2": "mirror",
                 "cycle_r2_z_q90": "cycle_z", "cycle_r2_w_q90": "cycle_w"}
    head = ("Section & $\\kappa$ & " + " & ".join(h + arrow(sweep_dir[k])
                                                for h, k, _ in SWEEP_READS)
            + f" & $\\I(\\text{{niche}}; \\vw)${arrow('mi')} \\\\")
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
        f"permutation floor, in nats. {ARROW_NOTE}; {ZERO_NOTE}. Unassigned "
        "cells are not targets.")
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


# ------------------------------------------------------------ table 6: timing

TIMING_COLS = [OVARIAN, LUNG, FF, GSE]
TIMING_HEAD = {OVARIAN: "Ovarian (FFPE)", LUNG: "Lung (FFPE)",
               FF: "Ovarian (FF)", GSE: "Lung TMA core"}


def _timing_md_numbers(text: str) -> dict:
    """The phi arm's s/epoch, parameters and peak train MiB, and the finalL
    wall-clock minutes, as printed in a timing_lineage.md."""
    out = {}
    m = re.search(r"^\| phi \| \d+ \| ([0-9.]+) \| [0-9.]+ \| (\d+) \| \d+ \| "
                  r"\d+ \| ([0-9,]+) \|", text, re.M)
    if m:
        out["s_per_epoch"] = float(m.group(1))
        out["peak_train_mib"] = int(m.group(2))
        out["n_params"] = int(m.group(3).replace(",", ""))
    out["minutes"] = {int(s): float(v) for s, v in
                      re.findall(r"finalL_s(\d) ([0-9.]+) min", text)}
    return out


def table_timing(command: str) -> tuple[str, Trace]:
    tr = Trace("tab:timing")
    rows = {k: [] for k in ("params", "peak", "epochs", "spe", "discell",
                            "resolVI", "Cellina", "SIMVI", "MintFlow")}
    notrun: list[str] = []
    pending_n = 0
    trunc_note: list[str] = []
    # run time of each fit per column (None: no successful whole fit), for
    # the bold on the fastest
    tval = {k: [] for k in ("discell", "resolVI", "Cellina", "SIMVI",
                            "MintFlow")}
    for ds in TIMING_COLS:
        tj = DATA / ds / "experiments" / "timing_lineage" / "timing_phi.json"
        t = tr.read_json(tj)
        md = tr.read_text(DATA / ds / "experiments" / "timing_lineage.md")
        printed = _timing_md_numbers(md)
        if (abs(printed["s_per_epoch"] - t["s_per_epoch"]) > 0.005
                or printed["n_params"] != t["n_params"]
                or printed["peak_train_mib"] != round(t["peak_train_mib"])):
            raise SystemExit(f"{ds}: timing_phi.json disagrees with "
                             "timing_lineage.md")
        rows["params"].append(tr.cell("parameters", SHORT[ds],
                                      rnd(t["n_params"] / 1e6, 2),
                                      [t["n_params"] / 1e6], 2, tj,
                                      "n_params / 1e6"))
        gib = t["peak_train_mib"] / 1024
        rows["peak"].append(tr.cell("peak memory", SHORT[ds], rnd(gib, 1),
                                    [gib], 1, tj, "peak_train_mib / 1024"))
        rows["spe"].append(tr.cell("s per epoch", SHORT[ds],
                                   rnd(t["s_per_epoch"], 2),
                                   [t["s_per_epoch"]], 2, tj, "s_per_epoch"))
        eps, mins = [], []
        for s in SEEDS:
            mp = DATA / ds / "runs" / FINAL.format(s=s) / "metrics.json"
            m = tr.read_json(mp)
            eps.append(m["last_epoch"] + 1)
            mins.append(m["minutes"])
            if abs(printed["minutes"][s] - m["minutes"]) > 0.05 + 1e-9:
                raise SystemExit(f"{rel(mp)}: minutes disagree with the md")
        mp = DATA / ds / "runs" / "finalL_s<s>" / "metrics.json"
        ep_txt = (str(min(eps)) if min(eps) == max(eps)
                  else f"{min(eps)}--{max(eps)}")
        rows["epochs"].append(tr.cell("epochs", SHORT[ds], ep_txt,
                                      [min(eps), max(eps)], 0, mp,
                                      "last_epoch + 1, min-max over s0-s2"))
        rows["discell"].append(tr.cell("DISCELL", SHORT[ds], vrange(mins, 1),
                                       [min(mins), max(mins)], 1, mp,
                                       "minutes, min-max over s0-s2"))
        tval["discell"].append(float(np.mean(mins)))
        tr.prov.append(f"DISCELL / {SHORT[ds]}: {rel(tj)} : n_params, "
                       "peak_train_mib, s_per_epoch (checked against "
                       f"timing_lineage.md); runs/finalL_s{{0,1,2}}/"
                       "metrics.json : minutes, last_epoch + 1")
        # comparison methods: the config.json of the fit compared in the paper
        cfgs = {"resolVI": BRES / "resolvi_lineage" / ds / "config.json",
                "Cellina": BRES / "cellina" / f"{ds}_lineage" / "config.json"}
        for meth in ("resolVI", "Cellina"):
            c = tr.read_json(cfgs[meth])
            rows[meth].append(tr.cell(meth, SHORT[ds],
                                      rnd(c["train_s"] / 60, 1),
                                      [c["train_s"] / 60], 1, cfgs[meth],
                                      "train_s / 60"))
            tval[meth].append(c["train_s"] / 60)
            tr.prov.append(f"{meth} / {SHORT[ds]}: {rel(cfgs[meth])} : "
                           "train_s / 60")
        for tool in ("SIMVI", "MintFlow"):
            cp = BRES / tool.lower() / f"{ds}_lineage" / "config.json"
            where = f"{tool} / {SHORT[ds]}"
            if ds == GSE:
                state, row = "ok", None
            else:
                state, row = baseline_state(tr, tool, ds)
            if state in RUNNABLE and cp.exists():
                c = tr.read_json(cp)
                if tool == "MintFlow" and c.get("epochs_completed") != 50:
                    trunc_note.append(f"{SHORT[ds]}: {c.get('epochs_completed')}"
                                      " epochs")
                mark = ("$^{b}$" if tool == "MintFlow"
                        and c.get("epochs_completed") != 50 else "")
                rows[tool].append(tr.cell(tool, SHORT[ds],
                                          rnd(c["train_s"] / 60, 1) + mark,
                                          [c["train_s"] / 60], 1, cp,
                                          "train_s / 60"))
                # a fit stopped by its time cap is not a finished fit
                tval[tool].append(None if mark else c["train_s"] / 60)
                tr.prov.append(f"{where}: {rel(cp)} : train_s / 60"
                               + (f"; feasibility.tsv: outcome {row['outcome']}"
                                  f", wall {row['wall_h']} h, peak GPU "
                                  f"{row['peak_gpu_gb']} GB, host "
                                  f"{row['peak_host_gb']} GB ({row['reason']})"
                                  if row else ""))
            elif state in ("running", "queued") or state in RUNNABLE:
                pending_n += 1
                why = {"running": "whole-section fit running",
                       "queued": "whole-section fit queued"}.get(
                    state, "fit done, config not yet written")
                rows[tool].append("{\\scriptsize " + tr.pend(
                    where, why + " (queue baselines_complete_2026-09-28)",
                    state if state in ("running", "queued") else "scoring")
                    + "}")
                tval[tool].append(None)
            else:
                notrun.append(f"{tool}, {INLINE[ds]}: "
                              + tex_escape(row["reason"]))
                rows[tool].append(f"not run$^{{a}}$")
                tval[tool].append(None)
                tr.prov.append(f"{where}: feasibility.tsv {row}")
    # bold: the fastest successful fit per section, DISCELL by its mean
    for i, ds in enumerate(TIMING_COLS):
        order = list(tval)
        vals = [tval[k][i] for k in order]
        for k in best(vals, "time", [rnd(v, 1) if finite(v) else ""
                                     for v in vals]):
            rows[order[k]][i] = bold(rows[order[k]][i])
            tr.notes.append(f"{SHORT[ds]}: fastest fit {order[k]} (bold).")
    L = ["\\multicolumn{5}{@{}l}{\\emph{DISCELL, model and training "
         "details}} \\\\",
         "\\quad parameters (M) & " + " & ".join(rows["params"]) + " \\\\",
         "\\quad peak memory (GiB) & " + " & ".join(rows["peak"]) + " \\\\",
         "\\quad epochs to the stop & " + " & ".join(rows["epochs"]) + " \\\\",
         f"\\quad s per epoch, training only{arrow('time')} & "
         + " & ".join(rows["spe"])
         + " \\\\", "\\addlinespace",
         "\\multicolumn{5}{@{}l}{\\emph{Run time of one fit (min)}"
         f"{arrow('time')}}} \\\\",
         "\\quad DISCELL & " + " & ".join(rows["discell"]) + " \\\\"]
    for meth, lab in (("resolVI", "resolVI"), ("Cellina", "Cellina"),
                      ("SIMVI", "SIMVI"),
                      ("MintFlow", "MintFlow, 50 epochs")):
        L.append(f"\\quad {lab} & " + " & ".join(span_pending(rows[meth]))
                 + " \\\\")
    tr.notes.append("DISCELL times per epoch are the timing arm 'phi' (20 "
                    "epochs, no evaluation, first epoch excluded; the final "
                    "configuration of finalL_s0). The comparison methods' "
                    "times are train_s of the fit compared in the paper "
                    "(lineage labels): resolvi_lineage/<ds>, cellina/"
                    "<ds>_lineage, simvi/<ds>_lineage, mintflow/<ds>_lineage. "
                    "For the whole-section MintFlow fit on lung FFPE, "
                    "feasibility.tsv's wall_h includes setup and prediction; "
                    "the table shows train_s like every other method.")
    foot = []
    if notrun:
        foot.append(f"$^{{a}}${NOT_RUN[0].upper() + NOT_RUN[1:]}; measured "
                    f"reason: {'; '.join(notrun)}.")
    if trunc_note:
        foot.append("$^{b}$Stopped by its time cap before 50 epochs ("
                    + "; ".join(trunc_note) + ").")
    caption = (
        "Model size and training cost on one GPU. \\emph{Peak memory}: "
        "largest allocation during training, with the section resident on "
        "the GPU. \\emph{Run time}: one whole fit, for DISCELL including its "
        "evaluations every five epochs and the early stop, as the range over "
        "the three seeds of the final configuration; for the comparison "
        "methods, the training time of the fit compared in this paper, at "
        "lineage labels. Every method is fitted on whole sections only. "
        f"{ARROW_NOTE}; {BOLD_NOTE}: the fastest finished fit, for DISCELL "
        "by its mean."
        + (" Whole-section fits of SIMVI and MintFlow still queued or "
           "running are marked as pending." if pending_n else "")
        + (" " + " ".join(foot) if foot else ""))
    tex = (tr.header(command, "DISCELL: the 20-epoch timing arm (s/epoch, "
                     "peak, parameters) and the three final fits' wall clock "
                     "(min-max); baselines: train_s of the lineage fit; "
                     "whole-section attempts from feasibility.tsv, else the "
                     "baselines_complete queue log")
           + "\\begin{table}[t]\n\\centering\n"
           f"\\caption{{{caption}}}\n\\label{{tab:timing}}\n\\footnotesize\n"
           "\\setlength{\\tabcolsep}{2.5pt}\n"
           "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n"
           "& " + " & ".join(TIMING_HEAD[d] for d in TIMING_COLS)
           + " \\\\\n\\midrule\n" + "\n".join(L)
           + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return tex, tr


# --------------------------------------------------------- table 7: breakdown

BD_SECTIONS = [("ovarian", OVARIAN), ("lung", LUNG), ("ff", FF),
               ("gse", GSE), ("gse_dual", DUAL)]
BD_MEMBERS = [
    ("cycle_asym_q90", "Cycle asymmetry, $R^2(\\vz) - R^2(\\vw)$"),
    ("w_niche_mi_excess", "$\\I(\\text{niche}; \\vw)$ $-$ its floor"),
    ("readA_minus_typemean", "Read A $-$ type-mean reference"),
    ("readB_twin_margin", "Twin margin"),
    ("transport_cf_minus_program", "Transport $-$ programme part"),
    ("transport_cf_minus_leak", "Transport $-$ leakage part"),
    ("signalling_response_lr_vs_other", "Signalling share, LR $-$ other"),
    ("marker_excl_minus_ctrl_dc", "Marker pairs, exclusive $-$ control"),
    ("axis_tau_true_minus_false", "Tumour axis, true $-$ false"),
]
BD_TRANSPORT = {"readA_minus_typemean", "readB_twin_margin",
                "transport_cf_minus_program", "transport_cf_minus_leak"}


def _bd_family(section: str, m_s: int) -> set[str]:
    """The members of a section's family (devlog, lean 8.19 entry)."""
    if section == "gse_dual":
        return {"cycle_asym_q90"}
    fam = {k for k, _ in BD_MEMBERS} - {"axis_tau_true_minus_false"}
    if section == "ovarian":
        fam |= {"axis_tau_true_minus_false"}
    if len(fam) != m_s:
        raise SystemExit(f"breakdown {section}: family of {len(fam)} "
                         f"members, file says m = {m_s}")
    return fam


def table_breakdown(command: str) -> tuple[str, Trace]:
    tr = Trace("tab:breakdown")
    p = BREAKDOWN / "breakdown_all.json"
    d = tr.read_json(p)
    md = tr.read_text(BREAKDOWN / "breakdown_all.md")
    # the md's cells, for a check of the statuses
    md_cells = {}
    head = None
    for ln in md.splitlines():
        if ln.startswith("| member"):
            head = [c.strip() for c in ln.strip("|").split("|")]
        elif ln.startswith("| ") and head and not ln.startswith("|---"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            md_cells[cells[0]] = dict(zip(head[1:], cells[1:]))
    foot = {"interval contains 0": "a", "a seed flips sign": "b",
            "seeds disagree in sign": "b"}
    used_foot: set[str] = set()
    neg = False
    body = []
    header = ["Contrast"]
    for sec, ds in BD_SECTIONS:
        header.append(f"\\shortstack{{{SHORT[ds]}\\\\($m = {d[sec]['m_s']}$)}}")
    for key, label in BD_MEMBERS:
        cells = []
        for sec, ds in BD_SECTIONS:
            s = d[sec]
            fam = _bd_family(sec, s["m_s"])
            where = f"{label} / {SHORT[ds]}"
            if key not in fam:
                cells.append("--")
                continue
            e = s["members"].get(key)
            if e is None:
                why = ("transport member being computed on the published "
                       "read (breakdown queue relaunched with "
                       "GATE_OVERRIDE=1, devlog 2026-09-30)"
                       if key in BD_TRANSPORT else "member missing from the "
                       "file")
                cells.append(tr.pend(where, why, "computing"))
                continue
            st = e["status"]
            sign = "$^{(-)}$" if e.get("sign") == -1 else ""
            neg |= bool(sign)
            if st == "above the grid":
                txt = "above the grid"
                vals, nd = [], 0
            elif st == "breaks":
                f = foot.get(e.get("reason"), "c")
                used_foot.add(f)
                txt = f"{e['kappa_star']:g}$^{{{f}}}$"
                vals, nd = [], 0
            elif st == "no finding":
                f = foot.get(e.get("reason"), "c")
                used_foot.add(f)
                txt = f"no finding$^{{{f}}}$"
                vals, nd = [], 0
            else:
                cells.append(tr.pend(where, f"status '{st}'", "gap"))
                continue
            txt = txt + sign
            # the md prints the same status
            mdv = md_cells.get(key, {}).get(
                next((h for h in md_cells.get(key, {}) if h.startswith(sec + " ")
                      or h == sec or h.startswith(sec + " (")), ""), None)
            if mdv is not None and st == "breaks" and not mdv.startswith(
                    f"{e['kappa_star']:g}"):
                raise SystemExit(f"breakdown md/json disagree: {key} {sec}")
            if mdv is not None and st != "breaks" and not mdv.startswith(st):
                raise SystemExit(f"breakdown md/json disagree: {key} {sec}")
            cells.append(tr.cell(label, SHORT[ds], txt, vals, nd, p,
                                 f"{sec}.members.{key}.status/kappa_star"))
            tr.prov.append(f"{where}: {rel(p)} : {sec}.members.{key} "
                           f"(status '{st}', kappa_star {e.get('kappa_star')},"
                           f" sign {e.get('sign')})")
        body.append(f"{label} & " + " & ".join(span_pending(cells))
                    + " \\\\")
    tr.notes.append("Statuses checked equal to the rendered "
                    f"{rel(BREAKDOWN / 'breakdown_all.md')}. family_complete: "
                    + ", ".join(f"{sec} {d[sec]['family_complete']}"
                                for sec, _ in BD_SECTIONS) + ".")
    tr.notes.append("Written after the freeze by the breakdown queue "
                    "(scripts/logs/breakdown_2026-09-29/queue.log); the "
                    "transport members fill in when the queue's final tables "
                    "step rewrites breakdown_all.json; regenerate then.")
    fl = []
    if "a" in used_foot:
        fl.append("$^{a}$the interval contains $0$")
    if "b" in used_foot:
        fl.append("$^{b}$a seed has the opposite sign")
    if "c" in used_foot:
        fl.append("$^{c}$see the section's record")
    grid = d["ovarian"]["grid"]
    caption = (
        "Breakdown points $\\kappa^*$ of the contrasts the results claim, per "
        "section, on the leakage sweep ($\\kappa \\in \\{"
        + ", ".join(f"{g:g}" for g in grid) + "\\}$, three seeds each). A "
        "contrast holds at a grid point if every seed has the sign it has "
        "at $\\kappa = 0$ and the interval of the mean over seeds (spatial "
        "block bootstrap over tiles, two-sided, Bonferroni-adjusted over the "
        "$m$ contrasts of the section) excludes $0$. $\\kappa^*$ is the first "
        "grid point where it fails; \\emph{above the grid}: it holds at every "
        "grid point; \\emph{no finding}: it fails already at $\\kappa = 0$"
        + ("; " + ", ".join(fl) if fl else "")
        + (". $^{(-)}$The contrast is negative at $\\kappa = 0$" if neg else "")
        + ". Signalling share: the response's share of ligand-receptor (LR) "
        "genes against other genes; tumour axis: Kendall's $\\tau$ of the "
        "response-predicted shift with the true against a false axis. "
        "On the serial section only the cycle asymmetry is read, through "
        "the core's fits. The tumour axis exists on the ovarian FFPE "
        "section only. --: not in that section's set.")
    tex = (tr.header(command, "kappa* per member and section from the lean "
                     "8.19 table; Bonferroni within each section (m_s); "
                     "status and kappa* as stored")
           + "\\begin{table*}[t]\n\\centering\n"
           f"\\caption{{{caption}}}\n\\label{{tab:breakdown}}\n\\footnotesize\n"
           "\\setlength{\\tabcolsep}{2pt}\n"
           "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
           + " & ".join(header) + " \\\\\n\\midrule\n" + "\n".join(body)
           + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    return tex, tr


TRAJ_READS = [
    ("transport_of_ceiling_trusted", "Transport, fraction of ceiling", 2),
    ("w_within_type_var_fraction", "Within-type share of $\\vw$ variance", 2),
    ("moran_mu_z", "Moran's $I$ of $\\vmu_z$", 3),
    ("moran_mu_w", "Moran's $I$ of $\\vmu_w$", 3),
    ("atlas_effective_rank", "Atlas effective rank", 1),  # range: integers
    ("kappa_survival_overlap", "Programme overlap with the reference", 2),
]


def table_breakdown_traj(command: str) -> tuple[str, Trace]:
    tr = Trace("tab:breakdown-traj")
    p = BREAKDOWN / "breakdown_all_trajectory.json"
    d = tr.read_json(p)
    grid = None
    body = []
    few = []
    for sec, ds in BD_SECTIONS:
        if sec not in d or not d[sec]["readouts"]:
            continue
        s = d[sec]
        grid = grid or s["grid"]
        if s["grid"] != grid:
            raise SystemExit("trajectory grids differ between sections")
        body.append(f"\\multicolumn{{{len(grid) + 1}}}{{@{{}}l}}"
                    f"{{\\emph{{{LONG[ds]}}}}} \\\\")
        for key, label, nd in TRAJ_READS:
            if s["sources"].get(key) is None:
                raise SystemExit(f"{sec}: no source recorded for {key}")
            mean_c, rng_c = [], []
            for g in grid:
                e = s["readouts"][key].get(f"{g:g}")
                where = f"{SHORT[ds]} / {label} / kappa {g:g}"
                if e is None or not finite(e.get("mean")):
                    mean_c.append(tr.pend(where, "no read", "missing"))
                    rng_c.append("")
                    continue
                mark = ""
                if e["n"] < 3:
                    mark = "$^{\\dagger}$"
                    few.append(f"{SHORT[ds]} {key} kappa {g:g}: n = {e['n']}")
                mean_c.append(tr.cell(f"{SHORT[ds]} {key}", f"{g:g}",
                                      num(e["mean"], nd) + mark, [e["mean"]],
                                      nd, p, f"{sec}.readouts.{key}.{g:g}.mean"))
                lo, hi, rd = e["min"], e["max"], nd
                if key == "atlas_effective_rank":   # counts of axes
                    if lo != int(lo) or hi != int(hi):
                        raise SystemExit(f"{sec}: non-integer atlas rank")
                    lo, hi, rd = int(lo), int(hi), 0
                rng_c.append(tr.cell(f"{SHORT[ds]} {key} range", f"{g:g}",
                                     (str(lo) if lo == hi else f"{lo}--{hi}")
                                     if rd == 0 else vrange([lo, hi], rd),
                                     [lo, hi], rd, p,
                                     f"{sec}.readouts.{key}.{g:g}.min, max"))
            mk = (arrow("transport")
                  if key == "transport_of_ceiling_trusted" else "")
            body.append(f"\\quad {label}{mk} & " + " & ".join(mean_c)
                        + " \\\\")
            body.append("\\quad\\quad range & " + " & ".join(rng_c) + " \\\\")
            tr.prov.append(f"{SHORT[ds]} / {label}: {rel(p)} : {sec}."
                           f"readouts.{key} (mean; min, max per kappa); "
                           f"underlying read: {s['sources'][key]}")
        body.append("\\addlinespace")
    body = body[:-1]
    head = ("Readout & " + " & ".join(
        f"$\\kappa = {g:g}$" if g != 0.1 else "$\\bm{\\kappa = 0.1}$"
        for g in grid) + " \\\\")
    tr.notes.append("Left out: the all-panel transport fraction (off the "
                    "tables, manifest), and w_within_type_var (unnormalised; "
                    "its share is shown). The serial section has no "
                    "trajectory reads." + (" Fewer seeds: " + "; ".join(few)
                                          if few else ""))
    caption = (
        "Readouts reported as trajectories across the leakage sweep, without "
        "a breakdown point: the mean over three seeds with the range. These "
        "are magnitudes, positive under any fit, or descriptions of what the "
        "invariance leaves, so no null applies. \\emph{Transport}: the "
        "fraction of the noise ceiling recovered by the mean transport read, "
        "trusted tier" + ("; $^{\\dagger}$over the seeds with a trusted panel"
                          if few else "")
        + ". \\emph{Within-type share of $\\vw$ variance}: the share of "
        "$\\vw$'s total variance that lies within types. Moran's $I$ is "
        "within type, variance-weighted over coordinates. \\emph{Atlas "
        "effective rank}: the number of programme axes kept. \\emph{Programme "
        "overlap}: the overlap of each fit's programme shift space with that "
        "of the first seed at $\\kappa = 0.1$, which is therefore $1$ for that "
        "fit. The operating point $\\kappa = 0.1$ is in bold. The arrow gives "
        "the preferred direction; the other readouts have none.")
    tex = (tr.header(command, "3-seed mean and [min, max] per kappa as "
                     "stored in the trajectory table")
           + "\\begin{table}[t]\n\\centering\n"
           f"\\caption{{{caption}}}\n\\label{{tab:breakdown-traj}}\n"
           "\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
           f"\\begin{{tabular}}{{@{{}}l{'c' * len(grid)}@{{}}}}\n\\toprule\n"
           f"{head}\n\\midrule\n" + "\n".join(body)
           + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return tex, tr


# ------------------------------------------------ table 8: transport held-out

def _heldout_records(tr: Trace) -> tuple[Path, list[dict]]:
    p = HELDOUT.with_suffix(".json")
    recs = tr.read_json(p)
    for r in recs:
        for side in ("published", "heldout"):
            em = r[side]
            if side == "heldout" and (em["scored_cells"] or {}).get(
                    "scored_cells") != "heldout-tiles":
                raise SystemExit(f"{rel(p)}: {r['dataset']} {r['run']} "
                                 "held-out side not scored on held-out tiles")
    return p, recs


def _md_flags(text: str) -> dict:
    """(tier, dataset label, method, run) -> 'OUTSIDE' | 'inside' | '--'."""
    out, tier = {}, None
    for ln in text.splitlines():
        if ln.startswith("## Fraction of ceiling, trusted"):
            tier = "trusted"
        elif ln.startswith("## Fraction of ceiling, all panels"):
            tier = "all"
        elif ln.startswith("## "):
            tier = None
        elif tier and ln.startswith("| ") and not ln.startswith("| dataset"):
            c = [x.strip() for x in ln.strip().strip("|").split("|")]
            out[(tier, c[0], c[1], c[2])] = c[-1].strip("*")
    return out


HO_LABEL = {GSE: "GSE solo", FF: "ovary FF", OVARIAN: "ovarian FFPE",
            LUNG: "lung FFPE"}


def table_transport_heldout(command: str) -> tuple[str, Trace]:
    tr = Trace("tab:transport-heldout")
    p, recs = _heldout_records(tr)
    flags_md = _md_flags(tr.read_text(HELDOUT.with_suffix(".md")))
    reads = [("trusted", "transport_of_ceiling_trusted", 3),
             ("all", "transport_of_ceiling", 3),
             ("readA", "readA_gap_own", 3),
             ("twin", "twin_margin_own", 3)]

    def val(side: dict, key: str):
        v = side[key]
        v = v.get("value") if isinstance(v, dict) else v
        return v if finite(v) else None

    count = {t: {"up": 0, "down": 0, "inside": 0, "n": 0}
             for t in ("trusted", "all")}
    reada = {"lower": 0, "higher": 0}
    twin_moves = []
    missing = []
    body = []
    for ds in TRAINED:
        mine = [r for r in recs if r["dataset"] == ds]
        for meth in ("DisCell", "Cellina"):
            rs = [r for r in mine if r["method"] == meth]
            if not rs:
                raise SystemExit(f"{rel(p)}: no {meth} record for {ds}")
            cells = []
            for tag, key, nd in reads:
                for side in ("published", "heldout"):
                    vals = [val(r[side], key) for r in rs]
                    have = [v for v in vals if v is not None]
                    where = f"{SHORT[ds]} / {meth} / {key} / {side}"
                    if not have:
                        cells.append("--")
                        missing.append(f"{where}: no value "
                                       + ("(no trusted panel)"
                                          if tag == "trusted" else
                                          "(NaN in the file)"))
                        continue
                    mark = ""
                    if len(have) < len(vals):
                        mark = "$^{\\dagger}$"
                        missing.append(f"{where}: {len(have)} of {len(vals)} "
                                       "seeds have a value")
                    txt = vrange(have, nd) + mark
                    cells.append(tr.cell(f"{SHORT[ds]} {meth}",
                                         f"{key} {side}", txt,
                                         [min(have), max(have)], nd, p,
                                         f"[{meth}, {ds}].{side}.{key}"
                                         + (".value" if tag in ("trusted",
                                                                "all",
                                                                "readA")
                                            else "") + " min, max"))
            # direction of every run, DisCell only for the caption counts
            for r in rs:
                for tag, key, _ in reads[:2]:
                    a, b = r["published"][key], r["heldout"][key]
                    if not (finite(a.get("value")) and finite(b.get("value"))):
                        fl = "--"
                    else:
                        lo, hi = a["ci95"]
                        fl = "inside" if lo <= b["value"] <= hi else "OUTSIDE"
                        if meth == "DisCell":
                            count[tag]["n"] += 1
                            if fl == "inside":
                                count[tag]["inside"] += 1
                            elif b["value"] > hi:
                                count[tag]["up"] += 1
                            else:
                                count[tag]["down"] += 1
                    md = flags_md.get((tag, HO_LABEL[ds], meth, r["run"]))
                    if md is not None and md != fl:
                        raise SystemExit(f"flag disagrees with the md: {ds} "
                                         f"{meth} {r['run']} {tag}: {fl} vs "
                                         f"{md}")
                if meth == "DisCell":
                    a = val(r["published"], "readA_gap_own")
                    b = val(r["heldout"], "readA_gap_own")
                    reada["lower" if b < a else "higher"] += 1
                    twin_moves.append(val(r["heldout"], "twin_margin_own")
                                      - val(r["published"],
                                            "twin_margin_own"))
            label = "DISCELL" if meth == "DisCell" else "Cellina"
            if meth == "DisCell":
                body.append(f"\\multicolumn{{9}}{{@{{}}l}}"
                            f"{{\\emph{{{LONG[ds]}}}}} \\\\")
            body.append(f"\\quad {label} & " + " & ".join(cells) + " \\\\")
            tr.prov.append(f"{SHORT[ds]} / {meth}: {rel(p)} : records with "
                           f"dataset {ds}, method {meth} (runs "
                           f"{', '.join(r['run'] for r in rs)}); published.* "
                           "and heldout.* of transport_of_ceiling_trusted, "
                           "transport_of_ceiling, readA_gap_own (value), "
                           "twin_margin_own")
        body.append("\\addlinespace")
    body = body[:-1]
    tr.notes.append("Flags recomputed here (held-out value outside the "
                    "published half-tile 95 % interval) and checked equal to "
                    f"the 'flag' column of {rel(HELDOUT.with_suffix('.md'))}.")
    tr.notes.append("Values missing or partial: " + "; ".join(missing) + ".")
    tr.notes.append(f"DISCELL counts: trusted {count['trusted']}; all panels "
                    f"{count['all']}; Read A held-out lower on "
                    f"{reada['lower']} of {sum(reada.values())} fits; twin "
                    f"margin moves {min(twin_moves):+.3f} to "
                    f"{max(twin_moves):+.3f}. NOTE: in the trusted tier the "
                    "held-out read leaves the interval downward more often "
                    "than upward; 'more often up' holds for all panels and "
                    "for the two tiers pooled.")
    tr.notes.append("Written after the freeze (queue transport_heldout_"
                    "2026-09-29, finished 2026-09-30 03:43); author's "
                    "decision 2026-09-30: published read = headline, this = "
                    "sensitivity row.")
    ct, ca = count["trusted"], count["all"]
    tw = max(abs(x) for x in twin_moves)
    caption = (
        "Transport scored on held-out tiles only, beside the published read. "
        "\\emph{Published}: the read of \\cref{tab:headline}, whose readout is "
        "cross-fitted over spatial folds drawn from every tile, so most "
        "scored cells lie in the model's training tiles. \\emph{Held-out "
        "tiles}: only cells of the model's held-out tiles are scored, and "
        "every model quantity is estimated from its training tiles; same "
        "niches, panels, ceilings and Unassigned mask. Fraction of the noise "
        "ceiling recovered by the mean read, for panels in the trusted tier "
        "and for all panels; Read A, the median gap to the target closed, "
        "own target; the twin margin, own target. For DISCELL the range over "
        "three seeds; Cellina is its neighbour-rewiring counterfactual, "
        "fitted on the training tiles and read on the same panels as the "
        "first seed. The held-out pool is about three quarters of the "
        "published one, so fewer panels reach the trusted tier. Against the "
        "published $95\\%$ interval, the held-out read moves in both "
        f"directions: in the trusted tier it lies above on {ct['up']}, below "
        f"on {ct['down']} and inside on {ct['inside']} of the {ct['n']} "
        f"DISCELL fits with both reads; over all panels, above on {ca['up']},"
        f" below on {ca['down']} and inside on {ca['inside']} of {ca['n']}. "
        f"Read A is lower on held-out tiles on {reada['lower']} of "
        f"{sum(reada.values())} fits; the twin margin moves by at most "
        f"{rnd(tw, 3)}. {ARROW_NOTE}. --: no panel in the tier, or no value"
        + ("; $^{\\dagger}$over the seeds with a value" if any(
            "seeds have a value" in m for m in missing) else "") + ".")
    tex = (tr.header(command, "DisCell: [min, max] over finalL_s0-s2 of each "
                     "read, published and held-out tiles; Cellina: the single "
                     "counterfactual fit mirroring finalL_s0; Unassigned "
                     "excluded as a target")
           + "\\begin{table*}[t]\n\\centering\n"
           f"\\caption{{{caption}}}\n\\label{{tab:transport-heldout}}\n"
           "\\footnotesize\n\\setlength{\\tabcolsep}{2.5pt}\n"
           "\\begin{tabular}{@{}lcccccccc@{}}\n\\toprule\n"
           f" & \\multicolumn{{2}}{{c}}{{Trusted tier{arrow('transport')}}} & "
           f"\\multicolumn{{2}}{{c}}{{All panels{arrow('transport')}}} & "
           f"\\multicolumn{{2}}{{c}}{{Read A{arrow('reada')}}} & "
           f"\\multicolumn{{2}}{{c}}{{Twin margin{arrow('twin')}}} "
           "\\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}"
           "\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
           "Method" + " & published & held-out" * 4
           + " \\\\\n\\midrule\n" + "\n".join(body)
           + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    return tex, tr


# -------------------------------------------- table 9: Cellina counterfactual

def _side_by_side(text: str) -> dict:
    """dataset -> read -> (DisCell text, Cellina text) from the combined md."""
    out, ds = {}, None
    for ln in text.splitlines():
        m = re.match(r"# Transport reads, DisCell finalL_s0 vs Cellina -- (\S+)",
                     ln)
        if m:
            ds = m.group(1)
            out[ds] = {}
        elif ds and ln.startswith("| ") and not ln.startswith("| read |"):
            c = [x.strip() for x in ln.strip().strip("|").split("|")]
            out[ds][c[0]] = (c[1], c[2])
    return out


def table_cellina_cf(command: str) -> tuple[str, Trace]:
    tr = Trace("tab:cellina-cf")
    p, recs = _heldout_records(tr)
    sbs = _side_by_side(tr.read_text(SIDE_BY_SIDE))
    cols = TRAINED
    rows: dict[str, list[str]] = {k: [] for k in
                                  ("frac", "frac_ci", "panels", "reada",
                                   "reada_ci", "twin", "prog", "leak")}
    vals: dict[str, list] = {"frac": [], "reada": [], "twin": []}
    for ds in cols:
        d0 = next(r for r in recs if r["dataset"] == ds
                  and r["method"] == "DisCell" and r["run"] == "finalL_s0")
        cf = next(r for r in recs if r["dataset"] == ds
                  and r["method"] == "Cellina")
        cdir = BRES / "cellina" / f"{ds}_lineage_cf"
        ct = tr.read_json(cdir / "transport" / "transport.json")
        if ct["replay"]["max_abs_diff"] != 0.0 or not ct["replay"]["available"]:
            raise SystemExit(f"{ds}: Cellina replay is not 0.0")
        if MASK not in ct["eval_mask"]["excluded_types"]:
            raise SystemExit(f"{ds}: Cellina cf not masked")
        cb = tr.read_json(cdir / "bootstrap_ci.json")["reads"]
        # published values: comparison json, checked against Cellina's own
        # bootstrap file and the side-by-side md
        pub_d, pub_c = d0["published"], cf["published"]
        checks = [("transport_of_ceiling_trusted",
                   pub_c["transport_of_ceiling_trusted"]["value"]),
                  ("readA_gap_own", pub_c["readA_gap_own"]["value"])]
        for k, v in checks:
            if finite(v) and abs(cb[k]["estimate"] - v) > 1e-9:
                raise SystemExit(f"{ds}: Cellina {k} differs from its "
                                 "bootstrap_ci.json")
        sb = sbs[ds]
        for k, dv, cv in (("transport_of_ceiling_trusted",
                           pub_d["transport_of_ceiling_trusted"]["value"],
                           pub_c["transport_of_ceiling_trusted"]["value"]),
                          ("readA_gap_own", pub_d["readA_gap_own"]["value"],
                           pub_c["readA_gap_own"]["value"]),
                          ("twin_margin_own", pub_d["twin_margin_own"],
                           pub_c["twin_margin_own"])):
            for v, t in ((dv, sb[k][0]), (cv, sb[k][1])):
                if finite(v) and not t.startswith(rnd(v, 3)):
                    raise SystemExit(f"{ds} {k}: {v} vs side-by-side {t}")
        # DisCell's decomposition from its own transport file (s0)
        tp = DATA / ds / "runs" / "finalL_s0" / "transport" / "transport.json"
        ts = tr.read_json(tp)["summary"]["extrapolation_trusted"]
        if abs(ts["counterfactual"] / ts["noise_ceiling"]
               - ts["counterfactual_of_ceiling"]) > 1e-9 or abs(
                ts["counterfactual_of_ceiling"]
                - pub_d["transport_of_ceiling_trusted"]["value"]) > 1e-9:
            raise SystemExit(f"{tp}: fraction of ceiling is not "
                             "counterfactual / noise_ceiling")
        not_mapped = set(ct["not_mapped"])
        for meth, pub in (("DisCell", pub_d), ("Cellina", pub_c)):
            fr = pub["transport_of_ceiling_trusted"]
            ra = pub["readA_gap_own"]
            vals["frac"].append(fr["value"])
            vals["reada"].append(ra["value"])
            vals["twin"].append(pub["twin_margin_own"])
            rows["frac"].append(tr.cell("fraction", f"{SHORT[ds]} {meth}",
                                        num(fr["value"], 3), [fr["value"]], 3,
                                        p, "published.transport_of_ceiling_"
                                        "trusted.value"))
            rows["frac_ci"].append(tr.cell("fraction CI", f"{SHORT[ds]} {meth}",
                                           interval(*fr["ci95"], 2),
                                           fr["ci95"], 2, p, "published."
                                           "transport_of_ceiling_trusted.ci95"))
            rows["panels"].append(tr.cell("panels", f"{SHORT[ds]} {meth}",
                                          str(fr["n_panels"]),
                                          [fr["n_panels"]], 0, p,
                                          "published.transport_of_ceiling_"
                                          "trusted.n_panels"))
            rows["reada"].append(tr.cell("Read A", f"{SHORT[ds]} {meth}",
                                         num(ra["value"], 3), [ra["value"]], 3,
                                         p, "published.readA_gap_own.value"))
            rows["reada_ci"].append(tr.cell("Read A CI", f"{SHORT[ds]} {meth}",
                                            interval(*ra["ci95"], 2),
                                            ra["ci95"], 2, p, "published."
                                            "readA_gap_own.ci95"))
            rows["twin"].append(tr.cell("twin", f"{SHORT[ds]} {meth}",
                                        num(pub["twin_margin_own"], 3),
                                        [pub["twin_margin_own"]], 3, p,
                                        "published.twin_margin_own"))
            if meth == "DisCell":
                for part, rk in (("program_only", "prog"),
                                 ("leak_only", "leak")):
                    v = ts[part] / ts["noise_ceiling"]
                    rows[rk].append(tr.cell(part, SHORT[ds], num(v, 3), [v],
                                            3, tp, f"summary.extrapolation_"
                                            f"trusted.{part} / noise_ceiling"))
            else:
                for part, rk in (("program_only", "prog"),
                                 ("leak_only", "leak")):
                    if part not in not_mapped or (
                            ct["summary"]["extrapolation_trusted"][part]
                            is not None):
                        raise SystemExit(f"{ds}: Cellina {part} is mapped?")
                    rows[rk].append("not mapped$^{a}$")
        tr.prov.append(f"{SHORT[ds]}: {rel(p)} : published side of DisCell "
                       f"finalL_s0 and Cellina cf (fraction trusted + ci95, "
                       "readA_gap_own + ci95, twin_margin_own); "
                       "programme-only and leak-only "
                       f"parts: {rel(tp)} : summary.extrapolation_trusted."
                       "{program_only, leak_only} / noise_ceiling; Cellina: "
                       f"{rel(cdir / 'transport' / 'transport.json')} "
                       f"not_mapped = {sorted(not_mapped)}; replay 0.0 "
                       "checked; values checked against "
                       f"{rel(cdir / 'bootstrap_ci.json')} and "
                       f"{rel(SIDE_BY_SIDE)}")
    # bold: per section, the better of the two estimates of each read
    for k, rd in (("frac", "transport"), ("reada", "reada"), ("twin", "twin")):
        for i, ds in enumerate(cols):
            pair = vals[k][2 * i:2 * i + 2]
            for j in best(pair, rd, [rnd(v, 3) if finite(v) else ""
                                     for v in pair]):
                rows[k][2 * i + j] = bold(rows[k][2 * i + j])
                tr.notes.append(f"{SHORT[ds]} / {k}: bold on "
                                + ("DISCELL" if j == 0 else "Cellina") + ".")
    L = [
        "\\multicolumn{9}{@{}l}{\\emph{Fraction of the noise ceiling, "
        f"trusted tier}}{arrow('transport')}}} \\\\",
        "\\quad estimate & " + " & ".join(rows["frac"]) + " \\\\",
        "\\quad $95\\%$ int. & " + " & ".join(rows["frac_ci"]) + " \\\\",
        "\\quad panels & " + " & ".join(rows["panels"]) + " \\\\",
        "\\addlinespace",
        f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{Read A, own target}}"
        f"{arrow('reada')}}} \\\\",
        "\\quad estimate & " + " & ".join(rows["reada"]) + " \\\\",
        "\\quad $95\\%$ int. & " + " & ".join(rows["reada_ci"]) + " \\\\",
        "\\addlinespace",
        f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{Twin margin, own target}}"
        f"{arrow('twin')}}} \\\\",
        "\\quad estimate & " + " & ".join(rows["twin"]) + " \\\\",
        "\\addlinespace",
        "\\multicolumn{9}{@{}l}{\\emph{Parts of the prediction, fraction of "
        "the noise ceiling, trusted tier}} \\\\",
        "\\quad programmes & " + " & ".join(rows["prog"]) + " \\\\",
        "\\quad leakage & " + " & ".join(rows["leak"]) + " \\\\",
    ]
    tr.notes.append("Written after the freeze by cellina_extra_2026-09-29 "
                    "(Cellina refit on training tiles only, posterior-mean "
                    "decode, DisCell's panels and draws replayed, replay = 0.0 "
                    "on every dataset) and transport_heldout_2026-09-29.")
    caption = (
        "The counterfactual claim head to head: DISCELL's transported "
        "prediction against Cellina's neighbour-rewiring counterfactual, "
        "scored with the same transport reads (published read, as in "
        "\\cref{tab:headline}). Cellina is refitted on the training tiles only, "
        "decoded at its posterior mean, and read on the panels, niches, "
        "ceilings and bootstrap draws of DISCELL's first seed, which is "
        "shown beside it with the $95\\%$ intervals (half-tile subsampling "
        "for the fraction, tile bootstrap for Read A; to two decimals); "
        "DISCELL's range over three seeds is in \\cref{tab:transport-heldout}. "
        "\\emph{Panels}: panels in the trusted tier. Read A: the median gap to the target closed; twin margin: how "
        "much closer a cell's prediction is to its own target than to a "
        "random cell's. \\emph{Parts of the prediction}: DISCELL's "
        "prediction from the response programmes alone and from the leakage "
        "alone (the parts are not additive). $^{a}$Not mapped: Cellina has no leakage channel, so its "
        "counterfactual cannot be split into these parts. "
        f"{ARROW_NOTE} (the parts have none); {BOLD_NOTE}, of the two "
        "estimates. Unassigned cells "
        "are not targets.")
    head = (" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SHORT[d]}}}"
                               for d in cols) + " \\\\\n"
            + "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}"
                      for i in range(len(cols))) + "\n"
            + " & " + " & ".join("DISCELL & Cellina" for _ in cols) + " \\\\")
    tex = (tr.header(command, "published transport read; DisCell finalL_s0 "
                     "(the seed Cellina's read mirrors) with its interval, "
                     "and [min, max] over finalL_s0-s2; Cellina single fit; "
                     "Unassigned excluded as a target")
           + "\\begin{table*}[t]\n\\centering\n"
           f"\\caption{{{caption}}}\n\\label{{tab:cellina-cf}}\n\\footnotesize\n"
           "\\setlength{\\tabcolsep}{2.5pt}\n"
           "\\begin{tabular}{@{}lcccccccc@{}}\n\\toprule\n"
           f"{head}\n\\midrule\n" + "\n".join(L)
           + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    return tex, tr


# ----------------------------------------------------- table 10: synthetic

SYN_READS = [("nmi_z", "NMI, $\\vmu_z$", 2),
             ("w_cca", "CCA, $\\vw$", 2),
             ("b_cosine", "Loadings", 2),
             ("comp_r2", "Comp.\\ $R^2$", 2)]
SYN_DIR = {"nmi_z": "syn_nmi", "w_cca": "syn_cca", "b_cosine": "syn_cosine",
           "comp_r2": "comp_r2"}


def table_synthetic(command: str) -> tuple[str, Trace]:
    tr = Trace("tab:synthetic")
    p = SYNTH / "synthetic_recovery.json"
    d = tr.read_json(p)
    if not d["simulator_checks"]["all_pass"]:
        raise SystemExit("synthetic simulator checks do not all pass")
    groups = d["groups"]

    def find(arm, pk, ak):
        g = [x for x in groups if x["arm"] == arm and x["planted_kappa"] == pk
             and x["assumed_kappa"] == ak]
        return g[0] if g else None

    def row_cells(g, label):
        cells = []
        for key, _, nd in SYN_READS:
            vals = [f[key] for f in g["fits"]]
            if (abs(min(vals) - g[key]["min"]) > 1e-12
                    or abs(max(vals) - g[key]["max"]) > 1e-12
                    or len(vals) != g[key]["n"]):
                raise SystemExit(f"synthetic {label} {key}: fits disagree "
                                 "with the group summary")
            cells.append(tr.cell(label, key, vrange(vals, nd),
                                 [min(vals), max(vals)], nd, p,
                                 f"groups[{g['arm']}, planted "
                                 f"{g['planted_kappa']:g}, assumed "
                                 f"{g['assumed_kappa']:g}].fits[*].{key} "
                                 "min, max"))
        if any(f.get("dead_w_channel") for f in g["fits"]):
            tr.notes.append(f"{label}: a fit has a dead w channel.")
        return cells

    body = ["\\multicolumn{8}{@{}l}{\\emph{Assumed $\\kappa$ = planted "
            "$\\kappa$}} \\\\"]
    order = [("final", 0.0, 0.0, "final configuration"),
             ("uncontrolled", 0.0, 0.0, "without the adversary"),
             ("final", 0.2, 0.2, "final configuration"),
             ("uncontrolled", 0.2, 0.2, "without the adversary")]
    for arm, pk, ak, lab in order:
        g = find(arm, pk, ak)
        if g is None:
            raise SystemExit(f"synthetic: no group {arm} {pk} {ak}")
        body.append(f"\\quad {lab} & {pk:g} & {ak:g} & {g['n_fits']} & "
                    + " & ".join(row_cells(g, f"{arm} {pk:g}/{ak:g}"))
                    + " \\\\")
    body.append("\\addlinespace")
    body.append("\\multicolumn{8}{@{}l}{\\emph{Misspecified $\\kappa$, "
                "planted $\\kappa = 0.2$, final configuration}} \\\\")
    for ak in sorted({x["assumed_kappa"] for x in groups
                      if x["arm"] == "final" and x["planted_kappa"] == 0.2}):
        g = find("final", 0.2, ak)
        a = f"\\textbf{{{ak:g}}}" if ak == 0.2 else f"{ak:g}"
        body.append(f"\\quad assumed $\\kappa$ & 0.2 & {a} & {g['n_fits']} & "
                    + " & ".join(row_cells(g, f"final 0.2/{ak:g}")) + " \\\\")
    body.append("\\addlinespace")
    # world references (range over the three worlds, both planted kappa)
    wr = d["world_references"]
    body.append("\\multicolumn{8}{@{}l}{\\emph{References of the simulated "
                "worlds}} \\\\")

    def wref(key, nd, label):
        vals = [wr[k][key]["min"] for k in wr] + [wr[k][key]["max"] for k in wr]
        return tr.cell(label, key, vrange(vals, nd), [min(vals), max(vals)],
                       nd, p, f"world_references[*].{key} min, max")

    bw = d["world_references_by_world"]
    for lab, cells in (
            ("clusters of the counts", [wref("nmi_counts", 2, "counts"),
                                        "--", "--", "--"]),
            ("planted $\\vz$", [wref("nmi_planted_z", 2, "planted z"), "--",
                                "--", wref("comp_r2_planted_z", 2,
                                           "planted z comp")]),
            ("random subspace, mean", ["--", "--",
                                       wref("b_cosine_random_mean", 2,
                                            "random mean"), "--"]),
            ("random subspace, 95th percentile",
             ["--", "--", wref("b_cosine_random_q95", 2, "random q95"),
              "--"])):
        body.append(f"\\quad {lab} & & & -- & " + " & ".join(cells)
                    + " \\\\")
    tr.prov.append(f"every row: {rel(p)} : groups[arm, planted_kappa, "
                   "assumed_kappa].fits[*] (min, max over world x model "
                   "seeds; checked against the group's stored summary); "
                   "references: world_references[planted kappa].<key> "
                   "min, max over both planted kappa")
    cfg = d["config_final_arm"]
    tr.notes.append(f"Final arm configuration (config_final_arm): "
                    f"adv_comp_weight {cfg['adv_comp_weight']}, alpha_w "
                    f"{cfg['alpha_w']}, w_warmup_epochs "
                    f"{cfg['w_warmup_epochs']}, alpha_z {cfg['alpha_z']:.4g}, "
                    f"epochs {cfg['epochs']}/patience {cfg['patience']}, "
                    f"d_z {cfg['d_z']}, d_w {cfg['d_w']}, tile_cells "
                    f"{cfg['tile_cells']}, val_fraction {cfg['val_fraction']}. "
                    "Simulator checks: all pass.")
    tr.notes.append("Written 2026-09-29 16:58 (after the freeze; a new run, "
                    "scripts/logs/synthetic_2026-09-29/AGENT_REPORT.md).")
    caption = (
        "Recovery on simulated sections at the final configuration, scaled "
        "to the simulated world. Three worlds per planted leakage rate, each "
        "fitted with three model seeds; the range over the nine fits "
        "(\\emph{Fits}). NMI of the intrinsic latent's posterior mean with "
        "the planted type; \\emph{CCA}: the first canonical correlation of "
        "the response's posterior mean with the planted response; "
        "\\emph{Loadings}: the principal-angle cosine of the fitted against "
        "the planted loadings; \\emph{Comp.\\ $R^2$}: the within-type "
        "$R^2$ of neighbour composition from the intrinsic latent. "
        "\\emph{Misspecified $\\kappa$}: the worlds planted at $\\kappa = "
        "0.2$, fitted at each assumed rate (the matched rate in bold). "
        "\\emph{References}: NMI of clusters of the counts and of the "
        "planted intrinsic state, the composition $R^2$ of the planted state, "
        "and the loading cosine of a random subspace (its mean and 95th "
        "percentile), each as the range over the three worlds and both "
        "planted rates. A fit is "
        "stopped early on $15\\%$ of held-out tiles; every read covers all "
        f"cells. {ARROW_NOTE}.")
    tex = (tr.header(command, "[min, max] over 3 world seeds x 3 model seeds "
                     "per arm; best checkpoint; all 6,000 cells")
           + "\\begin{table}[t]\n\\centering\n"
           f"\\caption{{{caption}}}\n\\label{{tab:synthetic}}\n\\small\n"
           "\\setlength{\\tabcolsep}{3pt}\n"
           "\\begin{tabular}{@{}lccccccc@{}}\n\\toprule\n"
           "& \\multicolumn{2}{c}{$\\kappa$} & & & & & \\\\\n"
           "\\cmidrule(lr){2-3}\n"
           "Arm & planted & assumed & Fits & "
           + " & ".join(h + arrow(SYN_DIR[k]) for k, h, _ in SYN_READS)
           + " \\\\\n\\midrule\n"
           + "\n".join(body)
           + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return tex, tr


# --------------------------------------------------- table 11: planted w

PP_READS = [("kl_auc", "KL AUC", 2), ("w_corr", "$\\vw$ corr.", 2),
            ("z_absorption", "$\\vz$ absorption", 2),
            ("recon_gain_planted", "Recon. gain", 3),
            ("spearman", "Spearman", 2),
            ("top_decile_overlap", "Top-decile overlap", 2)]


def table_planted_percell(command: str) -> tuple[str, Trace]:
    tr = Trace("tab:planted-percell")
    p = SYNTH / "planted_percell.json"
    d = tr.read_json(p)
    rule = d["rule"]
    body = []
    for w in sorted({g["world_seed"] for g in d["groups"]}):
        title = ("World 0 (decides)" if w == rule["world_seed"]
                 else f"World {w} (replication)")
        body.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for g in sorted((g for g in d["groups"] if g["world_seed"] == w),
                        key=lambda g: g["s"]):
            cells = []
            for key, _, nd in PP_READS:
                if key in ("spearman", "top_decile_overlap"):
                    vals = [c[key] for c in g["cross_seed"]]
                elif key == "recon_gain_planted":
                    vals = [x[key]["gain_per_count"] for x in g["per_seed"]]
                else:
                    vals = [x[key] for x in g["per_seed"]]
                if (abs(min(vals) - g[key]["min"]) > 1e-12
                        or abs(max(vals) - g[key]["max"]) > 1e-12):
                    raise SystemExit(f"planted w{w} s{g['s']} {key}: per-seed "
                                     "values disagree with the summary")
                cells.append(tr.cell(f"world {w} s {g['s']:g}", key,
                                     vrange(vals, nd), [min(vals), max(vals)],
                                     nd, p, f"groups[world {w}, s {g['s']:g}]"
                                     f".{key} min, max"))
            det = "yes" if g["detected"] else "no"
            if w == rule["world_seed"] and rule["by_s"][f"{g['s']:g}"][
                    "detected"] != g["detected"]:
                raise SystemExit("planted: rule and group verdict differ")
            shift = tr.cell(f"world {w} s {g['s']:g}", "shift",
                            rnd(g["shift_norm_lograte"], 1),
                            [g["shift_norm_lograte"]], 1, p,
                            "shift_norm_lograte")
            body.append(f"\\quad {g['s']:g} & {shift} & " + " & ".join(cells)
                        + f" & {det} \\\\")
        body.append("\\addlinespace")
    body = body[:-1]
    pr = d["predictability"]
    aucs = [v for w in pr.values() for k, v in w.items()
            if k.endswith("_auc")]
    smallest = rule["smallest_detected_s"]
    tr.prov.append(f"every row: {rel(p)} : groups[world, s] per_seed[*] "
                   "(KL AUC, w corr, z absorption, recon_gain_planted."
                   "gain_per_count) and cross_seed[*] (Spearman, top-decile "
                   "overlap), min-max, checked against the group summary; "
                   "detected from the group, checked against rule.by_s; "
                   "predictability[world].*_auc")
    tr.cells.append({"row": "caption", "col": "predictability",
                     "text": f"{rnd(min(aucs), 2)}--{rnd(max(aucs), 2)}",
                     "values": [min(aucs), max(aucs)], "nd": 2,
                     "src": rel(p), "key": "predictability[*].*_auc min, max"})
    tr.notes.append("Written 2026-09-29 17:49 (after the freeze; a new run, "
                    "scripts/logs/planted_percell_2026-09-29/AGENT_REPORT.md). "
                    f"Smallest detected s on world 0: {smallest}.")
    caption = (
        "A planted per-cell response on simulated sections (planted $\\kappa "
        "= 0.2$, final configuration unchanged). In $10\\%$ of the cells of "
        "one type, spread across niches, the log-rate is shifted along the "
        "planted response programme by $s$ times the size of the planted "
        "niche effect (\\emph{Shift}, its norm in log-rate units); $s = 0$ is "
        "the null. \\emph{KL AUC}: the per-cell divergence of the response "
        "posterior from its prior separating the planted cells from the "
        "other cells of their type; \\emph{$\\vw$ corr.}: correlation of the "
        "response's deviation from its prior mean, along the planted "
        "direction, with the plant; \\emph{$\\vz$ absorption}: the same for "
        "the intrinsic latent, read through the decoder along the planted direction; "
        "\\emph{Recon. gain}: held-out gain in nats per count of the full "
        "decode over the response at its prior mean, on planted cells; "
        "\\emph{Spearman} and \\emph{top-decile overlap}: agreement of the "
        "per-cell divergence between seeds. Ranges over three model seeds "
        "(seed pairs for the last two). The rule, fixed in advance and "
        "applied to world 0: the channel detects at $s$ if every seed has "
        "KL AUC $\\geq 0.8$ and every seed pair a Spearman correlation "
        "$\\geq 0.5$; worlds 1 and 2 replicate. "
        + ("No strength is detected. " if smallest is None else
           f"The smallest strength detected is $s = {smallest:g}$. ")
        + "The planted subset cannot be predicted from neighbour composition "
        "or from composition and the image descriptor (five-fold "
        f"cross-validated AUC {rnd(min(aucs), 2)}--{rnd(max(aucs), 2)} "
        "within the planted type, logistic and forest classifiers, all three "
        "worlds). The arrow gives the preferred direction.")
    tex = (tr.header(command, "[min, max] over model seeds 0-2 (seed pairs "
                     "for Spearman and top-decile overlap) per world and s")
           + "\\begin{table}[t]\n\\centering\n"
           f"\\caption{{{caption}}}\n\\label{{tab:planted-percell}}\n\\small\n"
           "\\setlength{\\tabcolsep}{3pt}\n"
           "\\begin{tabular}{@{}lcccccccc@{}}\n\\toprule\n"
           "$s$ & Shift & " + " & ".join(
               h + (arrow("kl_auc") if k == "kl_auc" else "")
               for k, h, _ in PP_READS)
           + " & Detected \\\\\n\\midrule\n" + "\n".join(body)
           + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return tex, tr


# ------------------------------------------------ table 12: context latent

#: tab:context's columns: (header, block key, family, readout of DIRECTION)
CONTEXT_PROBE_COLS = [("Comp., MLP", "comp", "mlp"),
                      ("Comp., ridge", "comp", "ridge"),
                      ("Image, MLP", "img", "mlp"),
                      ("Image, ridge", "img", "ridge")]
#: their printed heads, under a spanning "Composition" / "Image" row
CONTEXT_HEADS = ["MLP", "Ridge", "MLP", "Ridge"]


MI_WIDTH_NOTE = ("not comparable across widths: the kNN estimate depends on "
                 "the latent's width")


def _context_file(tr: Trace, ds: str) -> tuple[Path, dict]:
    p = DATA / ds / "experiments" / "context_grade.json"
    return p, tr.read_json(p)


def table_context(command: str) -> tuple[str, Trace]:
    tr = Trace("tab:context")
    files, methods = {}, {}
    for ds in SECTIONS:
        bp = DATA / ds / "experiments" / "baseline_battery_lineage.json"
        methods[ds] = section_methods(tr, ds, tr.read_json(bp))
        files[ds] = _context_file(tr, ds)
    # checks: the mask, exp(2 excess) - 1, DISCELL's w guard against its
    # degeneracy.json, and each baseline row graded on the table's latents
    for ds in SECTIONS:
        p, d = files[ds]
        for name, m in d["methods"].items():
            entries = (list(m["runs"].values()) if "runs" in m else [m])
            for e in entries:
                if e.get("absent"):
                    continue
                if MASK not in e["eval_mask"]["excluded_types"]:
                    raise SystemExit(f"{rel(p)} {name}: no Unassigned mask")
                for fam in ("mlp", "ridge"):
                    for blk in ("comp", "img"):
                        b = e["probe"][fam][blk]
                        if abs(b["var_fraction"]
                               - np.expm1(2 * b["excess"])) > 1e-9:
                            raise SystemExit(f"{rel(p)} {name} {fam}/{blk}: "
                                             "var_fraction is not "
                                             "exp(2 excess) - 1")
                c = e.get("check_degeneracy")
                if c is not None and not c["ok"]:
                    raise SystemExit(f"{rel(p)} {name}/{e['run']}: I(niche; w) "
                                     f"differs from degeneracy.json by "
                                     f"{c['diff']:+.1e}")
        checked = [f"{r}: {v['check_degeneracy']['diff']:+.1e}"
                   for r, v in d["methods"]["DISCELL"]["runs"].items()
                   if "check_degeneracy" in v]
        if checked:
            tr.notes.append(f"{SHORT[ds]}: DISCELL's I(niche; mu_w) excess "
                            "equals each run's degeneracy.json w-channel guard "
                            f"(differences {', '.join(checked)}; tolerance "
                            f"{MI_TOL}).")
        for m in methods[ds]:
            e = d["methods"].get(m["label"])
            if e is not None and m.get("probe") not in e.get("entries", []):
                raise SystemExit(f"{rel(p)} {m['label']}: graded entries "
                                 f"{e.get('entries')} do not include the "
                                 f"table's '{m.get('probe')}'")

    rest = len(CONTEXT_PROBE_COLS) + 2      # the columns after d
    body = []
    for ds in SECTIONS:
        p, d = files[ds]
        runs = d["methods"]["DISCELL"]["runs"]
        rows = []           # (label, width, probe cells, mi, nmi)
        # DISCELL: mean row and range row
        mean_cells, rng_cells = [], []
        for i, (h, blk, fam) in enumerate(CONTEXT_PROBE_COLS):
            vs = [probe_share(runs[r]["probe"][fam][blk]["excess"])
                  for r in sorted(runs)]
            m = float(np.mean(vs))
            key = f"DISCELL.runs.finalL_s{{0,1,2}}.probe.{fam}.{blk}: 1 - exp(-2 excess), x100"
            mean_cells.append(tr.cell(f"{SHORT[ds]} DISCELL", h, pct(m),
                                      [m], 1, p, key + " mean"))
            rng_cells.append(tr.cell(f"{SHORT[ds]} DISCELL range", h,
                                     pct_span(vs),
                                     [min(vs), max(vs)], 1, p,
                                     key + " min, max"))
        mis = [runs[r]["niche_mi"]["w_niche_mi_excess"] for r in sorted(runs)]
        nmis = [runs[r]["nmi"] for r in sorted(runs)]
        mi_key = "DISCELL.runs.finalL_s{0,1,2}.niche_mi.w_niche_mi_excess"
        nmi_key = "DISCELL.runs.finalL_s{0,1,2}.nmi"
        width = str(d["methods"]["DISCELL"]["width"])
        rows.append(("DISCELL", width, mean_cells,
                     tr.cell(f"{SHORT[ds]} DISCELL", "I(niche)",
                             num(float(np.mean(mis)), 2), [float(np.mean(mis))],
                             2, p, mi_key + " mean"),
                     tr.cell(f"{SHORT[ds]} DISCELL", "NMI",
                             num(float(np.mean(nmis)), 2),
                             [float(np.mean(nmis))], 2, p, nmi_key + " mean")))
        rows.append(("\\quad range", "", rng_cells,
                     tr.cell(f"{SHORT[ds]} DISCELL range", "I(niche)",
                             span(min(mis), max(mis), 2), [min(mis), max(mis)],
                             2, p, mi_key + " min, max"),
                     tr.cell(f"{SHORT[ds]} DISCELL range", "NMI",
                             span(min(nmis), max(nmis), 2),
                             [min(nmis), max(nmis)], 2, p,
                             nmi_key + " min, max")))
        tr.prov.append(f"{SHORT[ds]} / DISCELL: {rel(p)} : DISCELL.runs."
                       "finalL_s{0,1,2} (probe 1 - exp(-2 excess) x100, niche_mi "
                       "excess, nmi; mean and min, max)")
        for mt in methods[ds]:
            meth = mt["label"]
            shown = meth
            e = d["methods"].get(meth)
            if meth == "resolVI":
                rows.append((shown, "--", None,
                             "no context latent (a single cell latent)", None))
                tr.prov.append(f"{SHORT[ds]} / resolVI: {rel(p)} : "
                               "resolVI.absent (no context latent)")
                continue
            if e is None or e.get("absent"):
                state = mt.get("state", "not graded")
                txt = (NOT_RUN if state not in ("running", "queued",
                                                "scoring") else
                       tr.pend(f"{SHORT[ds]} / {meth}",
                               f"whole-section fit {state}", state))
                rows.append((shown, "--", None,
                             txt, None))
                tr.prov.append(f"{SHORT[ds]} / {meth}: not graded "
                               f"(whole-section fit: {state})")
                continue
            cells = []
            for i, (h, blk, fam) in enumerate(CONTEXT_PROBE_COLS):
                v = probe_share(e["probe"][fam][blk]["excess"])
                cells.append(tr.cell(f"{SHORT[ds]} {meth}", h, pct(v),
                                     [v], 1, p, f"'{meth}'.probe.{fam}.{blk}"
                                     ": 1 - exp(-2 excess), x100"))
            mi = e["niche_mi"]["w_niche_mi_excess"]
            rows.append((shown, str(e["width"]), cells,
                         tr.cell(f"{SHORT[ds]} {meth}", "I(niche)", num(mi, 2),
                                 [mi], 2, p,
                                 f"'{meth}'.niche_mi.w_niche_mi_excess"),
                         tr.cell(f"{SHORT[ds]} {meth}", "NMI",
                                 num(e["nmi"], 2), [e["nmi"]], 2, p,
                                 f"'{meth}'.nmi")))
            tr.prov.append(f"{SHORT[ds]} / {meth}: {rel(p)} : '{meth}' "
                           f"({e['key']} of {e['latents']}; probe "
                           "1 - exp(-2 excess) x100, niche_mi excess, nmi)")
        # bold: the most recovered per probe column in the section, DISCELL
        # by its mean; ties at the printed digits all bold
        scored = [r for r in rows if r[2] is not None and r[0] != "\\quad range"]
        for i in range(len(CONTEXT_PROBE_COLS)):
            texts = [r[2][i] for r in scored]
            for j in best([float(re.sub(r"[$]", "", t)) for t in texts],
                          "ctx_probe", texts):
                scored[j][2][i] = bold(scored[j][2][i])
        lab = f"\\multirow{{{len(rows)}}}{{*}}{{{SHORT[ds]}}}"
        for k, (name, w, cells, mi, nmi) in enumerate(rows):
            head = f"{lab if k == 0 else ''} & {name} & {w}"
            if cells is None:
                body.append(f"{head} & \\multicolumn{{{rest}}}{{c}}"
                            f"{{{mi}}} \\\\")
            else:
                body.append(f"{head} & " + " & ".join(cells + [mi, nmi])
                            + " \\\\")
        body.append("\\midrule")
    body = body[:-1]
    tr.notes.append("Bold: the highest share per probe column and section "
                    "(context latents are graded in the positive direction: "
                    "the context channel should carry the niche). "
                    f"I(niche; .) carries no arrow and no bold: {MI_WIDTH_NOTE} "
                    "(devlog 2026-09-30, rule set in advance). The NMI with "
                    "type has no preferred direction here and is descriptive.")
    tr.notes.append("Reads: discell/experiments/context_grade.py on the probe "
                    "regrade's cells and targets (DISCELL: each finalL fit's "
                    "own split; comparison methods: finalL_s0's split, as "
                    "their probe records).")
    caption = (
        "What each method's context latent carries, on the held-out cells of "
        "each section, with Unassigned cells not targets. The context latent "
        "is DISCELL's response posterior mean $\\vmu_w$, SIMVI's spatial "
        "latent, MintFlow's microenvironment latent and Cellina's "
        "microenvironment latent $s$; resolVI has a single cell latent and no "
        "context latent. \\emph{Composition}, \\emph{Image}: the share of the "
        "within-type variance of neighbour composition and of the image "
        "block that the held-out MLP and ridge probes recover from the "
        "context latent (with the type) beyond the within-type permutation "
        "floor, in \\% ($1 - e^{-2\\,\\mathrm{excess}}$, as in "
        "\\cref{tab:probe}). "
        "$\\I(\\text{niche}; \\cdot)$: the mutual information of the latent "
        "with ten composition niches beyond the within-type floor, in nats; "
        "its estimate depends on the latent's width $d$, so it is not "
        "compared across methods of different width. \\emph{NMI}: $k$-means "
        "on the context latent against the cell type. For DISCELL, the mean "
        "over three seeds and the range. "
        f"{ARROW_NOTE}; bold marks the highest recovery per section and "
        "probe column. On the serial section the Cellina rows and MintFlow "
        "are the core's models transferred, SIMVI is fitted on the section. "
        "Cellina's $s$ is inferred from the mean expression of the cell's "
        "neighbours. --: not applicable.")
    up = ARROWS[DIRECTION["ctx_probe"]]
    head = ("Section & Method & $d$ & "
            + " & ".join(f"{h} {up}" for h in CONTEXT_HEADS)
            + " & $\\I(\\text{niche}; \\cdot)$ & NMI \\\\")
    tex = (tr.header(command, "lineage labels; accepted checkpoint; one "
                     "Unassigned mask for every method; DisCell mean and "
                     "[min, max] over finalL_s0-s2, baselines single fit; "
                     "context latent graded on the probe regrade's cells")
           + "\\begin{table*}[t]\n\\centering\n"
           f"\\caption{{{caption}}}\n\\label{{tab:context}}\n\\small\n"
           "\\setlength{\\tabcolsep}{3pt}\n"
           "\\begin{tabular}{@{}llccccccc@{}}\n\\toprule\n"
           " & & & \\multicolumn{2}{c}{Composition (\\%)} & "
           "\\multicolumn{2}{c}{Image (\\%)} & & \\\\\n"
           "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
           f"{head}\n\\midrule\n" + "\n".join(body)
           + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    return tex, tr


# ----------------------------------------------------------------------- main

TABLES = {"headline": table_headline,
          "headline_full": table_headline_full,
          "probe": table_probe,
          "battery": table_battery,
          "sensitivity": table_sensitivity,
          "kappa_sweep": table_kappa_sweep,
          "timing": table_timing,
          "breakdown": table_breakdown,
          "breakdown_traj": table_breakdown_traj,
          "transport_heldout": table_transport_heldout,
          "cellina_cf": table_cellina_cf,
          "synthetic": table_synthetic,
          "planted_percell": table_planted_percell,
          "context": table_context}


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
