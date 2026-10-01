"""Generated paper tables: round trip from the frozen sources to the cells.

``scripts/paper_tables.py`` is loaded by path (scripts/ is not a package).
The round-trip tests re-read a sample of cells from the source files with
their own minimal parsing, round them, and look for them in the row of the
generated .tex file where they belong. They need the result files on disk
and are skipped without them.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import math
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "paper_tables.py"
DATA = REPO / "data" / "datasets"
OV = "xenium_prime_ovarian_cancer_ffpe"
LUNG = "xenium_prime_human_lung_cancer_ffpe"
FF = "xenium_prime_human_ovary_ff"
GSE = "gse315411_pdltma06_11_prime_solo"
DUAL = "gse315411_pdltma06_10_prime_dual"


@functools.lru_cache(maxsize=None)
def _load():
    spec = importlib.util.spec_from_file_location("paper_tables", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pin(p: Path) -> Path:
    """The file the tables read for *p*: its pre-refit MintFlow backup while
    the generator pins the MintFlow fits (MF_PIN), else *p*."""
    return _load().mf_pinned(p)


@pytest.fixture(scope="module")
def pt():
    spec = importlib.util.spec_from_file_location("paper_tables", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def built(pt, tmp_path_factory):
    if not (DATA / OV / "experiments" / "envelope_table_ci_at_best.md").exists():
        pytest.skip("frozen result files are not on disk")
    out = tmp_path_factory.mktemp("generated")
    traces = pt.build(list(pt.TABLES), out, "pytest")
    return out, traces


def rendered(path: Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines()
            if not ln.startswith("%")]


def row(path: Path, label: str, occurrence: int = 0) -> list[str]:
    hits = [ln for ln in rendered(path) if ln.strip().startswith(label)
            or f"& {label} &" in ln]
    assert len(hits) > occurrence, f"{label!r} not in {path.name}"
    cells = [c.strip() for c in hits[occurrence].rstrip("\\ ").split(" & ")]
    return cells


SHORT = {OV: "Ovarian FFPE", LUNG: "Lung FFPE", FF: "Ovarian FF",
         GSE: "TMA core", DUAL: "TMA serial"}
BASE = REPO.parent / "DisCell-baselines" / "results"
#: MintFlow's battery / probe entry per section (the TMA serial: transferred)
MF_ENTRY = {OV: "MintFlow (lineage)", LUNG: "MintFlow (lineage)",
            FF: "MintFlow (lineage)", GSE: "MintFlow (lineage)",
            DUAL: "MintFlow (lineage, transfer)"}


def section_rows(path: Path, ds: str) -> list[list[str]]:
    """The rendered rows of one section of a \\multirow-sectioned table
    (battery, context), each as its cells (cells[1] is the method)."""
    out, on = [], False
    for ln in rendered(path):
        if f"{{*}}{{{SHORT[ds]}}} &" in ln:
            on = True
        elif on and not ln.startswith(" & "):
            break
        if on:
            out.append([c.strip() for c in ln.rstrip("\\ ").split(" & ")])
    assert out, f"no {SHORT[ds]} section in {path.name}"
    return out


def method_row(path: Path, ds: str, method: str) -> list[str] | None:
    hits = [c for c in section_rows(path, ds) if len(c) > 1 and c[1] == method]
    return hits[0] if hits else None


def battery_json(ds: str) -> dict:
    return json.loads(_pin(DATA / ds / "experiments"
                       / "baseline_battery_lineage.json").read_text())


def mf_refit(entry: dict | None) -> bool:
    """A MintFlow battery entry from a refit with the corrected export
    (devlog 2026-10-01): its fit config names the decoded rate."""
    return bool(((entry or {}).get("config") or {}).get("decoded_rate"))


def mf_probe_excess(ds: str, block: str = "mlp_comp") -> float | None:
    """MintFlow's probe excess on a section as the table reads it: the final
    probe table's entry, else its own probe record; None if neither."""
    name = MF_ENTRY[ds]
    d = json.loads(_pin(DATA / ds / "experiments"
                    / "probe_regrade_lineage_final.json").read_text())
    if name in d:
        return d[name][block]["excess"]
    rec = _pin(DATA / ds / "experiments" / "probe_regrade_lineage"
               / (re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") + ".json"))
    if not rec.exists():
        return None
    fam, blk = block.split("_")
    return json.loads(rec.read_text())[fam][blk]["excess"]


def mf_config(ds: str) -> Path:
    """The config.json of the MintFlow fit a section's timing cell reads:
    the refit's once it exists, else the original fit's (in place or
    archived)."""
    new = BASE / "mintflow_refit_lineage" / ds / "config.json"
    if not _load().MF_PIN and new.exists() and json.loads(new.read_text()).get("decoded_rate"):
        return new
    old = BASE / "mintflow" / f"{ds}_lineage" / "config.json"
    return old if old.exists() else (BASE / "_archive_mintflow_export_bug"
                                     / "mintflow" / f"{ds}_lineage"
                                     / "config.json")


def unbold(cell: str) -> str:
    """A cell without the bold that marks the best value of its section."""
    m = re.fullmatch(r"\\textbf\{(?:\\boldmath )?(.*)\}", cell)
    return m.group(1) if m else cell


def slim(cell: str) -> tuple[str, str]:
    """(mean, range) of a cell of the slim headline: the mean with the range
    in brackets beside it, or under it where the range has a negative end."""
    m = (re.fullmatch(r"\\begin\{tabular\}\[c\]\{@\{\}c@\{\}\}(.*)"
                      r"\\\\\{\\scriptsize \((.*)\)\}\\end\{tabular\}", cell)
         or re.fullmatch(r"(.*) \{\\scriptsize \((.*)\)\}", cell))
    assert m, cell
    return m.group(1), m.group(2)


def is_bold(cell: str) -> bool:
    return cell.startswith("\\textbf{")


def share(excess: float) -> float:
    """Independent probe share: 1 - exp(-2 excess), in % (author,
    2026-09-30: the share of within-type variance explained)."""
    return 100 * (1 - math.exp(-2 * excess))


def r(x: float, nd: int) -> str:
    """Independent half-up rounding (numbers here are never ties in binary)."""
    s = f"{x + (5e-12 if x >= 0 else -5e-12):.{nd}f}"
    return f"${s}$" if s.startswith("-") else s


# ---------------------------------------------------------------- formatting

def test_rounding_is_half_up_on_the_stored_value(pt):
    assert pt.rnd(0.2500626945504267, 2) == "0.25"
    assert pt.rnd(0.125, 2) == "0.13"          # half-up, not half-even
    assert pt.rnd(-7.2321, 3) == "-7.232"
    assert pt.num(-0.0004, 3) == "$-0.000$"


def test_probe_share_is_one_minus_exp_minus_two_excess(pt):
    """Every probe % is 1 - exp(-2 excess) (author, 2026-09-30), bounded
    by 100 %; e^{2 excess} - 1 would read 172 % at excess 0.5."""
    assert abs(pt.probe_share(0.5) - 100 * (1 - math.exp(-1))) < 1e-12
    assert pt.probe_share(0.0) == 0.0
    assert pt.probe_share(-0.01) < 0
    assert pt.probe_share(10.0) < 100
    assert pt.pct(pt.probe_share(0.1)) == "18.1"


def test_no_table_prints_the_error_ratio_minus_one(built):
    out, _ = built
    for f in sorted(out.glob("*.tex")):
        text = "\n".join(rendered(f))
        assert "e^{2\\,\\mathrm{excess}}" not in text, f.name


def test_sensitivity_mlp_column_is_the_share(built):
    out, _ = built
    d = json.loads(_pin(DATA / OV / "experiments" / "sensF_adversary.json")
                   .read_text())
    ctrl = d["sections"][OV]["control"]["names"]
    arm = d["sections"][OV]["arms"]["comp5"]["names"]
    read = lambda run: share(json.loads(  # noqa: E731
        (DATA / OV / "runs" / run / "validation" / "probe_blocks.json")
        .read_text())["mlp"]["comp"]["excess"])
    c = [read(x) for x in ctrl]
    a = sum(read(x) for x in arm) / len(arm)
    mc = sum(c) / len(c)
    sd = math.sqrt(sum((x - mc) ** 2 for x in c) / (len(c) - 1))
    cells = row(out / "sensitivity.tex", "\\quad\\quad composition weight 5")
    assert cells[5] == f"{r(a, 1)} ({r((a - mc) / sd, 1)})"
    assert row(out / "sensitivity.tex", "\\quad final configuration")[5] \
        == r(mc, 1)


def test_ranges_with_a_negative_end_use_to(pt):
    assert pt.span(0.701, 0.723, 3) == "0.701--0.723"
    assert pt.span(-7.2321, -7.1475, 3) == "$-7.232$ to $-7.148$"


# ------------------------------------------------------------ whole-file rules

def test_every_file_has_the_traceability_header_and_one_label(built):
    out, _ = built
    for name, label in [("headline", "tab:headline"),
                        ("headline_full", "tab:headline-full"),
                        ("probe", "tab:probe"),
                        ("battery", "tab:battery"),
                        ("sensitivity", "tab:sensitivity"),
                        ("kappa_sweep", "tab:kappa-sweep"),
                        ("timing", "tab:timing"),
                        ("breakdown", "tab:breakdown"),
                        ("breakdown_traj", "tab:breakdown-traj"),
                        ("breakdown_traj_main", "tab:breakdown-traj"),
                        ("transport_heldout", "tab:transport-heldout"),
                        ("cellina_cf", "tab:cellina-cf"),
                        ("synthetic", "tab:synthetic"),
                        ("planted_percell", "tab:planted-percell"),
                        ("planted_spillover", "tab:S-planted-spillover"),
                        ("reloc_clean", "tab:S-reloc-clean"),
                        ("headline_main", "tab:headline-main"),
                        ("breakdown_main", "tab:kappa-star")]:
        text = (out / f"{name}.tex").read_text()
        for key in ("% Command:", "% Generated:", "% Sources read",
                    "% Reading convention:", "% Provenance",
                    "% Pending cells:"):
            assert key in text, (name, key)
        assert text.count(f"\\label{{{label}}}") == 1
        assert "\\toprule" in text and "\\bottomrule" in text
        assert "\\small" in text or "\\footnotesize" in text


def test_no_code_reference_in_the_rendered_part(built):
    out, _ = built
    bad = re.compile(r"finalL|uncontrolledL|sweepL|sensF|\.json|\.md\b|"
                     r"_s\d|--[a-z]|scripts/|discell|B-mf1|q90")
    for f in out.glob("*.tex"):
        for ln in rendered(f):
            assert not bad.search(ln), (f.name, ln)
            assert "_" not in re.sub(r"\$[^$]*\$", "", ln), (f.name, ln)


def test_every_traced_cell_is_in_its_file_and_matches_its_values(pt, built):
    out, traces = built
    for name, tr in traces.items():
        text = (out / f"{name}.tex").read_text()
        for c in tr.cells:
            assert c["text"] in text, (name, c)
            nds = (c["nd"] if isinstance(c["nd"], list)
                   else [c["nd"]] * len(c["values"]))
            for v, nd in zip(c["values"], nds):
                shown = pt.rnd(v, nd) if nd else str(v)
                assert shown in c["text"], (name, c)


def test_transport_is_released(built):
    out, _ = built
    head = row(out / "headline.tex", "\\TermReloc{}")
    assert not any("\\pending" in c for c in head)
    ov = [v for v in _raw_transport(OV) if v is not None]
    assert slim(head[1]) == (r(sum(ov) / 3, 2), f"{r(min(ov), 2)}--{r(max(ov), 2)}")
    tma = [v for v in _raw_transport(GSE) if v is not None]
    assert len(tma) == 2 and slim(head[4])[0] == r(sum(tma) / 2, 2) + "$^{\\dagger}$"
    sens = (out / "sensitivity.tex").read_text()
    assert "every arm: \\pending" not in sens


# ------------------------------------------------------------ round trips

def _envelope_row(ds: str, label: str) -> list[str]:
    text = (DATA / ds / "experiments" / "envelope_table_ci_at_best.md").read_text()
    for ln in text.splitlines():
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cells and cells[0] == label:
            return cells
    raise KeyError(label)


def _battery_at_best(ds: str, key: str) -> list[float]:
    """A read of the masked re-read of best.pt, per seed (raw)."""
    return [json.loads(_pin(DATA / ds / "runs" / f"finalL_s{s}"
                        / "degeneracy.json").read_text())["battery"][key]
            for s in range(3)]


def _raw_transport(ds: str) -> list:
    """The trusted-tier fraction of the ceiling per seed (raw; None where
    the seed has no trusted panel)."""
    out = []
    for s in range(3):
        t = json.loads(_pin(DATA / ds / "runs" / f"finalL_s{s}" / "transport"
                        / "transport.json").read_text())
        v = (t["summary"].get("extrapolation_trusted") or {}).get(
            "counterfactual_of_ceiling")
        out.append(v if v is not None and v == v else None)
    return out


def test_headline_round_trip(built):
    """Every cell is rounded once from the raw per-run values (audit item
    5: rounding the envelope's printed digits again had shown 0.7547 as
    0.76)."""
    out, _ = built
    for f in (out / "headline.tex", out / "headline_full.tex"):
        is_slim = f.name == "headline.tex"

        def mean_of(cell: str) -> str:      # the slim cell is "mean (min-max)"
            return slim(cell)[0] if is_slim else cell

        vals = _battery_at_best(OV, "nmi_targets")
        nmi = row(f, "NMI of")
        assert mean_of(nmi[1]) == r(sum(vals) / 3, 3)
        want = f"{r(min(vals), 3)}--{r(max(vals), 3)}"
        if is_slim:
            assert slim(nmi[1]) == (r(sum(vals) / 3, 3), want)
        else:
            assert row(f, "\\quad range &", 1)[1] == want  # NMI block
        vals = _battery_at_best(LUNG, "cycle_r2_z_q90")
        cz, mi = (("Cycle $R^2$ of $\\vz$", "$\\I(\\text{niche}; \\vw)$ excess")
                  if is_slim else ("Cycle $R^2$, $\\vz$", "$\\I(\\text{niche}; \\vw)$"))
        assert mean_of(row(f, cz)[2]) == r(sum(vals) / 3, 3)
        env = _envelope_row(GSE, "I(niche; w) excess over within-type floor (nats)")
        assert mean_of(row(f, mi)[4]) \
            == r(float(env[2]), 2)
        # the held-out section, from the three cross-slide files
        vals = [json.loads(_pin(DATA / GSE / "runs" / f"finalL_s{s}" / "crossslide"
                            / f"{DUAL}.json").read_text())
                ["held_out_section"]["nmi_targets"] for s in range(3)]
        assert mean_of(nmi[5]) == r(sum(vals) / 3, 3)


def test_probe_round_trip(built):
    out, _ = built
    f = out / "probe.tex"
    d = json.loads(_pin(DATA / LUNG / "experiments"
                    / "probe_regrade_lineage_final.json").read_text())
    us = [d[f"DisCell/finalL_s{s}"]["mlp_comp"]["fraction_of_uncontrolled"]
          for s in range(3)]
    left = row(f, "\\quad fraction left $\\downarrow$")    # composition MLP
    assert left[2] == f"{r(min(us), 2)}--{r(max(us), 2)}"
    vf = [share(d[f"DisCell/finalL_s{s}"]["mlp_comp"]["excess"])
          for s in range(3)]
    assert unbold(row(f, "\\quad with the adversary")[2]) == \
        f"{r(min(vf), 1)}--{r(max(vf), 1)}"
    g = json.loads(_pin(DATA / GSE / "experiments"
                    / "probe_regrade_lineage_final.json").read_text())
    mint = r(share(g["MintFlow (lineage)"]["mlp_comp"]["excess"]), 1)
    simvi = r(share(g["SIMVI (lineage)"]["mlp_comp"]["excess"]), 1)
    cells = row(f, "\\quad MintFlow, SIMVI")
    assert [unbold(c) for c in cells[4].split(", ")] == [mint, simvi]
    # every section's MintFlow share from its source, or -- without one
    for j, ds in enumerate((OV, LUNG, FF, GSE, DUAL)):
        ex = mf_probe_excess(ds)
        shown = cells[j + 1].split(", ")[0]
        assert unbold(shown) == ("--" if ex is None else r(share(ex), 1)), ds
    text = (f).read_text()
    assert "guard" not in "\n".join(rendered(f)).lower()
    assert "\\label{tab:probe}" in text


def test_battery_round_trip(built):
    out, _ = built
    f = out / "battery.tex"
    b = battery_json(GSE)
    cells = [unbold(c) for c in method_row(f, GSE, "resolVI")]
    assert cells[5] == r(b["resolVI"]["mirror"]["r2"], 3)
    assert cells[2] == r(b["resolVI"]["nmi"], 3)
    # MintFlow's reconstruction: the refit's value where the entry is a refit
    # (corrected export), otherwise never a number (B-mf1)
    for ds in (OV, LUNG, FF, GSE, DUAL):
        cells = method_row(f, ds, "MintFlow")
        e = battery_json(ds).get(MF_ENTRY[ds])
        if len(cells) == 3:                     # not run / pending: one span
            assert "not run$^{e}$" in cells[2] or "\\pending" in cells[2]
        elif mf_refit(e):
            assert unbold(cells[7]) == r(e["reconstruction"]["recon"], 3), ds
        else:
            assert cells[7] == "n/r$^{b}$", ds
            assert not any(r(e["reconstruction"]["recon"], 3) in ln
                           for ln in rendered(f)), ds


def test_sensitivity_round_trip(built):
    out, _ = built
    d = json.loads(_pin(DATA / OV / "experiments" / "sensF_adversary.json")
                   .read_text())
    arm = d["sections"][OV]["arms"]["comp5"]
    nmi = sum(x["nmi"] for x in arm["records"]) / len(arm["records"])
    sd = d["sections"][OV]["moves"]["comp5"]["nmi"]["in_sd"]
    cells = row(out / "sensitivity.tex", "\\quad\\quad composition weight 5")
    assert cells[2] == f"{r(nmi, 3)} ({r(sd, 1)})"
    assert len(cells) == 7                      # no Seeds column
    assert "Adversary capacity and weight} (2)" in \
        (out / "sensitivity.tex").read_text()


def test_kappa_sweep_round_trip(built):
    out, _ = built
    d = json.loads(_pin(DATA / FF / "experiments" / "kappa_sweep_sweepL.json")
                   .read_text())
    vals = [x["nmi"] for x in d["runs"] if x["kappa"] == 0.2]
    lines = [ln for ln in rendered(out / "kappa_sweep.tex")
             if re.match(r"^(\\multirow\{6\}\{\*\}\{Ovarian FF\})? & 0\.2 &", ln)]
    ff = [ln for ln in lines if ln]  # rows of every section at kappa 0.2
    cells = [c.strip() for c in ff[2].rstrip("\\ ").split(" & ")]  # FF third
    assert cells[3] == f"{r(min(vals), 3)}--{r(max(vals), 3)}"


def test_sensitivity_transport_round_trip(built):
    out, _ = built
    d = json.loads(_pin(DATA / OV / "experiments" / "sensF_adversary.json")
                   .read_text())
    sec = d["sections"][OV]

    def tt(run):
        t = json.loads(_pin(DATA / OV / "runs" / run / "transport"
                        / "transport.json").read_text())
        return t["summary"]["extrapolation_trusted"]["counterfactual_of_ceiling"]

    ctrl = [tt(n) for n in sec["control"]["names"]]
    arm = [tt(n) for n in sec["arms"]["comp5"]["names"]]
    m = sum(ctrl) / 3
    sd = (sum((x - m) ** 2 for x in ctrl) / 2) ** 0.5
    am = sum(arm) / len(arm)
    cells = row(out / "sensitivity.tex", "\\quad\\quad composition weight 5")
    assert cells[-1] == f"{r(am, 2)} ({r((am - m) / sd, 1)})"


# ---------------------------------------------- baselines added 2026-09-30

def test_battery_new_rows(built):
    out, _ = built
    f = out / "battery.tex"
    b = battery_json(LUNG)
    cells = [unbold(c) for c in method_row(f, LUNG, "MintFlow")]
    assert cells[2] == r(b["MintFlow (lineage)"]["nmi"], 3)
    assert cells[6] == r(b["MintFlow (lineage)"]["cycle_q90"]["z"]
                         ["r2_pooled"], 3)
    assert cells[7] == (r(b["MintFlow (lineage)"]["reconstruction"]["recon"], 3)
                        if mf_refit(b["MintFlow (lineage)"]) else "n/r$^{b}$")
    pm = json.loads(_pin(DATA / LUNG / "experiments" / "probe_regrade_lineage"
                     / "MintFlow_lineage.json").read_text())
    assert cells[4] == r(pm["mlp"]["comp"]["fraction_of_uncontrolled"], 2)
    # the niche-domain row on every section, the own-graph row on two
    niche = [ln for ln in rendered(f) if "& Cellina, niche$^{d}$ &" in ln]
    own = [ln for ln in rendered(f) if "& Cellina, own graph &" in ln]
    assert len(niche) == 5 and len(own) == 2
    g = json.loads(_pin(DATA / GSE / "experiments"
                    / "baseline_battery_lineage.json").read_text())
    cells = [unbold(c.strip()) for c in niche[3].rstrip("\\ ").split(" & ")]
    assert cells[2] == r(g["cellina_nicheadv"]["nmi"], 3)
    assert "best case on the probe, not its published setting" in \
        f.read_text()
    # SIMVI: numbers with no reconstruction (n/a) where fitted, else pending
    # or "not run" -- never a number for a whole section not fitted
    for ln in rendered(f):
        if "& SIMVI &" in ln and "\\pending" not in ln:
            assert "n/a" in ln or "not run$^{e}$" in ln, ln


def test_probe_new_rows(built):
    out, _ = built
    f = out / "probe.tex"
    pm = json.loads(_pin(DATA / LUNG / "experiments" / "probe_regrade_lineage"
                     / "MintFlow_lineage.json").read_text())
    cells = row(f, "\\quad MintFlow, SIMVI")
    mint, simvi = cells[2].split(", ")
    assert unbold(mint) == r(share(pm['mlp']['comp']['excess']), 1)
    assert simvi == "--"
    d = json.loads(_pin(DATA / FF / "experiments"
                    / "probe_regrade_lineage_final.json").read_text())
    cells = row(f, "\\quad Cellina niche domain, own graph")
    assert unbold(cells[3].split(", ")[0]) == r(share(d["cellina_nicheadv"]["mlp_comp"]
                                       ["excess"]), 1)


# ------------------------------------------------------------- new tables

def test_timing_round_trip(built):
    out, _ = built
    f = out / "timing.tex"
    t = json.loads(_pin(DATA / FF / "experiments" / "timing_lineage"
                    / "timing_phi.json").read_text())
    assert row(f, "\\quad s per epoch")[3] == r(t["s_per_epoch"], 2)
    assert row(f, "\\quad peak memory")[3] == r(t["peak_train_mib"] / 1024, 1)
    mins = [json.loads(_pin(DATA / OV / "runs" / f"finalL_s{s}" / "metrics.json")
                       .read_text())["minutes"] for s in range(3)]
    assert unbold(row(f, "\\quad DISCELL")[1]) == \
        f"{r(min(mins), 1)}--{r(max(mins), 1)}"
    base = BASE
    c = json.loads((base / "cellina" / f"{LUNG}_lineage" / "config.json")
                   .read_text())
    assert unbold(row(f, "\\quad Cellina")[2]) == r(c["train_s"] / 60, 1)
    feas = (base / "feasibility.tsv").read_text()
    mf = row(f, "\\quad MintFlow, 50 epochs")
    if f"MintFlow\t{LUNG}\t" in feas and "\tok\t" in feas:
        m = json.loads(mf_config(LUNG).read_text())
        assert unbold(mf[2]) == r(m["train_s"] / 60, 1)
    m = json.loads(mf_config(GSE).read_text())            # the TMA core
    assert unbold(mf[4]) == r(m["train_s"] / 60, 1)
    assert "\\pending" in mf[1] or f"MintFlow\t{OV}\t" in feas


def test_breakdown_round_trip(built):
    out, _ = built
    f = out / "breakdown.tex"
    d = json.loads((REPO / "scripts" / "logs" / "breakdown_2026-09-29"
                    / "breakdown_all.json").read_text())
    sig = row(f, "Signalling share")
    e = d["ff"]["members"]["signalling_response_lr_vs_other"]
    if e["status"] == "breaks":
        assert sig[3].startswith(f"{e['kappa_star']:g}")
    cyc = row(f, "Cycle asymmetry")
    assert all(c.startswith("above the grid") for c in cyc[1:]) == all(
        d[s]["members"]["cycle_asym_q90"]["status"] == "above the grid"
        for s in ("ovarian", "lung", "ff", "gse", "gse_dual"))
    readA = row(f, "\\TermReloc{} $>$ type mean")
    if "readA_minus_typemean" not in d["ovarian"]["members"]:
        assert "\\pending" in readA[1]
    # the serial section's family: the cycle asymmetry only before the
    # 2026-10-01 gap fill, the headline members (core fits applied) after
    dual_fam = d["gse_dual"].get("family") or ["cycle_asym_q90"]
    assert (readA[-1] == "--") == ("readA_minus_typemean" not in dual_fam)
    assert "($m = 7$)" in f.read_text()
    # not applicable (the TMA sections: no tumour cells) is said, not blank
    axis = row(f, "Tumour axis")
    for i, sec in enumerate(("ovarian", "lung", "ff", "gse", "gse_dual")):
        if "axis_tau_true_minus_false" in (d[sec].get("not_applicable")
                                           or {}):
            assert axis[1 + i] == "n/a"
            assert "no tumour cells" in f.read_text()
    # the marker pairs are a trajectory, not a claimed contrast (2026-09-30)
    assert not [ln for ln in rendered(f) if ln.startswith("Marker pairs")]
    # so is transport - programme part (2026-10-01)
    assert not [ln for ln in rendered(f)
                if ln.startswith("\\TermReloc{} $-$ programme")]


def test_breakdown_traj_round_trip(built):
    out, _ = built
    d = json.loads((REPO / "scripts" / "logs" / "breakdown_2026-09-29"
                    / "breakdown_all_trajectory.json").read_text())
    e = d["lung"]["readouts"]["moran_mu_z"]["0.2"]
    lines = [ln for ln in rendered(out / "breakdown_traj.tex")
             if ln.strip().startswith("\\quad Moran's $I$ of $\\vmu_z$")]
    cells = [c.strip() for c in lines[1].rstrip("\\ ").split(" & ")]  # lung
    assert cells[4] == r(e["mean"], 3)
    # the marker-pair contrast, in percentage points (2026-09-30)
    e = d["ff"]["readouts"]["marker_excl_minus_ctrl_dc"]["0"]
    lines = [ln for ln in rendered(out / "breakdown_traj.tex")
             if ln.strip().startswith("\\quad Marker pairs")]
    cells = [c.strip() for c in lines[2].rstrip("\\ ").split(" & ")]  # FF
    assert cells[1] == r(100 * e["mean"], 1)
    # transport - programme part, zero by construction at kappa = 0
    # (2026-10-01)
    lines = [ln for ln in rendered(out / "breakdown_traj.tex")
             if ln.strip().startswith("\\quad \\TermReloc{} $-$ programme")]
    for g in ("0", "0.2"):
        e = d["ff"]["readouts"]["transport_cf_minus_program"][g]
        cells = [c.strip() for c in lines[2].rstrip("\\ ").split(" & ")]
        assert cells[1 + ["0", "0.05", "0.1", "0.2"].index(g)] == r(
            e["mean"], 2)


def test_breakdown_traj_main_round_trip(pt, built):
    """The RECOMB supplement's compact trajectory table: the means of the
    trajectory file, and the probe row as the seed mean of the sweep
    re-grade's composition share (kappa = 0.1 are the final fits)."""
    out, _ = built
    f = out / "breakdown_traj_main.tex"
    text = f.read_text()
    assert "longtable" not in text and "\\begin{table}[!htbp]" in text
    d = json.loads((REPO / "scripts" / "logs" / "breakdown_2026-09-29"
                    / "breakdown_all_trajectory.json").read_text())
    lines = [ln for ln in rendered(f)
             if ln.strip().startswith("\\quad Moran's $I$ of $\\vmu_z$")]
    assert len(lines) == 4
    cells = [c.strip() for c in lines[1].rstrip("\\ ").split(" & ")]  # lung
    assert cells[4] == r(d["lung"]["readouts"]["moran_mu_z"]["0.2"]["mean"], 3)
    # the TMA relocation row carries the fewer-seeds mark
    lines = [ln for ln in rendered(f)
             if ln.strip().startswith("\\quad \\TermReloc{}, fraction")]
    assert [("dagger" in ln) for ln in lines] == [False, False, False, True]
    # probe: kappa = 0.1 on the primary section from the final fits
    rec = json.loads(_pin(DATA / OV / "experiments"
                      / "probe_regrade_lineage_sweep.json").read_text())
    vals = [pt.probe_share(v["mlp_comp"]["excess"]) for k, v in rec.items()
            if k.startswith("DisCell/finalL_s")]
    assert len(vals) == 3
    lines = [ln for ln in rendered(f)
             if ln.strip().startswith("\\quad Probe, composition")]
    assert len(lines) == 4
    cells = [c.strip() for c in lines[0].rstrip("\\ ").split(" & ")]
    assert cells[3] == pt.pct(sum(vals) / 3)


def test_breakdown_traj_builds_its_companion(pt, tmp_path):
    if not (DATA / OV / "experiments" / "envelope_table_ci_at_best.md").exists():
        pytest.skip("frozen result files are not on disk")
    pt.build(["breakdown_traj"], tmp_path, "pytest")
    assert (tmp_path / "breakdown_traj.tex").exists()
    assert (tmp_path / "breakdown_traj_main.tex").exists()


def test_kappa_star_caption_points_to_the_supplement_floats(built):
    out, _ = built
    text = "\n".join(rendered(out / "breakdown_main.tex"))
    assert "\\cref{tab:breakdown}" not in text
    assert "\\cref{tab:S-contrasts}" in text
    assert "\\cref{fig:breakdown-data}" in text


def _heldout():
    return json.loads((REPO / "scripts" / "logs" / "transport_heldout_2026-09-29"
                       / "transport_heldout_comparison.json").read_text())


def test_transport_heldout_round_trip(built):
    out, _ = built
    recs = _heldout()
    ff = [x for x in recs if x["dataset"] == FF and x["method"] == "DisCell"]
    pub = [x["published"]["transport_of_ceiling_trusted"]["value"] for x in ff]
    ho = [x["heldout"]["transport_of_ceiling_trusted"]["value"] for x in ff]
    cells = row(out / "transport_heldout.tex", "\\multirow{2}{*}{Ovarian FF}")
    assert cells[1] == "DISCELL" and len(cells) == 10
    assert cells[2] == f"{r(min(pub), 2)}--{r(max(pub), 2)}"
    assert cells[3] == f"{r(min(ho), 2)}--{r(max(ho), 2)}"
    cl = [ln for ln in rendered(out / "transport_heldout.tex")
          if ln.strip().startswith("& Cellina &")][1]                 # lung
    c = next(x for x in recs if x["dataset"] == LUNG and x["method"] == "Cellina")
    assert r(c["heldout"]["transport_of_ceiling_trusted"]["value"], 2) in cl
    text = (out / "transport_heldout.tex").read_text()
    assert "single-cell read is lower" in text


def test_cellina_cf_round_trip(built):
    out, _ = built
    f = out / "cellina_cf.tex"
    recs = _heldout()
    d0 = next(x for x in recs if x["dataset"] == FF and x["run"] == "finalL_s0")
    cf = next(x for x in recs if x["dataset"] == FF and x["method"] == "Cellina")
    first = [unbold(c) for c in row(f, "\\quad estimate")]   # fraction block
    assert first[5] == r(d0["published"]["transport_of_ceiling_trusted"]["value"], 2)
    assert first[6] == r(cf["published"]["transport_of_ceiling_trusted"]["value"], 2)
    twin = [unbold(c) for c in row(f, "\\quad estimate", 2)]
    assert twin[6] == r(cf["published"]["twin_margin_own"], 2)
    prog = row(f, "\\quad programmes")
    t = json.loads(_pin(DATA / FF / "runs" / "finalL_s0" / "transport"
                    / "transport.json").read_text())["summary"]["extrapolation_trusted"]
    assert prog[5] == r(t["program_only"] / t["noise_ceiling"], 2)
    assert prog[6] == "--$^{a}$"


def test_synthetic_round_trip(built):
    out, _ = built
    d = json.loads(_pin(DATA / "synthetic_smoke" / "experiments"
                    / "synthetic_recovery.json").read_text())
    g = next(x for x in d["groups"] if x["arm"] == "uncontrolled"
             and x["planted_kappa"] == 0.2)
    cells = row(out / "synthetic.tex",
                "\\quad without the adversary, $\\kappa = 0.2$")
    assert len(cells) == 5
    vals = [x["w_cca"] for x in g["fits"]]
    assert cells[2] == f"{r(min(vals), 2)}--{r(max(vals), 2)}"
    assert f"the range over the {len(g['fits'])} fits" in \
        (out / "synthetic.tex").read_text()


def test_planted_percell_round_trip(built):
    out, _ = built
    d = json.loads(_pin(DATA / "synthetic_smoke" / "experiments"
                    / "planted_percell.json").read_text())
    g = next(x for x in d["groups"] if x["world_seed"] == 0 and x["s"] == 1.0)
    lines = [ln for ln in rendered(out / "planted_percell.tex")
             if ln.strip().startswith("\\quad 1")]
    cells = [c.strip() for c in lines[0].rstrip("\\ ").split(" & ")]
    auc = [x["kl_auc"] for x in g["per_seed"]]
    assert cells[2] == f"{r(min(auc), 2)}--{r(max(auc), 2)}"
    rho = [x["spearman"] for x in g["cross_seed"]]
    assert cells[6] == f"{r(min(rho), 2)}--{r(max(rho), 2)}"
    assert len(cells) == 8                      # no 'Detected' column
    assert cells[0].endswith("$^{*}$") == bool(g["detected"])


def test_planted_spillover_round_trip(built):
    """tab:S-planted-spillover: corrected kappa* from the null diagnostic,
    the first definition and the genuine response from the control."""
    out, _ = built
    syn = DATA / "synthetic_smoke" / "experiments"
    first = json.loads((syn / "planted_spillover.json").read_text())
    corr = json.loads((syn / "null_diagnostic.json").read_text())
    lines = [ln for ln in rendered(out / "planted_spillover.tex")
             if re.match(r"0(\.\d+)? & ", ln)]
    assert len(lines) == len(first["worlds"]) == 3
    by_kt = {c[0]: c for c in ([x.strip() for x in ln.rstrip("\\ ")
                                .split(" & ")] for ln in lines)}
    # null world: the corrected contrast is no finding, the first one breaks
    assert corr["worlds"]["0"]["corrected"]["status"] == "no finding"
    assert by_kt["0"][1:3] == ["no finding", "no finding"]
    assert by_kt["0"][3].startswith(
        f"{first['worlds']['0']['table']['members']['spill_cf_minus_leak']['kappa_star']:g}")
    for w in ("0.1", "0.2"):
        c = corr["worlds"][w]["corrected"]
        m = first["worlds"][w]["table"]["members"]
        assert by_kt[w][1] == w                    # predicted: kappa_true
        assert by_kt[w][2].startswith(f"{c['kappa_star']:g}$")
        assert by_kt[w][4].startswith(
            f"{m['response_cf_minus_leak']['kappa_star']:g}$")
    assert by_kt["0"][4] == f"$>${max(first['grid']):g}"


def test_reloc_clean_round_trip(built):
    """tab:S-reloc-clean: each cell is the seed mean of the stored mean R^2,
    rounded half-up (independently recomputed)."""
    out, _ = built
    d = json.loads((DATA / "synthetic_smoke" / "experiments"
                    / "relocation_clean_truth.json").read_text())
    order = ["discell_programme", "discell_cf", "regression_log",
             "regression_rate"]
    lines = [ln for ln in rendered(out / "reloc_clean.tex")
             if re.match(r"0(\.\d+)? & 0(\.\d+)? & ", ln)]
    assert len(lines) == 5
    for ln in lines:
        cells = [c.strip() for c in ln.rstrip("\\ ").split(" & ")]
        kt, ka = float(cells[0]), float(cells[1])
        fs = [f for f in d["fits"]
              if f["kappa_true"] == kt and f["assumed"] == ka]
        assert len(fs) == 3
        want = [r(sum(f["summary"]["all"][t][m]["mean_r2"] for f in fs) / 3, 2)
                for t in ("clean", "observed") for m in order]
        assert cells[2:] == want, (kt, ka)
    # the headline of the paper's claim: the programmes are flat against the
    # clean truth at the planted fraction, the log regression degrades
    rows = {(c[0], c[1]): c for c in ([x.strip() for x in
            ln.rstrip("\\ ").split(" & ")] for ln in lines)}
    assert rows[("0", "0")][2] == rows[("0.1", "0.1")][2] == \
        rows[("0.2", "0.2")][2] == "0.76"
    assert rows[("0", "0")][4] == "0.40" and rows[("0.2", "0.2")][4] == "0.14"


# ------------------------------------------- audit corrections, 2026-09-30

def test_headline_transport_rounded_once_from_raw(built):
    """Audit item 5: the TMA core's transport maximum is 0.7547 -> 0.75 and
    its interval's lower bound 0.6350 -> 0.63, not 0.76 and 0.64 from the
    envelope's printed 0.755 and 0.635."""
    out, _ = built
    f = out / "headline_full.tex"
    tma = [v for v in _raw_transport(GSE) if v is not None]
    assert slim(row(out / "headline.tex", "\\TermReloc{}")[4])[1] == "0.71--0.75"
    rng = [ln for ln in rendered(f) if ln.startswith("\\quad range &")]
    ci = [ln for ln in rendered(f) if ln.startswith("\\quad $95\\%$ CI")]
    t_rng = [c.strip() for c in rng[-2].rstrip("\\ ").split("&")]
    assert t_rng[4] == f"{r(min(tma), 2)}--{r(max(tma), 2)}" == "0.71--0.75"
    lo = min(json.loads(_pin(DATA / GSE / "runs" / f"finalL_s{s}"
                         / "bootstrap_ci.json").read_text())
             ["reads"]["transport_of_ceiling_trusted"]["ci95"][0]
             for s in (0, 2))
    t_ci = [c.strip() for c in ci[-1].rstrip("\\ ").split("&")]
    assert t_ci[4].startswith(f"[{r(lo, 2)},") and r(lo, 2) == "0.63"


def test_tma_cycle_w_agrees_between_headline_and_sweep(built):
    """Audit item 17: the same three fits print the same range in
    tab:headline and at kappa = 0.1 in tab:kappa-sweep, rounded from the
    raw values (-0.000496 -> -0.000)."""
    out, _ = built
    vals = _battery_at_best(GSE, "cycle_r2_w_q90")
    want = r(min(vals), 3) + " to " + r(max(vals), 3)
    assert slim(row(out / "headline.tex", "Cycle $R^2$ of $\\vw$")[4])[1] == want
    f = out / "headline_full.tex"
    rng = [ln for ln in rendered(f) if ln.startswith("\\quad range &")]
    cyc_w = [c.strip() for c in rng[4].rstrip("\\ ").split(" & ")]
    assert cyc_w[4] == want == "$-0.000$ to 0.002"
    sweep = [ln for ln in rendered(out / "kappa_sweep.tex")
             if "\\textbf{0.1}" in ln]
    cells = [c.strip() for c in sweep[3].rstrip("\\ ").split(" & ")]  # TMA
    assert cells[6] == want


def test_battery_caption_explains_the_discell_rows(built):
    """Audit item 16."""
    out, _ = built
    text = " ".join(rendered(out / "battery.tex"))
    assert ("its rows are the battery's own reads, on the cell set "
            "shared with the comparison methods, so they differ slightly "
            "from \\cref{tab:headline-full}") in text


def test_battery_footnote_follows_the_whole_section_state(pt, built):
    """Audit item 18: the footnote names what is done, running and queued
    (feasibility.tsv, else the queue log), never 'three Xenium sections'."""
    out, _ = built
    f = out / "battery.tex"
    text = " ".join(rendered(f))
    assert "three Xenium sections" not in text
    names = {LUNG: "lung FFPE", OV: "ovarian FFPE", FF: "ovarian FF"}
    last = {}
    for ln in pt.FEAS.read_text().splitlines()[1:]:
        tool, ds, _, outcome, *_, reason = ln.split("\t")
        # a refit that did not finish leaves the original fit in place
        if reason.startswith(pt.MF_REFIT_TAG) and outcome not in ("ok", "truncated"):
            continue
        if ds in names:
            last[(tool, ds)] = outcome
    if "Whole-section fits of SIMVI and MintFlow:" in text:   # something pending
        for (tool, ds), outcome in last.items():
            if outcome == "ok":
                assert f"{tool} on {names[ds]}" in text
                assert "done and reported" in text
    else:                    # nothing pending: no footnote a, every row final
        assert "\\pending" not in text and "$^{a}$" not in text
        for (tool, ds), outcome in last.items():
            cells = method_row(f, ds, tool)
            if outcome in ("ok", "truncated"):
                assert len(cells) == 8, (tool, ds, cells)
            else:
                assert "not run$^{e}$" in cells[2], (tool, ds, cells)


# ------------------------------- directions and bold, 2026-09-30 (polish)

def test_best_picks_the_max_or_min_per_the_direction_map(pt):
    nan = float("nan")
    vals = [0.2, None, 0.9, nan, 0.5]
    assert pt.DIRECTION["nmi"] == "up" and pt.best(vals, "nmi") == {2}
    assert pt.DIRECTION["mirror"] == "down" and pt.best(vals, "mirror") == {0}
    assert pt.DIRECTION["time"] == "down" and pt.best([3.4, 1.3], "time") == {1}
    # negative reads: the higher log-likelihood is the better one
    assert pt.best([-7.231, -7.192, -7.278], "recon") == {1}
    # a tie at the printed digits bolds both
    assert pt.best([0.1234, 0.1231], "nmi", ["0.123", "0.123"]) == {0, 1}
    assert pt.best([None, nan], "nmi") == set()
    with pytest.raises(ValueError):            # no better direction
        pt.best([0.001, 0.003], "cycle_w")


def _number(cell: str) -> float | None:
    m = re.match(r"\$?(-?\d+\.\d+)", unbold(cell).replace("$", "").strip())
    return float(m.group(1)) if m else None


def _check_bold(cells: list[str], direction: str, label: str) -> None:
    """The bold cells are exactly those printing the best number."""
    vals = [_number(c) for c in cells]
    have = [v for v in vals if v is not None]
    if not have:
        assert not any(is_bold(c) for c in cells), label
        return
    top = (max if direction == "up" else min)(have)
    for c, v in zip(cells, vals):
        assert is_bold(c) == (v == top), (label, cells)


def test_battery_bold_is_the_best_per_section_and_column(pt, built):
    out, _ = built
    lines = rendered(out / "battery.tex")
    start = next(i for i, ln in enumerate(lines) if ln == "\\midrule")
    sections, cur = [], []
    for ln in lines[start + 1:]:
        if ln in ("\\midrule", "\\bottomrule"):
            sections.append(cur)
            cur = []
            if ln == "\\bottomrule":
                break
        elif "\\multicolumn" not in ln and "range" not in ln:
            cur.append([c.strip() for c in ln.rstrip("\\ ").split(" & ")][2:])
    assert len(sections) == 5
    for rows in sections:
        for j, (_, _, _, _, rd) in enumerate(pt.BATTERY_COLS):
            _check_bold([r[j] for r in rows], pt.DIRECTION[rd], f"col {j}")


def test_cellina_cf_and_timing_bold_the_better_value(pt, built):
    out, _ = built
    f = out / "cellina_cf.tex"
    for occ, rd in ((0, "transport"), (1, "reada"), (2, "twin")):
        cells = row(f, "\\quad estimate", occ)[1:]
        for i in range(0, 8, 2):
            _check_bold(cells[i:i + 2], pt.DIRECTION[rd], f"cellina {rd} {i}")
    t = out / "timing.tex"
    runs = [_expand(row(t, f"\\quad {m}")[1:]) for m in
            ("DISCELL", "resolVI", "Cellina", "SIMVI", "MintFlow")]
    for j, ds in enumerate((OV, LUNG, FF, GSE)):
        mins = [json.loads(_pin(DATA / ds / "runs" / f"finalL_s{s}"
                            / "metrics.json").read_text())["minutes"]
                for s in range(3)]
        col = [r[j] for r in runs]
        # DISCELL prints a range: its mean decides. Pending cells and fits
        # stopped by their time cap ($^{b}$) are not finished fits.
        vals = [sum(mins) / 3] + [_number(c) for c in col[1:]]
        vals = [None if ("pending" in c or "^{b}" in c) else v
                for c, v in zip(col, vals)]
        top = r(min(v for v in vals if v is not None), 1)
        for c, v in zip(col, vals):
            assert is_bold(c) == (v is not None and r(v, 1) == top), (ds, col)


def _expand(cells: list[str]) -> list[str]:
    """Cells with every multicolumn spread over the columns it spans."""
    out = []
    for c in cells:
        m = re.fullmatch(r"\\multicolumn\{(\d+)\}\{c\}\{(.*)\}", c)
        out += [m.group(2)] * int(m.group(1)) if m else [c]
    return out


def test_probe_bold_marks_the_lowest_method(built):
    """Per block and section, bold is on the lowest share among DISCELL
    with the adversary (its mean over seeds, from the file) and the
    comparison methods; the fits without the adversary never compete."""
    out, _ = built
    lines = rendered(out / "probe.tex")
    blocks, cur = [], None
    for ln in lines:
        if ln.startswith("\\multicolumn{6}"):
            cur = []
            blocks.append(cur)
        elif cur is not None and ln.startswith("\\quad ") and \
                "fraction left" not in ln:
            cur.append([c.strip() for c in ln.rstrip("\\ ").split(" & ")])
    assert len(blocks) == 4
    order = ("mlp_comp", "mlp_img", "ridge_comp", "ridge_img")
    for block, rows in zip(order, blocks):
        assert rows[0][0].startswith("\\quad without")
        assert not any(is_bold(c) for c in rows[0])
        for j, ds in enumerate((OV, LUNG, FF, GSE, DUAL)):
            d = json.loads(_pin(DATA / ds / "experiments"
                            / "probe_regrade_lineage_final.json").read_text())
            mean = sum(share(d[f"DisCell/finalL_s{s}"][block]["excess"])
                       for s in range(3)) / 3
            cells = [rows[1][j + 1]] + [p for rw in rows[2:]
                                        for p in rw[j + 1].split(", ")]
            vals = [mean] + [_number(c) for c in cells[1:]]
            top = r(min(v for v in vals if v is not None), 1)
            for c, v in zip(cells, vals):
                assert is_bold(c) == (v is not None and r(v, 1) == top), \
                    (block, ds, cells)
    assert "\\textbf{fraction left}" not in "\n".join(lines)


def test_no_bold_where_nothing_competes(built):
    out, _ = built
    for name in ("headline", "headline_full", "sensitivity"):
        assert "\\textbf" not in "\n".join(rendered(out / f"{name}.tex")), name
    sweep = "\n".join(rendered(out / "kappa_sweep.tex"))
    assert sweep.count("\\textbf") == 4           # the kappa = 0.1 labels only


def test_arrows_follow_the_direction_map(pt, built):
    out, _ = built
    for name in ("headline", "headline_full"):
        f = out / f"{name}.tex"
        for label, *_, rd in pt.HEADLINE_ROWS:
            if name == "headline_full":
                label = pt.FULL_LABEL.get(label, label)
            assert row(f, pt.tex_label(label))[0] == (
                pt.tex_label(label) + pt.arrow(rd)), (name, label)
        assert "Arrows give the preferred direction" in f.read_text()
    assert row(out / "headline.tex", "Cycle $R^2$ of $\\vw$")[0].endswith(
        "($\\approx 0$)")
    for name in ("probe", "battery", "cellina_cf", "timing", "sensitivity",
                 "kappa_sweep", "transport_heldout", "synthetic"):
        text = " ".join(rendered(out / f"{name}.tex"))
        assert "Arrows give the preferred direction" in text, name
        if name in ("probe", "battery", "cellina_cf", "timing"):
            assert "bold marks the best value per section" in text, name


def test_headline_slim_and_full(built):
    out, _ = built
    text = " ".join(rendered(out / "headline.tex"))
    full = " ".join(rendered(out / "headline_full.tex"))
    assert "range over seeds &" not in text and "interval &" not in text
    assert "\\cref{tab:headline-full}" in text
    assert "tile-bootstrap" not in text and "half-tile" not in text
    assert "$95\\%$ CI &" in full and "half-tile subsampling" in full
    assert "$^{\\dagger}$seeds with a trusted panel" in text
    for ln in rendered(out / "headline.tex"):     # every number cell parses
        if " & " in ln and not ln.startswith("Read &"):
            cells = [c.strip() for c in ln.rstrip("\\ ").split(" & ")][1:]
            assert all(c == "--" or slim(c) for c in cells), ln


def test_context_round_trip_and_bold(pt, built):
    """tab:context: cells re-read from context_grade.json (1 - exp(-2
    excess)), bold only on the four probe columns, the best per section."""
    import numpy as np

    out, _ = built
    f = out / "context.tex"
    lung = json.loads(_pin(DATA / LUNG / "experiments"
                       / "context_grade.json").read_text())["methods"]
    lines = rendered(f)
    mf = method_row(f, LUNG, "MintFlow")
    assert mf[2] == str(lung["MintFlow"]["width"])
    assert unbold(mf[3]) == r(share(lung["MintFlow"]["probe"]["mlp"]["comp"]
                                    ["excess"]), 1)
    assert mf[7] == r(lung["MintFlow"]["niche_mi"]["w_niche_mi_excess"], 2)
    for ds in (OV, GSE, DUAL):          # every other section that has it
        ctx = json.loads(_pin(DATA / ds / "experiments"
                          / "context_grade.json").read_text())["methods"]
        mf = method_row(f, ds, "MintFlow")
        if "MintFlow" in ctx and len(mf) > 3:
            assert unbold(mf[3]) == r(share(ctx["MintFlow"]["probe"]["mlp"]
                                            ["comp"]["excess"]), 1), ds
    runs = lung["DISCELL"]["runs"].values()
    dc = method_row(f, LUNG, "DISCELL")
    assert unbold(dc[5]) == r(np.mean([share(e["probe"]["mlp"]["img"]["excess"])
                                       for e in runs]), 1)
    start = next(i for i, ln in enumerate(lines) if ln == "\\midrule")
    sections, cur = [], []
    for ln in lines[start + 1:]:
        if ln in ("\\midrule", "\\bottomrule"):
            sections.append(cur)
            cur = []
            if ln == "\\bottomrule":
                break
        elif "\\multicolumn" not in ln and "range" not in ln:
            cur.append([c.strip() for c in ln.rstrip("\\ ").split(" & ")][3:])
    assert len(sections) == 5
    for rows in sections:
        for j in range(4):
            _check_bold([row_[j] for row_ in rows], "up", f"context col {j}")
        assert not any(is_bold(row_[k]) for row_ in rows for k in (4, 5))


# ------------------------------- MintFlow refits (corrected export), 2026-10-01

def test_mintflow_refit_switch(pt, tmp_path, monkeypatch):
    """The refit's config is used once it exists with the decoded rate; a
    refit entry wins over the original; a refit that did not finish leaves
    the original fit's feasibility state."""
    monkeypatch.setattr(pt, "MF_REFIT", tmp_path)
    monkeypatch.setattr(pt, "MF_PIN", False)
    cfg = tmp_path / LUNG / "config.json"
    assert pt.mf_config(LUNG) != cfg                       # no refit yet
    cfg.parent.mkdir()
    cfg.write_text(json.dumps({"train_s": 1.0}))
    assert pt.mf_config(LUNG) != cfg                       # not a corrected export
    cfg.write_text(json.dumps({"train_s": 1.0, "decoded_rate": "decode_mintflow"}))
    assert pt.mf_config(LUNG) == cfg
    bat = {"MintFlow (lineage)": {"config": {}},
           "MintFlow (lineage, truncated 9/50 epochs)":
               {"config": {"decoded_rate": "decode_mintflow"}}}
    assert pt.whole_entry(bat, "MintFlow").startswith("MintFlow (lineage, trunc")
    assert pt.whole_entry({"MintFlow (lineage)": {}}, "MintFlow") == "MintFlow (lineage)"
    feas = tmp_path / "feasibility.tsv"
    head = "tool\tdataset\tcells\toutcome\twall_h\tpeak_gpu_gb\tpeak_host_gb\treason\n"
    old = f"MintFlow\t{LUNG}\t1\tok\t1\t1\t1\twhole section\n"
    feas.write_text(head + old + f"MintFlow\t{LUNG}\t1\tfailed\t1\t1\t1\t"
                    f"{pt.MF_REFIT_TAG}: CUDA OOM\n")
    monkeypatch.setattr(pt, "FEAS", feas)
    assert pt.baseline_state(pt.Trace("t"), "MintFlow", LUNG)[0] == "ok"
    feas.write_text(head + old + f"MintFlow\t{LUNG}\t1\tok\t2\t1\t1\t"
                    f"{pt.MF_REFIT_TAG}: whole section\n")
    state, row_ = pt.baseline_state(pt.Trace("t"), "MintFlow", LUNG)
    assert state == "ok" and row_["wall_h"] == "2"


# ------------------------------------------- RECOMB main-text tables (one column)

def test_headline_main_round_trip(pt, built):
    """tab:headline-main: means only, from the same raw values as
    tab:headline (fewer digits), and the probe row is the mean over the
    final fits of tab:probe's MLP composition share."""
    out, _ = built
    f = out / "headline_main.tex"
    text = f.read_text()
    assert "\\begin{table}[t]" in text and "table*" not in text
    body = "\n".join(ln for ln in rendered(f) if not ln.startswith("\\caption"))
    assert "{\\scriptsize" not in text
    assert "\\textbf" not in body.replace(pt.BD_PENDING_MARK, "")
    vals = _battery_at_best(OV, "nmi_targets")
    assert row(f, "NMI of")[1] == r(sum(vals) / 3, 2)
    vals = _battery_at_best(LUNG, "cycle_r2_z_q90")
    assert row(f, "Cycle $R^2$ of $\\vz$")[2] == r(sum(vals) / 3, 2)
    env = _envelope_row(GSE, "I(niche; w) excess over within-type floor (nats)")
    assert row(f, "$\\I(\\text{niche}; \\vw)$")[4] == r(float(env[2]), 2)
    # the serial section's I(niche; w): the breakdown gaps queue's read at
    # the final kappa once its record holds it, else tab:kappa-star's mark
    bd = json.loads((REPO / "scripts" / "logs" / "breakdown_2026-09-29"
                     / "breakdown_all.json").read_text())
    serial = row(f, "$\\I(\\text{niche}; \\vw)$")[5]
    m = bd["gse_dual"]["members"].get("w_niche_mi_excess")
    if m is None:
        assert serial == pt.BD_PENDING_MARK
        assert (f"{pt.BD_PENDING_MARK}: \\pending{{breakdown gaps queue}}"
                in text) and "--: not read" not in text
    else:
        assert serial == r(m["trajectory"]["0.1"]["estimate"], 2)
    # that point is the final fits' read: so on the core
    core = bd["gse"]["members"]["w_niche_mi_excess"]["trajectory"]["0.1"]
    assert r(core["estimate"], 2) == row(f, "$\\I(\\text{niche}; \\vw)$")[4]
    for j, ds in enumerate((OV, LUNG, FF, GSE, DUAL)):
        d = json.loads(_pin(DATA / ds / "experiments"
                        / "probe_regrade_lineage_final.json").read_text())
        v = [share(d[f"DisCell/finalL_s{s}"]["mlp_comp"]["excess"])
             for s in range(3)]
        assert row(f, "Residual niche signal")[j + 1] == r(sum(v) / 3, 1), ds
    # the rows of the story map, in its order, with their arrows
    labels = [ln.split(" & ")[0] for ln in rendered(f)
              if " & " in ln and not ln.startswith("Read &")]
    assert labels == [lab + pt.arrow(rd)
                      for lab, _h, _nd, rd in pt.HEADLINE_MAIN_ROWS]
    assert [lab for lab, *_ in pt.HEADLINE_MAIN_ROWS] == [
        "NMI of $\\vz$ with type", "Cycle $R^2$ of $\\vz$",
        "Cycle $R^2$ of $\\vw$", "Residual niche signal (\\%)",
        "Mirror $R^2$", "$\\I(\\text{niche}; \\vw)$ (nats)"]
    assert "$\\downarrow$" in labels[3] and "$\\approx 0$" in labels[2]


def test_breakdown_main_round_trip(pt, built):
    """tab:kappa-star: every cell follows the record's status (>top, the
    bold kappa* with its footnote, n/a), and a member of the expected family
    the record lacks is pending, never blank."""
    out, _ = built
    f = out / "breakdown_main.tex"
    text = f.read_text()
    assert "\\begin{table}[t]" in text and "table*" not in text
    assert "\\emph{Biological claims}" in text
    assert "\\emph{Allocation checks}" in text
    d = json.loads((REPO / "scripts" / "logs" / "breakdown_2026-09-29"
                    / "breakdown_all.json").read_text())
    fam, na_all = pt._bd_expected()
    secs = ("ovarian", "lung", "ff", "gse", "gse_dual")
    top = f"{max(d['ovarian']['grid']):g}"
    for _group, members in pt.BD_MAIN_GROUPS:
        for key, label in members:
            cells = _expand(row(f, pt.tex_label(label))[1:])
            assert len(cells) == 5, label
            for c, sec in zip(cells, secs):
                e = d[sec]["members"].get(key)
                na = ((d[sec].get("not_applicable") or {}).get(key)
                      or na_all.get(sec, {}).get(key))
                if na:
                    assert c == "n/a", (key, sec)
                elif e is None:
                    assert (c == pt.BD_PENDING_MARK) == (key in fam[sec]), \
                        (key, sec)
                elif e["status"] == "above the grid":
                    assert c.startswith(f"$>${top}"), (key, sec)
                elif e["status"] == "breaks":
                    assert c.startswith(f"\\textbf{{{e['kappa_star']:g}}}$^"), \
                        (key, sec)
    assert "no tumour cells" in text


def test_vocabulary_is_neutral(built):
    """The tables are shared by two manuscripts with different words
    (leakage / spill-over, transport / relocation): rendered text uses the
    vocabulary macros, the provenance comments keep the raw names."""
    out, _ = built
    raw = re.compile(r"leak|transport|spill|relocat|influx", re.I)
    for f in sorted(out.glob("*.tex")):
        for ln in rendered(f):
            ln = re.sub(r"\\(label|cref)\{[^}]*\}|\\[tT]erm[A-Za-z]+", "",
                        ln)
            assert not raw.search(ln), (f.name, ln)
    for name in ("headline", "breakdown", "breakdown_main"):
        comments = [ln for ln in (out / f"{name}.tex").read_text().splitlines()
                    if ln.startswith("%")]
        assert not any("\\termSpill" in ln or "\\TermReloc" in ln
                       for ln in comments), name
    assert any("Relocation $>$ spill-over part" in ln for ln in
               (out / "breakdown.tex").read_text().splitlines()
               if ln.startswith("%"))


def test_internal_read_names_stay_out_of_the_rendered_tables(built):
    """'Read A', 'twin margin' and 'published read' are internal names: the
    rendered tables say single-cell read, twin read and cross-fitted read
    (comments may keep them)."""
    out, _ = built
    bad = re.compile(r"Read A|[Tt]win margin|[Pp]ublished read|"
                     r"& published &")
    for f in sorted(out.glob("*.tex")):
        for ln in rendered(f):
            assert not bad.search(ln), (f.name, ln)
    th = " ".join(rendered(out / "transport_heldout.tex"))
    assert "Single-cell read" in th and "Twin read" in th
    assert "cross-fitted & held-out" in th


def test_mintflow_pinned_to_the_fits_the_tables_were_rendered_with(pt, built):
    """While MF_PIN holds (refit queue still running, 2026-10-01), every
    table reads MintFlow from the queue's pre-refit backup: the battery's
    MintFlow rows keep the original fits' values and n/r reconstruction."""
    if not pt.MF_PIN:
        pytest.skip("MintFlow refits released")
    out, _ = built
    f = out / "battery.tex"
    for ds in (GSE, DUAL):
        live = DATA / ds / "experiments" / "baseline_battery_lineage.json"
        src = pt.mf_pinned(live)
        if src == live:
            continue                         # no refit on this section yet
        assert src.is_relative_to(pt.MF_BACKUP)
        e = json.loads(src.read_text())[MF_ENTRY[ds]]
        assert not mf_refit(e)
        cells = [unbold(c) for c in method_row(f, ds, "MintFlow")]
        assert cells[2] == r(e["nmi"], 3) and cells[7] == "n/r$^{b}$"


def test_battery_fits_its_page(built):
    """Layout: eight columns, a short not-run cell, a caption that points to
    the measured reasons instead of repeating them."""
    out, _ = built
    text = (out / "battery.tex").read_text()
    assert "\\begin{tabular}{@{}llcccccc@{}}" in text
    assert "could not be run" not in " ".join(rendered(out / "battery.tex"))
    assert "(measured reasons: \\cref{tab:timing})" in text
    assert "\\renewcommand{\\arraystretch}{0.9}" in text
