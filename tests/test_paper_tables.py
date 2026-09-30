"""Generated paper tables: round trip from the frozen sources to the cells.

``scripts/paper_tables.py`` is loaded by path (scripts/ is not a package).
The round-trip tests re-read a sample of cells from the source files with
their own minimal parsing, round them, and look for them in the row of the
generated .tex file where they belong. They need the result files on disk
and are skipped without them.
"""

from __future__ import annotations

import importlib.util
import json
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


def test_ranges_with_a_negative_end_use_to(pt):
    assert pt.span(0.701, 0.723, 3) == "0.701--0.723"
    assert pt.span(-7.2321, -7.1475, 3) == "$-7.232$ to $-7.148$"


# ------------------------------------------------------------ whole-file rules

def test_every_file_has_the_traceability_header_and_one_label(built):
    out, _ = built
    for name, label in [("headline", "tab:headline"),
                        ("probe", "tab:probe"),
                        ("battery", "tab:battery"),
                        ("sensitivity", "tab:sensitivity"),
                        ("kappa_sweep", "tab:kappa-sweep")]:
        text = (out / f"{name}.tex").read_text()
        for key in ("% Command:", "% Generated:", "% Sources read",
                    "% Reading convention:", "% Provenance",
                    "% Pending cells:"):
            assert key in text, (name, key)
        assert text.count(f"\\label{{{label}}}") == 1
        assert "\\toprule" in text and "\\bottomrule" in text
        assert "\\small" in text


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


def test_transport_is_pending_everywhere(built):
    out, _ = built
    head = row(out / "headline.tex", "Transport")
    assert "\\pending{held-out-tile check}" in head[1]
    sens = (out / "sensitivity.tex").read_text()
    assert "Transport, fraction of ceiling, every arm: \\pending" in sens


# ------------------------------------------------------------ round trips

def _envelope_row(ds: str, label: str) -> list[str]:
    text = (DATA / ds / "experiments" / "envelope_table_ci_at_best.md").read_text()
    for ln in text.splitlines():
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cells and cells[0] == label:
            return cells
    raise KeyError(label)


def test_headline_round_trip(built):
    out, _ = built
    f = out / "headline.tex"
    env = _envelope_row(OV, "NMI vs the type labels")
    nmi = row(f, "NMI of")
    assert nmi[1] == r(float(env[2]), 3)
    rng = row(f, "\\quad range over seeds", 1)       # the NMI block
    assert rng[1] == f"{r(float(env[1]), 3)}--{r(float(env[3]), 3)}"
    env = _envelope_row(LUNG, "cycle R² (z, top-decile set)")
    assert row(f, "Cycle $R^2$ of $\\vz$")[2] == r(float(env[2]), 3)
    env = _envelope_row(GSE, "I(niche; w) excess over within-type floor (nats)")
    assert row(f, "$\\I(\\text{niche}; \\vw)$ excess")[4] == r(float(env[2]), 2)
    # the held-out section, from the three cross-slide files
    vals = [json.loads((DATA / GSE / "runs" / f"finalL_s{s}" / "crossslide"
                        / f"{DUAL}.json").read_text())
            ["held_out_section"]["nmi_targets"] for s in range(3)]
    assert nmi[5] == r(sum(vals) / 3, 3)


def test_probe_round_trip(built):
    out, _ = built
    f = out / "probe.tex"
    d = json.loads((DATA / LUNG / "experiments"
                    / "probe_regrade_lineage_final.json").read_text())
    us = [d[f"DisCell/finalL_s{s}"]["mlp_comp"]["fraction_of_uncontrolled"]
          for s in range(3)]
    left = row(f, "\\quad \\textbf{fraction left}")      # composition MLP
    assert left[2] == f"\\textbf{{{r(min(us), 2)}--{r(max(us), 2)}}}"
    vf = [100 * d[f"DisCell/finalL_s{s}"]["mlp_comp"]["var_fraction"]
          for s in range(3)]
    assert row(f, "\\quad with the adversary")[2] == \
        f"{r(min(vf), 1)}--{r(max(vf), 1)}"
    g = json.loads((DATA / GSE / "experiments"
                    / "probe_regrade_lineage_final.json").read_text())
    mint = r(100 * g["MintFlow (lineage)"]["mlp_comp"]["var_fraction"], 1)
    simvi = r(100 * g["SIMVI (lineage)"]["mlp_comp"]["var_fraction"], 1)
    cells = row(f, "\\quad MintFlow, SIMVI")
    assert cells[4] == f"{mint}, {simvi}" and cells[1] == "--"
    text = (f).read_text()
    assert "guard" not in "\n".join(rendered(f)).lower()
    assert "\\label{tab:probe}" in text


def test_battery_round_trip(built):
    out, _ = built
    f = out / "battery.tex"
    b = json.loads((DATA / GSE / "experiments"
                    / "baseline_battery_lineage.json").read_text())
    resolvi = [ln for ln in rendered(f) if "& resolVI &" in ln][3]  # TMA core
    cells = [c.strip() for c in resolvi.rstrip("\\ ").split(" & ")]
    assert cells[5] == r(b["resolVI"]["mirror"]["r2"], 3)
    assert cells[2] == r(b["resolVI"]["nmi"], 3)
    mint = [ln for ln in rendered(f) if "& MintFlow &" in ln]
    assert all("n/r" in ln or "\\pending" in ln for ln in mint)
    assert not any(r(b["MintFlow (lineage)"]["reconstruction"]["recon"], 3)
                   in ln for ln in rendered(f))


def test_sensitivity_round_trip(built):
    out, _ = built
    d = json.loads((DATA / OV / "experiments" / "sensF_adversary.json")
                   .read_text())
    arm = d["sections"][OV]["arms"]["comp5"]
    nmi = sum(x["nmi"] for x in arm["records"]) / len(arm["records"])
    sd = d["sections"][OV]["moves"]["comp5"]["nmi"]["in_sd"]
    cells = row(out / "sensitivity.tex", "\\quad\\quad composition weight 5")
    assert cells[3] == f"{r(nmi, 3)} ({r(sd, 1)})"


def test_kappa_sweep_round_trip(built):
    out, _ = built
    d = json.loads((DATA / FF / "experiments" / "kappa_sweep_sweepL.json")
                   .read_text())
    vals = [x["nmi"] for x in d["runs"] if x["kappa"] == 0.2]
    lines = [ln for ln in rendered(out / "kappa_sweep.tex")
             if re.match(r"^(\\multirow\{6\}\{\*\}\{Ovarian FF\})? & 0\.2 &", ln)]
    ff = [ln for ln in lines if ln]  # rows of every section at kappa 0.2
    cells = [c.strip() for c in ff[2].rstrip("\\ ").split(" & ")]  # FF third
    assert cells[3] == f"{r(min(vals), 3)}--{r(max(vals), 3)}"
