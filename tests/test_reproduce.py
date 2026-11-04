"""reproduce/: the label mapping, the section configs and the tolerance rule."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

import discell as dc

REPRODUCE = Path(__file__).resolve().parents[1] / "reproduce"
sys.path.insert(0, str(REPRODUCE))
import lineage as L  # noqa: E402
import run as R  # noqa: E402


def test_lineage_map_choices_and_per_cell(tmp_path):
    table = tmp_path / "map.csv"
    pd.DataFrame({"source_label": ["A", "B", "SOX2-OT+ Tumor Cells", "P", "Unassigned"],
                  "lineage": ["X", "Y", "Tumour", "per-cell: X|Y", "Unassigned"]}
                 ).to_csv(table, index=False)
    re = tmp_path / "re.csv"
    pd.DataFrame({"cell_id": ["c4"], "assigned_lineage": ["Y"], "source_label": ["P"]}
                 ).to_csv(re, index=False)
    source = pd.Series(["A", "B", "SOX2-OT+ Tumor Cells", None, "P"],
                       index=["c0", "c1", "c2", "c3", "c4"])
    out = L.lineage(source, table, choices={"sox2ot": "unassigned"}, reassignments=[re])
    assert out.tolist() == ["X", "Y", "Unassigned", "Unassigned", "Y"]
    with pytest.raises(ValueError, match="choose sox2ot"):
        L.lineage(source, table, reassignments=[re])
    with pytest.raises(ValueError, match="reassignment"):
        L.lineage(source, table, choices={"sox2ot": "tumor"})
    with pytest.raises(ValueError, match="missing from the map"):
        L.lineage(pd.Series(["Z"], index=["c9"]), table, choices={"sox2ot": "tumor"})


@pytest.mark.parametrize("section", ["tma_core", "ovarian_ffpe", "lung_ffpe", "ovary_ff"])
def test_section_configs_build_and_cover_the_maps(section):
    spec = R.load_section(section)
    config = dc.Config(**spec["config"])
    assert config.kappa == spec["final_kappa"] == 0.1 and config.label_key == "lineage"
    expected = json.loads((REPRODUCE / "expected" / section / "seed0" / "config.json")
                          .read_text())
    assert config.alpha_z == expected["alpha_z"] and config.tile_cells == expected["tile_cells"]
    points = R.grid_points(spec, quick=False)
    assert len(points) == 18 and points[:3] == [(0.1, 0), (0.1, 1), (0.1, 2)]
    assert R.grid_points(spec, quick=True) == [(0.1, 0)]
    if spec["labels"]["map"]:
        table = pd.read_csv(REPRODUCE / spec["labels"]["map"])
        L.resolve_map(table, spec["labels"]["choices"])
        assert "Unassigned" in set(table["source_label"])
    # KRONOS2 only; the paper's descriptor was KRONOS v1
    from types import SimpleNamespace

    assert spec["embed"]["encoder"] == "kronos2" and R.PAPER_ENCODER == "kronos-v1"
    args = SimpleNamespace(no_image=False, device="cpu")
    assert R.config_of(spec, args).image_key == spec["embed"]["key"] == "X_kronos_masked"
    assert R.config_of(spec, SimpleNamespace(no_image=True, device="cpu")).image_key is None

def test_tolerance_band():
    ok = R.band([1.0, 1.1], [1.05, 1.15], floor=0.01)
    assert ok["pass"] and ok["tolerance"] == pytest.approx(0.2)
    assert not R.band([2.0, 2.1], [1.0, 1.01], floor=0.01)["pass"]
    assert R.band([1.02], [1.0, 1.01], floor=0.05)["pass"]


def test_serial_stage_applies_the_core_fits_to_a_second_section(tmp_path):
    """--stages serial on CPU: a tiny core fit applied unchanged to another section."""
    from types import SimpleNamespace

    from discell.sim import simulate

    spec = R.load_section("tma_core")
    assert spec["serial"]["expected"] == "tma_serial"
    assert spec["serial"]["breakdown_key"] == "gse_dual"
    core = simulate(n_cells=900, n_genes=30, kappa=0.1, seed=0).to_anndata()
    other = simulate(n_cells=700, n_genes=30, kappa=0.1, seed=1).to_anndata()
    for a in (core, other):
        a.obs["lineage"] = a.obs["cell_type"]
        dc.build_graph(a, "lineage")
    config = dc.Config(label_key="lineage", image_key="X_image", kappa=0.1, seed=0, epochs=3,
                       tile_cells=256, d_z=4, d_w=2, hidden=32, gat_dim=8, v_pcs=4,
                       device="cpu", eval_every=1, w_warmup_epochs=1)
    dc.fit(core, config, out_dir=R.fit_dir(tmp_path, 0.1, 0))
    (tmp_path / "serial").mkdir()
    other.write_h5ad(tmp_path / "serial" / "section.h5ad")       # ingest is skipped
    # no_image only skips the embed stage here: the sim carries its own X_image
    args = SimpleNamespace(work=tmp_path, quick=True, no_image=True, device="cpu",
                           part="0/1", lr_genes=None, cellchatdb=None,
                           data_root=str(tmp_path), serial_outs=None,
                           serial_cell_groups="labels.csv", serial_label_dir=str(tmp_path))
    report = R.stage_serial(spec, args)
    fit = R.fit_dir(tmp_path / "serial", 0.1, 0)
    assert (fit / "probe.json").exists() and (fit / "cross_section.json").exists()
    assert any((fit / "draws").glob("*.npz"))
    assert report["n_cells"] == 700 and report["seeds"] == [0]
    assert set(report["reads"]) == {f"probe_{b}_var_fraction" for b in R.PROBE_BLOCKS}
    assert "breakdown" not in report                             # quick: no breakdown
    assert (tmp_path / "serial" / "report.md").read_text().startswith("# tma_core serial")
    with pytest.raises(SystemExit, match="serial-cell-groups"):
        R.stage_serial(spec, SimpleNamespace(**{**vars(args), "serial_cell_groups": None}))
    with pytest.raises(SystemExit, match="no serial section"):
        R.stage_serial(R.load_section("lung_ffpe"), args)


@pytest.mark.parametrize("section", ["tma_core", "ovarian_ffpe", "lung_ffpe", "ovary_ff"])
def test_variant_configs_override_only_their_fields(section):
    from types import SimpleNamespace

    spec = R.load_section(section)
    final = R.config_of(spec, SimpleNamespace(no_image=False, device="cpu"))
    assert set(spec["variants"]) == {"no_adversary", "timing"}
    for name, variant in spec["variants"].items():
        args = SimpleNamespace(no_image=False, device="cpu", variant=name, quick=False,
                               part="0/1", work=Path("w"))
        config = R.config_of(spec, args)
        changed = {k for k, v in config.to_dict().items() if v != final.to_dict()[k]}
        assert changed == {k for k, v in variant.get("config", {}).items()
                           if v != final.to_dict()[k]}
        if "timing" in variant:
            continue
        assert R.my_points(spec, args) == [(0.1, s) for s in variant["seeds"]]
        assert R.fits_root(args) == Path("w") / "variants" / name / "fits"
    with pytest.raises(SystemExit, match="no variant"):
        R.variant_of(spec, SimpleNamespace(variant="nope"))


def test_variant_fits_reads_and_timing_on_a_simulated_section(tmp_path, monkeypatch):
    """--variant: fits + reads into <work>/variants/<NAME>, resumable; the timing stage."""
    from types import SimpleNamespace

    from discell.sim import simulate

    adata = simulate(n_cells=900, n_genes=30, kappa=0.1, seed=0).to_anndata()
    adata.obs["lineage"] = adata.obs["cell_type"]
    dc.build_graph(adata, "lineage")
    small = dict(label_key="lineage", image_key="X_image", kappa=0.1, alpha_z=0.01, epochs=3,
                 tile_cells=256, d_z=4, d_w=2, hidden=32, gat_dim=8, v_pcs=4, eval_every=1,
                 w_warmup_epochs=1)
    spec = {"section": "sim", "final_kappa": 0.1, "config": small,
            "variants": {"no_adversary": {"config": {"alpha_a": 0.0}, "seeds": [0, 1]},
                         "timing": {"timing": {"epochs": 2, "seed": 0}}}}
    args = SimpleNamespace(work=tmp_path, quick=False, no_image=False, device="cpu",
                           part="1/2", variant="no_adversary")
    R.stage_fits(spec, args, adata)
    R.stage_reads(spec, args, adata)
    fit = tmp_path / "variants" / "no_adversary" / "fits" / "k0.1_s1"
    assert not (tmp_path / "variants" / "no_adversary" / "fits" / "k0.1_s0").exists()
    reads = json.loads((fit / "reads.json").read_text())
    assert json.loads((fit / "config.json").read_text())["alpha_a"] == 0.0
    assert reads["nmi_z_targets"] == pytest.approx(reads["fit"]["nmi_targets"], abs=1e-12)
    assert 0.0 <= reads["nmi_w_targets"] <= 1.0
    assert set(reads["probe"]) >= {"ridge", "mlp"} and (fit / "probe.json").exists()
    # resumable: nothing is refitted or reread
    monkeypatch.setattr(R, "fit_reads", lambda *a: pytest.fail("reread"))
    R.stage_reads(spec, args, adata)
    timing = R.stage_timing(spec, SimpleNamespace(**{**vars(args), "variant": "timing"}),
                            adata)
    assert timing["n_epochs"] == 2 and len(timing["epoch_s"]) == 2 and timing["n_params"] > 0
    assert (tmp_path / "variants" / "timing" / "timing.json").exists()
