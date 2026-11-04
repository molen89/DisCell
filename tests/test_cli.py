"""The ``discell`` command line, end to end on CPU (fake Xenium + simulation)."""

from __future__ import annotations

import json

import anndata as ad

from discell.cli import main
from discell.sim import simulate
from test_io_xenium import write_fake_xenium

TINY = {"d_z": 4, "d_w": 2, "hidden": 32, "gat_dim": 8, "v_pcs": 4, "tile_cells": 300,
        "epochs": 4, "eval_every": 2, "w_warmup_epochs": 2, "patience": 100,
        "image_key": "X_image", "device": "cpu"}


def test_ingest_writes_the_contract(tmp_path):
    outs, _ = write_fake_xenium(tmp_path / "outs")
    out = tmp_path / "section.h5ad"
    assert main(["ingest", str(outs), "-o", str(out), "--donors", "A", "B"]) == 0
    adata = ad.read_h5ad(out)
    assert set(adata.obs["donor"].astype(str)) == {"A", "B"}
    assert adata.obsm["spatial"].shape == (adata.n_obs, 2)
    assert not any(v.startswith("NegControl") for v in adata.var_names)


def test_fit_sweep_breakdown_relocation(tmp_path, capsys):
    adata = simulate(n_cells=1200, n_genes=30, kappa=0.1, seed=0).to_anndata()
    section = tmp_path / "sim.h5ad"
    adata.write_h5ad(section)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(TINY))
    base = ["--label-key", "cell_type", "--config", str(config)]

    assert main(["fit", str(section), "-o", str(tmp_path / "one"), "--kappa", "0.2",
                 "--alpha-z", "0.01", "--write-h5ad", str(tmp_path / "fitted.h5ad"),
                 *base]) == 0
    saved = json.loads((tmp_path / "one" / "config.json").read_text())
    assert saved["kappa"] == 0.2 and saved["alpha_z"] == 0.01 and saved["epochs"] == 4
    assert "X_discell_w" in ad.read_h5ad(tmp_path / "fitted.h5ad").obsm

    fits = tmp_path / "fits"
    assert main(["sweep", str(section), "-o", str(fits), "--kappas", "0,0.2",
                 "--seeds", "0,1", *base]) == 0
    assert sorted(p.name for p in fits.iterdir()) == ["k0.2_s0", "k0.2_s1", "k0_s0", "k0_s1"]
    stamp = (fits / "k0_s0" / "model.pt").stat().st_mtime
    assert main(["sweep", str(section), "-o", str(fits), "--kappas", "0,0.2",
                 "--seeds", "0,1", *base]) == 0          # resumes: nothing refitted
    assert (fits / "k0_s0" / "model.pt").stat().st_mtime == stamp

    table = tmp_path / "breakdown.json"
    assert main(["breakdown", str(fits), str(section), "--groups", "w_mi", "--n", "20",
                 "-o", str(table)]) == 0
    result = json.loads(table.read_text())
    assert result["grid"] == [0.0, 0.2] and result["seeds"] == [0, 1]
    assert set(result["members"]) == {"niche_information_check"}
    assert (fits / "k0.2_s1" / "draws" / "w_mi.npz").exists()
    printed = capsys.readouterr().out
    assert "niche_information_check" in printed and "kappa_bar" in printed
    assert result["kappa_bar"]["source"] == "counts only"      # no fit at kappa = 0.1
    assert result["decision"]["niche_information_check"]["status"] == \
        result["members"]["niche_information_check"]["status"]
    assert main(["breakdown", str(fits), str(section), "--groups", "w_mi", "--n", "20",
                 "--operating-kappa", "0.2", "-o", str(table)]) == 0
    result = json.loads(table.read_text())
    assert result["kappa_bar"]["source"].startswith("model-based, mean over 2 fits")
    assert main(["breakdown", str(fits), str(section), "--groups", "w_mi", "--n", "20",
                 "--kappa-bar", "0.15", "-o", str(table)]) == 0
    result = json.loads(table.read_text())
    assert result["kappa_bar"] == {"value": 0.15, "source": "given"}
    assert result["decision"]["niche_information_check"]["first_tested_at_or_above"] == 0.2

    report = tmp_path / "relocation.json"
    assert main(["relocation", str(fits / "k0_s0"), str(section), "-o", str(report)]) == 0
    out = json.loads(report.read_text())
    assert {"pairs", "summary", "alignment"} <= set(out)
