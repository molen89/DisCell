"""Planted-answer tests for the shared baseline battery (todo 8.5).

The battery must give a *latent from any tool* the reads DisCell gets, so the
tests plant two latents with known content -- one that knows type and cycle,
one that knows only the niche -- and check that every column points the way
the planting says it should.
"""

from __future__ import annotations

import json

import numpy as np
import scipy.sparse as sp

from discell.experiments import baseline_battery as B


def _world(seed: int = 0, n: int = 4000, k: int = 4, d: int = 6):
    """Cells with a type, a niche, a cycle score, and counts that follow both."""
    rng = np.random.default_rng(seed)
    t = rng.integers(0, k, n)
    niche = rng.standard_normal((n, 3))
    cycle_score = rng.standard_normal(n)
    # the intrinsic latent: type identity + cycle, blind to the niche
    z = np.eye(k)[t] @ rng.standard_normal((k, d)) \
        + cycle_score[:, None] * rng.standard_normal((1, d)) \
        + 0.05 * rng.standard_normal((n, d))
    # the spatial latent: the niche, and nothing else
    s = niche @ rng.standard_normal((3, 4)) + 0.05 * rng.standard_normal((n, 4))
    c = niche @ rng.standard_normal((3, 5)) + 0.05 * rng.standard_normal((n, 5))
    v_block = niche + 0.01 * rng.standard_normal((n, 3))
    vbar_t = np.stack([v_block[t == g].mean(axis=0) for g in range(k)])
    train = rng.random(n) < 0.7
    profile = rng.random((k, 20)) + 0.1
    profile /= profile.sum(axis=1, keepdims=True)
    counts = np.stack([rng.multinomial(60, profile[g]) for g in t]).astype(float)
    cycle = {"types": np.arange(k),
             "scores": np.stack([cycle_score, cycle_score * 0.5], axis=1)}
    return dict(t=t, z=z, s=s, c=c, v_block=v_block, vbar_t=vbar_t,
                train=train, test=~train, cycle=cycle, counts=counts,
                profile=profile)


def test_intrinsic_latent_reads_as_intrinsic():
    w = _world()
    out = B.battery(w["z"], w["t"], w["train"], w["test"], w["v_block"],
                    w["vbar_t"], w["c"], spatial=w["s"], cycle=w["cycle"])
    assert out["nmi"] > 0.6                       # it knows type
    assert out["mirror"]["r2"] < 0.1              # it does not mirror context
    assert abs(out["probe"]["delta_ce"]) < 0.05   # it carries no niche
    assert out["cycle"]["z"]["r2_pooled"] > 0.5   # it carries cycle
    assert out["cycle"]["spatial"]["r2_pooled"] < 0.1   # the niche does not
    assert out["degeneracy"]["mi_ratio"] > 0.5
    assert 0.0 < out["spatial_var_fraction"] < 1.0


def test_niche_latent_fails_the_intrinsic_reads():
    """A latent that is really the niche must light up probe and mirror."""
    w = _world()
    out = B.battery(w["s"], w["t"], w["train"], w["test"], w["v_block"],
                    w["vbar_t"], w["c"], cycle=w["cycle"])
    assert out["nmi"] < 0.1
    assert out["mirror"]["r2"] > 0.8
    assert out["probe"]["delta_ce"] > 0.5
    assert out["cycle"]["z"]["r2_pooled"] < 0.1
    assert out["d_spatial"] == 0
    assert "spatial_var_fraction" not in out


def test_reconstruction_beats_the_type_profile_when_the_rate_is_right():
    w = _world()
    rows = np.flatnonzero(w["test"])[:500]
    x = sp.csr_matrix(w["counts"])
    perfect = w["profile"][w["t"][rows]] * (1 + 1e-9)
    out = B.reconstruction(x, w["t"], w["train"], rows, perfect)
    ref = B.reconstruction(x, w["t"], w["train"], rows, None)
    assert abs(out["recon"] - out["recon_type_profile"]) < 0.02
    assert "recon" not in ref                     # no decoder, no column
    worse = B.reconstruction(x, w["t"], w["train"], rows,
                             np.ones_like(perfect))
    assert worse["recon"] < out["recon"]          # a flat rate is worse


def test_latents_h5ad_round_trip_keeps_the_cell_ids(tmp_path):
    ad = __import__("anndata")
    import pandas as pd

    rows = np.array([3, 7, 11])
    obs = pd.DataFrame({"split": ["train", "val", "val"]},
                       index=[f"cell_{i}" for i in rows])
    adata = ad.AnnData(X=np.zeros((3, 1), dtype=np.float32), obs=obs)
    adata.obsm["X_simvi_intrinsic"] = np.arange(6, dtype=np.float32).reshape(3, 2)
    adata.obsm["X_simvi_spatial"] = np.zeros((3, 2), dtype=np.float32)
    path = tmp_path / "latents.h5ad"
    adata.write_h5ad(path)
    (tmp_path / "config.json").write_text(json.dumps({"tool": "SIMVI"}))

    got_rows, z, s, cfg = B.load_latents(path, "simvi")
    assert got_rows.tolist() == rows.tolist()
    assert z.shape == (3, 2) and s.shape == (3, 2)
    assert cfg["tool"] == "SIMVI"


def test_markdown_table_has_one_column_per_method():
    w = _world()
    entry = B.battery(w["z"], w["t"], w["train"], w["test"], w["v_block"],
                      w["vbar_t"], w["c"], spatial=w["s"], cycle=w["cycle"])
    entry["reconstruction"] = {"n_cells": 10, "recon_type_profile": -3.0}
    entry["config"] = {"tool": "SIMVI", "graph": "discell"}
    table = B.markdown({"DisCell/best": entry, "SIMVI": entry}, "ds", "")
    assert "| read | DisCell/best | SIMVI |" in table
    assert "no decoder" in table                  # a tool without a decoder
    assert table.count("\n| NMI(z, type) |") == 1


def test_top_decile_block_reads_the_shared_set_for_every_method():
    """With the slide's cycling set in the cycle inputs the battery adds the
    top-decile block (z, spatial, linear reference); a tool that scored a
    subset of cells is read on the same set restricted to its rows."""
    from discell.model.cell_cycle import cycling_set

    w = _world(n=20000)                # a decile of the held-out cells >= 200
    rng = np.random.default_rng(1)
    cells = cycling_set(w["cycle"]["scores"], w["test"])
    x_pcs = w["z"] + rng.standard_normal(w["z"].shape)
    cycle = {**w["cycle"], "cells": cells, "x_pcs": x_pcs}
    out = B.battery(w["z"], w["t"], w["train"], w["test"], w["v_block"],
                    w["vbar_t"], w["c"], spatial=w["s"], cycle=cycle)
    block = out["cycle_q90"]
    assert block["n_heldout"] == round(0.1 * w["test"].sum())
    assert block["z"]["r2_pooled"] > 0.5
    assert abs(block["spatial"]["r2_pooled"]) < 0.1
    assert "linear_ref" in block
    assert "cycle" in out                     # the retired read, kept
    # the subset tool: the shared set's restriction, not its own decile
    rows = np.sort(rng.choice(len(w["t"]), 12000, replace=False))
    sub = {"types": cycle["types"], "scores": cycle["scores"][rows],
           "cells": cells[rows], "x_pcs": x_pcs[rows]}
    part = B.battery(w["z"][rows], w["t"][rows], w["train"][rows],
                     w["test"][rows], w["v_block"][rows], w["vbar_t"],
                     w["c"][rows], cycle=sub)
    assert part["cycle_q90"]["n_heldout"] == (cells[rows] & w["test"][rows]).sum()
    md = B.markdown({"a": out}, "ds", "")
    assert "cycle R2, intrinsic (top-decile set)" in md
    assert "cycle R2, intrinsic (label-derived set, retired)" in md
