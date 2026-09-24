"""The GO localisation read recovers a planted localisation (todo 6b.10)."""

import numpy as np

from discell.experiments.go_localisation import (benjamini_hochberg,
                                                 rank_enrichment)


def planted():
    """A panel of 300 genes, 60 of them 'secreted', carrying the loading."""
    rng = np.random.default_rng(0)
    genes = np.asarray([f"G{i:03d}" for i in range(300)])
    members = set(genes[:60])
    loading = rng.normal(0, 0.1, 300)
    loading[:60] += np.where(rng.random(60) < 0.5, 1.0, -1.0) * 1.5
    sets = {"extracellular": members,
            "nucleus": set(genes[60:120]),
            "small": set(genes[:3])}
    return loading, genes, np.ones(300, dtype=bool), sets


def test_planted_set_is_enriched_and_the_complement_depleted():
    loading, genes, expressed, sets = planted()
    rows = rank_enrichment(loading, genes, expressed, sets)
    benjamini_hochberg(rows)
    by = {r["set"]: r for r in rows}

    assert "small" not in by, "a set below MIN_SET panel genes is untestable"
    assert by["extracellular"]["n_panel"] == 60
    assert by["extracellular"]["effect"] > 0.8
    assert by["extracellular"]["q"] <= 0.05
    assert by["extracellular"]["significant"]
    # the planted signs are balanced, so the set is enriched in |loading|
    # while its mean signed loading stays near zero -- direction orients the
    # row, it does not gate it
    assert abs(by["extracellular"]["direction"]) < 0.5
    # the untouched set is pushed down in rank by the planted one
    assert by["nucleus"]["effect"] < 0
    assert np.isclose(by["extracellular"]["effect"],
                      2 * by["extracellular"]["auc"] - 1)


def test_no_signal_gives_no_significant_set():
    rng = np.random.default_rng(1)
    genes = np.asarray([f"G{i:03d}" for i in range(300)])
    rows = rank_enrichment(rng.normal(size=300), genes,
                           np.ones(300, dtype=bool),
                           {"extracellular": set(genes[:60]),
                            "nucleus": set(genes[60:120])})
    benjamini_hochberg(rows)
    assert not any(r["significant"] for r in rows)
