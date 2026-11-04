"""The plots (``discell.pl``): each draws from the stored readouts without error."""

from __future__ import annotations

import numpy as np
import pytest

import discell as dc

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def test_breakdown_trajectory():
    rng = np.random.default_rng(0)
    draws = {"finding": {k: {s: {"estimate": 1.0 - 4 * k + 0.01 * s,
                                  "draws": rng.normal(1.0 - 4 * k, 0.05, 500),
                                  "method": "percentile", "c": 1.0}
                              for s in (0, 1)} for k in (0.0, 0.1, 0.2, 0.3)}}
    table = dc.sensitivity.breakdown(draws)
    fig, ax = plt.subplots()
    out = dc.pl.breakdown_trajectory(table, "finding", ax=ax)
    assert out is ax and table["members"]["finding"]["kappa_star"] == pytest.approx(0.3)
    assert any("kappa*" in t.get_text() for t in ax.get_legend().get_texts())
    plt.close(fig)
