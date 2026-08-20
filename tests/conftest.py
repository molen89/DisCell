"""A small real bundle, built once per session into a throwaway data root.

The fixtures run the actual preprocessing entry point rather than hand-building
an AnnData, so the tests cover the pipeline that produces the artefacts as well
as the loader that consumes them. ``DISCELL_DATA`` is redirected before any
discell module is imported, so nothing touches the developer's own tree.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

#: Slide the fixtures are cut from. Skipped, not failed, when it is absent --
#: the raw Xenium outputs are tens of GB and are not in the repo.
SAMPLE = Path(__file__).resolve().parent.parent / (
    "data/interim/xenium/Xenium_Prime_Ovarian_Cancer_FFPE")

#: Enough cells for a connected graph with several cell types, small enough that
#: the whole suite runs in well under a minute.
N_CELLS = 1500
VARIANT = "test"


@pytest.fixture(scope="session")
def data_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("discell-data")
    os.environ["DISCELL_DATA"] = str(root)
    return root


@pytest.fixture(scope="session")
def bundle(data_root):
    """``(dataset, variant)`` for a freshly built bundle."""
    if not SAMPLE.exists():
        pytest.skip(f"no sample slide at {SAMPLE}")

    from discell import paths
    from discell.preprocess.main import main

    # Only the bundle: the embedding stage needs gated KRONOS weights, and
    # nothing in the batch contract depends on the vectors being real.
    assert main(["--sample", str(SAMPLE), "--variant", VARIANT, "--only", "bundle",
                 "--max-cells", str(N_CELLS), "--quiet"]) == 0
    return paths.dataset(SAMPLE), VARIANT


@pytest.fixture(scope="session")
def dataset(bundle):
    """An opened :class:`CellGraphDataset` over that bundle."""
    from discell.data.loader import CellGraphDataset

    ds, variant = bundle
    return CellGraphDataset.from_dataset(ds, variant, batch_cells=16)


@pytest.fixture(scope="session")
def batch(dataset):
    return next(iter(dataset.batches(shuffle=False)))
