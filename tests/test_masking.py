"""The ego mask must hide the cell completely, and nothing about which cell."""

from __future__ import annotations

import numpy as np
import pytest

from discell.preprocess.crops import (
    DEFAULT_MASK_RADIUS_UM,
    _ego_disk,
    _polygon_mask,
    covering_radius_um,
)


def test_ego_disk_is_identical_for_every_cell():
    """The whole argument for a fixed disk: the hole says nothing about the cell.

    It is a pure function of geometry, so two calls with the same patch spec
    cannot differ -- which is what makes the artefact uninformative.
    """
    a = _ego_disk(256, 25.0, 128.0)
    b = _ego_disk(256, 25.0, 128.0)
    assert np.array_equal(a, b)


def test_ego_disk_area_matches_the_analytic_circle():
    keep = _ego_disk(256, 25.0, 128.0)
    assert (~keep).mean() == pytest.approx(np.pi * 25.0**2 / 128.0**2, abs=2e-3)


def test_ego_disk_is_centred_and_round():
    keep = _ego_disk(256, 25.0, 128.0)
    hidden = ~keep
    ys, xs = np.nonzero(hidden)
    assert xs.mean() == pytest.approx(127.5, abs=0.5)
    assert ys.mean() == pytest.approx(127.5, abs=0.5)
    # round, not square: the bounding box is far larger than the mask itself
    box = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
    assert hidden.sum() / box == pytest.approx(np.pi / 4, abs=0.02)


def test_covering_radius_exceeds_the_equivalent_radius(bundle):
    """Why the mask is not sized from ``equiv_diameter_um``.

    Area-derived radius underestimates anything elongated, and the elongated
    tail is exactly what decides how big the hole has to be.
    """
    from discell.data.bundle import load_bundle

    ds, variant = bundle
    adata = load_bundle(ds.bundle_dir, variant)
    covering = covering_radius_um(adata)
    equivalent = adata.obs["equiv_diameter_um"].to_numpy() / 2
    assert (covering >= equivalent - 1e-9).all()
    assert covering.max() > equivalent.max()


def test_default_radius_covers_all_but_a_handful(bundle):
    from discell.data.bundle import load_bundle

    ds, variant = bundle
    adata = load_bundle(ds.bundle_dir, variant)
    covered = covering_radius_um(adata) <= DEFAULT_MASK_RADIUS_UM
    assert covered.mean() > 0.999


def test_polygon_mask_contains_the_cell_and_little_else(bundle):

    from discell.data.bundle import load_bundle

    ds, variant = bundle
    adata = load_bundle(ds.bundle_dir, variant)
    centres = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    size, out_size = 602, 256
    for cell in (0, adata.n_obs // 2, adata.n_obs - 1):
        polygon = adata.uns["polygons"][cell]
        x0 = round(centres[cell][0] - size / 2)
        y0 = round(centres[cell][1] - size / 2)
        keep = _polygon_mask(polygon, x0, y0, size, out_size)
        assert keep.any(), "the cell vanished from its own crop"
        # the kept area should match the polygon's, within rasterisation error
        scale = out_size / size
        assert keep.sum() == pytest.approx(polygon.area * scale**2, rel=0.35)
        # and it should sit where the polygon does
        ys, xs = np.nonzero(keep)
        cx = (polygon.centroid.x - x0) * scale
        cy = (polygon.centroid.y - y0) * scale
        assert xs.mean() == pytest.approx(cx, abs=3.0)
        assert ys.mean() == pytest.approx(cy, abs=3.0)
