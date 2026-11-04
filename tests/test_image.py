"""The image extra on CPU: crop geometry, the ego disc and the embed plumbing,
with an in-memory image and a stub encoder (no weights, no tifffile)."""

from __future__ import annotations

import numpy as np
import pytest

MPP = 0.2125


def ramp_image(height=1200, width=1400, channels=4):
    """uint16 image whose value encodes the position: ch0 = x, ch1 = y, ch2 = 1000, ch3 = c."""
    y, x = np.mgrid[:height, :width]
    img = np.zeros((height, width, channels), dtype=np.uint16)
    img[..., 0] = x
    img[..., 1] = y
    img[..., 2] = 1000
    img[..., 3] = 7
    return img


def test_ego_disc_geometry():
    from discell.image import ego_disc

    keep = ego_disc(256, 25.0, 128.0)            # 0.5 um/px -> radius 50 px
    assert keep.shape == (256, 256)
    hole = ~keep
    # the hole is round and centered: area pi r^2 within 1%, symmetric
    assert abs(hole.sum() - np.pi * 50**2) / (np.pi * 50**2) < 0.01
    assert (hole == hole[::-1, :]).all() and (hole == hole[:, ::-1]).all()
    assert hole[127, 127] and hole[128, 128]
    assert not hole[127, 127 + 51] and hole[127, 127 + 49]
    assert keep[0, 0] and keep[255, 255]


def test_crop_size_centre_and_resampling():
    from discell.image import iter_crop_blocks

    img = ramp_image()
    centers = np.array([[700.3, 600.8], [650.0, 640.0]])
    out = dict()
    for idx, crops in iter_crop_blocks(img, centers, MPP, half_um=64.0, out_size=None,
                                       mask="none"):
        for k, c in zip(idx, crops):
            out[int(k)] = c
    size = int(round(128.0 / MPP))               # 602 level-0 pixels
    c0 = out[0]
    assert c0.shape == (size, size, 4) and c0.dtype == np.uint16
    x0 = int(round(700.3 - size / 2))
    y0 = int(round(600.8 - size / 2))
    assert c0[0, 0, 0] == x0 and c0[0, 0, 1] == y0
    assert c0[-1, -1, 0] == x0 + size - 1

    for _idx, crops in iter_crop_blocks(img, centers, MPP, half_um=64.0, out_size=256,
                                        mask="none"):
        assert crops.shape[1:] == (256, 256, 4)
        # bilinear resampling of a ramp stays a ramp spanning the same field
        assert abs(int(crops[0, 0, 0, 0]) - x0) <= 1
        assert abs(int(crops[0, 0, -1, 0]) - (x0 + size - 1)) <= 1


def test_edge_crops_are_zero_padded_with_constant_shape():
    from discell.image import iter_crop_blocks

    img = ramp_image()
    img[..., 2] = 1000
    centers = np.array([[10.0, 10.0], [1395.0, 1195.0]])
    for idx, crops in iter_crop_blocks(img, centers, MPP, half_um=64.0, out_size=None,
                                       mask="none"):
        for k, c in zip(idx, crops):
            assert c.shape == (602, 602, 4)
            if k == 0:      # runs off the top-left: padded rows/cols are zero
                assert (c[:250, :, 2] == 0).all() and (c[-1, -1, 2] == 1000)
            else:           # bottom-right
                assert (c[-250:, :, 2] == 0).all() and (c[0, 0, 2] == 1000)


def test_ego_mask_zeroes_the_disc_only():
    from discell.image import ego_disc, iter_crop_blocks

    img = ramp_image()
    centers = np.array([[700.0, 600.0]])
    (_, crops), = list(iter_crop_blocks(img, centers, MPP, half_um=64.0, out_size=256,
                                        mask="ego", mask_radius_um=25.0))
    crop = crops[0]
    hole = ~ego_disc(256, 25.0, 128.0)
    assert (crop[hole] == 0).all()
    assert (crop[~hole][:, 2] == 1000).all()


def test_blocks_cover_every_cell_once():
    from discell.image import iter_crop_blocks

    img = ramp_image(3000, 3000)
    rng = np.random.default_rng(0)
    centers = rng.uniform(0, 3000, (300, 2))
    seen = np.concatenate([idx for idx, _ in iter_crop_blocks(
        img, centers, MPP, half_um=20.0, out_size=32, block_px=512, mask="none")])
    assert sorted(seen.tolist()) == list(range(300))


def _adata(n=40, seed=0):
    import anndata as ad

    rng = np.random.default_rng(seed)
    pos_px = rng.uniform(350, 850, (n, 2))      # windows inside the image
    a = ad.AnnData(X=np.zeros((n, 3), dtype=np.float32))
    a.obsm["spatial_px"] = pos_px
    a.obsm["spatial"] = pos_px * MPP
    a.obs["discell_covering_radius_um"] = rng.uniform(3, 15, n)
    a.obs.iloc[3, a.obs.columns.get_loc("discell_covering_radius_um")] = 30.0
    a.uns["discell"] = {"xenium": {"pixel_size_um": MPP}}
    return a


def stub(array):
    """Per-channel mean and the pixel at the center and at the corner of channel 0."""
    assert array.dtype == np.float32 and array.shape[1:] == (4, 256, 256)
    return np.concatenate([array.mean(axis=(2, 3)), array[:, :1, 128, 128],
                           array[:, :1, 0, 0]], axis=1)


def test_embed_with_a_stub_encoder_follows_the_recipe():
    from discell.image import embed

    adata = _adata()
    out = embed(adata, ramp_image(), encoder=stub, batch_size=7)
    assert out.shape == (40, 6) and out.dtype == np.float32
    assert adata.obsm["X_kronos_masked"] is out
    # the too-large cell (covering radius 30 > 25 um) is embedded by default
    assert adata.obs["X_kronos_masked_embedded"].all() and (out[3] != 0).any()
    assert adata.uns["discell"]["image"]["large_cells"] == "embed"
    # disc center zeroed; uint16 scaled by 65535
    assert (out[:, 4] == 0).all()
    corner_x = np.round(adata.obsm["spatial_px"][:, 0] - 301) / 65535
    np.testing.assert_allclose(out[np.arange(40) != 3, 5], corner_x[np.arange(40) != 3],
                               atol=2 / 65535)
    rec = adata.uns["discell"]["image"]
    assert rec["patch_px"] == 256 and rec["target_mpp"] == 0.5 and rec["field_um"] == 128.0
    assert rec["mask"] == "ego" and rec["mask_radius_um"] == 25.0
    assert rec["channels"] == [0, 1, 2, 3]


def test_embed_rows_follow_cells_not_block_order():
    from discell.image import embed

    adata = _adata(seed=1)
    full = embed(adata, ramp_image(), encoder=stub, mask="none", key="X_full")
    part = embed(adata, ramp_image(), encoder=stub, mask="none", cells=[5, 1, 9], key="X_p")
    np.testing.assert_array_equal(part[[1, 5, 9]], full[[1, 5, 9]])
    assert (np.delete(part, [1, 5, 9], axis=0) == 0).all()


def test_defaults_are_the_papers_recipe():
    import inspect

    from discell.image import XENIUM_MARKER_NAMES, embed

    sig = inspect.signature(embed).parameters
    assert sig["encoder"].default == "kronos2"
    assert tuple(sig["channels"].default) == (0, 1, 2, 3)
    assert sig["patch_px"].default == 256 and sig["target_mpp"].default == 0.5
    assert sig["mask"].default == "ego" and sig["mask_radius_um"].default == 25.0
    assert list(XENIUM_MARKER_NAMES.values()) == ["DAPI", "NAKATP", "18S", "A-SMA"]


def test_intensity_scaling():
    from discell.image import intensity_scaling_factor

    assert intensity_scaling_factor(np.uint16) == 65535.0
    assert intensity_scaling_factor(np.uint8) == 255.0
    assert intensity_scaling_factor(np.float32) == 400.0


def test_without_the_kronos2_extra_the_message_explains_access(monkeypatch):
    import importlib.util

    from discell.image import kronos2

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "transformers" else real(name, *a))
    with pytest.raises(ImportError, match="not installed") as info:
        kronos2.require_kronos2()
    message = str(info.value)
    assert "transformers" in message.split("not installed")[0]
    assert "huggingface.co/MahmoodLab/KRONOS2" in message
    assert "closed to new access requests" in message and "not supported" in message
    with pytest.raises(ImportError, match="MahmoodLab/KRONOS2"):
        from discell.image import embed

        embed(_adata(), ramp_image())               # the default encoder is KRONOS2


def test_unknown_encoder_and_kronos_v1_are_refused():
    from discell.image import embed

    for name in ("kronos3", "kronos-v1"):
        with pytest.raises(ValueError, match="use 'kronos2'"):
            embed(_adata(), ramp_image(), encoder=name)


class FakeKronos2:
    """Stands in for the hub model: records what it is given, z-scores like KRONOS2."""

    def __init__(self):
        self.registered = None
        self.calls = []

    def register_additional_markers(self, path):
        import pandas as pd

        self.registered = pd.read_csv(path)

    def preprocess(self, patches, names, preferred_dapi=None):
        assert preferred_dapi == "DAPI"
        self.calls.append(list(names))
        return (np.asarray(patches, dtype=np.float32) - 0.5) / 2.0

    def __call__(self, batch, names):
        import torch

        assert isinstance(batch, torch.Tensor) and batch.dtype == torch.float32
        means = batch.mean(dim=(2, 3))                            # (B, 4)
        center = batch[:, :1, 128, 128]
        return torch.cat([means, center, torch.zeros(len(batch), 763)], dim=1)


def test_kronos2_path_crops_names_and_normalization(monkeypatch):
    """The KRONOS2 recipe with a stub model: 768-d, markers by name, 18S registered
    with statistics of native-resolution unmasked crops, the ego disc zeroed."""
    from discell.image import embed, kronos2

    fake = FakeKronos2()
    seen = {}

    def load(hf_token, cache_dir, checkpoint, revision, device):
        seen.update(revision=revision, device=device)
        return fake

    monkeypatch.setattr(kronos2, "load_kronos2", load)
    adata = _adata()
    img = ramp_image()
    img[..., 2] = 13107                                          # 18S = 0.2 after /65535
    out = embed(adata, img, device="cpu", batch_size=16)
    assert out.shape == (40, 768) and out.dtype == np.float32
    assert seen == {"revision": kronos2.KRONOS2_REVISION, "device": "cpu"}
    # markers by name, every batch; 18S registered as a novel marker with its statistics
    assert all(c == ["DAPI", "NAKATP", "18S", "A-SMA"] for c in fake.calls)
    row = fake.registered.iloc[0]
    assert len(fake.registered) == 1 and row["marker_name"] == "18S"
    assert (row["compartment"], row["family"], row["pretraining"]) == ("cytoplasm", "qc_mask",
                                                                       "no")
    np.testing.assert_allclose([row["mean"], row["std"]], [13107 / 65535, 0.0], atol=1e-6)
    # measured unmasked: the disc would pull the mean below 0.2
    rec = adata.uns["discell"]["image"]
    np.testing.assert_allclose(rec["novel_marker_stats"]["18S"], [row["mean"], row["std"]])
    assert rec["model"] == "kronos2"
    assert rec["marker_names"] == ["DAPI", "NAKATP", "18S", "A-SMA"]
    assert rec["revision"] == kronos2.KRONOS2_REVISION
    assert rec["field_um"] == 128.0 and rec["mask_radius_um"] == 25.0
    # crops: 128 um field at 0.5 um/px, disc zeroed, scaled to [0, 1], then preprocess
    hole = ~__import__("discell.image", fromlist=["ego_disc"]).ego_disc(256, 25.0, 128.0)
    frac = hole.mean()
    np.testing.assert_allclose(out[:, 2], ((1 - frac) * 0.2 + frac * 0.0 - 0.5) / 2, atol=1e-5)
    np.testing.assert_allclose(out[:, 4], (0.0 - 0.5) / 2)       # center pixel: in the disc
    assert (out[:, 5:] == 0).all()


def test_kronos2_novel_marker_stats_follow_the_research_sampling():
    """512 cells drawn with seed 0 over all cells, native 256 px windows, unmasked."""
    from discell.image import kronos2

    img = ramp_image(3000, 3000)
    rng = np.random.default_rng(1)
    centers = rng.uniform(400, 2600, (700, 2))
    stats = kronos2.novel_marker_stats(img, centers, MPP, [0, 1, 2, 3],
                                       ["DAPI", "NAKATP", "18S", "A-SMA"])
    assert list(stats) == ["18S"]
    picked = np.sort(np.random.default_rng(0).choice(700, 512, replace=False))
    assert abs(stats["18S"][0] - 1000 / 65535) < 1e-6
    # channel 0 (x ramp) through the same sampler gives the mean of the picked windows
    x0 = np.round(centers[picked, 0] - 128).astype(int)
    expect = (x0 + 127.5).mean() / 65535
    got = kronos2.novel_marker_stats(img[..., [2, 1, 0, 3]], centers, MPP, [2], ["18S"])
    assert abs(got["18S"][0] - expect) < 1e-6
    assert kronos2.novel_marker_stats(img, centers, MPP, [0], ["DAPI"]) == {}
    with pytest.raises(ValueError, match="unknown channel"):
        kronos2.marker_names([0, 4])


def test_large_cells_zero_gives_the_too_large_cell_a_zero_row():
    from discell.image import embed

    adata = _adata()
    out = embed(adata, ramp_image(), encoder=stub, large_cells="zero")
    assert (out[3] == 0).all() and not adata.obs["X_kronos_masked_embedded"].iloc[3]
    assert adata.obs["X_kronos_masked_embedded"].sum() == 39
    assert adata.uns["discell"]["image"]["large_cells"] == "zero"
    with pytest.raises(ValueError, match="large_cells"):
        embed(adata, ramp_image(), encoder=stub, large_cells="drop")
