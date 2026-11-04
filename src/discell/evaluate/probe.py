"""The held-out probe: neighborhood information in ``z`` per block, two graders.

A grader is fitted on training-tile cells to predict the niche block
``v = [y minus its last column, PCs of Phi]`` from ``(z, t)`` and scored on the
held-out tiles. Per column ``k`` the gain is
``1/2 log(MSE_k(type mean) / MSE_k(probe))``, the Gaussian cross-entropy gain,
so a column's scale cancels. Each block (composition, image, pooled) reports
its mean gain against a floor: the same probe on within-type permutations of
``z``. ``excess = gain - floor mean`` is the neighborhood information, reported as
``var_fraction = 1 - exp(-2 excess)``: the share of the block's within-type
variance that the probe explains beyond the permutation floor (exactly so when
every column gains equally).

Graders: a ridge regression, and a small MLP (one network per block, inputs
and targets standardized on the training cells).
"""

from __future__ import annotations

import numpy as np

from discell.evaluate.mask import mask_record, metric_target_mask
from discell.evaluate.metrics import _subsample_rows

#: The withdrawn 2-sd rule, kept as ``pass_2sd_legacy``.
PROBE_BAND_SD = 2.0
#: Keeps the log ratio finite; negligible against any graded column's MSE.
_MSE_EPS = 1e-12
#: A column whose held-out variance is at most this is constant on the held-out
#: cells and is left out of the block means.
_CONSTANT_VAR = 1e-12

GUARD_BLOCKS = (("ridge", "comp"), ("ridge", "img"), ("mlp", "comp"), ("mlp", "img"))


def mlp_fit_predict(design: np.ndarray, target: np.ndarray, rows: np.ndarray,
                    test_rows: np.ndarray, seed: int = 0) -> np.ndarray:
    """The probe MLP (64-64 ReLU, 300 Adam steps of 2,048 rows at lr 1e-3,
    squared error) fitted on *rows*; its predictions on *test_rows*.

    Seeded with *seed*; the caller's global torch RNG state is left untouched.
    """
    import torch
    import torch.nn as nn

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        design = np.asarray(design, dtype=np.float32)
        net = nn.Sequential(nn.Linear(design.shape[1], 64), nn.ReLU(),
                            nn.Linear(64, 64), nn.ReLU(),
                            nn.Linear(64, target.shape[1]))
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        d_train = torch.tensor(design[rows])
        v_train = torch.tensor(target[rows], dtype=torch.float32)
        for _ in range(300):
            pick = torch.randint(0, len(rows), (2048,))
            loss = ((net(d_train[pick]) - v_train[pick]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            return net(torch.tensor(design[test_rows])).numpy()


def _permute_within_type(z: np.ndarray, t: np.ndarray, rng) -> np.ndarray:
    out = z.copy()
    for k in np.unique(t):
        rows = np.flatnonzero(t == k)
        out[rows] = z[rows[rng.permutation(len(rows))]]
    return out


def _ridge_fit_predict(train_design: np.ndarray, train_target: np.ndarray,
                       test_design: np.ndarray) -> np.ndarray:
    """Ridge with an intercept, penalty 1e-3."""
    d = np.hstack([train_design, np.ones((len(train_design), 1))])
    gram = d.T @ d + 1e-3 * np.eye(d.shape[1])
    coef = np.linalg.solve(gram, d.T @ train_target)
    return np.hstack([test_design, np.ones((len(test_design), 1))]) @ coef


def _probe_columns(z, t, v, vbar_t, train, test, seed, n_perm, fit_predict) -> dict:
    """Held-out MSE per column of v: type-mean baseline, probe on ``(z, t)``, and
    the probe on *n_perm* within-type permutations of z.

    Draw order: first permutation, training subsample, test subsample, the
    remaining permutations (permutation 0 and the split are those of
    :func:`discell.evaluate.metrics.probe_delta_ce`).
    """
    rng = np.random.default_rng(seed)
    permuted = [_permute_within_type(z, t, rng)]
    train_rows = _subsample_rows(np.flatnonzero(train), rng)
    test_rows = _subsample_rows(np.flatnonzero(test), rng)
    permuted += [_permute_within_type(z, t, rng) for _ in range(n_perm - 1)]
    onehot = np.eye(int(t.max()) + 1, dtype=np.float64)
    v_test = v[test_rows]

    def fit(latent: np.ndarray) -> tuple[np.ndarray, float]:
        def design(rows):
            return np.hstack([latent[rows], onehot[t[rows]]])

        sq = (v_test - fit_predict(design(train_rows), v[train_rows],
                                   design(test_rows))) ** 2
        return sq.mean(axis=0).astype(np.float64), float(sq.mean())

    base = (v_test - vbar_t[t[test_rows]]) ** 2
    probe, probe_pooled = fit(z)
    floors = [fit(p) for p in permuted]
    return {"graded": v_test.astype(np.float64).var(axis=0) > _CONSTANT_VAR,
            "baseline": base.mean(axis=0).astype(np.float64),
            "baseline_ce": float(base.mean()),
            "probe": probe, "probe_pooled": probe_pooled,
            "floor": [f[0] for f in floors], "floor_pooled": [f[1] for f in floors],
            "n_train": int(len(train_rows)), "n_test": int(len(test_rows))}


def _block(base: np.ndarray, probe: np.ndarray, floors: list, cols: np.ndarray) -> dict:
    """Mean per-column gain of one block against its permutation floor."""
    def gain(mse: np.ndarray) -> np.ndarray:
        return 0.5 * np.log((base[cols] + _MSE_EPS) / (mse[cols] + _MSE_EPS))

    per_col = gain(probe)
    draws = [float(gain(f).mean()) for f in floors]
    mean, sd = float(np.mean(draws)), float(np.std(draws, ddof=1))
    excess = float(per_col.mean()) - mean
    return {"gain": float(per_col.mean()), "floor_mean": mean, "floor_sd": sd,
            "excess": excess,
            # the share of within-type variance the excess implies, 1 - e^{-2 excess}
            "var_fraction": float(-np.expm1(-2.0 * excess)),
            "excess_in_sd": excess / sd if sd > 0 else float("inf"),
            "pass_2sd_legacy": bool(abs(excess) <= PROBE_BAND_SD * sd),
            "n_cols": int(len(cols)), "gain_per_col": per_col.tolist(),
            "floor_draws": draws}


def _blocks(cols: dict, n_comp: int, n_perm: int, seed: int, family: str,
            legacy_scale: np.ndarray | None = None) -> dict:
    """The per-block record. *legacy_scale* (per-column variances) converts
    standardized MSEs back to the original units for the pooled ``legacy`` dCE."""
    n_cols = len(cols["baseline"])
    out = {"family": family, "n_comp": int(n_comp), "n_img": n_cols - n_comp,
           "n_perm": int(n_perm), "seed": int(seed),
           "n_train": cols["n_train"], "n_test": cols["n_test"]}
    for name, sel in (("comp", np.arange(n_comp)), ("img", np.arange(n_comp, n_cols)),
                      ("pooled", np.arange(n_cols))):
        if not len(sel):                 # image-free v: no image block
            out[name] = None
            continue
        graded = sel[cols["graded"][sel]]
        out[name] = _block(cols["baseline"], cols["probe"], cols["floor"], graded)
        out[name]["constant_cols"] = sel[~cols["graded"][sel]].tolist()
    if legacy_scale is None:
        base, probe, floor = (cols["baseline_ce"], cols["probe_pooled"],
                              cols["floor_pooled"][0])
    else:
        def unscale(mse):
            return float(np.mean(mse * legacy_scale))

        base, probe, floor = (unscale(cols["baseline"]), unscale(cols["probe"]),
                              unscale(cols["floor"][0]))
    out["legacy"] = {"delta_ce": base - probe, "noise_floor": base - floor,
                     "baseline_ce": base}
    return out


def _check_blocks(v: np.ndarray, n_comp: int, n_perm: int) -> None:
    if not 0 < n_comp <= v.shape[1]:
        raise ValueError(f"n_comp={n_comp} must split v's {v.shape[1]} columns")
    if n_perm < 2:
        raise ValueError("a floor sd needs n_perm >= 2")


def probe_gain_per_block(z: np.ndarray, t: np.ndarray, v: np.ndarray,
                         vbar_t: np.ndarray, train: np.ndarray, test: np.ndarray,
                         n_comp: int, seed: int = 0, n_perm: int = 5) -> dict:
    """The ridge grader per column and block (see module doc).

    The composition block is the first *n_comp* columns of *v*, the image block
    the rest. ``legacy`` is :func:`discell.evaluate.metrics.probe_delta_ce`.
    """
    _check_blocks(v, n_comp, n_perm)
    cols = _probe_columns(z, t, v, vbar_t, train, test, seed, n_perm, _ridge_fit_predict)
    return _blocks(cols, n_comp, n_perm, seed, "ridge")


def _standardize(values: np.ndarray, rows: np.ndarray):
    """Column mean and sd over *rows*; a constant column keeps sd 1."""
    mean = values[rows].mean(axis=0)
    sd = values[rows].std(axis=0)
    return mean, np.where(sd > 0, sd, 1.0)


def probe_gain_per_block_mlp(z: np.ndarray, t: np.ndarray, v: np.ndarray,
                             vbar_t: np.ndarray, train: np.ndarray, test: np.ndarray,
                             n_comp: int, seed: int = 0, n_perm: int = 5) -> dict:
    """:func:`probe_gain_per_block` with the MLP grader.

    Same split, subsample, seed and permutations as the ridge. ``z`` and ``v``
    are standardized by their training-cell mean and sd, and each block gets
    its own network, so neither a column's scale nor the other block decides a
    block's grade. The ``legacy`` dCE is reported in v's original units.
    """
    _check_blocks(v, n_comp, n_perm)
    train_rows = np.flatnonzero(train)
    z_mean, z_sd = _standardize(np.asarray(z, dtype=np.float64), train_rows)
    v_mean, v_sd = _standardize(np.asarray(v, dtype=np.float64), train_rows)
    z_std = (z - z_mean) / z_sd
    v_std = (v - v_mean) / v_sd
    vbar_std = (vbar_t - v_mean) / v_sd
    blocks = ((slice(0, n_comp),) if n_comp == v.shape[1]
              else (slice(0, n_comp), slice(n_comp, None)))

    def fit(d_train, v_train, d_test):
        design = np.vstack([d_train, d_test])
        rows = np.arange(len(d_train))
        test_rows = np.arange(len(d_train), len(design))
        return np.hstack([mlp_fit_predict(design, v_train[:, block], rows, test_rows,
                                          seed) for block in blocks])

    cols = _probe_columns(z_std, t, v_std, vbar_std, train, test, seed, n_perm, fit)
    return _blocks(cols, n_comp, n_perm, seed, "mlp", legacy_scale=v_sd ** 2)


def probe_blocks(z: np.ndarray, t: np.ndarray, v: np.ndarray, vbar_t: np.ndarray,
                 train: np.ndarray, test: np.ndarray, n_comp: int, seed: int = 0,
                 n_perm: int = 5) -> dict:
    """Both graders per block.

    The pass/fail guard needs the uncontrolled (``alpha_a = 0``) reference fits
    and is not decided here; ``pass_2sd_legacy`` is the withdrawn 2-sd rule.
    """
    out = {family: fn(z, t, v, vbar_t, train, test, n_comp, seed=seed, n_perm=n_perm)
           for family, fn in (("ridge", probe_gain_per_block),
                              ("mlp", probe_gain_per_block_mlp))}
    out["pass_2sd_legacy"] = {f"{f}_{b}": out[f][b]["pass_2sd_legacy"]
                              for f, b in GUARD_BLOCKS if out[f][b] is not None}
    out["invariance_pass_2sd_legacy"] = bool(all(out["pass_2sd_legacy"].values()))
    return out


def heldout_probe(model, adata, n_perm: int = 5, *, data=None) -> dict:
    """:func:`probe_blocks` of a fitted *model* on *adata*'s held-out tiles.

    *adata* is the section the model was fitted on, or another section with the
    same type vocabulary and genes (its own tiles and split at the model's seed).
    The training tiles fit the graders, the held-out tiles grade; target cells
    only. *data*: an assembly of *adata* already made with the model's settings.
    """
    data = data if data is not None else model.assemble(adata)
    latents = model.latents(data)
    rows = latents["nodes"]
    n_train = sum(len(tile) for tile in data.train_tiles)
    train = np.arange(len(rows)) < n_train
    exclude = (model.config.unassigned_label,)
    record_mask = mask_record(data.type_names, data.t[rows], exclude)
    targets = metric_target_mask(data.t[rows], data.type_names, exclude)
    rows, train, mu_z = rows[targets], train[targets], latents["mu_z"][targets]
    record = probe_blocks(mu_z, data.t[rows], data.v_block[rows], data.vbar_t,
                          train, ~train, n_comp=data.n_comp, seed=model.config.seed,
                          n_perm=n_perm)
    record.update(n_cells=int(len(rows)), n_heldout=int((~train).sum()),
                  eval_mask=record_mask)
    return record
