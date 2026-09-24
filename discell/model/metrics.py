#!/usr/bin/env python3
"""The diagnostics of spec 7.10, each as a function of collected arrays.

Everything here consumes plain numpy -- the trainer collects ``mu_z``, ``mu_w``,
``c`` per cell and hands them over -- so each diagnostic can be tested on
synthetic inputs with a known answer, and none can silently read training state.

The probe (spec 4.6) is deliberately a *fresh* model fitted here, never the
training-time covariance tracker or adversary: an invariance check evaluated
with the machinery being checked would be circular.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("discell.model.metrics")

#: Cells used for k-means / probe fits; more adds cost, not information.
MAX_EVAL_CELLS = 30_000


def _subsample(n: int, limit: int, rng) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    return np.sort(rng.choice(n, limit, replace=False))


def _subsample_rows(rows: np.ndarray, rng, limit: int = MAX_EVAL_CELLS) -> np.ndarray:
    return rows if len(rows) <= limit else np.sort(rng.choice(rows, limit, replace=False))


def z_type_nmi(z: np.ndarray, t: np.ndarray, seed: int = 0) -> float:
    """k-means over ``z`` against the labels -- the hard floor of spec 7.10.

    The CSVAE failure mode is this number collapsing while the invariance
    penalty looks great; it is half of the joint early-stopping criterion.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score

    rng = np.random.default_rng(seed)
    rows = _subsample(len(z), MAX_EVAL_CELLS, rng)
    k = int(t.max()) + 1
    labels = KMeans(k, n_init=4, random_state=seed).fit_predict(z[rows])
    return float(normalized_mutual_info_score(t[rows], labels))


def mirror_r2(z: np.ndarray, c: np.ndarray, t: np.ndarray,
              seed: int = 0) -> dict:
    """Within-type ``R^2`` of ``z`` on ``c``, against a within-type permuted control.

    The mirror attractor (spec 7.2): attention reconstructing the cell through
    similar neighbours shows up as ``c`` predicting ``z``. **Within type**
    (spec 7.10 as patched): the pooled regression partly measures type
    separability, which ``z`` and ``c`` both carry legitimately -- both are
    residualised against their per-type means, and the control permutes the
    residualised ``c`` within each type.
    """
    rng = np.random.default_rng(seed)
    rows = _subsample(len(z), MAX_EVAL_CELLS, rng)
    z_s, c_s, t_s = z[rows].copy(), c[rows].copy(), t[rows]
    permuted = np.empty_like(c_s)
    for k in np.unique(t_s):
        members = np.flatnonzero(t_s == k)
        z_s[members] -= z_s[members].mean(axis=0)
        c_s[members] -= c_s[members].mean(axis=0)
        permuted[members] = c_s[members[rng.permutation(len(members))]]

    def fit_r2(design: np.ndarray) -> float:
        design = np.hstack([design, np.ones((len(design), 1))])
        coef, *_ = np.linalg.lstsq(design, z_s, rcond=None)
        return float(1.0 - (z_s - design @ coef).var() / z_s.var())

    return {"r2": fit_r2(c_s), "r2_permuted": fit_r2(permuted)}


def probe_delta_ce(z: np.ndarray, t: np.ndarray, v: np.ndarray,
                   vbar_t: np.ndarray, train: np.ndarray, test: np.ndarray,
                   seed: int = 0) -> dict:
    """Excess held-out skill of ``(z, t)`` at predicting the niche block ``v``.

    ``dCE = CE(v, vbar(t)) - CE(v, probe(z, t))`` on held-out cells, where the
    probe is a fresh ridge regression and CE is Gaussian (mean squared error):
    the niche block here is continuous ([y minus one column, PCs of Phi]), so
    squared error is its cross-entropy up to constants. Two reference scales
    (spec 4.6): the noise floor (z permuted within type; should be ~0) tells
    you what zero looks like, and the run at ``alpha_a = 0`` is the
    uncontrolled baseline to compare against.
    """
    rng = np.random.default_rng(seed)

    def ridge(design: np.ndarray, target: np.ndarray, rows: np.ndarray):
        d = np.hstack([design[rows], np.ones((len(rows), 1))])
        gram = d.T @ d + 1e-3 * np.eye(d.shape[1])
        return np.linalg.solve(gram, d.T @ target[rows])

    def mse(design: np.ndarray, coef: np.ndarray, rows: np.ndarray) -> float:
        d = np.hstack([design[rows], np.ones((len(rows), 1))])
        return float(((v[rows] - d @ coef) ** 2).mean())

    onehot = np.eye(int(t.max()) + 1, dtype=np.float64)[t]
    design = np.hstack([z, onehot])
    z_permuted = z.copy()
    for k in np.unique(t):                        # permute z within type
        rows = np.flatnonzero(t == k)
        z_permuted[rows] = z[rows[rng.permutation(len(rows))]]
    design_floor = np.hstack([z_permuted, onehot])

    # random subsample: the first rows follow tile order, i.e. space
    train = _subsample_rows(np.flatnonzero(train), rng)
    test_rows = _subsample_rows(np.flatnonzero(test), rng)

    baseline = float(((v[test_rows] - vbar_t[t[test_rows]]) ** 2).mean())
    with_z = mse(design, ridge(design, v, train), test_rows)
    floor = mse(design_floor, ridge(design_floor, v, train), test_rows)
    return {"delta_ce": baseline - with_z,
            "noise_floor": baseline - floor,
            "baseline_ce": baseline}


# -- the probe per block (review R20 + R22, devlog 2026-09-24 15:00) ---------

#: a block passes when |gain - floor mean| <= this many floor sd
PROBE_BAND_SD = 2.0
#: keeps the log ratio finite; negligible against any graded column's MSE
_MSE_EPS = 1e-12
#: a column whose held-out variance is at most this is constant on the
#: held-out cells (a type that is no held-out cell's neighbour): it has
#: nothing to grade, and its gain would be a ratio of rounding errors
#: (-0.3 nats on two such columns of the GSE dual section)
_CONSTANT_VAR = 1e-12


def _permute_within_type(z: np.ndarray, t: np.ndarray, rng) -> np.ndarray:
    out = z.copy()
    for k in np.unique(t):
        rows = np.flatnonzero(t == k)
        out[rows] = z[rows[rng.permutation(len(rows))]]
    return out


def _ridge_fit_predict(train_design: np.ndarray, train_target: np.ndarray,
                       test_design: np.ndarray) -> np.ndarray:
    """The ridge of :func:`probe_delta_ce` (intercept, penalty 1e-3)."""
    d = np.hstack([train_design, np.ones((len(train_design), 1))])
    gram = d.T @ d + 1e-3 * np.eye(d.shape[1])
    coef = np.linalg.solve(gram, d.T @ train_target)
    return np.hstack([test_design, np.ones((len(test_design), 1))]) @ coef


def _probe_columns(z, t, v, vbar_t, train, test, seed, n_perm, fit_predict):
    """Held-out MSE per column of v: type-mean baseline, probe on (z, t), and
    the probe on *n_perm* within-type permutations of z.

    The random draws come in :func:`probe_delta_ce`'s order (first
    permutation, training subsample, test subsample), then the remaining
    permutations, so permutation 0 and the split are the legacy probe's.
    """
    rng = np.random.default_rng(seed)
    permuted = [_permute_within_type(z, t, rng)]
    train_rows = _subsample_rows(np.flatnonzero(train), rng)
    test_rows = _subsample_rows(np.flatnonzero(test), rng)
    permuted += [_permute_within_type(z, t, rng) for _ in range(n_perm - 1)]
    onehot = np.eye(int(t.max()) + 1, dtype=np.float64)
    v_test = v[test_rows]

    def fit(latent: np.ndarray) -> tuple[np.ndarray, float]:
        design = lambda rows: np.hstack([latent[rows], onehot[t[rows]]])
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
            "floor": [f[0] for f in floors],
            "floor_pooled": [f[1] for f in floors],
            "n_train": int(len(train_rows)), "n_test": int(len(test_rows))}


def _block(base: np.ndarray, probe: np.ndarray, floors: list,
           cols: np.ndarray) -> dict:
    """Mean per-column gain of one block against its permutation floor."""
    def gain(mse: np.ndarray) -> np.ndarray:
        return 0.5 * np.log((base[cols] + _MSE_EPS) / (mse[cols] + _MSE_EPS))

    per_col = gain(probe)
    draws = [float(gain(f).mean()) for f in floors]
    mean, sd = float(np.mean(draws)), float(np.std(draws, ddof=1))
    excess = float(per_col.mean()) - mean
    return {"gain": float(per_col.mean()), "floor_mean": mean, "floor_sd": sd,
            "excess": excess,
            "excess_in_sd": excess / sd if sd > 0 else float("inf"),
            "pass": bool(abs(excess) <= PROBE_BAND_SD * sd),
            "n_cols": int(len(cols)), "gain_per_col": per_col.tolist(),
            "floor_draws": draws}


def _blocks(cols: dict, n_comp: int, n_perm: int, seed: int, family: str,
            legacy_scale: np.ndarray | None = None) -> dict:
    """Assemble the per-block record from :func:`_probe_columns` output.

    *legacy_scale* (per-column variances) converts standardised MSEs back to
    the original units for the pooled legacy dCE.
    """
    n_cols = len(cols["baseline"])
    out = {"family": family, "n_comp": int(n_comp), "n_img": n_cols - n_comp,
           "n_perm": int(n_perm), "seed": int(seed),
           "n_train": cols["n_train"], "n_test": cols["n_test"]}
    for name, sel in (("comp", np.arange(n_comp)),
                      ("img", np.arange(n_comp, n_cols)),
                      ("pooled", np.arange(n_cols))):
        graded = sel[cols["graded"][sel]]
        out[name] = _block(cols["baseline"], cols["probe"], cols["floor"],
                           graded)
        out[name]["constant_cols"] = sel[~cols["graded"][sel]].tolist()
    if legacy_scale is None:
        base, probe, floor = (cols["baseline_ce"], cols["probe_pooled"],
                              cols["floor_pooled"][0])
    else:
        unscale = lambda mse: float(np.mean(mse * legacy_scale))
        base, probe, floor = (unscale(cols["baseline"]), unscale(cols["probe"]),
                              unscale(cols["floor"][0]))
    out["legacy"] = {"delta_ce": base - probe, "noise_floor": base - floor,
                     "baseline_ce": base}
    return out


def _check_blocks(v: np.ndarray, n_comp: int, n_perm: int) -> None:
    if not 0 < n_comp < v.shape[1]:
        raise ValueError(f"n_comp={n_comp} must split v's {v.shape[1]} columns")
    if n_perm < 2:
        raise ValueError("a floor sd needs n_perm >= 2")


def probe_gain_per_block(z: np.ndarray, t: np.ndarray, v: np.ndarray,
                         vbar_t: np.ndarray, train: np.ndarray,
                         test: np.ndarray, n_comp: int, seed: int = 0,
                         n_perm: int = 5) -> dict:
    """The ridge probe of :func:`probe_delta_ce`, graded per column and block.

    Per column k of v, ``gain_k = 1/2 log(MSE_k(type mean) / MSE_k(probe))``
    on held-out cells -- the Gaussian cross-entropy gain with fitted
    variances, so a column's scale cancels. The composition block is the
    first *n_comp* columns (K-1: y minus one column), the image block the
    rest (PCs of Phi). Each block reports its mean gain, the mean and sample
    sd of the same over *n_perm* within-type permutations of z (the floor),
    ``excess`` = gain - floor mean, and ``pass`` = |excess| <= 2 floor sd.
    Columns constant on the held-out cells are left out of the block means
    (listed in ``constant_cols``). ``pooled`` is the same over all columns;
    ``legacy`` is
    :func:`probe_delta_ce` itself (pooled squared error), reproduced exactly.
    """
    _check_blocks(v, n_comp, n_perm)
    cols = _probe_columns(z, t, v, vbar_t, train, test, seed, n_perm,
                          _ridge_fit_predict)
    return _blocks(cols, n_comp, n_perm, seed, "ridge")


def _standardise(values: np.ndarray, rows: np.ndarray):
    """Column mean and sd over *rows*; a constant column keeps sd 1."""
    mean = values[rows].mean(axis=0)
    sd = values[rows].std(axis=0)
    return mean, np.where(sd > 0, sd, 1.0)


def probe_gain_per_block_mlp(z: np.ndarray, t: np.ndarray, v: np.ndarray,
                             vbar_t: np.ndarray, train: np.ndarray,
                             test: np.ndarray, n_comp: int, seed: int = 0,
                             n_perm: int = 5) -> dict:
    """:func:`probe_gain_per_block` with the calibration MLP as the grader.

    Same split, subsample, seed and permutations as the ridge; the network is
    :func:`discell.model.calibrate.mlp_fit_predict` (64->64, 300 Adam steps),
    fitted with the same torch seed for the probe and every floor draw.
    Three changes from the calibration probe, so that neither a column's
    scale, a latent's scale, nor the other block can decide a block's grade:

    * z's and v's columns are standardised by their training-cell mean and
      sd before the fit (a pooled squared-error loss would otherwise spend
      the network on the image PCs -- the R20 defect again);
    * one network **per block**: with shared hidden layers, fitting
      composition signal moved the image outputs off their floor (planted
      composition-only z: image excess -2.2 floor sd), and the floor, which
      permutes z whole, cannot reproduce that.

    Gains are ratios, so standardising leaves them unchanged; the legacy
    dCE is reported in v's original units.
    """
    from discell.model.calibrate import mlp_fit_predict

    _check_blocks(v, n_comp, n_perm)
    train_rows = np.flatnonzero(train)
    z_mean, z_sd = _standardise(np.asarray(z, dtype=np.float64), train_rows)
    v_mean, v_sd = _standardise(np.asarray(v, dtype=np.float64), train_rows)
    z_std = (z - z_mean) / z_sd
    v_std = (v - v_mean) / v_sd
    vbar_std = (vbar_t - v_mean) / v_sd

    def fit(d_train, v_train, d_test):
        design = np.vstack([d_train, d_test])
        rows = np.arange(len(d_train))
        test_rows = np.arange(len(d_train), len(design))
        return np.hstack([mlp_fit_predict(design, v_train[:, block], rows,
                                          test_rows, seed)
                          for block in (slice(0, n_comp),
                                        slice(n_comp, None))])

    cols = _probe_columns(z_std, t, v_std, vbar_std, train, test, seed,
                          n_perm, fit)
    return _blocks(cols, n_comp, n_perm, seed, "mlp", legacy_scale=v_sd ** 2)


def probe_blocks(z: np.ndarray, t: np.ndarray, v: np.ndarray,
                 vbar_t: np.ndarray, train: np.ndarray, test: np.ndarray,
                 n_comp: int, seed: int = 0, n_perm: int = 5) -> dict:
    """Both graders per block, and the pre-registered invariance guard:
    ``invariance_pass`` iff the composition and the image block pass for
    both the ridge and the MLP."""
    out = {family: fn(z, t, v, vbar_t, train, test, n_comp, seed=seed,
                      n_perm=n_perm)
           for family, fn in (("ridge", probe_gain_per_block),
                              ("mlp", probe_gain_per_block_mlp))}
    out["pass"] = {f"{family}_{block}": out[family][block]["pass"]
                   for family in ("ridge", "mlp") for block in ("comp", "img")}
    out["invariance_pass"] = bool(all(out["pass"].values()))
    return out


def w_mirror_delta_r2(mu_w: np.ndarray, neighbour_z: np.ndarray,
                      y: np.ndarray, t: np.ndarray, train: np.ndarray,
                      test: np.ndarray, seed: int = 0) -> dict:
    """The w-mirror detector: neighbour *state* beyond neighbour *composition*.

    ``delta_r2 = R2(mu_w ~ [y, N]) - R2(mu_w ~ y)`` held-out, within-type
    centred, where ``N_i = sum_j beta_ij mu_z_j``. An honest w is a
    composition-level field (delta ~ 0); the mirror basin -- the prior
    reconstructing the cell from per-neighbour z detail -- reads large.
    Certified against labelled basin/honest runs before being trusted
    (devlog, 2026-09-11).
    """
    rng = np.random.default_rng(seed)
    rows_train = _subsample_rows(np.flatnonzero(train), rng)
    rows_test = _subsample_rows(np.flatnonzero(test), rng)
    keep = np.concatenate([rows_train, rows_test])
    w_c, y_c, n_c = mu_w.astype(np.float64).copy(), \
        y.astype(np.float64).copy(), neighbour_z.astype(np.float64).copy()
    for g in np.unique(t[keep]):
        members = keep[t[keep] == g]
        for block in (w_c, y_c, n_c):
            block[members] -= block[members].mean(axis=0)

    def held_out_sse(design_blocks) -> float:
        design = np.hstack([b[rows_train] for b in design_blocks]
                           + [np.ones((len(rows_train), 1))])
        gram = design.T @ design + 1e-3 * np.eye(design.shape[1])
        coef = np.linalg.solve(gram, design.T @ w_c[rows_train])
        held = np.hstack([b[rows_test] for b in design_blocks]
                         + [np.ones((len(rows_test), 1))])
        return float(((w_c[rows_test] - held @ coef) ** 2).sum())

    total = float((w_c[rows_test] ** 2).sum())
    r2_y = 1.0 - held_out_sse([y_c]) / max(total, 1e-12)
    r2_yn = 1.0 - held_out_sse([y_c, n_c]) / max(total, 1e-12)
    return {"delta_r2": r2_yn - r2_y, "r2_y": r2_y, "r2_yn": r2_yn}


def held_out_reconstruction(x: np.ndarray, log_p: np.ndarray) -> float:
    """Mean per-count log-likelihood on held-out seeds; the early-stop signal."""
    totals = x.sum(axis=1).clip(min=1.0)
    return float(((x * log_p).sum(axis=1) / totals).mean())


def attention_beta_correlation(alpha: np.ndarray, beta: np.ndarray) -> float:
    """corr(alpha_ij, beta_ij) -- only meaningful if edge features are ever
    added back (spec 7.3): high correlation means the GAT rediscovered the
    leak kernel and ``w`` is becoming collinear with ``rho_bar``."""
    if len(alpha) < 2:
        return float("nan")
    return float(np.corrcoef(alpha, beta)[0, 1])


def principal_curve(coords: np.ndarray, n_iter: int = 4,
                    resolution: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Pseudotime along a 1-D principal curve through a 2-D embedding.

    Hastie-Stuetzle by moving average: order by the first principal component,
    smooth the ordered coordinates into a polyline, project every point onto
    it, re-parameterise by arclength, iterate. Returns ``(pseudotime in [0,1],
    curve polyline)``. The direction is arbitrary -- an embedding has no
    preferred end -- so downstream reads are about *ordering*, never sign.
    """
    from scipy.spatial import cKDTree
    from sklearn.decomposition import PCA

    order_value = PCA(1).fit_transform(coords).ravel()
    curve = coords[np.argsort(order_value)][:: max(len(coords) // resolution, 1)]
    for _ in range(n_iter):
        ranked = coords[np.argsort(order_value)]
        window = max(len(coords) // 50, 15)
        pad = (window // 2, window - 1 - window // 2)
        smooth = np.stack([
            np.convolve(np.pad(ranked[:, d], pad, mode="edge"),
                        np.ones(window) / window, mode="valid")
            for d in range(coords.shape[1])], axis=1)
        keep = np.linspace(0, len(smooth) - 1,
                           min(resolution, len(smooth))).astype(int)
        curve = smooth[keep]
        arc = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(curve, axis=0), axis=1))])
        order_value = arc[cKDTree(curve).query(coords)[1]]
    span = order_value.max() - order_value.min()
    return (order_value - order_value.min()) / max(span, 1e-9), curve


def cycle_r2(latent: np.ndarray, t: np.ndarray, scores: np.ndarray,
             types: np.ndarray, train: np.ndarray, test: np.ndarray,
             seed: int = 0) -> dict:
    """Within-type ridge R^2 of continuous S/G2M scores from a latent.

    The disentanglement read: cycle is intrinsic state, so z should score well
    above the within-type-permuted control and w should sit at it -- if w
    predicts cycle, identity is leaking into the context channel. Restricted
    to *types* (the MKI67-ranked cycling ones); pooled across them with the
    per-type means removed so type identity itself carries nothing.
    """
    rng = np.random.default_rng(seed)
    keep = np.isin(t, types)
    rows_train = np.flatnonzero(train & keep)
    rows_test = np.flatnonzero(test & keep)
    if len(rows_train) < 200 or len(rows_test) < 200:
        return {"r2_pooled": float("nan"), "r2_mean_types": float("nan"),
                "r2_permuted": float("nan"), "by_type": {}}
    rows_train = _subsample_rows(rows_train, rng)
    rows_test = _subsample_rows(rows_test, rng)

    latent = latent.copy().astype(np.float64)
    target = scores.copy().astype(np.float64)
    permuted = latent.copy()
    for g in types:                       # centre per type; permute within type
        members = np.flatnonzero(t == g)
        latent[members] -= latent[members].mean(axis=0)
        target[members] -= target[members].mean(axis=0)
        permuted[members] = latent[members[rng.permutation(len(members))]]

    def fit(design_all: np.ndarray):
        design = np.hstack([design_all[rows_train],
                            np.ones((len(rows_train), 1))])
        gram = design.T @ design + 1e-3 * np.eye(design.shape[1])
        coef = np.linalg.solve(gram, design.T @ target[rows_train])
        held = np.hstack([design_all[rows_test], np.ones((len(rows_test), 1))])
        return target[rows_test] - held @ coef

    def r2(residual: np.ndarray, sel: np.ndarray) -> float:
        return float(1.0 - residual[sel].var()
                     / max(target[rows_test][sel].var(), 1e-12))

    residual = fit(latent)
    residual_permuted = fit(permuted)
    everything = np.ones(len(rows_test), dtype=bool)
    # one shared fit, evaluated per type: how well the shared cycle axis
    # transfers into each type, plus the pooled and mean-of-types summaries
    by_type = {int(g): r2(residual, t[rows_test] == g)
               for g in types if (t[rows_test] == g).sum() >= 100}
    return {"r2_pooled": r2(residual, everything),
            "r2_mean_types": float(np.mean(list(by_type.values())))
            if by_type else float("nan"),
            "r2_permuted": r2(residual_permuted, everything),
            "by_type": by_type}


def type_means(z: np.ndarray, t: np.ndarray, n_types: int) -> np.ndarray:
    """Per-type mean of ``z`` (K, d); a type with no cells gets the global mean."""
    out = np.tile(z.mean(axis=0), (n_types, 1))
    for g in np.unique(t):
        out[g] = z[t == g].mean(axis=0)
    return out.astype(z.dtype)


def type_degeneracy(z: np.ndarray, t: np.ndarray, train: np.ndarray,
                    test: np.ndarray, seed: int = 0) -> dict:
    """Is ``z`` just ``t``? (spec 7.10: ``I(z;t)/H(t), var(z|t)``).

    ``mi = h_t - ce_heldout``: ``ce_heldout`` is the held-out cross-entropy of
    a multinomial logistic regression ``mu_z -> t`` fitted on training cells,
    ``h_t`` that of the training-set type frequencies (the intercept-only
    probe), so the difference is a held-out, probe-based *lower* bound on the
    mutual information; ``mi_ratio = mi / h_t`` is in [0, 1] up to estimation
    noise. ``within_var_fraction = tr Cov(z|t) / tr Cov(z)`` (population-
    weighted pooled within-type covariance over the total, law of total
    variance) on every cell passed in -- deliberately not capped at
    ``MAX_EVAL_CELLS`` like the probe above it, since it is an exact O(n)
    variance pass with no fit, cheap even at 407k cells. Directions: within
    fraction 0 = z is a function of t (degenerate); 1 = the type means
    coincide (z blind to type, the failure NMI guards). A high ratio alone is expected -- the decoder
    has no t, so z must carry it; degenerate = ratio near 1 AND within
    fraction near 0 (and a null type-mean reconstruction gap).
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    rows_train = _subsample_rows(np.flatnonzero(train), rng)
    rows_test = _subsample_rows(np.flatnonzero(test), rng)
    k = int(t.max()) + 1
    z = z.astype(np.float64)
    scale = z[rows_train].std(axis=0) + 1e-8
    standardised = (z - z[rows_train].mean(axis=0)) / scale
    probe = LogisticRegression(C=1.0, max_iter=500).fit(
        standardised[rows_train], t[rows_train])
    log_prob = np.full((len(rows_test), k), np.log(1e-12))
    log_prob[:, probe.classes_] = np.log(
        probe.predict_proba(standardised[rows_test]).clip(1e-12))
    ce = float(-log_prob[np.arange(len(rows_test)), t[rows_test]].mean())
    freq = (np.bincount(t[rows_train], minlength=k) / len(rows_train)).clip(1e-12)
    h_t = float(-np.log(freq)[t[rows_test]].mean())
    # log_prob is k-wide (absent classes at log 1e-12), so its argmax is
    # already a type index -- never index probe.classes_ with it (a type
    # absent from the training rows made that overflow, GSE core 2026-09-17)
    accuracy = float((log_prob.argmax(axis=1) == t[rows_test]).mean())

    total = z.var(axis=0)
    within = np.zeros(z.shape[1])
    for g in np.unique(t):
        members = t == g
        within += members.mean() * z[members].var(axis=0)
    return {"mi_ratio": (h_t - ce) / h_t, "mi": h_t - ce, "h_t": h_t,
            "ce_heldout": ce, "accuracy": accuracy,
            "within_var_fraction": float(within.sum() / max(total.sum(), 1e-12)),
            "within_var_fraction_per_dim": (within / total.clip(min=1e-12)).tolist()}


def type_profile_reconstruction(x_train, t_train: np.ndarray, x_test: np.ndarray,
                                t_test: np.ndarray) -> float:
    """Held-out per-count log-likelihood of the empirical per-type profile.

    Training counts pooled per type and normalised (floor 1e-8, the EPS of
    ``leakage_mix``): what a lookup from ``t`` alone achieves with no w and
    no leak -- the reference line for the type-mean-z reconstruction gap.
    *x_train* may be sparse; *x_test* is dense.
    """
    import scipy.sparse as sp

    k = int(max(t_train.max(), t_test.max())) + 1
    onehot = sp.csr_matrix((np.ones(len(t_train)), (t_train, np.arange(len(t_train)))),
                           shape=(k, len(t_train)))
    pooled = np.asarray((onehot @ x_train).todense() if sp.issparse(x_train)
                        else onehot @ x_train, dtype=np.float64)
    log_profile = np.log((pooled / pooled.sum(axis=1, keepdims=True).clip(min=1.0)
                          ).clip(min=1e-8))
    totals = x_test.sum(axis=1).clip(min=1.0)
    per_cell = np.empty(len(t_test))
    for g in np.unique(t_test):
        members = t_test == g
        per_cell[members] = (x_test[members] @ log_profile[g]) / totals[members]
    return float(per_cell.mean())
