#!/usr/bin/env python3
"""Residual composition in z on a planted world: capacity or weighting?

Devlog 2026-09-24 16:00, "Diagnostic first". On real slides the per-block
probe finds z carrying a small composition residual beyond type. A planted
world has no biology channel -- ``z`` is independent of the niche given the
type by construction -- so whatever a fitted ``mu_z`` carries there is the
adversary's failure: capacity or weighting.

**The world.** The x~-gate world (:func:`xtilde_gate.build_world`, defaults):
6 000 cells, 8 types, kappa 0.2, a programme planted in 30 % of two types.
``z_true = m_t + 0.5 eps`` with eps independent of position, so
``z_true`` is independent of (y, phi) given t. The niche block is the one
:func:`planted.fit_synthetic` penalises, ``v = [y minus its last column,
phi]`` (K - 1 = 7 composition columns, 8 image columns); the adversary's
targets are built as :func:`prepare.assemble` builds them -- y itself, and
``e_phi`` = soft K-means (K = 8) of phi -- with phi in place of its PCs (the
world's phi is 8-dimensional; PCA without whitening preserves distances).

**Why a local fit loop.** :func:`planted.fit_synthetic` has no adversary: it
fits the closed-form covariance penalty at a hard-coded alpha_a = 0.02 and
exposes no alpha_a, head, warm-up or loss hook. :func:`fit_planted` is its
loop verbatim (model, tiles, optimiser, schedule, epochs, the world's kappa,
last-epoch read) with the invariance term replaced by the trainer's
adversary branch (``Trainer._step``) at the pinned knobs -- alpha_a 0.3,
6 head steps, head lr 2e-3, head width 64 -- plus the final configuration's
w-only warm-up (30 epochs), and 25 % of the spatial tiles held out from
training (the probe grades those). ``--check-identity`` verifies that the
loop in fit_synthetic's own mode (closed form, all tiles, no warm-up)
reproduces fit_synthetic bit for bit. Nothing in ``discell/model`` changes.

**The weighting arm** raises the composition half of the *encoder* term
three-fold, ``alpha_a * mean(3 excess_y + excess_phi)``; the heads' own loss
is untouched (the two heads are separate networks under Adam, which is
invariant to a per-parameter loss scale, so weighting their CE would do
nothing).

**Evaluated by** :func:`metrics.probe_blocks` (ridge and MLP, per block,
within-type permutation floor, n_perm = 5) on the held-out tiles, training
tiles fitting the probe -- the protocol of ``validate.probe_blocks_for_run``.
Excess is reported in nats per column, as the within-type variance fraction
exp(2 excess) - 1, and as a fraction of the same seed's alpha_a = 0 fit.
A latent block is *at floor* when |excess| <= 2 floor sd (the rule for exact
independence; with 5 floor draws a truly independent latent still falls
outside it ~14 % of the time per block). **Planted test:** the world's own
z_true must be at floor in both blocks for both graders. Guard against a
trivial pass: the within-type R^2 of z_true from mu_z, held out.

**Pre-registered reading (devlog 16:00).** Pinned fits' composition residual
at floor on >= 2 of 3 seeds for both graders -> adversary capacity is
sufficient, and the real-data residual is weighting or biology; residual
above floor -> capacity (or weighting): the x3 arm adjudicates -- composition
residual back at floor on >= 2 of 3 seeds -> weighting; lower than the pinned
fit's on every seed but still above floor -> partly weighting; not lower ->
capacity.

Usage::

    python -m discell.experiments.planted_probe --check-identity
    CUDA_VISIBLE_DEVICES=0 python -m discell.experiments.planted_probe \\
        --seeds 0 1 2 --arms uncontrolled pinned comp3
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("discell.experiments.planted_probe")

OUT_DIR = Path("scripts/logs/planted_probe_2026-09-24")
# the pinned adversary (TrainConfig defaults) and the final configuration's
# w-only warm-up (devlog 2026-09-24 10:00)
ALPHA_A = 0.3
ADV_STEPS = 6
ADV_LR = 2e-3
ADV_HIDDEN = 64
W_WARMUP = 30
# fit_synthetic's own settings, kept
EPOCHS = 400
TILE_CELLS = 512
D_Z = 8
# the probe
VAL_FRACTION = 0.25
N_PERM = 5
#: arm -> (alpha_a, composition weight in the encoder term)
ARMS = {"uncontrolled": (0.0, 1.0), "pinned": (ALPHA_A, 1.0),
        "comp3": (ALPHA_A, 3.0)}


# -- the world's targets ---------------------------------------------------

def world_targets(sim, seed: int) -> dict:
    """v, its per-type mean, and the adversary's targets, as ``assemble``
    builds them (per-type means over connected cells only)."""
    from discell.model.prepare import soft_clusters

    k = sim.n_types
    y = sim.graph.y
    v = np.hstack([y[:, :-1], sim.phi]).astype(np.float32)
    connected = sim.graph.degrees > 0
    e_phi, _ = soft_clusters(sim.phi, k, seed=seed)

    def type_mean(values):
        return np.stack([values[(sim.t == g) & connected].mean(axis=0)
                         for g in range(k)]).astype(np.float32)

    return {"v": v, "vbar_t": type_mean(v), "n_comp": k - 1,
            "e_phi": e_phi, "phibar_t": type_mean(e_phi),
            "ybar_t": sim.graph.ybar_t}


def split_tiles(sim, seed: int, val_fraction: float = VAL_FRACTION):
    """fit_synthetic's tiles, a fraction held out with assemble's split rng."""
    from discell.model.prepare import spatial_tiles

    tiles = spatial_tiles(sim.positions, TILE_CELLS)
    order = np.random.default_rng([seed, 2]).permutation(len(tiles))
    n_val = max(1, int(round(val_fraction * len(tiles))))
    return [tiles[i] for i in order[n_val:]], [tiles[i] for i in order[:n_val]]


# -- the fit ---------------------------------------------------------------

def weighted_encoder_term(heads, mu_z, t, y, e_phi, ybar_t, phibar_t,
                          comp_weight: float):
    """``elbo.adversary_terms``' encoder half with the composition excess
    scaled by *comp_weight* (1 reproduces it exactly)."""
    from discell.model.networks import soft_cross_entropy

    log_y, log_phi = heads(mu_z, t)
    base_y = soft_cross_entropy(y, ybar_t.clamp(min=1e-8).log()[t])
    base_phi = soft_cross_entropy(e_phi, phibar_t.clamp(min=1e-8).log()[t])
    excess_y = base_y - soft_cross_entropy(y, log_y)
    excess_phi = base_phi - soft_cross_entropy(e_phi, log_phi)
    return (comp_weight * excess_y + excess_phi).mean()


def fit_planted(sim, targets: dict | None, seed: int, alpha_a: float,
                comp_weight: float = 1.0, invariance: str = "adversary",
                w_warmup: int = W_WARMUP, epochs: int = EPOCHS,
                train_tiles=None, eval_tiles=None, device: str = "cuda") -> dict:
    """fit_synthetic's loop with the trainer's adversary branch.

    ``invariance="closed_form"`` with all tiles, ``alpha_a=0.02`` and
    ``w_warmup=0`` is fit_synthetic itself (``--check-identity``).
    *train_tiles* default to every tile; *eval_tiles* (posterior means) to
    the training tiles plus nothing else. Returns node-ordered ``mu_z`` for
    the evaluated cells (NaN elsewhere) and the head's last losses.
    """
    import torch

    from discell.model.elbo import Weights, adversary_terms, discell_loss
    from discell.model.equations import TypeCovariances
    from discell.model.networks import Adversary, DisCell, soft_cross_entropy
    from discell.model.prepare import spatial_tiles, tile_batch

    torch.manual_seed(seed)
    device = device if torch.cuda.is_available() else "cpu"
    genes, k = sim.x.shape[1], sim.n_types
    d_w = 2
    kappa = float(sim.kappa)
    model = DisCell(genes, k, sim.phi.shape[1],
                    median_counts=float(np.median(sim.totals)),
                    d_z=D_Z, d_w=d_w, hidden=128, gat_dim=16,
                    subtract_leak=False).to(device)
    cov = heads = head_opt = None
    if alpha_a and invariance == "closed_form":
        cov = TypeCovariances(k, D_Z, k - 1 + sim.phi.shape[1],
                              ema=0.05, min_count=100).to(device)
    elif alpha_a and invariance == "adversary":
        heads = Adversary(D_Z, k, targets["e_phi"].shape[1],
                          hidden=ADV_HIDDEN).to(device)
        head_opt = torch.optim.Adam(heads.parameters(), lr=ADV_LR)
        e_phi_all = torch.tensor(targets["e_phi"], device=device)
        ybar_t = torch.tensor(targets["ybar_t"], device=device)
        phibar_t = torch.tensor(targets["phibar_t"], device=device)
    elif alpha_a:
        raise ValueError(f"unknown invariance {invariance!r}")
    base = Weights(omega=1.0, alpha_z=0.007, alpha_w=0.1,
                   alpha_a=alpha_a if cov is not None else 0.0)
    p_t = torch.tensor(np.bincount(sim.t, minlength=k) / len(sim.t),
                       dtype=torch.float32, device=device)
    y_all = torch.tensor(sim.graph.y, dtype=torch.float32, device=device)
    phi_all = torch.tensor(sim.phi, device=device)

    def to_batch(tile):
        b = tile_batch(sim.graph, tile)
        return (b.nodes, dict(
            x=torch.tensor(sim.x[b.nodes], device=device),
            t=torch.tensor(sim.t[b.nodes], device=device),
            phi=phi_all[torch.tensor(b.nodes, device=device)],
            isolated=torch.tensor(sim.graph.isolated[b.nodes], device=device),
            gat_src=torch.tensor(b.gat_src, device=device),
            gat_dst=torch.tensor(b.gat_dst, device=device),
            leak_src=torch.tensor(b.leak_src, device=device),
            leak_dst=torch.tensor(b.leak_dst, device=device),
            leak_beta=torch.tensor(b.leak_beta, dtype=torch.float32,
                                   device=device),
            n_seeds=b.n_seeds, n_context=b.n_context))

    if train_tiles is None:
        train_tiles = spatial_tiles(sim.positions, TILE_CELLS)
    batches = [to_batch(tile) for tile in train_tiles]
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    rng = np.random.default_rng(seed)
    head_loss = torch.zeros(())
    for epoch in range(epochs):
        weights = (dataclasses.replace(
            base, alpha_w=base.alpha_w * min(1.0, epoch / w_warmup))
            if w_warmup > 0 else base)
        for j in rng.permutation(len(batches)):
            nodes, tensors = batches[j]
            fwd = model(**tensors, kappa=kappa)
            n = tensors["n_seeds"]
            seeds_t = torch.tensor(nodes[:n], device=device)
            terms = discell_loss(
                fwd, tensors["x"][:n], tensors["t"][:n], weights=weights,
                v=torch.cat([y_all[seeds_t][:, :-1], phi_all[seeds_t]], dim=-1),
                covariances=cov, p_t=p_t)
            loss = terms.loss
            if heads is not None:
                t_n, y_n, e_n = tensors["t"][:n], y_all[seeds_t], e_phi_all[seeds_t]
                if comp_weight == 1.0:
                    enc = adversary_terms(heads, fwd.mu_z[:n], t_n, y_n, e_n,
                                          ybar_t, phibar_t).encoder_term
                else:
                    enc = weighted_encoder_term(heads, fwd.mu_z[:n], t_n, y_n,
                                                e_n, ybar_t, phibar_t,
                                                comp_weight)
                loss = loss + alpha_a * enc
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimiser.step()
            if heads is not None:
                z_frozen = fwd.mu_z[:n].detach()
                for _ in range(ADV_STEPS):
                    head_opt.zero_grad()
                    log_y, log_phi = heads(z_frozen, t_n)
                    head_loss = (soft_cross_entropy(y_n, log_y)
                                 + soft_cross_entropy(e_n, log_phi)).mean()
                    head_loss.backward()
                    head_opt.step()
        schedule.step()
    model.eval()
    z_hat = np.full((len(sim.t), D_Z), np.nan, dtype=np.float32)
    eval_batches = batches if eval_tiles is None else [
        to_batch(tile) for tile in eval_tiles]
    with torch.no_grad():
        for nodes, tensors in eval_batches:
            fwd = model(**tensors, kappa=kappa, sample=False)
            s = tensors["n_seeds"]
            z_hat[nodes[:s]] = fwd.mu_z[:s].cpu().numpy()
    return {"z": z_hat, "head_loss": float(head_loss)}


# -- the grade -------------------------------------------------------------

def intrinsic_r2(latent, z_true, t, train, test) -> float:
    """Within-type held-out R^2 of the world's z from *latent* (least squares
    on type-centred values, training cells -> held-out cells): a latent that
    reads at floor by carrying nothing fails this."""
    a = np.asarray(latent, dtype=np.float64).copy()
    b = np.asarray(z_true, dtype=np.float64).copy()
    for g in np.unique(t):
        rows = t == g
        a[rows] -= a[rows & train].mean(axis=0)
        b[rows] -= b[rows & train].mean(axis=0)
    design = np.hstack([a, np.ones((len(a), 1))])
    coef, *_ = np.linalg.lstsq(design[train], b[train], rcond=None)
    resid = b[test] - design[test] @ coef
    return float(np.mean(1.0 - (resid ** 2).sum(0) / (b[test] ** 2).sum(0)))


def grade(latent, sim, targets, train_tiles, val_tiles, seed: int,
          n_perm: int = N_PERM) -> dict:
    """``metrics.probe_blocks`` in ``validate.probe_blocks_for_run``'s order:
    training-tile cells fit the probe, held-out-tile cells grade it."""
    from discell.model import metrics as M

    rows = np.concatenate(train_tiles + val_tiles)
    n_train = sum(len(tile) for tile in train_tiles)
    train = np.arange(len(rows)) < n_train
    z = np.asarray(latent, dtype=np.float64)[rows]
    assert np.isfinite(z).all(), "latent missing on graded cells"
    record = M.probe_blocks(z, sim.t[rows], targets["v"][rows],
                            targets["vbar_t"], train, ~train,
                            n_comp=targets["n_comp"], seed=seed, n_perm=n_perm)
    record.update(n_cells=int(len(rows)), n_heldout=int((~train).sum()),
                  z_true_r2=intrinsic_r2(z, sim.z_true[rows], sim.t[rows],
                                         train, ~train))
    return record


# -- the reading -----------------------------------------------------------

def at_floor(block: dict) -> bool:
    return bool(abs(block["excess"]) <= 2.0 * block["floor_sd"])


def above_floor(block: dict) -> bool:
    return bool(block["excess"] > 2.0 * block["floor_sd"])


def summarise(records: dict, seeds, out_dir: Path, suffix: str = "") -> dict:
    """The table (nats, variance fraction, floor sd, fraction of the same
    seed's uncontrolled fit) and the pre-registered reading."""
    fams, blocks = ("ridge", "mlp"), ("comp", "img")
    lines = ["| latent | seed | grader | block | excess (nats/col) | "
             "exp(2e)-1 | floor sd | excess / sd | at floor | "
             "fraction of α_a=0 | z_true R² |", "|" + "---|" * 11]
    for name in ("z_true", "uncontrolled", "pinned", "comp3"):
        for seed in seeds:
            rec = records.get(f"{name}_s{seed}")
            if rec is None:
                continue
            ref = records.get(f"uncontrolled_s{seed}")
            for f in fams:
                for b in blocks:
                    e = rec[f][b]
                    frac = ("" if ref is None or name == "uncontrolled"
                            else f"{e['excess'] / ref[f][b]['excess']:.2f}")
                    lines.append(
                        f"| {name} | {seed} | {f} | {b} | {e['excess']:+.4f} | "
                        f"{np.expm1(2 * e['excess']) * 100:.2f} % | "
                        f"{e['floor_sd']:.4f} | {e['excess_in_sd']:+.1f} | "
                        f"{'yes' if at_floor(e) else 'no'} | {frac} | "
                        f"{rec['z_true_r2']:.3f} |")

    def arm_seeds(arm):
        return [records[f"{arm}_s{s}"] for s in seeds if f"{arm}_s{s}" in records]

    verdict = {}
    truth = arm_seeds("z_true")
    verdict["planted_test"] = {
        "at_floor": {f"{f}_{b}_s{s}": at_floor(records[f"z_true_s{s}"][f][b])
                     for s in seeds for f in fams for b in blocks
                     if f"z_true_s{s}" in records},
        "note": "chance rate outside a 2-sd band of 5 floor draws for an "
                "exactly independent latent ~14 % per block"}
    verdict["planted_test"]["passed"] = bool(truth) and all(
        verdict["planted_test"]["at_floor"].values())
    pinned, comp3 = arm_seeds("pinned"), arm_seeds("comp3")
    if pinned:
        floor_both = sum(all(at_floor(r[f]["comp"]) for f in fams) for r in pinned)
        above_any = sum(any(above_floor(r[f]["comp"]) for f in fams) for r in pinned)
        verdict["pinned_comp"] = {
            "seeds": len(pinned), "at_floor_both_graders": floor_both,
            "above_floor_either_grader": above_any,
            "reading": ("capacity sufficient" if floor_both >= 2 else
                        "residual present (capacity or weighting)"
                        if above_any >= 2 else "mixed")}
        verdict["pinned_img"] = {
            "at_floor_both_graders": sum(all(at_floor(r[f]["img"]) for f in fams)
                                         for r in pinned)}
    if pinned and comp3 and len(comp3) == len(pinned):
        floor_both = sum(all(at_floor(r[f]["comp"]) for f in fams) for r in comp3)
        lower_all = {f: all(c[f]["comp"]["excess"] < p[f]["comp"]["excess"]
                            for c, p in zip(comp3, pinned)) for f in fams}
        verdict["comp3"] = {
            "at_floor_both_graders": floor_both, "lower_than_pinned_every_seed":
            lower_all, "reading": ("weighting" if floor_both >= 2 else
                                   "partly weighting" if all(lower_all.values())
                                   else "capacity")}
    text = "\n".join(lines) + "\n\n```json\n" + json.dumps(verdict, indent=1) + "\n```\n"
    (out_dir / f"planted_probe{suffix}.md").write_text(text)
    (out_dir / f"verdict{suffix}.json").write_text(json.dumps(verdict, indent=1))
    log.info("verdict: %s", json.dumps(verdict))
    return verdict


# -- driver ----------------------------------------------------------------

def check_identity(epochs: int = 3) -> dict:
    """The local loop in fit_synthetic's mode vs fit_synthetic, CPU, seed 0,
    one thread (multi-threaded CPU scatter-adds are not bitwise
    reproducible: fit_synthetic against itself differs by ~7e-5 at 8)."""
    import torch

    from discell.applications.planted import fit_synthetic
    from discell.applications.xtilde_gate import build_world

    torch.set_num_threads(1)
    sim = build_world(0).sim
    reference = fit_synthetic(sim, epochs=epochs, device="cpu", seed=0)["z"]
    local = fit_planted(sim, None, seed=0, alpha_a=0.02,
                        invariance="closed_form", w_warmup=0, epochs=epochs,
                        device="cpu")["z"]
    out = {"epochs": epochs, "identical": bool(np.array_equal(reference, local)),
           "max_abs_diff": float(np.abs(reference - local).max())}
    log.info("identity check: %s", json.dumps(out))
    return out


def run(seeds, arms, out_dir: Path, device: str, n_perm: int = N_PERM) -> dict:
    """Fit (or load the cached mu_z) and grade; a non-default *n_perm*
    re-grades into its own files (``*_nperm<n>``), the fits unchanged."""
    from discell.applications.xtilde_gate import build_world

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if n_perm == N_PERM else f"_nperm{n_perm}"
    record_path = out_dir / f"planted_probe{suffix}.json"
    records = (json.loads(record_path.read_text()) if record_path.exists()
               else {})
    for seed in seeds:
        sim = build_world(seed).sim
        targets = world_targets(sim, seed)
        train_tiles, val_tiles = split_tiles(sim, seed)
        key = f"z_true_s{seed}"
        if key not in records:
            records[key] = grade(sim.z_true, sim, targets, train_tiles,
                                 val_tiles, seed, n_perm)
            record_path.write_text(json.dumps(records, indent=1))
        for arm in arms:
            key = f"{arm}_s{seed}"
            if key in records:
                log.info("%s graded already, skipped", key)
                continue
            cache = out_dir / f"mu_z_{key}.npy"
            alpha_a, comp_weight = ARMS[arm]
            started = time.time()
            if cache.exists():
                z = np.load(cache)
            else:
                fit = fit_planted(sim, targets, seed, alpha_a, comp_weight,
                                  train_tiles=train_tiles,
                                  eval_tiles=train_tiles + val_tiles,
                                  device=device)
                z = fit["z"]
                np.save(cache, z)
            records[key] = grade(z, sim, targets, train_tiles, val_tiles, seed,
                                 n_perm)
            records[key].update(arm=arm, alpha_a=alpha_a,
                                comp_weight=comp_weight, seed=seed,
                                seconds=round(time.time() - started, 1))
            log.info("%s: comp ridge %+.4f / mlp %+.4f, img ridge %+.4f / "
                     "mlp %+.4f, z_true R2 %.3f (%.0f s)", key,
                     records[key]["ridge"]["comp"]["excess"],
                     records[key]["mlp"]["comp"]["excess"],
                     records[key]["ridge"]["img"]["excess"],
                     records[key]["mlp"]["img"]["excess"],
                     records[key]["z_true_r2"], records[key]["seconds"])
            record_path.write_text(json.dumps(records, indent=1))
    summarise(records, seeds, out_dir, suffix)
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--arms", nargs="+", default=list(ARMS),
                        choices=list(ARMS))
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-perm", type=int, default=N_PERM,
                        help="floor draws; non-default re-grades cached fits "
                             "into *_nperm<n> files")
    parser.add_argument("--check-identity", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    if args.check_identity:
        return 0 if check_identity()["identical"] else 1
    run(args.seeds, args.arms, args.out, args.device, args.n_perm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
