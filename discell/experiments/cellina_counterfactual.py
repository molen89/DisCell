#!/usr/bin/env python3
"""Cellina's neighbour-rewiring counterfactual, scored with DisCell's transport
reads (devlog "Cellina: three additions (motivation, 2026-09-29; author)",
addition 3). Stage B; stage A (``DisCell-baselines/cellina/cellina_cf_encode.py``)
refits Cellina on the training tiles and writes posterior-mean encodings.

**The counterfactual.** Cellina decodes ``px_scale(cat(z, s))``, z from the
cell's own counts, s from phi(v), the mean of its neighbours' log-normalised
expression. Its edge perturbation (``make_counterfactual_adata``,
``precomputed=True``) gives a cell the phi of a donor, i.e. rewires the cell to
the donor's neighbours (exact on our binary, degree-normalised graph); s
depends on phi only, so the rewired s is the donor's posterior mean ``qsm``.
A cell moved into niche B of its type g keeps its own ``qzm`` and takes the
``qsm`` of a random type-g training-fold cell of niche B (one donor per cell,
``default_rng([seed, 29])`` / ``[seed, 30]``) -- the neighbourhood a type-g
cell in B actually has, as DisCell's context is B's type-g group mean.
Decoding is the plug-in posterior mean (no sample): deterministic.

**Mirrored from ``discell.model.transport``, draw for draw.** Niches
(``validate.niche_labels``, K = 10, run seed; thread-count dependent, run at 8),
held-out tiles (fold 0 of the prepare tiles), panel eligibility, pair order,
the overlap guard, gene masks, the observed shifts, both noise ceilings and
every subsample / bootstrap stream (shared ``default_rng(seed)``, fresh
``seed + 1 / + 2 / + 3 / + 101 ...``) are DisCell's, so the ceilings and the
data-side floors reproduce DisCell's run exactly (``replay`` in the output).
The Unassigned panels are drawn, then dropped (``eval_mask``).

Read by read (DisCell -> Cellina):

* mean read ``counterfactual``: B(m_psi_B - m_psi_A) + influx(A->B), z fixed
  -> log mean p(z_i, s_donor(B)) - log mean p(z_i, s_i), i = A's type-g
  training-fold cells. ``full``: log mean p over B's cells - over A's.
  **Not mapped** (no counterpart in Cellina, recorded as ``not_mapped``):
  ``program_only``, ``leak_only`` (Cellina has no leak channel),
  ``counterfactual_phi_fixed`` / ``program_phi_fixed`` / interventionable
  share (s has no composition-only input to freeze), ``full_beats_both``.
* Read A (distribution, pairwise and leave-one-out): transported = source at
  its own z with a target-niche donor s; untransported = with a donor s from
  its own niche (DisCell: its own niche's group w); group-w target = target
  cells with a target-niche donor s; own-w target = target cells at their own
  (qzm, qsm). Type-mean and raw-target variants unchanged.
* Read B (twins): nearest source cell in Cellina's qzm (64-d), otherwise as
  DisCell's.
* CIs: ``bootstrap.transport_mean_bootstrap`` (half-tile subsampling) and
  ``bootstrap._mmd_draws`` / ``transport_dist_summary`` (tile bootstrap), same
  seeds and tile draws as DisCell's ``bootstrap_ci.json``.

    python -m discell.experiments.cellina_counterfactual --dataset D \\
        --cellina-dir .../results/cellina/D_lineage_cf

Writes ``<dir>/transport/transport{,_distribution,_twins}.json``,
``<dir>/bootstrap_ci.json`` and ``<dir>/transport_side_by_side.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from discell.experiments import bootstrap as BS
from discell.model import eval_mask as EM
from discell.model import transport as T

log = logging.getLogger("discell.cellina_counterfactual")

NA = {"r2": float("nan"), "slope": float("nan"), "corr": float("nan"),
      "not_mapped": True}
NOT_MAPPED = ("program_only", "leak_only", "counterfactual_phi_fixed",
              "program_phi_fixed", "interventionable_share",
              "median_slope_phi_fixed", "full_beats_both")


class Decoder:
    """Cellina's decoder at the posterior means: ``px_scale(cat(z, s))``.

    *net* maps ``(n, 2d)`` shifted latents to ``(n, G)`` probabilities; built
    by :func:`cellina_decoder` from stage A's weights (or planted, in tests).
    Deterministic: eval mode, no sampling anywhere."""

    def __init__(self, net, device: str = "cpu", chunk: int = 4096):
        self.net, self.device, self.chunk = net.eval().to(device), device, chunk

    def _batches(self, z, s):
        import torch
        for a in range(0, len(z), self.chunk):
            x = np.hstack([z[a:a + self.chunk], s[a:a + self.chunk]])
            with torch.no_grad():
                yield self.net(torch.as_tensor(x, dtype=torch.float32,
                                               device=self.device))

    def decode(self, z, s) -> np.ndarray:
        return np.vstack([p.cpu().numpy() for p in self._batches(z, s)])

    def mean(self, z, s) -> np.ndarray:
        total = sum(p.double().sum(0) for p in self._batches(z, s))
        return (total / len(z)).cpu().numpy()


def cellina_decoder(state: dict):
    """DecoderSCVI's px_scale path (scvi FCLayers: Linear -> BatchNorm1d(eps
    1e-3, eval) -> ReLU per layer, dropout 0; Linear -> softmax), from the
    state dict stage A saved."""
    import torch

    n = sum(1 for k in state if k.endswith(".0.weight")
            and k.startswith("px_decoder.fc_layers."))
    layers = []
    for i in range(n):
        pre = f"px_decoder.fc_layers.Layer {i}."
        lin = torch.nn.Linear(*state[pre + "0.weight"].shape[::-1])
        bn = torch.nn.BatchNorm1d(state[pre + "0.weight"].shape[0], eps=1e-3)
        lin.load_state_dict({"weight": state[pre + "0.weight"],
                             "bias": state[pre + "0.bias"]})
        bn.load_state_dict({k: state[pre + "1." + k] for k in (
            "weight", "bias", "running_mean", "running_var",
            "num_batches_tracked")})
        layers += [lin, bn, torch.nn.ReLU()]
    out = torch.nn.Linear(*state["px_scale_decoder.0.weight"].shape[::-1])
    out.load_state_dict({"weight": state["px_scale_decoder.0.weight"],
                         "bias": state["px_scale_decoder.0.bias"]})
    return torch.nn.Sequential(*layers, out, torch.nn.Softmax(dim=-1))


def _draw(rng, pool, n):
    return pool[rng.integers(0, len(pool), n)]


def _members(sl, k, g):
    base = sl["connected"] & (sl["t"] == g) & (sl["labels"] == k)
    return (np.flatnonzero(base & ~sl["held_out"]),
            np.flatnonzero(base & sl["held_out"]))


def mean_read(sl: dict, z, s, dec: Decoder, seed: int):
    """``transport_check``'s loop with Cellina's prediction; returns (the JSON
    record, the per-panel artefacts ``bootstrap.transport_mean_bootstrap``
    reads)."""
    rng, rng_tiles = np.random.default_rng(seed), np.random.default_rng([seed, 1])
    donors = np.random.default_rng([seed, 29])
    x_rate, names = sl["x_rate"], sl["names"]
    own: dict = {}

    def own_mean(key, rows):
        if key not in own:
            own[key] = dec.mean(z[rows], s[rows])
        return own[key]

    panels, boot = [], []
    for niche_a, niche_b in T.pick_pairs(sl["labels"], sl["y"], sl["connected"]):
        for g in range(len(names)):
            members = {}
            for niche, side in ((niche_a, "A"), (niche_b, "B")):
                tr, te = _members(sl, niche, g)
                if len(tr) < T.MIN_CELLS or len(te) < T.MIN_CELLS // 5:
                    break
                members[side] = (tr, te)
            if len(members) < 2:
                continue
            (tr_a, te_a), (tr_b, te_b) = members["A"], members["B"]
            y_a, y_b = sl["y"][tr_a], sl["y"][tr_b]
            gap_vec = y_b.mean(0) - y_a.mean(0)
            gap = np.linalg.norm(gap_vec)
            unit = gap_vec / max(gap, 1e-12)
            spread = 0.5 * (float((y_a @ unit).std()) + float((y_b @ unit).std()))
            p_a, p_b = own_mean((niche_a, g), tr_a), own_mean((niche_b, g), tr_b)
            p_cf = dec.mean(z[tr_a], s[_draw(donors, tr_b, len(tr_a))])
            cf = np.log(p_cf + T.EPS) - np.log(p_a + T.EPS)
            full = np.log(p_b + T.EPS) - np.log(p_a + T.EPS)
            obs_a = np.asarray(x_rate[te_a].mean(axis=0)).ravel()
            obs_b = np.asarray(x_rate[te_b].mean(axis=0)).ravel()
            keep = (obs_a > T.MIN_RATE) & (obs_b > T.MIN_RATE)
            observed = np.log(obs_b[keep] + T.EPS) - np.log(obs_a[keep] + T.EPS)
            p = {"pair": (int(niche_a), int(niche_b)), "type": names[g],
                 "n_train": [len(tr_a), len(tr_b)], "n_test": [len(te_a), len(te_b)],
                 "n_genes": int(keep.sum()), "overlap_flag": bool(gap > 3 * spread),
                 "counterfactual": T.score_shift(cf[keep], observed),
                 "full": T.score_shift(full[keep], observed),
                 **{k: dict(NA) for k in NOT_MAPPED[:4]},
                 "noise_ceiling": T.noise_ceiling(x_rate, te_a, te_b, keep, rng),
                 "top_gene_overlap": T.top_gene_overlap(cf[keep], observed),
                 "zero_null_r2": 0.0}
            p["trusted"] = bool(p["noise_ceiling"] >= T.TRUST_CEILING
                                and p["n_genes"] >= T.TRUST_GENES)
            p["counterfactual_of_ceiling"] = float(
                p["counterfactual"]["r2"] / p["noise_ceiling"]
                ) if p["noise_ceiling"] >= 0.05 else float("nan")
            p["interventionable_share"] = None
            p["ceiling_tiles"] = T.noise_ceiling(x_rate, te_a, te_b, keep, rng_tiles,
                                                 split="tiles", tile_of=sl["tile_of"])
            p["trusted_tiles"] = bool(p["ceiling_tiles"] >= T.TRUST_CEILING
                                      and p["n_genes"] >= T.TRUST_GENES)
            p["fraction_of_ceiling_tiles"] = float(
                p["counterfactual"]["r2"] / p["ceiling_tiles"]
                ) if p["ceiling_tiles"] >= 0.05 else float("nan")
            panels.append(p)
            boot.append({"pair": p["pair"], "type": p["type"], "rows_a": te_a,
                         "rows_b": te_b, "keep": keep, "prediction": cf[keep],
                         "overlap_flag": p["overlap_flag"], "trusted": p["trusted"],
                         "r2": p["counterfactual"]["r2"],
                         "noise_ceiling": p["noise_ceiling"]})
    res = {"eval_mask": EM.record(names, sl["t"])}
    kept = [p for p in panels if not EM.is_excluded(p["type"])]
    res["eval_mask"]["n_panels_dropped"] = len(panels) - len(kept)
    res["panels"] = kept
    tiers = {"supported": [p for p in kept if not p["overlap_flag"]],
             "extrapolation": [p for p in kept if p["overlap_flag"]]}
    for name in ("supported", "extrapolation"):
        tiers[f"{name}_trusted"] = [p for p in tiers[name] if p["trusted"]]
        tiers[f"{name}_trusted_tiles"] = [p for p in tiers[name] if p["trusted_tiles"]]
    res["summary"] = {}
    for name, tier in tiers.items():
        if tier:
            summary = T.tier_summary(tier)
            res["summary"][name] = {k: (None if k in NOT_MAPPED else v)
                                    for k, v in summary.items()}
    return res, [b for b in boot if not EM.is_excluded(b["type"])]


def distribution_read(sl: dict, z, s, dec: Decoder, seed: int, boot: int,
                      device: str, n_draws: int = BS.N_BOOT):
    """``distribution_check`` (+ its own-target pass and twins) with Cellina's
    decodes, and ``bootstrap.transport_dist_panels``' tile draws on the
    pairwise panels. Returns (distribution JSON, twins JSON, CI reads)."""
    rng, donors = np.random.default_rng(seed), np.random.default_rng([seed, 30])
    names, x_rate, hvg = sl["names"], sl["x_rate"], sl["hvg"]
    n_types, n_niches = len(names), int(sl["labels"].max()) + 1
    on_hvg = (lambda a: T.restrict_renormalise(a, hvg)) if hvg is not None else None
    train, test = {}, {}
    for k in range(n_niches):
        for g in range(n_types):
            tr, te = _members(sl, k, g)
            if len(tr) >= T.MIN_CELLS and len(te) >= T.MIN_CELLS // 5:
                train[(k, g)], test[(k, g)] = tr, te
    deferred = []

    def panel(src, src_niche, target):
        cut = rng.permutation(len(src))[:T.SIZE_CAP]
        rows, own = src[cut], src_niche[cut]
        g = target[1]
        d_to = _draw(donors, train[target], len(rows))
        d_own = np.empty(len(rows), dtype=np.int64)
        for k in np.unique(own):
            d_own[own == k] = _draw(donors, train[(int(k), g)], int((own == k).sum()))
        p_tr, p_un = dec.decode(z[rows], s[d_to]), dec.decode(z[rows], s[d_own])
        tgt = test[target]
        tgt = tgt[rng.permutation(len(tgt))[:T.SIZE_CAP]]
        p_target = np.asarray(x_rate[tgt].todense())
        p_src = np.asarray(x_rate[rows].todense())
        out = {"scores": T.distribution_scores(p_tr, p_un, p_target, p_src, rng,
                                               device=device, n_boot=boot),
               "scores_count_matched": T.distribution_scores(
                   p_tr, p_un, p_target, p_src, np.random.default_rng(seed),
                   device=device, n_boot=boot, count_depths=sl["totals"][tgt])}
        d_tgt = _draw(donors, train[target], len(tgt))
        p_tg = dec.decode(z[tgt], s[d_tgt])
        out["scores_model"] = T.distribution_scores(
            p_tr, p_un, p_tg, p_src, np.random.default_rng(seed + 1),
            device=device, n_boot=boot)
        if on_hvg:
            out["scores_model_hvg"] = T.distribution_scores(
                on_hvg(p_tr), on_hvg(p_un), on_hvg(p_tg), on_hvg(p_src),
                np.random.default_rng(seed + 2), device=device, n_boot=boot)
        out["twins"] = T.twin_scores(p_tr, p_un, p_tg, z[rows], z[tgt],
                                     np.random.default_rng(seed + 3), n_boot=boot)
        if on_hvg:
            out["twins_hvg"] = T.twin_scores(
                on_hvg(p_tr), on_hvg(p_un), on_hvg(p_tg), z[rows], z[tgt],
                np.random.default_rng(seed + 3), n_boot=boot)
        deferred.append({"rows": rows, "tgt": tgt, "d_to": d_to, "d_own": d_own,
                         "d_tgt": d_tgt})
        return out

    res = {"pairwise": [], "leave_one_out": []}
    for a, b in T.pick_pairs(sl["labels"], sl["y"], sl["connected"]):
        for g in range(n_types):
            if (a, g) in train and (b, g) in train:
                src = test[(a, g)]
                out = panel(src, np.full(len(src), a), (b, g))
                res["pairwise"].append({"pair": [int(a), int(b)], "type": names[g],
                                        "target": int(b), "n_source": int(len(src)),
                                        "n_target": int(len(test[(b, g)])), **out})
                deferred[-1]["entry"] = res["pairwise"][-1]
    for (a, g) in sorted(train):
        others = [k for k in range(n_niches) if k != a and (k, g) in train]
        if not others:
            continue
        src = np.concatenate([test[(k, g)] for k in others])
        own = np.concatenate([np.full(len(test[(k, g)]), k) for k in others])
        out = panel(src, own, (a, g))
        res["leave_one_out"].append({"target": int(a), "type": names[g],
                                     "n_source": int(len(src)),
                                     "n_source_niches": len(others),
                                     "n_target": int(len(test[(a, g)])), **out})
        deferred[-1]["entry"] = res["leave_one_out"][-1]

    # the tile draws of bootstrap.transport_dist_panels: pairwise, mask applied
    pw = [d for d in deferred[:len(res["pairwise"])]
          if not EM.is_excluded(d["entry"]["type"])]
    universe = np.unique(np.concatenate([d["rows"] for d in pw])) if pw else []
    pos = np.full(len(sl["t"]), -1, dtype=np.int64)
    pos[universe] = np.arange(len(universe))
    tiles, n_tiles = BS.tile_index(sl["positions"][universe]) if pw else ([], 0)
    draw_rng = np.random.default_rng(0)
    mults = np.stack([np.bincount(draw_rng.integers(0, n_tiles, n_tiles),
                                  minlength=n_tiles)[tiles]
                      for _ in range(n_draws)]) if pw else None
    ci_panels = []
    for i, d in enumerate(deferred):     # the own-target pass (seed + 101 ...)
        e, rows, tgt = d["entry"], d["rows"], d["tgt"]
        p_tr, p_un = dec.decode(z[rows], s[d["d_to"]]), dec.decode(z[rows], s[d["d_own"]])
        p_own = dec.decode(z[tgt], s[tgt])
        p_src = np.asarray(x_rate[rows].todense())
        e["scores_model_own"] = T.distribution_scores(
            p_tr, p_un, p_own, p_src, np.random.default_rng(seed + 101),
            device=device, n_boot=boot)
        if on_hvg:
            e["scores_model_own_hvg"] = T.distribution_scores(
                on_hvg(p_tr), on_hvg(p_un), on_hvg(p_own), on_hvg(p_src),
                np.random.default_rng(seed + 102), device=device, n_boot=boot)
        e["twins_own"] = T.twin_scores(p_tr, p_un, p_own, z[rows], z[tgt],
                                       np.random.default_rng(seed + 103), n_boot=boot)
        if on_hvg:
            e["twins_own_hvg"] = T.twin_scores(
                on_hvg(p_tr), on_hvg(p_un), on_hvg(p_own), z[rows], z[tgt],
                np.random.default_rng(seed + 103), n_boot=boot)
        if i < len(res["pairwise"]) and not EM.is_excluded(e["type"]):
            p_tg = dec.decode(z[tgt], s[d["d_tgt"]])
            ci = {"pair": e["pair"], "type": e["type"]}
            for name, p_t, sd in (("group", p_tg, seed + 1), ("own", p_own, seed + 101)):
                ci[name] = BS._mmd_draws(p_tr, p_un, p_t, rows, np.random.default_rng(sd),
                                         device, lambda c: mults[:, pos[c]])
            ci_panels.append(ci)
    res["eval_mask"] = EM.record(names, sl["t"])
    for version in ("pairwise", "leave_one_out"):
        kept = [p for p in res[version] if not EM.is_excluded(p["type"])]
        res["eval_mask"][f"n_{version}_dropped"] = len(res[version]) - len(kept)
        res[version] = kept
    res["summary"] = {v: T.distribution_summary(res[v]) for v in ("pairwise", "leave_one_out")}
    res["summary_count_matched"] = {v: T.distribution_summary(res[v], "scores_count_matched")
                                    for v in ("pairwise", "leave_one_out")}
    for suffix in T.MODEL_SUFFIXES:
        sm = {v: T.distribution_summary(res[v], f"scores_model{suffix}")
              for v in ("pairwise", "leave_one_out")}
        if sm["pairwise"].get("n_panels"):
            res[f"summary_model{suffix}"] = sm
    twins = {"eval_mask": res["eval_mask"], "n_hvg": int(hvg.sum()) if hvg is not None else 0}
    for v in ("pairwise", "leave_one_out"):
        twins[v] = [{k: x for k, x in p.items() if not k.startswith("scores")}
                    for p in res[v]]
    for suffix in ("", "_hvg", "_own", "_own_hvg"):
        twins[f"summary{suffix}"] = {v: T.twin_summary(twins[v], f"twins{suffix}")
                                     for v in ("pairwise", "leave_one_out")}
    ci = BS.transport_dist_summary(ci_panels)
    for entry in ci.values():
        entry.update(n_boot=n_draws, n_tiles=int(n_tiles), n_cells=int(len(universe)))
    return res, twins, ci


def slide(data, labels, fold, hvg=None) -> dict:
    """The per-cell arrays both reads use, from an assembled dataset."""
    tile_of = np.full(data.graph.n_cells, -1, dtype=np.int64)
    for k, tile in enumerate(data.train_tiles + data.val_tiles):
        tile_of[tile] = k
    return {"labels": labels, "held_out": fold == 0, "t": np.asarray(data.t),
            "connected": data.graph.degrees > 0, "y": data.graph.y,
            "x_rate": data.x.multiply(1.0 / data.totals.clip(min=1.0)[:, None]).tocsr(),
            "tile_of": tile_of, "positions": np.asarray(data.positions, np.float64),
            "names": [str(n) for n in data.type_names],
            "totals": np.asarray(data.totals), "hvg": hvg}


def replay(ours: dict, theirs: dict | None, version: str | None = None) -> dict:
    """How far the data-side quantities (ceilings; the raw-target floor) are
    from the DisCell run's: 0 when the panels and draws are the run's."""
    if not theirs:
        return {"available": False}
    if version is None:
        by = {(tuple(p["pair"]), p["type"]): p for p in theirs["panels"]}
        diffs = [max(abs(p["noise_ceiling"] - q["noise_ceiling"]),
                     abs(np.nan_to_num(p["ceiling_tiles"]) - np.nan_to_num(q.get("ceiling_tiles", 0))))
                 for p in ours["panels"] if (q := by.get((tuple(p["pair"]), p["type"])))]
        n = len(ours["panels"])
    else:
        by = {(str(p.get("pair", p["target"])), p["type"]): p for p in theirs[version]}
        diffs = [abs(p["scores"]["mmd2"]["floor"] - q["scores"]["mmd2"]["floor"])
                 for p in ours[version]
                 if (q := by.get((str(p.get("pair", p["target"])), p["type"])))
                 and not p["scores"].get("insufficient")]
        n = len(ours[version])
    return {"available": True, "n_panels": n, "n_matched": len(diffs),
            "n_theirs": len(theirs["panels" if version is None else version]),
            "max_abs_diff": float(max(diffs)) if diffs else None}


def main(argv=None) -> int:
    import torch

    from discell import paths
    from discell.experiments.baseline_battery import run_config
    from discell.experiments.cellina_niche_domain import assemble_like
    from discell.model.validate import N_FOLDS, niche_labels

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--cellina-dir", required=True)
    p.add_argument("--run", default="finalL_s0", help="the DisCell run mirrored")
    p.add_argument("--niches", type=int, default=10)
    p.add_argument("--hvg", type=int, default=1000)
    p.add_argument("--boot", type=int, default=T.N_BOOT)
    p.add_argument("--n", type=int, default=BS.N_BOOT, help="CI draws")
    p.add_argument("--device", default="cuda")
    p.add_argument("--min-cells", type=int, default=None,
                   help="SMOKE ONLY: lower transport.MIN_CELLS (a window has "
                        "too few cells per panel); recorded in the output")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    if args.min_cells:
        T.MIN_CELLS = args.min_cells
    device = args.device if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    cdir = Path(args.cellina_dir)
    out = cdir / "transport"
    out.mkdir(parents=True, exist_ok=True)
    config = run_config(args.dataset, args.run)
    seed = int(config["seed"])
    data = assemble_like(args.dataset, config)
    n = data.graph.n_cells
    fold = np.zeros(n, dtype=np.int64)
    for k, tile in enumerate(data.train_tiles + data.val_tiles):
        fold[tile] = k % N_FOLDS
    labels = niche_labels(data, args.niches, seed)
    connected = data.graph.degrees > 0
    hvg = (T.hvg_mask(data, np.flatnonzero(connected & (fold != 0)), args.hvg, seed=seed)
           if args.hvg else None)
    sl = slide(data, labels, fold, hvg)

    enc = np.load(cdir / "encoded.npz")
    ids = enc["cell_id"]
    z = np.full((n, enc["qzm"].shape[1]), np.nan, dtype=np.float32)
    s = np.full((n, enc["qsm"].shape[1]), np.nan, dtype=np.float32)
    z[ids], s[ids] = enc["qzm"], enc["qsm"]
    missing = int(np.isnan(z[:, 0]).sum())
    if missing and not args.min_cells:
        raise ValueError(f"{missing} cells have no Cellina encoding")
    if missing:        # smoke on a window: unencoded cells join no panel
        sl["connected"] = sl["connected"] & ~np.isnan(z[:, 0])
        log.warning("SMOKE: %d cells without an encoding left out of every panel",
                    missing)
    state = torch.load(cdir / "decoder_state.pt", map_location="cpu")
    dec = Decoder(cellina_decoder(state), device)
    chk = np.load(cdir / "decoder_check.npz")
    mine = dec.decode(chk["qzm"], chk["qsm"])
    decoder_check = {"max_abs_diff": float(np.abs(mine - chk["px_scale"]).max()),
                     "max_px_scale": float(chk["px_scale"].max()),
                     "repeat_max_abs_diff": float(np.abs(
                         mine - dec.decode(chk["qzm"], chk["qsm"])).max())}
    if not np.allclose(mine, chk["px_scale"], rtol=1e-4, atol=1e-7):
        raise RuntimeError(f"decoder does not reproduce Cellina's: {decoder_check}")
    cellina_cfg = json.loads((cdir / "config.json").read_text())
    theirs = paths.dataset(args.dataset).root / "runs" / args.run / "transport"

    def load(name):
        f = theirs / name
        return json.loads(f.read_text()) if f.exists() else None

    head = {"tool": "Cellina", "mirrors": f"{args.dataset}/{args.run}",
            "kappa": None, "niche_source": "kmeans", "seed": seed,
            "counterfactual": "edge perturbation: own qzm + a target-niche type-g "
                              "training-fold donor's qsm (precomputed rewiring)",
            "decode": "px_scale(cat(qzm, qsm)), posterior means, no sample",
            "not_mapped": list(NOT_MAPPED), "decoder_check": decoder_check,
            "min_cells": T.MIN_CELLS, "smoke_min_cells": bool(args.min_cells),
            "cellina_fit": {k: cellina_cfg.get(k) for k in (
                "fit_cells", "n_fit", "epochs_completed", "best_checkpoint", "train_s")},
            "fold0_in_fit_tiles": float(np.isin(np.flatnonzero(fold == 0), np.concatenate(
                data.train_tiles)).mean())}
    mean, mean_boot = mean_read(sl, z, s, dec, seed)
    mean = {**head, **mean, "replay": replay(mean, load("transport.json"))}
    (out / "transport.json").write_text(json.dumps(mean, indent=2, default=float))
    log.info("mean read: %d panels (%.0f s); replay %s", len(mean["panels"]),
             time.time() - t0, mean["replay"])
    dist, twins, dist_ci = distribution_read(sl, z, s, dec, seed, args.boot, device, args.n)
    dist = {**head, **dist, "replay_pairwise": replay(dist, load("transport_distribution.json"),
                                                       "pairwise")}
    (out / "transport_distribution.json").write_text(json.dumps(dist, indent=2, default=float))
    (out / "transport_twins.json").write_text(json.dumps({**head, **twins}, indent=2,
                                                         default=float))
    log.info("distribution + twins done (%.0f s); replay %s", time.time() - t0,
             dist["replay_pairwise"])

    reads = {}
    if mean_boot:
        mb = BS.transport_mean_bootstrap(mean_boot, sl["x_rate"], sl["positions"], args.n,
                                         seed=0, device=device)
        for key, entry in mb["reads"].items():
            reads[key] = {**entry, "n_boot": args.n, "n_tiles": mb["n_tiles"],
                          "n_cells": mb["n_cells"], "method": mb["method"],
                          "m_tiles": mb["m_tiles"]}
    reads.update(dist_ci)
    (cdir / "bootstrap_ci.json").write_text(json.dumps(
        {"tool": "Cellina", "mirrors": head["mirrors"], "n_boot": args.n,
         "tile_um": BS.TILE_UM, "seed": 0, "eval_mask": mean["eval_mask"],
         "reads": reads}, indent=2, default=float))
    side_by_side(cdir, paths.dataset(args.dataset).root / "runs" / args.run, args.dataset)
    log.info("wrote %s (%.0f s)", out, time.time() - t0)
    return 0


def side_by_side(cdir: Path, run_dir: Path, dataset: str) -> None:
    """The table keys (``scripts/envelope_tables.transport_row``) for DisCell
    and Cellina, with the tile CIs where both have them."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "envelope_tables", Path(__file__).resolve().parents[2] / "scripts" / "envelope_tables.py")
    env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env)
    rows = {"DisCell": env.transport_row(run_dir), "Cellina": env.transport_row(cdir)}
    cis = {}
    for name, d in (("DisCell", run_dir), ("Cellina", cdir)):
        f = d / "bootstrap_ci.json"
        cis[name] = json.loads(f.read_text())["reads"] if f.exists() else {}

    def cell(name, key):
        v, c = rows[name].get(key), cis[name].get(key, {}).get("ci95")
        if v is None:
            return "--"
        return f"{v:.3f}" + (f" [{c[0]:.3f}, {c[1]:.3f}]" if c else "")
    keys = sorted(set(rows["DisCell"]) | set(rows["Cellina"]))
    lines = [f"# Transport reads, DisCell {run_dir.name} vs Cellina -- {dataset}", "",
             "Same panels, niches, held-out tiles, ceilings and draws; Cellina's "
             "counterfactual is its neighbour rewiring (cellina_counterfactual.py). "
             "Tile 95 % CIs in brackets.", "", "| read | DisCell | Cellina |", "|---|---|---|"]
    lines += [f"| {k} | {cell('DisCell', k)} | {cell('Cellina', k)} |" for k in keys]
    (cdir / "transport_side_by_side.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
