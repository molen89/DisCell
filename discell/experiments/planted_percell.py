#!/usr/bin/env python3
"""Planted per-cell response (devlog 2026-09-29, "Planted per-cell response").

Does the response channel detect a per-cell response beyond the niche when
one exists? **The world** is the synthetic-recovery world
(:mod:`discell.experiments.synthetic_recovery`: ``simulate`` at N 6,000,
K 8, planted kappa 0.2), with one addition passed through
``simulate(w_shift=...)`` -- a keyword whose default reproduces every
earlier call bit for bit, and which moves no random draw, so s = 0 IS the
recovery world of the same seed.

**The plant.** The planted type is the world's most abundant type; 10 % of
its cells, drawn uniformly from it with the ``[world_seed, 5]`` stream (so
spread across niches), get ``log rho = z A + (w + u) B`` with ``u = s R
e / ||e B||``, e = the first planted-w axis. The log-rate shift ``u B`` lies
in the column space of the planted B (spec choice 1) and its norm is s R,
where R = the across-cell RMS of ``||(w_i - w_bar) B||`` (spec choice 3),
the size of the planted niche effect in log-rate units. ``w_true`` is not
changed: the shift is what the cell does beyond its niche.

**The fit** is ``synthetic_recovery.fit_config`` unchanged (the final
configuration at the recovery widths), assumed kappa = planted = 0.2.

**Readouts** (the planted type's cells; posterior means, best checkpoint):

* ``kl_auc``: AUC of per-cell KL(q(w) || p(w)), planted vs the type's other
  cells;
* ``w_corr``: correlation, over the type's cells, of (mu_w - m_psi(c, t))
  projected on v -- the planted gene direction d = e B, centred over genes,
  mapped into the fitted w-space by least squares on the fitted B -- with
  the planted shift (0 / s R; the planted indicator, so s = 0 is defined);
* ``z_absorption`` (outside the rule, spec choice 2): the same correlation
  for dec_a(mu_z), centred over genes, projected on the centred d;
* ``recon_gain``: held-out (val-tile) per-count log-likelihood of the full
  decode minus the decode with w at its prior mean, on the planted cells
  (and on the type's other val cells, for reference);
* across seeds (aggregate): Spearman of per-cell KL_w and the overlap of
  the top deciles, over all cells, for every seed pair.

**Rule (fixed in advance).** Detected at s if on world 0 every seed has
kl_auc >= 0.8 and every seed pair has Spearman >= 0.5.

Usage::

    python -m discell.experiments.planted_percell fit --s 1 --world-seed 0 --model-seed 0
    python -m discell.experiments.planted_percell aggregate
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from discell.experiments import synthetic_recovery as SR

log = logging.getLogger("discell.experiments.planted_percell")

KAPPA = 0.2
FRACTION = 0.10
S_GRID = (0.0, 0.5, 1.0, 2.0)
AUC_MIN, RHO_MIN = 0.8, 0.5
RULE_WORLD = 0
OUT = Path("scripts/logs/planted_percell_2026-09-29")


# -- the world ---------------------------------------------------------------

def plant(world_seed: int) -> dict:
    """The planted subset, direction and effect size (s-independent)."""
    base = SR.world(KAPPA, world_seed)
    counts = np.bincount(base.t, minlength=base.n_types)
    planted_type = int(counts.argmax())
    members = np.flatnonzero(base.t == planted_type)
    rng = np.random.default_rng([world_seed, 5])
    chosen = rng.choice(members, int(round(FRACTION * len(members))),
                        replace=False)
    mask = np.zeros(base.t.size, dtype=bool)
    mask[chosen] = True
    b = base.B_true                                   # (d_w, G)
    niche = (base.w_true - base.w_true.mean(axis=0)) @ b
    centred = niche - niche.mean(axis=1, keepdims=True)
    e = np.zeros(b.shape[0])
    e[0] = 1.0
    d = e @ b
    return {"planted_type": planted_type, "mask": mask, "e": e,
            "d": d, "R": float(np.sqrt((niche ** 2).sum(axis=1).mean())),
            "R_gene_centred": float(np.sqrt((centred ** 2).sum(axis=1).mean())),
            "base": base}


def shifted_world(world_seed: int, s: float):
    from discell.model.synthetic import simulate

    p = plant(world_seed)
    shift = np.zeros_like(p["base"].w_true)
    shift[p["mask"]] = s * p["R"] / np.linalg.norm(p["d"]) * p["e"]
    sim = simulate(n_cells=SR.N_CELLS, n_types=SR.N_TYPES, kappa=KAPPA,
                   seed=world_seed, w_shift=shift)
    return sim, p


def predictability(world_seed: int, folds: int = 5) -> dict:
    """Is the subset predictable from context within its type? 5-fold CV AUC
    of a logistic regression and a random forest on [y, Phi] (the niche
    composition and the image descriptor), cells of the planted type only
    (type itself is constant there; across all cells it trivially marks the
    candidate pool)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    p = plant(world_seed)
    sim = p["base"]
    rows = sim.t == p["planted_type"]
    label = p["mask"][rows]
    out = {"n_type": int(rows.sum()), "n_planted": int(label.sum())}
    cv = StratifiedKFold(folds, shuffle=True, random_state=0)
    for feats_name, feats in (("composition", sim.graph.y[rows]),
                              ("composition_phi",
                               np.hstack([sim.graph.y[rows], sim.phi[rows]]))):
        for name, clf in (
                ("logistic", make_pipeline(StandardScaler(),
                                           LogisticRegression(max_iter=2000))),
                ("forest", RandomForestClassifier(300, min_samples_leaf=5,
                                                  random_state=0, n_jobs=4))):
            prob = cross_val_predict(clf, feats, label, cv=cv,
                                     method="predict_proba")[:, 1]
            out[f"{feats_name}_{name}_auc"] = float(roc_auc_score(label, prob))
    return out


# -- one fit -----------------------------------------------------------------

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def fit_one(s: float, world_seed: int, model_seed: int,
            epochs: int | None = None, patience: int | None = None,
            device: str = "cuda", out_dir: Path = OUT / "fits") -> dict:
    import torch
    from sklearn.metrics import roc_auc_score

    from discell.model.train import Trainer

    started = time.time()
    sim, p = shifted_world(world_seed, s)
    data = SR.model_data(sim, model_seed)
    name = f"percell_s{s:g}_w{world_seed}_m{model_seed}"
    config = SR.fit_config(sim, KAPPA, model_seed, name, False, epochs,
                           patience, device)
    trainer = Trainer(config, data)
    summary = trainer.fit()                         # best checkpoint restored
    swept = trainer._sweep(trainer.train_batches + trainer.val_batches)
    order = np.argsort(swept["nodes"])
    assert np.array_equal(swept["nodes"][order], np.arange(sim.t.size))
    mu_z, mu_w = swept["mu_z"][order], swept["mu_w"][order]
    prior_w, kl_w = swept["prior_w"][order], swept["kl_w"][order].sum(axis=1)

    model = trainer.model
    rows = sim.t == p["planted_type"]
    planted = p["mask"][rows].astype(float)
    d_c = p["d"] - p["d"].mean()
    b_hat = model.B.weight.detach().cpu().numpy()        # (G, d_w)
    v, *_ = np.linalg.lstsq(b_hat, d_c, rcond=None)
    w_proj = (mu_w - prior_w)[rows] @ v
    with torch.no_grad():
        a_z = model.dec_a(torch.tensor(mu_z[rows], device=trainer.device)
                          ).cpu().numpy()
    a_z -= a_z.mean(axis=1, keepdims=True)
    z_proj = a_z @ (d_c / np.linalg.norm(d_c))

    # held-out reconstruction gain of the full decode over w at its prior mean
    gain_rows, gain_ll = [], []
    model.eval()
    with torch.no_grad():
        for batch in trainer.val_batches:
            fwd = model(**Trainer._forward_kwargs(batch), kappa=config.kappa,
                        sample=False)
            n = batch["n_seeds"]
            x = batch["x"][:n].float()
            full = (x * fwd.log_p).sum(-1)
            prior = (x * trainer._decode_seeds(
                fwd, fwd.mu_z[:n], n, w_seeds=fwd.prior_mean_w[:n])).sum(-1)
            gain_rows.append(batch["nodes"][:n])
            gain_ll.append(torch.stack([full, prior, x.sum(-1)], 1).cpu().numpy())
    model.train()
    gain_rows, gain_ll = np.concatenate(gain_rows), np.vstack(gain_ll)

    def gain(select):
        sel = select[gain_rows]
        if not sel.any():
            return {"n": 0, "gain_per_count": None}
        f, pr, tot = gain_ll[sel].sum(axis=0)
        return {"n": int(sel.sum()), "gain_per_count": float((f - pr) / tot)}

    record = {
        "name": name, "s": s, "world_seed": world_seed,
        "model_seed": model_seed, "planted_kappa": KAPPA,
        "planted_type": p["planted_type"], "n_planted": int(p["mask"].sum()),
        "R": p["R"], "R_gene_centred": p["R_gene_centred"],
        "shift_norm_lograte": s * p["R"],
        "readouts": {
            "kl_auc": float(roc_auc_score(planted, kl_w[rows])),
            "w_corr": _corr(w_proj, planted),
            "z_absorption": _corr(z_proj, planted),
            "recon_gain_planted": gain(p["mask"]),
            "recon_gain_type_other": gain((sim.t == p["planted_type"])
                                          & ~p["mask"]),
            "kl_w_mean_planted": float(kl_w[p["mask"]].mean()),
            "kl_w_mean_type_other": float(kl_w[rows & ~p["mask"]].mean()),
            "kl_w_mean_all": float(kl_w.mean())},
        "fit": {"best_epoch": summary["best"]["epoch"],
                "last_epoch": summary["last_epoch"],
                "dead_w_channel": summary["dead_w_channel"],
                "run_dir": str(trainer.run_dir)},
        "config": json.loads((trainer.run_dir / "config.json").read_text()),
        "wall_minutes": (time.time() - started) / 60}
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}_klw.npy", kl_w.astype(np.float32))
    (out_dir / f"{name}.json").write_text(json.dumps(record, indent=1))
    log.info("%s: %s (%.1f min)", name, record["readouts"],
             record["wall_minutes"])
    return record


# -- aggregation ---------------------------------------------------------------

def _range(values) -> dict:
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": float(min(values)),
            "max": float(max(values))}


def cross_seed(kl: dict) -> list[dict]:
    from scipy.stats import spearmanr

    out = []
    for a, b in itertools.combinations(sorted(kl), 2):
        ka, kb = kl[a], kl[b]
        q = int(round(0.1 * len(ka)))
        top_a, top_b = set(np.argsort(-ka)[:q]), set(np.argsort(-kb)[:q])
        out.append({"seeds": [a, b],
                    "spearman": float(spearmanr(ka, kb).correlation),
                    "top_decile_overlap": len(top_a & top_b) / q})
    return out


def aggregate(fits_dir: Path, out_stem: Path, pred_path: Path) -> dict:
    records = [json.loads(p.read_text()) for p in sorted(fits_dir.glob("*.json"))]
    groups: dict = {}
    for r in records:
        groups.setdefault((r["world_seed"], r["s"]), []).append(r)
    rows = []
    for (world_seed, s), members in sorted(groups.items()):
        members.sort(key=lambda m: m["model_seed"])
        kl = {m["model_seed"]: np.load(fits_dir / f"{m['name']}_klw.npy")
              for m in members}
        pairs = cross_seed(kl)
        ro = [m["readouts"] for m in members]
        row = {"world_seed": world_seed, "s": s, "n_seeds": len(members),
               "R": members[0]["R"],
               "shift_norm_lograte": members[0]["shift_norm_lograte"],
               "per_seed": [{"model_seed": m["model_seed"], **m["readouts"],
                             "best_epoch": m["fit"]["best_epoch"]}
                            for m in members],
               "cross_seed": pairs,
               "kl_auc": _range([r["kl_auc"] for r in ro]),
               "w_corr": _range([r["w_corr"] for r in ro]),
               "z_absorption": _range([r["z_absorption"] for r in ro]),
               "recon_gain_planted": _range(
                   [r["recon_gain_planted"]["gain_per_count"] for r in ro]),
               "spearman": _range([p["spearman"] for p in pairs]),
               "top_decile_overlap": _range(
                   [p["top_decile_overlap"] for p in pairs])}
        complete = len(members) == 3 and len(pairs) == 3
        row["detected"] = bool(
            complete and all(r["kl_auc"] >= AUC_MIN for r in ro)
            and all(p["spearman"] >= RHO_MIN for p in pairs))
        row["rule_complete"] = complete
        rows.append(row)
    rule = {f"{r['s']:g}": {"detected": r["detected"],
                            "complete": r["rule_complete"]}
            for r in rows if r["world_seed"] == RULE_WORLD}
    detected = sorted(float(s) for s, v in rule.items() if v["detected"])
    result = {
        "rule": {"world_seed": RULE_WORLD, "auc_min": AUC_MIN,
                 "spearman_min": RHO_MIN, "by_s": rule,
                 "smallest_detected_s": detected[0] if detected else None},
        "groups": rows,
        "predictability": (json.loads(pred_path.read_text())
                           if pred_path.exists() else None),
        "n_records": len(records)}
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    out_stem.with_suffix(".json").write_text(json.dumps(result, indent=1))
    out_stem.with_suffix(".md").write_text(render(result))
    return result


def _fmt(r: dict) -> str:
    return "n/a" if not r.get("n") else f"{r['min']:.2f}–{r['max']:.2f}"


def render(result: dict) -> str:
    rule = result["rule"]
    lines = ["# Planted per-cell response", "",
             f"Rule (world {rule['world_seed']}): detected at s when every "
             f"seed has KL AUC ≥ {rule['auc_min']} and every seed pair has "
             f"Spearman ≥ {rule['spearman_min']}. Smallest detected s: "
             f"**{rule['smallest_detected_s']}**.", "",
             "| world | s | shift ‖·‖ (log-rate) | KL AUC | w corr | "
             "z absorption | recon gain (planted, /count) | Spearman | "
             "top-decile overlap | detected |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["groups"]:
        lines.append(
            f"| {r['world_seed']} | {r['s']:g} | {r['shift_norm_lograte']:.2f} "
            f"| {_fmt(r['kl_auc'])} | {_fmt(r['w_corr'])} | "
            f"{_fmt(r['z_absorption'])} | "
            f"{_fmt(r['recon_gain_planted'])} | {_fmt(r['spearman'])} | "
            f"{_fmt(r['top_decile_overlap'])} | "
            f"{r['detected'] if r['rule_complete'] else 'incomplete'} |")
    lines += ["", "Ranges are min–max over the 3 model seeds (seed pairs for "
              "Spearman and overlap). World 0 decides; worlds 1–2 replicate.",
              "", "## Subset predictability (within the planted type, 5-fold "
              "CV AUC)", ""]
    for world_seed, pred in (result["predictability"] or {}).items():
        lines.append(f"- world {world_seed}: " + ", ".join(
            f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
            for k, v in pred.items()))
    return "\n".join(lines) + "\n"


# -- CLI -----------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--s", type=float, required=True)
    f.add_argument("--world-seed", type=int, required=True)
    f.add_argument("--model-seed", type=int, required=True)
    f.add_argument("--epochs", type=int, default=None, help="smoke only")
    f.add_argument("--patience", type=int, default=None, help="smoke only")
    f.add_argument("--device", default="cuda")
    f.add_argument("--out", type=Path, default=OUT / "fits")
    c = sub.add_parser("predictability")
    c.add_argument("--out", type=Path, default=OUT / "predictability.json")
    a = sub.add_parser("aggregate")
    a.add_argument("--fits", type=Path, default=OUT / "fits")
    a.add_argument("--predictability", type=Path,
                   default=OUT / "predictability.json")
    a.add_argument("--out", type=Path, default=Path(
        f"data/datasets/{SR.DATASET}/experiments/planted_percell"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    if args.cmd == "fit":
        fit_one(args.s, args.world_seed, args.model_seed, args.epochs,
                args.patience, args.device, args.out)
    elif args.cmd == "predictability":
        result = {}
        for w in (0, 1, 2):
            p = plant(w)
            result[str(w)] = {**predictability(w), "planted_type":
                              p["planted_type"], "R": p["R"],
                              "R_gene_centred": p["R_gene_centred"]}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
    else:
        aggregate(args.fits, args.out, args.predictability)
    return 0


if __name__ == "__main__":
    sys.exit(main())
