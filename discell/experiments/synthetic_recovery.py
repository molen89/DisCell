#!/usr/bin/env python3
"""Synthetic recovery at the final configuration (devlog 2026-09-29).

The appendix's recovery study (``appendix/synthetic.tex``) rerun with the
final objective. **The world** is :func:`discell.model.synthetic.simulate` at
the appendix's settings: N = 6,000 cells, K = 8 types, G = 60 genes, planted
d_z = 4, d_w = 2, an 8-dimensional Phi, totals ~ max(Pois(150), 20).

**The fit** is the production :class:`~discell.model.train.Trainer` -- its
loop, adversary branch, w warm-up, joint early stopping and best-checkpoint
restore -- on a :class:`~discell.model.prepare.ModelData` built from the
simulation exactly as :func:`~discell.model.prepare.assemble` builds one from
a bundle (``v_block`` = [y minus its last column, PCs(Phi)], ``e_phi`` = soft
K-means of those PCs, per-type means over connected cells, spatial tiles
split with the ``[seed, 2]`` stream). Every TrainConfig default is kept (the
final configuration: adversary, alpha_a 0.3, composition weight 3, alpha_w
0.1 with a 30-epoch warm-up, 500 epochs / patience 40, val fraction 0.15);
overridden are only the appendix's reduced widths (d_z 8, d_w 2, hidden 128,
attention 16), tiles of 512 cells, v_pcs = 8 (Phi has 8 columns), alpha_z =
1/2 / the section's median total count, the assumed kappa (explicit), and
figures_every set past the budget (figures are a logging knob with their
own RNG stream). The uncontrolled arm is the same fit at alpha_a = 0.

**Readouts** (all cells, posterior means from the restored best checkpoint),
the appendix's and ``tests/test_model_recovery.py``'s definitions:

* ``nmi_z``: NMI of K-means (K, n_init 10, random_state 0) on mu_z against
  the planted types; world references ``nmi_counts`` (the same on PCA(8) of
  log1p(10^2 x / l)) and ``nmi_planted_z`` (the same on the planted z);
* ``w_cca``: first canonical correlation of mu_w with the planted w;
* ``b_cosine``: largest principal-angle cosine between span(B_hat) and
  span(B_true); world reference ``b_cosine_random``: the same for random
  d_w-dimensional subspaces of R^G (mean and 95th percentile of 2,000);
* ``comp_r2``: within-type R^2 of the type-centred composition y - ybar_t
  on [mu_z, 1]; its reference is the alpha_a = 0 fit of the same world and
  model seed.

Usage::

    python -m discell.experiments.synthetic_recovery fit --planted 0.2 \\
        --assumed 0.2 --world-seed 0 --model-seed 0 [--uncontrolled]
    python -m discell.experiments.synthetic_recovery checks
    python -m discell.experiments.synthetic_recovery aggregate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("discell.experiments.synthetic_recovery")

N_CELLS, N_TYPES = 6000, 8
WIDTHS = dict(d_z=8, d_w=2, hidden=128, gat_dim=16)
TILE_CELLS = 512
V_PCS = 8                       # Phi is 8-dimensional
DATASET = "synthetic_smoke"     # run directories only; no bundle is read
OUT = Path("scripts/logs/synthetic_2026-09-29")
N_RANDOM = 2000
READOUTS = ("nmi_z", "w_cca", "b_cosine", "comp_r2")


# -- the world ---------------------------------------------------------------

def world(kappa: float, seed: int):
    from discell.model.synthetic import simulate

    return simulate(n_cells=N_CELLS, n_types=N_TYPES, kappa=kappa, seed=seed)


def model_data(sim, seed: int, val_fraction: float = 0.15):
    """A ModelData from *sim*, built as ``prepare.assemble`` builds one."""
    import scipy.sparse as sp
    from sklearn.decomposition import PCA

    from discell.model.prepare import ModelData, soft_clusters, spatial_tiles

    k, t = sim.n_types, sim.t.astype(np.int64)
    graph = sim.graph
    phi_pcs = PCA(min(V_PCS, sim.phi.shape[1]),
                  random_state=seed).fit_transform(sim.phi)
    v_block = np.hstack([graph.y[:, :-1], phi_pcs]).astype(np.float32)
    connected = graph.degrees > 0

    def type_mean(values):
        return np.stack([values[(t == g) & connected].mean(axis=0)
                         for g in range(k)]).astype(np.float32)

    e_phi, _ = soft_clusters(phi_pcs, k, seed=seed)
    tiles = spatial_tiles(sim.positions, TILE_CELLS)
    order = np.random.default_rng([seed, 2]).permutation(len(tiles))
    n_val = max(1, int(round(val_fraction * len(tiles))))
    return ModelData(
        graph=graph, x=sp.csr_matrix(sim.x), t=t, phi=sim.phi,
        positions=sim.positions, totals=sim.totals,
        median_counts=float(np.median(sim.totals)),
        p_t=(np.bincount(t, minlength=k) / len(t)).astype(np.float32),
        type_names=np.array([f"type{g}" for g in range(k)]),
        v_block=v_block, vbar_t=type_mean(v_block),
        train_tiles=[tiles[i] for i in order[n_val:]],
        val_tiles=[tiles[i] for i in order[:n_val]],
        e_phi=e_phi, phibar_t=type_mean(e_phi))


def run_name(planted: float, assumed: float, world_seed: int,
             model_seed: int, uncontrolled: bool) -> str:
    return (f"synrec_p{planted:g}_a{assumed:g}_w{world_seed}_m{model_seed}"
            + ("_unc" if uncontrolled else ""))


def fit_config(sim, assumed: float, model_seed: int, name: str,
               uncontrolled: bool, epochs: int | None = None,
               patience: int | None = None, device: str = "cuda"):
    """TrainConfig defaults, plus only the overrides the module doc lists."""
    from discell.model.train import TrainConfig

    extra = {}
    if uncontrolled:
        extra["alpha_a"] = 0.0
    if epochs is not None:
        extra["epochs"] = epochs
    if patience is not None:
        extra["patience"] = patience
    return TrainConfig(
        dataset=DATASET, run_name=name, **WIDTHS, tile_cells=TILE_CELLS,
        v_pcs=V_PCS, kappa=float(assumed),
        alpha_z=0.5 / float(np.median(sim.totals)),
        figures_every=10 ** 6, seed=model_seed, device=device, **extra)


# -- readouts ----------------------------------------------------------------

def type_nmi(points: np.ndarray, t: np.ndarray) -> float:
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score

    k = int(t.max()) + 1
    return float(normalized_mutual_info_score(
        t, KMeans(k, n_init=10, random_state=0).fit_predict(points)))


def first_cca(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.cross_decomposition import CCA

    n = min(a.shape[1], b.shape[1])
    u, v = CCA(n).fit(a, b).transform(a, b)
    return float(abs(np.corrcoef(u[:, 0], v[:, 0])[0, 1]))


def principal_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Largest principal-angle cosine between the column spaces of a and b."""
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    return float(np.linalg.svd(qa.T @ qb, compute_uv=False)[0])


def comp_r2(sim, latent: np.ndarray) -> float:
    """Within-type R^2 of the type-centred composition on [latent, 1]."""
    residual = sim.graph.y - sim.graph.ybar_t[sim.t]
    design = np.hstack([latent, np.ones((len(latent), 1))])
    coef, *_ = np.linalg.lstsq(design, residual, rcond=None)
    return float(1 - (residual - design @ coef).var() / residual.var())


def world_references(sim, seed: int) -> dict:
    from sklearn.decomposition import PCA

    x_rep = PCA(8).fit_transform(np.log1p(sim.x / sim.totals[:, None] * 100))
    rng = np.random.default_rng([seed, 11])
    b_true = sim.B_true.T                                   # (G, d_w)
    random = [principal_cosine(rng.standard_normal(b_true.shape), b_true)
              for _ in range(N_RANDOM)]
    return {"nmi_counts": type_nmi(x_rep, sim.t),
            "nmi_planted_z": type_nmi(sim.z_true, sim.t),
            "b_cosine_random_mean": float(np.mean(random)),
            "b_cosine_random_q95": float(np.quantile(random, 0.95)),
            "comp_r2_planted_z": comp_r2(sim, sim.z_true)}


def readouts(sim, mu_z: np.ndarray, mu_w: np.ndarray, b_hat: np.ndarray) -> dict:
    return {"nmi_z": type_nmi(mu_z, sim.t),
            "w_cca": first_cca(mu_w, sim.w_true),
            "b_cosine": principal_cosine(b_hat, sim.B_true.T),
            "comp_r2": comp_r2(sim, mu_z)}


# -- one fit -----------------------------------------------------------------

def fit_one(planted: float, assumed: float, world_seed: int, model_seed: int,
            uncontrolled: bool = False, epochs: int | None = None,
            patience: int | None = None, device: str = "cuda",
            out_dir: Path = OUT / "fits") -> dict:
    import torch

    from discell.model.train import Trainer

    started = time.time()
    sim = world(planted, world_seed)
    data = model_data(sim, model_seed)
    name = run_name(planted, assumed, world_seed, model_seed, uncontrolled)
    config = fit_config(sim, assumed, model_seed, name, uncontrolled,
                        epochs, patience, device)
    trainer = Trainer(config, data)
    summary = trainer.fit()                    # restores the best checkpoint
    swept = trainer._sweep(trainer.train_batches + trainer.val_batches)
    order = np.argsort(swept["nodes"])
    assert np.array_equal(swept["nodes"][order], np.arange(sim.t.size))
    mu_z, mu_w = swept["mu_z"][order], swept["mu_w"][order]
    b_hat = trainer.model.B.weight.detach().cpu().numpy()
    record = {
        "name": name, "planted_kappa": planted, "assumed_kappa": assumed,
        "world_seed": world_seed, "model_seed": model_seed,
        "arm": "uncontrolled" if uncontrolled else "final",
        "readouts": readouts(sim, mu_z, mu_w, b_hat),
        "world": world_references(sim, world_seed),
        "fit": {"best_epoch": summary["best"]["epoch"],
                "last_epoch": summary["last_epoch"],
                "best_nmi_trainer": summary["best"]["nmi"],
                "dead_w_channel": summary["dead_w_channel"],
                "train_minutes": summary["minutes"],
                "run_dir": str(trainer.run_dir),
                "peak_gpu_mib": (torch.cuda.max_memory_allocated() / 2 ** 20
                                 if torch.cuda.is_available() else None)},
        "config": {k: v for k, v in json.loads(
            (trainer.run_dir / "config.json").read_text()).items()},
        "wall_minutes": (time.time() - started) / 60,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(record, indent=1))
    log.info("%s: %s (%.1f min)", name, record["readouts"],
             record["wall_minutes"])
    return record


# -- simulator checks ----------------------------------------------------------

def neighbour_cosine(sim) -> float:
    comp = sim.x / sim.totals[:, None]
    neigh = sim.graph.in_edges @ comp
    connected = sim.graph.degrees > 0
    a, b = comp[connected], neigh[connected]
    return float(((a * b).sum(1) / (np.linalg.norm(a, axis=1)
                                     * np.linalg.norm(b, axis=1) + 1e-12)).mean())


def simulator_checks(seeds=(0, 1, 2), kappas=(0.0, 0.2)) -> dict:
    """The appendix's three checks on the recovery worlds."""
    out = {"leak_cosine": {}, "response_r2": {}, "mixture_exact": {}}
    for seed in seeds:
        c0 = neighbour_cosine(world(0.0, seed))
        c4 = neighbour_cosine(world(0.4, seed))
        out["leak_cosine"][str(seed)] = {"kappa0": c0, "kappa0.4": c4,
                                         "delta": c4 - c0,
                                         "pass": bool(c4 - c0 > 0.02)}
        for kappa in kappas:
            sim = world(kappa, seed)
            key = f"k{kappa:g}_w{seed}"
            y, w = sim.graph.y, sim.w_true
            coef, *_ = np.linalg.lstsq(y, w, rcond=None)
            r2 = float(1 - (w - y @ coef).var() / w.var())
            out["response_r2"][key] = {"r2": r2, "pass": bool(r2 > 0.7)}
            connected = sim.graph.degrees > 0
            rho_bar = sim.graph.in_edges @ sim.rho_true
            expected = (1 - kappa) * sim.rho_true + kappa * rho_bar
            expected /= expected.sum(axis=1, keepdims=True)
            convex = float(np.abs(sim.p_true[connected]
                                  - expected[connected]).max())
            isolated = (float(np.abs(sim.p_true[~connected]
                                     - sim.rho_true[~connected]).max())
                        if (~connected).any() else None)
            out["mixture_exact"][key] = {
                "max_abs_convex": convex, "n_isolated": int((~connected).sum()),
                "max_abs_isolated": isolated,
                "pass": bool(convex < 1e-12 and (isolated is None
                                                 or isolated < 1e-12))}
    out["all_pass"] = all(v["pass"] for block in
                          ("leak_cosine", "response_r2", "mixture_exact")
                          for v in out[block].values())
    return out


# -- aggregation ---------------------------------------------------------------

def _range(values) -> dict:
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": float(min(values)),
            "max": float(max(values)), "median": float(np.median(values))}


def aggregate(fits_dir: Path, out_stem: Path, checks_path: Path) -> dict:
    records = [json.loads(p.read_text()) for p in sorted(fits_dir.glob("*.json"))]
    unc = {(r["planted_kappa"], r["world_seed"], r["model_seed"]):
           r["readouts"]["comp_r2"]
           for r in records if r["arm"] == "uncontrolled"}
    groups: dict = {}
    for r in records:
        key = (r["arm"], r["planted_kappa"], r["assumed_kappa"])
        groups.setdefault(key, []).append(r)
    rows = []
    for (arm, planted, assumed), members in sorted(groups.items()):
        row = {"arm": arm, "planted_kappa": planted, "assumed_kappa": assumed,
               "n_fits": len(members),
               "fits": [{"world_seed": m["world_seed"],
                         "model_seed": m["model_seed"], **m["readouts"],
                         "best_epoch": m["fit"]["best_epoch"],
                         "dead_w_channel": m["fit"]["dead_w_channel"]}
                        for m in sorted(members, key=lambda m: (
                            m["world_seed"], m["model_seed"]))]}
        for key in READOUTS:
            row[key] = _range([m["readouts"][key] for m in members])
            row[key]["by_world"] = {
                str(s): _range([m["readouts"][key] for m in members
                                if m["world_seed"] == s])
                for s in sorted({m["world_seed"] for m in members})}
        if arm == "final":
            # the uncontrolled reference: same planted world and model seed
            pairs = [(m["readouts"]["comp_r2"],
                      unc.get((planted, m["world_seed"], m["model_seed"])))
                     for m in members]
            row["comp_r2_uncontrolled"] = _range([u for _, u in pairs])
            row["comp_r2_minus_uncontrolled"] = _range(
                [c - u for c, u in pairs if u is not None])
        rows.append(row)
    worlds = {}
    for r in records:
        worlds.setdefault(f"k{r['planted_kappa']:g}_w{r['world_seed']}",
                          r["world"])
    world_ranges = {}
    for kappa in sorted({r["planted_kappa"] for r in records}):
        members = [v for k, v in worlds.items() if k.startswith(f"k{kappa:g}_")]
        world_ranges[f"{kappa:g}"] = {
            key: _range([m[key] for m in members]) for key in members[0]}
    checks = (json.loads(checks_path.read_text())
              if checks_path.exists() else None)
    config = next((r["config"] for r in records if r["arm"] == "final"), None)
    result = {"groups": rows, "world_references": world_ranges,
              "world_references_by_world": worlds,
              "simulator_checks": checks, "config_final_arm": config,
              "n_records": len(records)}
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    out_stem.with_suffix(".json").write_text(json.dumps(result, indent=1))
    out_stem.with_suffix(".md").write_text(render(result))
    return result


def _fmt(r: dict) -> str:
    return "n/a" if not r.get("n") else (
        f"{r['min']:.2f}–{r['max']:.2f} (n={r['n']})")


def render(result: dict) -> str:
    lines = ["# Synthetic recovery at the final configuration", "",
             "Seed ranges (min–max over 3 world seeds × 3 model seeds; n = "
             "fits present). Every fit on all 6,000 cells, best checkpoint.",
             "", "## Planted κ = assumed κ (final arm) and uncontrolled", "",
             "| arm | planted κ | assumed κ | NMI μ_z | w CCA | B cosine | "
             "comp R² | comp R² uncontrolled |",
             "|---|---|---|---|---|---|---|---|"]
    for row in result["groups"]:
        lines.append(
            f"| {row['arm']} | {row['planted_kappa']:g} | "
            f"{row['assumed_kappa']:g} | {_fmt(row['nmi_z'])} | "
            f"{_fmt(row['w_cca'])} | {_fmt(row['b_cosine'])} | "
            f"{_fmt(row['comp_r2'])} | "
            f"{_fmt(row.get('comp_r2_uncontrolled', {}))} |")
    lines += ["", "## World references (range over the 3 world seeds)", "",
              "| planted κ | NMI counts (K-means on PCs) | NMI planted z | "
              "B cosine random mean | random q95 | comp R² planted z |",
              "|---|---|---|---|---|---|"]
    for kappa, refs in result["world_references"].items():
        lines.append(
            f"| {kappa} | {_fmt(refs['nmi_counts'])} | "
            f"{_fmt(refs['nmi_planted_z'])} | "
            f"{_fmt(refs['b_cosine_random_mean'])} | "
            f"{_fmt(refs['b_cosine_random_q95'])} | "
            f"{_fmt(refs['comp_r2_planted_z'])} |")
    lines += ["", "## B cosine by world (seed-bistability read)", "",
              "| arm | planted κ | assumed κ | world | B cosine |",
              "|---|---|---|---|---|"]
    for row in result["groups"]:
        for world_seed, r in row["b_cosine"].get("by_world", {}).items():
            lines.append(f"| {row['arm']} | {row['planted_kappa']:g} | "
                         f"{row['assumed_kappa']:g} | {world_seed} | {_fmt(r)} |")
    checks = result["simulator_checks"]
    lines += ["", "## Simulator checks", ""]
    if checks is None:
        lines.append("not run")
    else:
        lines.append(f"all pass: **{checks['all_pass']}**")
        for seed, c in checks["leak_cosine"].items():
            lines.append(f"- leak cosine, world {seed}: κ0 {c['kappa0']:.4f}, "
                         f"κ0.4 {c['kappa0.4']:.4f}, Δ {c['delta']:.4f} "
                         f"(> 0.02: {c['pass']})")
        for key, c in checks["response_r2"].items():
            lines.append(f"- response R² {key}: {c['r2']:.3f} (> 0.7: {c['pass']})")
        for key, c in checks["mixture_exact"].items():
            lines.append(f"- mixture {key}: max |Δ| {c['max_abs_convex']:.2e}, "
                         f"isolated {c['n_isolated']} (pass: {c['pass']})")
    return "\n".join(lines) + "\n"


# -- CLI -----------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--planted", type=float, required=True)
    f.add_argument("--assumed", type=float, required=True)
    f.add_argument("--world-seed", type=int, required=True)
    f.add_argument("--model-seed", type=int, required=True)
    f.add_argument("--uncontrolled", action="store_true")
    f.add_argument("--epochs", type=int, default=None, help="smoke only")
    f.add_argument("--patience", type=int, default=None, help="smoke only")
    f.add_argument("--device", default="cuda")
    f.add_argument("--out", type=Path, default=OUT / "fits")
    c = sub.add_parser("checks")
    c.add_argument("--out", type=Path, default=OUT / "simulator_checks.json")
    a = sub.add_parser("aggregate")
    a.add_argument("--fits", type=Path, default=OUT / "fits")
    a.add_argument("--checks", type=Path, default=OUT / "simulator_checks.json")
    a.add_argument("--out", type=Path, default=Path(
        f"data/datasets/{DATASET}/experiments/synthetic_recovery"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    if args.cmd == "fit":
        fit_one(args.planted, args.assumed, args.world_seed, args.model_seed,
                args.uncontrolled, args.epochs, args.patience, args.device,
                args.out)
    elif args.cmd == "checks":
        result = simulator_checks()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        print(json.dumps({"all_pass": result["all_pass"]}))
        return 0 if result["all_pass"] else 1
    else:
        aggregate(args.fits, args.out, args.checks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
