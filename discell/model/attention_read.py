#!/usr/bin/env python3
"""What the attention looks at: the post-hoc read for arm (ii) (2026-09-23).

Two quantities, on one held-out tile set, from the ``alpha`` the forward pass
already returns:

* **entropy of alpha_ij by type** -- per destination cell, the entropy of its
  in-edge attention distribution (heads averaged, the sink's leftover mass
  counted as one more outcome when the model has one), grouped by the
  destination's type. A type-free or image query is only interesting if the
  spread of attention it produces differs from the pinned query's;
* **correlation of alpha with the top-5 PCs of Phi** -- edge-level, against
  the destination's and the source's image embedding, and cell-level against
  the entropy. Under ``--query image`` the query *is* a linear map of Phi_i,
  so a nonzero destination correlation is the read that says the histology is
  doing the selecting.

Nothing here is part of a fit: it loads a finished run and writes JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import numpy as np


def _entropy_per_destination(alpha: np.ndarray, dst: np.ndarray, n_dst: int,
                             sink: bool) -> tuple[np.ndarray, np.ndarray]:
    """``(entropy (n_dst,) in nats, in-degree)``; heads averaged.

    Empty destinations (the isolated contract: no in-edges) come out NaN.
    """
    heads = alpha.shape[1]
    degree = np.bincount(dst, minlength=n_dst)
    ent = np.zeros((n_dst, heads))
    term = -alpha * np.log(np.clip(alpha, 1e-30, None))
    for h in range(heads):
        ent[:, h] = np.bincount(dst, weights=term[:, h], minlength=n_dst)
    if sink:
        mass = np.zeros((n_dst, heads))
        for h in range(heads):
            mass[:, h] = np.bincount(dst, weights=alpha[:, h], minlength=n_dst)
        leftover = np.clip(1.0 - mass, 1e-30, None)
        ent = ent - leftover * np.log(leftover)
    out = ent.mean(axis=1)
    out[degree == 0] = np.nan
    return out, degree


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or a[keep].std() == 0 or b[keep].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def attention_read(trainer, data, batches, n_pcs: int = 5) -> dict:
    """Entropy by type and alpha-vs-Phi-PC correlations over *batches*."""
    import torch
    from sklearn.decomposition import PCA

    sink = bool(trainer.model.gat.sink)
    rows, ent, deg, alpha_edge, dst_node, src_node = [], [], [], [], [], []
    with torch.no_grad():
        for batch in batches:
            fwd = trainer.model(**trainer._forward_kwargs(batch),
                                kappa=trainer.config.kappa, sample=False)
            alpha = fwd.alpha.float().cpu().numpy()
            dst = batch["gat_dst"].cpu().numpy()
            src = batch["gat_src"].cpu().numpy()
            n_context = batch["n_context"]
            nodes = np.asarray(batch["nodes"])
            e, d = _entropy_per_destination(alpha, dst, n_context, sink)
            rows.append(nodes[:n_context])
            ent.append(e)
            deg.append(d)
            alpha_edge.append(alpha.mean(axis=1))
            dst_node.append(nodes[dst])
            src_node.append(nodes[src])
    rows = np.concatenate(rows)
    ent = np.concatenate(ent)
    deg = np.concatenate(deg)
    alpha_edge = np.concatenate(alpha_edge)
    dst_node = np.concatenate(dst_node)
    src_node = np.concatenate(src_node)

    phi = np.asarray(data.phi, dtype=np.float64)
    n_pcs = int(min(n_pcs, phi.shape[1], max(phi.shape[0] - 1, 1)))
    pca = PCA(n_pcs).fit(phi[np.unique(rows)])
    scores = pca.transform(phi)                      # (N, n_pcs), all cells

    t = np.asarray(data.t)
    by_type = {}
    for k, name in enumerate(np.asarray(data.type_names)):
        take = (t[rows] == k) & np.isfinite(ent)
        if take.sum() == 0:
            continue
        by_type[str(name)] = {
            "n": int(take.sum()),
            "entropy_mean": float(ent[take].mean()),
            "entropy_sd": float(ent[take].std()),
            "degree_mean": float(deg[take].mean()),
            # the reference an entropy is read against: log(in-degree),
            # the maximum a distribution over that many edges can carry
            "entropy_over_log_degree": float(np.nanmean(
                ent[take] / np.log(np.clip(deg[take], 2, None)))),
        }

    finite = np.isfinite(ent)
    return {
        "n_cells": int(finite.sum()),
        "n_edges": int(len(alpha_edge)),
        "attention_sink": sink,
        "entropy_mean": float(ent[finite].mean()),
        "entropy_sd": float(ent[finite].std()),
        "entropy_by_type": by_type,
        "phi_pc_explained_variance": [float(v) for v in
                                      pca.explained_variance_ratio_],
        # edge level: does an edge's weight follow the images?
        "alpha_vs_dst_phi_pc": [_corr(alpha_edge, scores[dst_node, k])
                                for k in range(n_pcs)],
        "alpha_vs_src_phi_pc": [_corr(alpha_edge, scores[src_node, k])
                                for k in range(n_pcs)],
        # cell level: does how *spread* a cell's attention is follow its image?
        "entropy_vs_phi_pc": [_corr(ent, scores[rows, k])
                              for k in range(n_pcs)],
    }


def main(argv: Sequence[str] | None = None) -> int:
    from discell.model.validate import load_run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="val", choices=("val", "train", "all"))
    parser.add_argument("--pcs", type=int, default=5)
    parser.add_argument("--out", default=None,
                        help="default: <run dir>/attention_read.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    config, data, trainer, run_dir, _ = load_run(args.dataset, args.run,
                                                 device=args.device)
    batches = {"val": trainer.val_batches, "train": trainer.train_batches,
               "all": trainer.train_batches + trainer.val_batches}[args.split]
    out = attention_read(trainer, data, batches, n_pcs=args.pcs)
    out |= {"run": args.run, "dataset": args.dataset, "split": args.split,
            "query": config.query}
    path = args.out or (run_dir / "attention_read.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
