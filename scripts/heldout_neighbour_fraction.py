"""Fraction of training seeds with a held-out graph neighbour (method §2.6 pending).
Split geometry only (finalL_s{0,1,2} splits, pruned graph); no model read.
    uv run python scripts/heldout_neighbour_fraction.py <dataset> ...
Output 2026-09-29: scripts/logs/heldout_neighbour_fraction_2026-09-29.jsonl"""
import json, sys
import numpy as np, scipy.sparse as sp
from discell.experiments.probe_regrade import run_config, Assemblies

out = []
for ds in sys.argv[1:]:
    for s in (0, 1, 2):
        cfg = run_config(ds, f"finalL_s{s}")
        d = Assemblies().get(ds, cfg)
        g, n = d.graph, d.graph.n_cells
        A = sp.coo_matrix((np.ones(2 * len(g.edge_i)), (np.r_[g.edge_i, g.edge_j], np.r_[g.edge_j, g.edge_i])), shape=(n, n)).tocsr()
        A.data[:] = 1
        tr = np.zeros(n, bool); tr[np.concatenate(d.train_tiles)] = True
        va = np.zeros(n, bool); va[np.concatenate(d.val_tiles)] = True
        hv = A @ va.astype(float)            # held-out neighbours per cell
        deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
        seeds_hit = tr & (hv > 0)
        two = (A @ (hv > 0).astype(float) > 0) | (hv > 0)
        exposed = va & ((A @ tr.astype(float)) > 0)   # held-out cells in some training ring one
        tiles_hit = np.mean([np.any(hv[t] > 0) for t in d.train_tiles])
        r = dict(dataset=ds, seed=s, n_train=int(tr.sum()), n_heldout=int(va.sum()),
                 n_train_tiles=len(d.train_tiles), n_val_tiles=len(d.val_tiles),
                 frac_train_seeds_heldout_nbr=float(seeds_hit.sum() / tr.sum()),
                 frac_train_seeds_heldout_within_2hops=float((tr & two).sum() / tr.sum()),
                 mean_heldout_share_among_hit=float((hv[seeds_hit] / deg[seeds_hit]).mean()),
                 frac_heldout_cells_in_training_ring1=float(exposed.sum() / va.sum()),
                 frac_train_tiles_touching_heldout=float(tiles_hit))
        print(json.dumps(r), flush=True); out.append(r)
