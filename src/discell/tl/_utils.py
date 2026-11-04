"""What the readouts read from an AnnData, and the small numerics they share.

Every readout reads the fit's outputs from the AnnData only (``.obsm``,
``.varm``, ``.obsp``, ``.uns['discell']``, the counts and the labels); the
spatial folds are rebuilt from the centroids and the stored configuration.
Records are kept in ``.uns['discell'][key]`` as JSON strings, so an h5ad round
trip keeps them whole; :func:`record` reads one back.
"""

from __future__ import annotations

import json

import numpy as np
import scipy.sparse as sp

from discell.graph.kernel import ModelGraph, read_graph
from discell.graph.tiles import spatial_tiles, split_tiles
from discell.io.anndata import counts_matrix, fold_labels, positions_um
from discell.model.config import Config

#: Spatial folds of the cross-validated reads: tile k goes to fold k mod N_FOLDS.
N_FOLDS = 5
#: A gene counts as expressed when at least this share of cells detect it.
MIN_PREVALENCE = 0.01


def fit_config(adata) -> Config:
    """The configuration of the fit whose outputs *adata* carries."""
    fit = adata.uns.get("discell", {}).get("fit")
    if fit is None:
        raise KeyError("no DisCell fit in adata: run discell.fit or Model.encode first")
    return Config.from_dict(json.loads(fit["config"]))


def obsm(adata, key: str) -> np.ndarray:
    if key not in adata.obsm:
        raise KeyError(f"adata.obsm has no {key!r}")
    return np.asarray(adata.obsm[key])


def types(adata) -> tuple[np.ndarray, np.ndarray]:
    """``(type_index, type_names)`` under the graph's label folding."""
    meta = adata.uns["discell"]["graph"]
    return fold_labels(adata.obs[meta["label_key"]], meta["unassigned_label"])


def graph(adata) -> ModelGraph:
    return read_graph(adata)


def expressed_genes(adata, config: Config, min_prevalence: float = MIN_PREVALENCE
                    ) -> np.ndarray:
    """``(G,)`` bool: genes detected in at least *min_prevalence* of the cells."""
    x = counts_matrix(adata, config.counts_layer)
    return np.asarray((x > 0).mean(axis=0)).ravel() >= min_prevalence


def tiles(adata, config: Config) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The fit's ``(train, held-out)`` tiles, rebuilt from the centroids."""
    return split_tiles(spatial_tiles(positions_um(adata), config.tile_cells),
                       config.val_fraction, config.seed)


def spatial_folds(adata, config: Config | None = None) -> np.ndarray:
    """``(N,)`` fold per cell: the fit's tiles (training, then held-out) in turn,
    tile k in fold ``k mod N_FOLDS``, so no tile straddles two folds. *config*
    defaults to the fit's."""
    train, val = tiles(adata, fit_config(adata) if config is None else config)
    fold = np.zeros(adata.n_obs, dtype=np.int64)
    for k, tile in enumerate(train + val):
        fold[tile] = k % N_FOLDS
    return fold


def store(adata, key: str, rec: dict) -> None:
    adata.uns.setdefault("discell", {})[key] = json.dumps(rec, default=_jsonable)


def record(adata, key: str) -> dict:
    """A readout's record from ``.uns['discell'][key]``."""
    value = adata.uns.get("discell", {}).get(key)
    if value is None:
        raise KeyError(f"no {key!r} record in adata.uns['discell']")
    return json.loads(value) if isinstance(value, str) else value


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON-serialisable: {type(value)}")


# -- shared numerics -----------------------------------------------------------

def center_per_type(values: np.ndarray, t: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Subtract the per-type mean over the *mask* cells; zero elsewhere (float64)."""
    out = np.zeros_like(values, dtype=np.float64)
    for g in np.unique(t[mask]):
        members = mask & (t == g)
        out[members] = values[members] - values[members].mean(axis=0)
    return out


def ridge_cv(design: np.ndarray, target: np.ndarray, fold: np.ndarray,
             lam: float = 1e-3) -> np.ndarray:
    """Held-out predictions of per-fold ridge fits (z-scored on the training folds)."""
    prediction = np.full(len(target), np.nan)
    for f in np.unique(fold):
        train, test = fold != f, fold == f
        mean = design[train].mean(axis=0)
        scale = design[train].std(axis=0) + 1e-12
        d_train = np.hstack([(design[train] - mean) / scale, np.ones((train.sum(), 1))])
        gram = d_train.T @ d_train + lam * np.eye(d_train.shape[1])
        coef = np.linalg.solve(gram, d_train.T @ target[train])
        d_test = np.hstack([(design[test] - mean) / scale, np.ones((test.sum(), 1))])
        prediction[test] = d_test @ coef
    return prediction


def r2(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(1.0 - (target - prediction).var() / max(target.var(), 1e-12))


def row_normalized_graph(g: ModelGraph, keep: np.ndarray | None = None
                         ) -> tuple[sp.csr_matrix, np.ndarray]:
    """``(W, connected)``: the binary symmetric adjacency, row-normalized.

    *keep*: the graph induced on those cells; a cell whose neighbors are all
    dropped counts as not connected.
    """
    n = g.n_cells
    i, j = g.edge_i, g.edge_j
    if keep is not None and not np.all(keep):
        inside = keep[i] & keep[j]
        i, j = i[inside], j[inside]
    adj = sp.coo_matrix((np.ones(2 * len(i)), (np.concatenate([i, j]), np.concatenate([j, i]))),
                        shape=(n, n)).tocsr()
    degrees = np.asarray(adj.sum(axis=1)).ravel()
    connected = degrees > 0
    inv = np.zeros(n)
    inv[connected] = 1.0 / degrees[connected]
    return sp.diags(inv) @ adj, connected
