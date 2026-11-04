"""The arrays one fit consumes, assembled from an AnnData with a DisCell graph.

Per fit and seed:

* ``v_block = [y minus its last column, PCA_12(Phi)]``, the niche block the
  held-out probe reads (the composition block alone when image-free);
* ``e_Phi``: soft k-means memberships (K clusters) on the image PCs, the
  adversary's image target, and ``phibar_t`` its mean per type;
* spatial tiles and the held-out tile split.

The PCA subsample uses the RNG stream ``[seed, 1]`` and the split ``[seed, 2]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

from discell.graph.kernel import ModelGraph, read_graph
from discell.graph.tiles import spatial_tiles, split_tiles
from discell.io.anndata import counts_matrix, fold_labels, positions_um

#: Thread count for every k-means and the PCA before it, pinned in code.
#: scikit-learn's multithreaded k-means is not reproducible run to run (the
#: result depends on the OpenMP thread count and on scheduling); one thread is
#: deterministic. The paper's fits ran with 8; values move only within that
#: setting's own run-to-run spread.
NUM_THREADS = 1
#: Image PCs in ``v_block`` and under ``e_Phi``.
DEFAULT_V_PCS = 12
#: Cells subsampled to fit the image PCA.
PCA_SUBSAMPLE = 50_000


def soft_clusters(points: np.ndarray, k: int, seed: int = 0,
                  n_threads: int = NUM_THREADS) -> tuple[np.ndarray, np.ndarray]:
    """``(memberships, centers)`` of soft k-means: ``softmax(-d^2 / T)``.

    ``T`` is the median squared distance to the nearest center. Memberships are
    float32 ``(N, k)``.
    """
    from sklearn.cluster import KMeans

    with threadpool_limits(limits=n_threads):
        centers = KMeans(k, n_init=4, random_state=seed).fit(points).cluster_centers_
    d2 = ((points[:, None, :] - centers[None]) ** 2).sum(-1)
    temperature = max(float(np.median(d2.min(axis=1))), 1e-8)
    logits = -d2 / temperature
    logits -= logits.max(axis=1, keepdims=True)
    soft = np.exp(logits)
    return (soft / soft.sum(axis=1, keepdims=True)).astype(np.float32), centers


def type_means(values: np.ndarray, t: np.ndarray, connected: np.ndarray, k: int,
               empty: float = 0.0) -> np.ndarray:
    """Per-type mean of *values* over connected cells; *empty* fills absent types."""
    return np.stack([
        values[(t == g) & connected].mean(axis=0) if ((t == g) & connected).any()
        else np.full(values.shape[1], empty, dtype=np.float32)
        for g in range(k)])


@dataclass
class ModelData:
    """Everything one fit consumes."""

    graph: ModelGraph
    x: sp.csr_matrix              # float32 counts
    t: np.ndarray                 # int64 type index
    phi: np.ndarray               # (N, D) float32; D = 0 when image-free
    positions: np.ndarray         # (N, 2) um
    totals: np.ndarray            # (N,) total counts
    median_counts: float
    p_t: np.ndarray               # (K,) type frequencies
    type_names: np.ndarray
    gene_names: np.ndarray
    v_block: np.ndarray           # (N, K - 1 [+ v_pcs])
    vbar_t: np.ndarray            # (K, width of v_block)
    train_tiles: list[np.ndarray]
    val_tiles: list[np.ndarray]
    e_phi: np.ndarray | None = None     # (N, K); None when image-free
    phibar_t: np.ndarray | None = None  # (K, K)

    @property
    def n_cells(self) -> int:
        return self.graph.n_cells

    @property
    def n_comp(self) -> int:
        """Width of the composition block of ``v_block`` (K - 1)."""
        return self.graph.y.shape[1] - 1

    @property
    def image_free(self) -> bool:
        return self.phi.shape[1] == 0


def assemble(adata, label_key: str, image_key: str | None = None, *,
             counts_layer: str | None = None, spatial_key: str = "spatial",
             tile_cells: int = 4096, val_fraction: float = 0.15,
             v_pcs: int = DEFAULT_V_PCS, seed: int = 0,
             n_threads: int = NUM_THREADS) -> ModelData:
    """Build the arrays, tiles and split for one fit from *adata*.

    *adata* must carry the graph written by :func:`discell.build_graph` for
    *label_key*. ``image_key=None`` gives the image-free model; a cell without
    an image descriptor should carry a zero row.
    """
    graph = read_graph(adata, label_key)
    meta = adata.uns["discell"]["graph"]
    t, type_names = fold_labels(adata.obs[label_key], meta["unassigned_label"])
    k = len(type_names)
    if image_key is not None and image_key not in adata.obsm:
        raise KeyError(f"adata.obsm has no {image_key!r}; pass image_key=None "
                       "for the image-free model")
    image_free = image_key is None
    if image_free:
        phi = np.zeros((adata.n_obs, 0), dtype=np.float32)
        v_block = graph.y[:, :-1].astype(np.float32)
    else:
        from sklearn.decomposition import PCA

        phi = np.asarray(adata.obsm[image_key], dtype=np.float32)
        rng_pca = np.random.default_rng([seed, 1])
        rows = rng_pca.choice(len(phi), min(len(phi), PCA_SUBSAMPLE), replace=False)
        with threadpool_limits(limits=n_threads):
            phi_pcs = PCA(v_pcs, random_state=seed).fit(phi[rows]).transform(phi)
        v_block = np.hstack([graph.y[:, :-1], phi_pcs]).astype(np.float32)

    connected = graph.degrees > 0
    vbar_t = type_means(v_block, t, connected, k)
    e_phi = phibar_t = None
    if not image_free:
        e_phi, _ = soft_clusters(phi_pcs, k, seed=seed, n_threads=n_threads)
        phibar_t = type_means(e_phi, t, connected, k, empty=1.0 / k)

    positions = positions_um(adata, spatial_key)
    train, val = split_tiles(spatial_tiles(positions, tile_cells), val_fraction, seed)
    x = counts_matrix(adata, counts_layer)
    totals = np.asarray(x.sum(axis=1)).ravel()
    return ModelData(
        graph=graph, x=x, t=t, phi=phi, positions=positions, totals=totals,
        median_counts=float(np.median(totals)),
        p_t=(np.bincount(t, minlength=k) / len(t)).astype(np.float32),
        type_names=type_names, gene_names=np.asarray(adata.var_names, dtype=str),
        v_block=v_block, vbar_t=vbar_t, train_tiles=train, val_tiles=val,
        e_phi=e_phi, phibar_t=phibar_t)
