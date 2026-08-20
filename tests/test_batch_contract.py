"""What a consumer of the dataloader is entitled to rely on.

These are the assertions that used to live in ``main.check_invariants`` and
``loader.self_test``, which were only ever run by hand.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# -- shape agreement ------------------------------------------------------


def test_node_aligned_fields_share_a_length(batch):
    assert batch.x.shape[0] == batch.y.shape[0] == len(batch.node_ids)
    assert batch.pos.shape[0] == batch.node_index.shape[0] == batch.n_nodes


def test_feature_axes_match_the_dataset(batch, dataset):
    assert batch.x.shape[1] == dataset.adata.n_vars
    assert batch.y.shape[1] == dataset.n_types


def test_edge_aligned_fields_share_a_length(batch):
    n_edges = batch.edge_index.shape[1]
    assert batch.edge_attr.shape == (n_edges, len(batch.edge_attr_names))
    assert batch.edge_id.shape == (n_edges,)
    assert batch.into_j.shape == (n_edges,)


def test_seed_aligned_fields_share_a_length(batch):
    assert batch.image_embedding.shape[0] == batch.n_seeds
    assert batch.seed_composition.shape[0] == batch.n_seeds
    assert batch.n_seeds <= batch.n_nodes


# -- graph structure ------------------------------------------------------


def test_every_edge_points_into_a_seed(batch):
    """Edges between two neighbours are dropped: they never reach a seed."""
    seeds = set(batch.seed_index.tolist())
    assert all(int(v) in seeds for v in batch.edge_index[1].tolist())


def test_no_self_loops(batch):
    assert bool((batch.edge_index[0] != batch.edge_index[1]).all())


def test_edge_endpoints_are_valid_node_rows(batch):
    assert int(batch.edge_index.min()) >= 0
    assert int(batch.edge_index.max()) < batch.n_nodes


def test_seeds_are_not_repeated_within_a_batch(batch):
    seeds = [batch.node_ids[k] for k in batch.seed_index.tolist()]
    assert len(set(seeds)) == len(seeds)


# -- values ---------------------------------------------------------------


def test_counts_are_raw_integers(batch):
    assert float(batch.x.min()) >= 0
    assert bool(torch.allclose(batch.x, batch.x.round()))


def test_labels_are_one_hot(batch):
    assert bool((batch.y.sum(dim=1) == 1).all())


def test_edge_attr_distance_agrees_with_pos(batch):
    """The precomputed centroid distance must match the coordinates carried."""
    src, dst = batch.edge_index[0], batch.edge_index[1]
    column = batch.edge_attr_names.index("centroid_dist_um")
    assert torch.allclose((batch.pos[dst] - batch.pos[src]).norm(dim=1),
                          batch.edge_attr[:, column], atol=1e-2)


def test_edge_attr_columns_are_non_negative(batch):
    for k, name in enumerate(batch.edge_attr_names):
        column = batch.edge_attr[:, k]
        assert float(column.min()) >= 0, name
        if name == "centroid_dist_um":
            assert float(column.min()) > 0, "two cells cannot share a centroid"


# -- the label space ------------------------------------------------------


def test_no_two_labels_differ_only_by_case(dataset):
    """The NaN fill must not add a second class meaning 'unassigned'."""
    folded = {name.casefold() for name in dataset.type_names}
    assert len(folded) == dataset.n_types


# -- the constants --------------------------------------------------------


def test_constants_are_not_carried_on_the_batch(batch):
    fields = type(batch).__dataclass_fields__
    assert "neighbour_composition" not in fields
    assert "beta" not in fields


def test_composition_rows_are_distributions(dataset):
    table = dataset.neighbour_composition.to_numpy()
    used = dataset.composition_support.to_numpy() > 0
    assert (table >= 0).all()
    assert np.allclose(table[used].sum(axis=1), 1.0)


def test_seed_composition_is_the_row_for_the_seed_s_own_type(batch, dataset):
    table = dataset.constant.neighbour_composition
    types = dataset.type_index[batch.node_index[batch.seed_index].cpu().numpy()]
    assert torch.allclose(batch.seed_composition, table[types])


def test_edge_beta_is_aligned_to_edge_index(batch, dataset):
    beta = dataset.constant.edge_beta(batch)
    assert beta.shape == (batch.edge_index.shape[1],)
    direct = torch.where(batch.into_j,
                         dataset.constant.beta[batch.edge_id, 0],
                         dataset.constant.beta[batch.edge_id, 1])
    assert torch.allclose(beta, direct)


def test_beta_is_row_stochastic_over_every_in_edge(dataset):
    """Summed over a cell's in-edges beta is 1, so smoothing stays on the simplex.

    The exceptions are exactly the isolated cells, which have no in-edges.
    """
    total = np.zeros(dataset.adata.n_obs)
    np.add.at(total, dataset.edge_j, dataset.constant.beta[:, 0].numpy())
    np.add.at(total, dataset.edge_i, dataset.constant.beta[:, 1].numpy())
    isolated = dataset.degrees == 0
    assert np.allclose(total[~isolated], 1.0)
    assert (total[isolated] == 0).all()


def test_niche_rows_are_distributions(dataset):
    niche = dataset.constant.niche.numpy()
    connected = dataset.degrees > 0
    assert np.allclose(niche[connected].sum(axis=1), 1.0)


# -- epoch semantics ------------------------------------------------------


def test_one_epoch_makes_every_cell_a_seed_exactly_once(dataset):
    seen = [b.node_ids[k] for b in dataset.batches(shuffle=True)
            for k in b.seed_index.tolist()]
    assert len(seen) == dataset.adata.n_obs
    assert len(set(seen)) == dataset.adata.n_obs


def test_len_counts_the_batches_an_epoch_yields(dataset):
    assert len(dataset) == sum(1 for _ in dataset.batches(shuffle=False))


# -- residency ------------------------------------------------------------


@pytest.mark.parametrize("resident", [None, "cpu"])
def test_residency_does_not_change_the_batch(bundle, resident):
    """An index_select on a resident tensor must serve what numpy served."""
    from discell.data.loader import CellGraphDataset

    ds, variant = bundle
    data = CellGraphDataset.from_dataset(ds, variant, batch_cells=16,
                                         resident=resident)
    batch = next(iter(data.batches(shuffle=False)))
    assert batch.x.dtype == torch.float32
    assert batch.y.dtype == torch.float32
    assert bool(torch.allclose(batch.x, batch.x.round()))
    assert bool((batch.y.sum(dim=1) == 1).all())


def test_a_seedless_batch_is_empty_not_broken(dataset):
    """An isolated seed yields no edges; the fields must still line up."""
    isolated = np.flatnonzero(dataset.degrees == 0)
    if not len(isolated):
        pytest.skip("no isolated cells in this bundle")
    batch = dataset.batch_from_seeds(isolated[:1])
    assert batch.edge_index.shape == (2, 0)
    assert batch.edge_attr.shape == (0, len(dataset.edge_metrics))
    assert batch.n_seeds == 1
