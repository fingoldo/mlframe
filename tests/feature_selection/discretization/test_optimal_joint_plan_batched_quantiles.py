"""The optimal_joint plan builds every candidate's fold edges from one percentile call; the validation bins equal the per-candidate edges."""

import numpy as np

from mlframe.feature_selection.filters import _adaptive_nbins as A
from mlframe.feature_selection.filters._optimal_joint_edges import _optimal_joint_plan


def _reference_cells(x, candidates, n_splits, random_state):
    """Per fold, ``[(M, bins of the validation values under _edges_from_quantiles(train, M))]`` as the plan built them before the batching."""
    fold_idx = np.random.default_rng(random_state).permutation(x.size) % n_splits
    out = []
    for k in range(n_splits):
        train, val = fold_idx != k, fold_idx == k
        if val.sum() < 4:
            continue
        cells = []
        for m in candidates:
            if m < 2 or int(train.sum()) < m:
                continue
            edges = A._edges_from_quantiles(x[train], m)
            if edges.size:
                cells.append((m, np.searchsorted(edges, x[val], side="right")))
        out.append(cells)
    return out


def test_batched_fold_edges_give_the_per_candidate_bins():
    """Columns with ties, NaNs and few rows: same candidates and the same validation bins in every fold."""
    rng = np.random.default_rng(0)
    for t in range(40):
        n = int(rng.integers(30, 4000))
        x = np.round(rng.normal(size=n), int(rng.integers(0, 4)))
        x[rng.random(n) < 0.02] = np.nan
        candidates = (2, 4, 8, 16, 32)
        ref = _reference_cells(x, candidates, 3, t)
        got = [fold[2] for fold in _optimal_joint_plan(x, candidates, 3, "quantile", t)["folds"]]
        assert len(ref) == len(got)
        for ref_cells, got_cells in zip(ref, got):
            assert [m for m, _ in ref_cells] == [m for m, _b, _k in got_cells]
            assert all(np.array_equal(a, b) for (_m, a), (_m2, b, _k) in zip(ref_cells, got_cells))


def test_fold_masks_are_shared_and_read_only():
    """Columns of one length share one set of fold masks (drawn once), and the shared masks cannot be written through."""
    import pytest

    rng = np.random.default_rng(3)
    a = _optimal_joint_plan(rng.normal(size=500), (4, 8), 3, "quantile", 0)["folds"]
    b = _optimal_joint_plan(rng.normal(size=500), (4, 8), 3, "quantile", 0)["folds"]
    assert all(fa[1] is fb[1] for fa, fb in zip(a, b))
    with pytest.raises(ValueError):
        a[0][1][0] = not a[0][1][0]
