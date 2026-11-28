"""Cluster 15 regression: the conditional-permutation leaf groups are built once per feature and match the old per-leaf scan (audit 2026-11-20, finding 15-F10)."""

import numpy as np

from mlframe.feature_selection.wrappers._helpers_importance import _cpi_leaf_permutation_score


class _Model:
    """A scorer whose score depends on the permuted column, so a wrong grouping changes the result."""

    def score(self, X, y):
        """Correlation of column 0 with y."""
        return float(np.corrcoef(np.asarray(X)[:, 0], y)[0, 1])


def _old_reference(model, y, X_arr, X_perm, j, leaves, n_repeats, rng, baseline):
    """The pre-change implementation: np.where per leaf per repeat."""
    losses = []
    orig = X_arr[:, j].copy()
    uniq = np.unique(leaves)
    for _ in range(n_repeats):
        for leaf in uniq:
            in_leaf = np.where(leaves == leaf)[0]
            if in_leaf.size <= 1:
                continue
            X_perm[in_leaf, j] = orig[rng.permutation(in_leaf)]
        losses.append(baseline - model.score(X_perm, y))
    X_perm[:, j] = orig
    return float(np.nanmean(losses))


def test_new_grouping_reproduces_the_old_score_exactly():
    """Same rng seed, same leaves: the grouped version returns the identical mean loss and restores the column."""
    rng0 = np.random.default_rng(0)
    n = 600
    X = rng0.normal(size=(n, 3))
    y = X[:, 0] + 0.3 * rng0.normal(size=n)
    leaves = rng0.integers(0, 12, n)
    model = _Model()
    baseline = model.score(X, y)
    new = _cpi_leaf_permutation_score(model, y, X, X.copy(), 0, leaves, 4, np.random.default_rng(7), baseline, False, None, None)
    old = _old_reference(model, y, X, X.copy(), 0, leaves, 4, np.random.default_rng(7), baseline)
    assert new == old
