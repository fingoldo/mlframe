"""CPX34: reused (n, p-1) conditioning-set scratch must be bit-identical to per-feature np.delete.

The old _conditional_permutation_importance built X_{-j} via ``np.delete(X_arr, j, axis=1)``
each iteration (a fresh allocation per feature). The optimization refills one reused
C-contiguous (n, p-1) buffer with two contiguous block-copies. This must produce exactly
the same conditioning set the estimator sees, hence bit-identical importances.

This test pins the buffer-build correctness directly (the np.delete equivalence) and the
end-to-end importance identity vs a verbatim np.delete reference.
"""

import numpy as np
import pytest

from mlframe.feature_selection.wrappers._helpers_importance import _conditional_permutation_importance


def _delete_reference(X, j):
    """Delete reference."""
    return np.delete(X, j, axis=1)


def _scratch_build(X, j):
    """Mirror of the production reused-buffer fill."""
    n, p = X.shape
    buf = np.empty((n, p - 1), dtype=X.dtype)
    if j > 0:
        buf[:, :j] = X[:, :j]
    if j < p - 1:
        buf[:, j:] = X[:, j + 1 :]
    return buf


@pytest.mark.parametrize("p", [2, 3, 7, 20])
def test_scratch_build_byte_identical_to_np_delete(p):
    """Scratch build byte identical to np delete."""
    rng = np.random.default_rng(p)
    X = rng.standard_normal((50, p))
    for j in range(p):
        assert np.array_equal(_scratch_build(X, j), _delete_reference(X, j)), f"mismatch at j={j}, p={p}"


def _make(n=4000, p=30, seed=0):
    """Builds seeded synthetic test data; returns ``(X, y)``."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    X[:, 1] += 0.7 * X[:, 0]
    X[:, 2] += 0.5 * X[:, 0]
    y = X[:, 0] * 1.5 + X[:, 3] - 0.8 * X[:, 5] + rng.standard_normal(n) * 0.3
    return X, y


def _reference_cpi(model, X, y, n_repeats, random_state):
    """Verbatim np.delete-based reference of the importance loop."""
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

    X_arr = np.asarray(X)
    _n, p = X_arr.shape
    rng = np.random.default_rng(random_state)
    baseline = float(model.score(X, y))
    importances = np.zeros(p, dtype=float)

    def _is_discrete_v2(col):
        """Is discrete v2."""
        if np.issubdtype(col.dtype, np.integer):
            return True
        mask = ~np.isnan(col.astype(float, copy=False))
        uniq = np.unique(col[mask])
        _n = max(int(mask.sum()), 1)
        return uniq.size <= max(5, int(np.sqrt(_n))) and uniq.size <= 0.5 * _n

    X_perm = X_arr.copy()
    for j in range(p):
        Xj = X_arr[:, j]
        Xnotj = np.delete(X_arr, j, axis=1)
        if Xnotj.shape[1] == 0:
            score_losses = []
            orig_col = X_arr[:, j].copy()
            try:
                for _ in range(n_repeats):
                    X_perm[:, j] = rng.permutation(orig_col)
                    score_losses.append(baseline - float(model.score(X_perm, y)))
            finally:
                X_perm[:, j] = orig_col
            importances[j] = float(np.mean(score_losses))
            continue
        tree = (DecisionTreeClassifier if _is_discrete_v2(Xj) else DecisionTreeRegressor)(max_depth=None, min_samples_leaf=10, random_state=random_state)
        try:
            tree.fit(Xnotj, Xj)
            leaves = tree.apply(Xnotj)
        except (ValueError, TypeError, MemoryError, RuntimeError):
            importances[j] = 0.0
            continue
        score_losses = []
        orig_col = X_arr[:, j].copy()
        unique_leaves = np.unique(leaves)
        try:
            for _ in range(n_repeats):
                for leaf_id in unique_leaves:
                    in_leaf = np.where(leaves == leaf_id)[0]
                    if in_leaf.size <= 1:
                        continue
                    shuffled_positions = rng.permutation(in_leaf)
                    X_perm[in_leaf, j] = orig_col[shuffled_positions]
                try:
                    score_losses.append(baseline - float(model.score(X_perm, y)))
                except Exception:
                    score_losses.append(np.nan)
        finally:
            X_perm[:, j] = orig_col
        importances[j] = float(np.nanmean(score_losses)) if any(not np.isnan(s) for s in score_losses) else 0.0
    return importances


def test_importances_bit_identical_to_np_delete_reference():
    """Importances bit identical to np delete reference."""
    from sklearn.linear_model import Ridge

    X, y = _make()
    model = Ridge().fit(X, y)
    ref = _reference_cpi(model, X, y, n_repeats=2, random_state=0)
    got = _conditional_permutation_importance(model, X, y, n_repeats=2, random_state=0)
    assert np.array_equal(ref, got), f"max|diff|={np.max(np.abs(ref - got))}"


def test_single_feature_path_runs():
    """Single feature path runs."""
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 1))
    y = X[:, 0] + rng.standard_normal(200) * 0.1
    model = Ridge().fit(X, y)
    imp = _conditional_permutation_importance(model, X, y, n_repeats=2, random_state=0)
    assert imp.shape == (1,)
