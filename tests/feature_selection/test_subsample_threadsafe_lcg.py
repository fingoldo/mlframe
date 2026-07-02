"""Regression test for audit2 H1: stratified-subsample njit kernels must not touch the process-global
np.random state (they used np.random.seed inside njit -> concurrent joblib-threading callers clobbered
each other). The kernels now use a per-call inline LCG; assert determinism AND that the global stream
is left untouched.
"""
import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_subsample import stratified_subsample_idx


@pytest.mark.parametrize("is_clf", [True, False])
def test_subsample_deterministic_and_does_not_mutate_global_rng(is_clf):
    if is_clf:
        y = np.array([0] * 80 + [1] * 20, dtype=np.int64)
    else:
        y = np.linspace(0.0, 1.0, 100).astype(np.float64)

    np.random.seed(12345)
    before = np.random.get_state()[1].copy()

    r1 = np.sort(stratified_subsample_idx(np.random.default_rng(7), y, 50, is_clf=is_clf))
    r2 = np.sort(stratified_subsample_idx(np.random.default_rng(7), y, 50, is_clf=is_clf))

    after = np.random.get_state()[1]

    assert np.array_equal(r1, r2), "same seed must yield identical subsample (determinism)"
    assert np.array_equal(before, after), "kernel must NOT mutate the global np.random MT19937 state"
    if is_clf:
        assert (y[r1] == 0).any() and (y[r1] == 1).any(), "both strata must be represented"


def test_different_seeds_give_different_subsamples():
    y = np.array([0] * 80 + [1] * 20, dtype=np.int64)
    a = set(stratified_subsample_idx(np.random.default_rng(1), y, 50, is_clf=True).tolist())
    b = set(stratified_subsample_idx(np.random.default_rng(2), y, 50, is_clf=True).tolist())
    assert a != b, "distinct seeds should draw distinct rows (LCG actually consumes the seed)"
