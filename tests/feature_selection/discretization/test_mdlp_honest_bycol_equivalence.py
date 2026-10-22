"""The column-parallel honest null kernel returns exactly what the permutation-parallel reference returns."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._analytic_mi_null import _min_expected_cell, analytic_null_min_n
from mlframe.feature_selection.filters._mdlp_honest_bycol_njit import honest_null_columns_bycol, permuted_labels
from mlframe.feature_selection.filters._mdlp_honest_njit import honest_null_columns


def _inputs(n, k_cont, k_fixed, n_perm, n_classes, seed):
    """Mixed pool: continuous columns (some with heavy ties), fixed-code columns, an informative continuous column, a row-subsample for the MI."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, k_cont))
    raw[:, 0] = np.round(raw[:, 0] * 3)
    y = rng.integers(0, n_classes, n).astype(np.int64)
    raw[:, 1] += 1.5 * (y == 1)
    k = k_cont + k_fixed
    fixed = rng.integers(0, 4, (n, max(1, k_fixed))).astype(np.int8)
    is_cont = np.array([True] * k_cont + [False] * k_fixed)
    slot = np.array(list(range(k_cont)) + list(range(k_fixed)), dtype=np.int64)
    qf = np.quantile(raw, [0.25, 0.5, 0.75], axis=0).T.copy()
    q = np.zeros((k, 3))
    q[:k_cont] = qf
    perms = np.stack([rng.permutation(n) for _ in range(n_perm)]).astype(np.int64)
    ev = np.sort(rng.choice(n, n * 3 // 5, replace=False)).astype(np.int64)
    common = dict(
        raw_cont=raw, is_cont=is_cont, fixed_codes=fixed, fixed_nbins=np.full(k, 4, dtype=np.int64), n_classes_total=n_classes, min_split_size=20,
        max_depth=8, alpha=0.05, n_permutations=100, seed=0, analytic_min_n=int(analytic_null_min_n()), analytic_min_cell=float(_min_expected_cell()),
        q_fallback=q, q_fallback_len=np.full(k, 3, dtype=np.int64), q_refine=q.copy(), q_refine_len=np.full(k, 3, dtype=np.int64), eval_rows=ev, col_slot=slot,
    )
    return common, y, perms


@pytest.mark.parametrize("n_classes", [2, 4])
def test_bycol_matches_the_permutation_parallel_reference(n_classes):
    """Same plain and corrected null MI for every (permutation, column), continuous and fixed alike."""
    common, y, perms = _inputs(6000, 5, 2, 12, n_classes, seed=n_classes)
    common["analytic_min_n"] = 1000  # analytic regime: no sampled root test, so the two kernels must agree bit for bit
    ref = honest_null_columns(y_raw=y, y_codes=y, perms=perms, **common)
    y_raw_perm, y_codes_perm = permuted_labels(y, y, iter(perms), len(perms))
    got = honest_null_columns_bycol(y_raw_perm=y_raw_perm, y_codes_perm=y_codes_perm, nan_cont=np.zeros(common["raw_cont"].shape, dtype=bool), **common)
    assert len(ref) == len(got) == 2, "plain and corrected null MI must both be compared"
    for a, b in zip(ref, got):
        assert np.array_equal(a, b)


def test_root_null_threshold_is_the_null_quantile_of_the_best_split_gain():
    """The shared root threshold accepts about ``alpha`` of tie-free columns under a null target, matching the per-column permutation test it replaces."""
    from mlframe.feature_selection.filters._mdlp_honest_bycol_njit import root_null_gain_threshold
    from mlframe.feature_selection.filters._mdlp_honest_njit import _mdlp_best_split_njit

    rng = np.random.default_rng(3)
    n = 1500
    labels = rng.integers(0, 3, n).astype(np.int64)
    thr = root_null_gain_threshold(labels, 3, 20, 0.05, 2000, 17)
    xs = np.arange(n, dtype=np.float64)
    hits = sum(_mdlp_best_split_njit(xs, rng.permutation(labels), 3, 20)[1] > thr for _ in range(1500))
    assert 0.02 < hits / 1500 < 0.08
    assert root_null_gain_threshold(np.zeros(50, dtype=np.int64), 3, 5, 0.05, 50, 1) == np.inf


def test_permuted_labels_uses_the_narrowest_integer_type_and_reads_the_rows_as_drawn():
    """The label matrices are ``labels[perm]`` row by row, narrow when the labels allow."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 100).astype(np.int64)
    perms = [rng.permutation(100) for _ in range(3)]
    a, b = permuted_labels(y, y + 200, iter(perms), 3)
    assert a.dtype == np.int8 and b.dtype == np.int16
    assert all(np.array_equal(a[i], y[p]) and np.array_equal(b[i], (y + 200)[p]) for i, p in enumerate(perms))
