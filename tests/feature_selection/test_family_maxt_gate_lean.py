"""The memory-lean family gate (int16 codes, per-candidate early stop) decides exactly as the full permutation null."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._family_maxt_gate import _family_codes, exact_pvalue_pass, family_maxt_pass, family_null_max, mi_all_columns_njit


def _frame(n=3000, k=40, seed=0):
    """Frame."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n) > 0).astype(np.int64)
    X[rng.random(n) < 0.01, 5] = np.nan
    return X, y


def test_codes_are_int16_and_match_column_by_column():
    """Codes are int16 and match column by column."""
    X, _ = _frame()
    cols, nx = _family_codes(X, 10)
    assert cols.dtype == np.int16 and cols.shape == (X.shape[1], X.shape[0])
    cols_df, nx_df = _family_codes(pd.DataFrame(X), 10)
    np.testing.assert_array_equal(cols, cols_df)
    np.testing.assert_array_equal(nx, nx_df)


def test_early_stop_pass_equals_full_null():
    """Early stop pass equals full null."""
    for seed in range(3):
        X, y = _frame(seed=seed)
        for n_perm in (99, 999):
            cols, nx = _family_codes(X, 10)
            _, y_codes = np.unique(y, return_inverse=True)
            y_codes = y_codes.astype(np.int64)
            n_y = int(y_codes.max()) + 1
            observed = mi_all_columns_njit(cols, nx, y_codes, n_y)
            null_max = family_null_max(cols, nx, y_codes, n_y, n_perm, np.random.default_rng(seed))
            full = exact_pvalue_pass(observed, null_max, 0.01)
            lean = family_maxt_pass(X, y, n_bins=10, alpha=0.01, n_perm=n_perm, seed=seed)
            np.testing.assert_array_equal(full, lean)
            if n_perm == 999:
                assert lean[0] and not lean[2:].any()


def _reference_family_pass(batches, y_codes, alpha, n_perm, seed=0):
    """The family decision computed on int64 stacked codes (how the gate was built before the narrow-width stack)."""
    from mlframe.feature_selection.filters._family_maxt_gate import exact_pvalue_pass as _unused  # noqa: F401

    y = np.ascontiguousarray(np.asarray(y_codes, dtype=np.int64).ravel())
    n_y = int(y.max()) + 1
    mats = [np.asarray(c, dtype=np.int64) for c in batches]
    widths = [m.shape[1] for m in mats]
    cols = np.ascontiguousarray(np.hstack(mats).T)
    n_x = (cols.max(axis=1) + 1).astype(np.int64)
    observed = mi_all_columns_njit(cols, n_x, y, n_y)
    bounds = np.cumsum([0, *widths])
    obs_max = np.array([float(observed[bounds[j] : bounds[j + 1]].max()) for j in range(len(widths))])
    null_max = family_null_max(cols, n_x, y, n_y, n_perm, np.random.default_rng(seed))
    r = (null_max[:, None] >= obs_max[None, :]).sum(axis=0)
    return [bool((1.0 + k) / (n_perm + 1.0) <= alpha) for k in r]


def test_family_pass_batches_narrow_stack_equals_int64_reference():
    """Family pass batches narrow stack equals int64 reference."""
    from mlframe.feature_selection.filters._family_maxt_gate import _stack_codes_transposed, family_pass_batches

    rng = np.random.default_rng(3)
    n = 2000
    y = rng.integers(0, 2, n)
    noise = [rng.integers(0, 6, (n, w)).astype(np.int8) for w in (5, 8)]
    sig = rng.integers(0, 6, (n, 3)).astype(np.int8)
    sig[:, 0] = np.clip(y * 3 + rng.integers(0, 3, n), 0, 5)
    batches = [noise[0], sig, noise[1]]
    got = family_pass_batches(batches, y, alpha=0.01, n_perm=999)
    assert got == _reference_family_pass(batches, y, 0.01, 999)
    assert got == [False, True, False]
    stacked = _stack_codes_transposed([np.asarray(b) for b in batches], [b.shape[1] for b in batches])
    assert stacked.dtype == np.int8 and stacked.shape == (16, n)
    wide = [np.asarray(b, dtype=np.int64) * 300 for b in batches]
    assert _stack_codes_transposed(wide, [b.shape[1] for b in wide]).dtype == np.int16
