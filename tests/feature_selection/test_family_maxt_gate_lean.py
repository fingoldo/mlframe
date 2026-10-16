"""The memory-lean family gate (int16 codes, per-candidate early stop) decides exactly as the full permutation null."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._family_maxt_gate import _family_codes, exact_pvalue_pass, family_maxt_pass, family_null_max, mi_all_columns_njit


def _frame(n=3000, k=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n) > 0).astype(np.int64)
    X[rng.random(n) < 0.01, 5] = np.nan
    return X, y


def test_codes_are_int16_and_match_column_by_column():
    X, _ = _frame()
    cols, nx = _family_codes(X, 10)
    assert cols.dtype == np.int16 and cols.shape == (X.shape[1], X.shape[0])
    cols_df, nx_df = _family_codes(pd.DataFrame(X), 10)
    np.testing.assert_array_equal(cols, cols_df)
    np.testing.assert_array_equal(nx, nx_df)


def test_early_stop_pass_equals_full_null():
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
