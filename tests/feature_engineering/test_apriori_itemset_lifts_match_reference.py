"""The vectorised itemset-lift scorer returns the same lifts as the per-row reference loop it replaced."""

import numpy as np
import pandas as pd

from mlframe.feature_engineering.transformer.apriori_itemsets import _itemset_lifts


def _reference(frequent, bin_matrix_train, col_index, y_t, task):
    """The original per-row loop: iterrows and a fancy-index AND for each itemset."""
    target_mean = float(y_t.mean()) if task == "binary" else float(np.median(y_t))
    out = []
    for _, row in frequent.iterrows():
        cols_idx = [col_index[it] for it in list(row["itemsets"])]
        mask = np.all(bin_matrix_train[:, cols_idx], axis=1)
        if mask.sum() == 0:
            out.append(0.0)
            continue
        if task == "binary":
            out.append(max(float(y_t[mask].mean()), 1e-6) / max(target_mean, 1e-6))
        else:
            out.append(abs(float(np.mean(y_t[mask])) - target_mean) / max(float(np.std(y_t)), 1e-9))
    return out


def test_itemset_lifts_match_the_reference_loop_for_binary_and_regression():
    """Random bin matrix and itemsets: both tasks give the reference lifts exactly."""
    rng = np.random.default_rng(0)
    n, d = 300, 6
    bins = rng.random((n, d)) < 0.5
    col_index = {f"c{j}": j for j in range(d)}
    frequent = pd.DataFrame({"itemsets": [("c0",), ("c1", "c2"), ("c3", "c4", "c5"), ("c0", "c5")]})
    y_bin = rng.integers(0, 2, n).astype(float)
    y_reg = rng.normal(size=n)
    for task, y in (("binary", y_bin), ("regression", y_reg)):
        got = _itemset_lifts(frequent, bins, col_index, y, task)
        np.testing.assert_allclose(got, _reference(frequent, bins, col_index, y, task), rtol=0, atol=0)
