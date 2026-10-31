"""The block binner of the CMI candidate scorer gives the codes of the per-column ``_quantile_bin`` for every kind of column."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _prebinned_candidates, score_candidates_by_cmi
from mlframe.feature_selection.filters._mi_greedy_cmi_fe_binning import _quantile_bin


def _candidates(n=4000, seed=0):
    """Continuous, tie-heavy, two-valued, constant and NaN-bearing columns."""
    rng = np.random.default_rng(seed)
    cols = {f"c{i}": rng.normal(size=n) for i in range(14)}
    cols["ties"] = np.round(rng.normal(size=n), 1)
    cols["binary"] = (rng.random(n) > 0.3).astype(float)
    cols["const"] = np.full(n, 2.0)
    cols["few"] = rng.integers(0, 4, size=n).astype(float)
    nan_col = rng.normal(size=n)
    nan_col[5] = np.nan
    cols["nan"] = nan_col
    return pd.DataFrame(cols)


def test_block_codes_equal_per_column_codes():
    """Every finite column of the block has exactly the codes ``_quantile_bin`` gives it; the NaN column is left to the per-column path."""
    X = _candidates()
    pre = _prebinned_candidates(X, list(X.columns), 10)
    assert "nan" not in pre and len(pre) == X.shape[1] - 1
    for c, codes in pre.items():
        np.testing.assert_array_equal(codes.astype(np.int64), _quantile_bin(X[c].to_numpy(), nbins=10), err_msg=c)


def test_scores_equal_the_per_column_scores(monkeypatch):
    """The pooled scores with pre-binned codes equal the scores with the block binner switched off."""
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe as M

    X = _candidates(seed=1)
    y = (X["c0"].to_numpy() + np.random.default_rng(2).normal(size=len(X)) > 0).astype(int)
    support = X[["c1", "c2", "ties"]]
    got = score_candidates_by_cmi(X, y, support, nbins=10)
    monkeypatch.setattr(M, "_prebinned_candidates", lambda *a, **k: {})
    ref = score_candidates_by_cmi(X, y, support, nbins=10)
    pd.testing.assert_series_equal(got, ref, check_exact=True)
