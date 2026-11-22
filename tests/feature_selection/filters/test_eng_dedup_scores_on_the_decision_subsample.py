"""The cross-stage engineered dedup scores its candidates on the fit's shared decision subsample, and drops the same duplicates as a full-row scan."""

from __future__ import annotations

import types

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._engineered_dedup import _dedup_scan_view
from mlframe.feature_selection.filters._mrmr_fit_impl._eng_dedup_scan import scan_engineered_duplicates


def _frame(n=6000, seed=0):
    """Two engineered monotone twins of one column, an independent column and a noisy near-twin below the 0.99 rank correlation."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "eng_a": a,
            "eng_a_twin": np.exp(a / 3.0),
            "eng_b": rng.standard_normal(n),
            "eng_a_noisy": a + 1.5 * rng.standard_normal(n),
        }
    )
    y = (a + 0.3 * rng.standard_normal(n) > 0).astype(np.int64)
    return X, y


def _fit_stub(idx, subsample_n):
    """A stand-in for the estimator with only the attributes ``fe_subsample_kwargs`` reads."""
    return types.SimpleNamespace(fe_check_pairs_subsample_n=subsample_n, random_seed=0, _fe_shared_subsample_idx=idx)


def test_the_view_is_the_shared_subsample_of_the_candidate_columns_only():
    """Rows are the shared draw (in order), columns are the candidates, and y is cut at the same rows."""
    X, y = _frame()
    X["raw_extra"] = 1.0
    idx = np.sort(np.random.default_rng(1).choice(len(X), size=1500, replace=False))
    view, y_view = _dedup_scan_view(_fit_stub(idx, 1500), X, ["eng_a", "eng_a_twin", "eng_b", "eng_a_noisy"], y)
    assert list(view.columns) == ["eng_a", "eng_a_twin", "eng_b", "eng_a_noisy"]
    np.testing.assert_array_equal(view["eng_b"].to_numpy(), X["eng_b"].to_numpy()[idx])
    np.testing.assert_array_equal(y_view, y[idx])


def test_no_subsample_or_a_single_candidate_leaves_the_inputs_alone():
    """Without a draw (or with fewer than two candidates) the frame and the target come back as they are."""
    X, y = _frame()
    same_X, same_y = _dedup_scan_view(_fit_stub(None, 0), X, ["eng_a", "eng_a_twin"], y)
    assert same_X is X and same_y is y
    idx = np.arange(0, len(X), 4)
    one_X, one_y = _dedup_scan_view(_fit_stub(idx, len(idx)), X, ["eng_a"], y)
    assert one_X is X and one_y is y


def test_the_subsample_scan_drops_the_same_columns_as_the_full_scan():
    """Monotone twins drop and the noisy near-twin and the independent column stay, on 1500 of 6000 rows as on all of them."""
    X, y = _frame()
    cols = ["eng_a", "eng_a_twin", "eng_b", "eng_a_noisy"]
    idx = np.sort(np.random.default_rng(2).choice(len(X), size=1500, replace=False))
    view, _y = _dedup_scan_view(_fit_stub(idx, 1500), X, cols, y)

    def prefer(cand, kept):
        """Within one stage the first appended column stays."""
        return False

    full_drop = scan_engineered_duplicates(X, cols, set(), prefer)[1]
    sub_drop = scan_engineered_duplicates(view, cols, set(), prefer)[1]
    assert full_drop == sub_drop == {"eng_a_twin"}
