"""The OOF holdout returns the same rows, in the same order, for a pandas and a polars training frame.

On the time-sorted holdout path the polars branch built X with a boolean mask (row order) while y followed the time order
(index order), so every holdout prediction was matched with another row's target (max |pred - y| 39 against 0 for pandas).
An identity component makes the check exact: its prediction is the row's own y, so any misalignment is nonzero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from mlframe.training.composite import compute_oof_holdout_predictions


class _Identity(BaseEstimator, RegressorMixin):
    """Predicts the frame's ``y_copy`` column: the prediction of a row is its own target."""

    def fit(self, X, y, **_):
        """Nothing to learn."""
        return self

    def predict(self, X):
        """The row's own target."""
        return np.asarray(X["y_copy"].to_numpy() if hasattr(X, "to_numpy") else X["y_copy"], dtype=np.float64)


@pytest.mark.parametrize("carrier", ["pandas", "polars"])
@pytest.mark.parametrize("order", ["monotone", "reversed", "shuffled"])
@pytest.mark.parametrize("kfold", [1, 3])
def test_the_holdout_rows_of_X_match_the_holdout_targets(carrier, order, kfold):
    """For every carrier and time order, the identity component's OOF prediction equals the returned holdout y exactly."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.normal(0.0, 10.0, n)
    frame = pd.DataFrame({"x": rng.normal(size=n), "y_copy": y})
    X = pl.from_pandas(frame) if carrier == "polars" else frame
    t = {"monotone": np.arange(n), "reversed": np.arange(n)[::-1], "shuffled": rng.permutation(n)}[order].astype(float)
    P, y_h, names = compute_oof_holdout_predictions(
        component_models=[_Identity()], component_names=["id"], component_specs=[None], train_X=X, y_train_full=y,
        base_train_full_per_spec={}, holdout_frac=0.3, random_state=0, time_ordering=t, kfold=kfold,
    )
    assert names == ["id"] and P.shape[0] == y_h.shape[0] > 0
    np.testing.assert_array_equal(P[:, 0], y_h)


def _row_slicers():
    """Every row-subsetting helper that branches on the frame type, as ``name -> fn(frame, idx)``."""
    from mlframe.training.composite.row_level_average_importance import _subset_rows
    from mlframe.training.core._phase_composite_post_xt_ensemble import _slice_frame_rows
    from mlframe.training.slicing._slice_helpers import _row_select

    return {"_subset_rows": _subset_rows, "_slice_frame_rows": _slice_frame_rows, "_row_select": _row_select}


@pytest.mark.parametrize("name", ["_subset_rows", "_slice_frame_rows", "_row_select"])
@pytest.mark.parametrize("order", ["monotone", "reversed", "shuffled"])
def test_row_slicers_return_the_same_rows_in_the_same_order_for_pandas_and_polars(name, order):
    """A polars frame gives the pandas rows, in the index's order (a boolean-mask filter returned them in row order)."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"a": np.arange(50, dtype=float), "b": rng.normal(size=50)})
    idx = {"monotone": np.arange(0, 50, 3), "reversed": np.arange(0, 50, 3)[::-1], "shuffled": rng.permutation(50)[:20]}[order]
    fn = _row_slicers()[name]
    got_pd = np.asarray(fn(frame, idx)["a"])
    got_pl = np.asarray(fn(pl.from_pandas(frame), idx)["a"].to_numpy())
    np.testing.assert_array_equal(got_pd, idx.astype(float))
    np.testing.assert_array_equal(got_pl, got_pd)


@pytest.mark.parametrize("carrier", ["pandas", "polars"])
@pytest.mark.parametrize("order", ["monotone", "shuffled"])
def test_the_external_holdout_rows_match_their_targets(carrier, order):
    """``oof_holdout_source='external_val'`` predicts on the caller's val frame; the returned y is that frame's, row for row.

    The K-fold and train-tail sources are covered above. This is the third source, where the holdout is a separate frame:
    a polars val frame gathered in a different order from its targets would misalign every weight the stack fits on it.
    """
    rng = np.random.default_rng(3)
    n, n_val = 200, 80
    y = rng.normal(0.0, 10.0, n)
    y_val = rng.normal(0.0, 10.0, n_val)
    train = pd.DataFrame({"x": rng.normal(size=n), "y_copy": y})
    val = pd.DataFrame({"x": rng.normal(size=n_val), "y_copy": y_val})
    if order == "shuffled":
        perm = rng.permutation(n_val)
        val, y_val = val.iloc[perm].reset_index(drop=True), y_val[perm]
    X, X_val = (pl.from_pandas(train), pl.from_pandas(val)) if carrier == "polars" else (train, val)
    P, y_h, names = compute_oof_holdout_predictions(
        component_models=[_Identity()], component_names=["id"], component_specs=[None], train_X=X, y_train_full=y,
        base_train_full_per_spec={}, holdout_frac=0.3, random_state=0, external_holdout_X=X_val, external_holdout_y=y_val,
    )
    assert names == ["id"] and P.shape[0] == n_val
    np.testing.assert_array_equal(y_h, y_val)
    np.testing.assert_array_equal(P[:, 0], y_h)
