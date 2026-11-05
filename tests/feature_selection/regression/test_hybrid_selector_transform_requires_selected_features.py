"""``HybridSelector.transform`` must produce every selected feature or raise, not hand a model fewer or different columns.

A selector fitted on 8 raw columns selected 9 features (raw plus engineered). ``transform`` on a frame missing one of the selected raw columns, or
when the engineered-feature replay failed, silently dropped every selected feature it could not produce and returned the rest, so a model trained
on 9 columns received 4. The only signal was a warning that Python's default filter shows once.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.hybrid_selector import HybridSelector


@pytest.fixture(scope="module")
def fitted():
    """A selector fitted on a frame whose selection includes engineered columns, with the frame it was fitted on."""
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"f{i}" for i in range(8)])
    y = pd.Series((X["f0"] + X["f1"] + 0.3 * rng.normal(size=n) > 0).astype(int))
    return HybridSelector(prescreen=True).fit(X, y), X


def test_transform_on_the_fitted_frame_returns_exactly_the_selection(fitted):
    """The normal path is unchanged: the selected columns, in selection order."""
    sel, X = fitted
    assert list(sel.transform(X).columns) == list(sel.raw_selected_)


def test_transform_raises_when_a_selected_raw_column_is_missing(fitted):
    """A frame without a selected raw column raises instead of returning the features that remain."""
    sel, X = fitted
    raw_selected = [c for c in sel.raw_selected_ if c in X.columns]
    assert raw_selected, "the fixture must select at least one raw column"
    with pytest.raises(ValueError, match="cannot be produced"):
        sel.transform(X.drop(columns=[raw_selected[0]]))


def test_transform_raises_when_the_engineered_replay_fails(fitted, monkeypatch):
    """A selected engineered column whose replay fails raises, with the failure named, rather than being dropped."""
    sel, X = fitted
    engineered = [c for c in sel.raw_selected_ if str(c).startswith("eng_")]
    if not engineered or sel._mrmr_member is None:
        pytest.skip("the fixture selection holds no MRMR-engineered column")

    def _boom(_X):
        """Stand-in for a replay that fails on serving data."""
        raise RuntimeError("replay failed")

    monkeypatch.setattr(sel._mrmr_member, "transform", _boom)
    with pytest.warns(UserWarning, match="FE augment failed"), pytest.raises(ValueError, match="cannot be produced"):
        sel.transform(X)
