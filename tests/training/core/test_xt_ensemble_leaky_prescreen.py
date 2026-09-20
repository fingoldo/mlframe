"""The refit-saving pre-screen must run under the default OOF source, not only under ``external_val``.

The dummy-floor gate at the end of the cross-target ensemble phase discards most components (14 of 21 in prod) only
after each has been K-fold OOF-refit, at roughly 5 minutes a booster and 10 a MLP. The cheap leaky-RMSE screen written
to avoid that was reachable only when ``oof_holdout_source='external_val'`` supplied the frame, so under the shipped
default it never ran. It now falls back to the validation split, which is available on both paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.core._phase_composite_post_xt_ensemble._prescreen import (
    PRESCREEN_SAFETY,
    dummy_floor_from_metadata,
    leaky_rmse_keep_mask,
    prescreen_frame,
)


class _Constant:
    """A component whose prediction is a fixed offset from the truth, and which records every predict call."""

    def __init__(self, offset: float):
        self.offset = offset
        self.calls = 0

    def predict(self, X):
        """Return the stored offset for every row."""
        self.calls += 1
        return np.full(len(X), self.offset, dtype=np.float64)


class _Broken:
    """A component whose predict raises, as a half-fitted model can."""

    def predict(self, X):
        """Fail the way a degenerate component does."""
        raise RuntimeError("no model")


def _frame(n: int = 50) -> pd.DataFrame:
    """A frame of the shape the components predict on; only its length matters here."""
    return pd.DataFrame({"f": np.arange(n, dtype=np.float64)})


def test_the_validation_split_is_used_when_the_oof_path_supplies_no_frame():
    """The K-fold source leaves the OOF frame empty; the screen must still get the val frame and its y."""
    val_df = _frame()
    y_full = np.arange(200, dtype=np.float64)
    val_idx = np.arange(100, 150)

    X, y = prescreen_frame(None, None, val_df, y_full, val_idx)
    assert X is val_df, "the val frame must be used when the OOF source supplies none"
    np.testing.assert_array_equal(y, y_full[val_idx])


def test_an_oof_frame_takes_precedence_over_the_validation_split():
    """When the OOF source does supply a frame, the screen keeps using it: the fallback only fills a gap."""
    ext_X, ext_y = _frame(10), np.zeros(10)
    X, y = prescreen_frame(ext_X, ext_y, _frame(50), np.arange(200.0), np.arange(50))
    assert X is ext_X and y is ext_y


def test_without_a_validation_split_the_screen_is_skipped():
    """No frame means no screen; every component goes to the honest refit, as before."""
    assert prescreen_frame(None, None, None, np.arange(10.0), np.arange(5)) == (None, None)
    assert prescreen_frame(None, None, _frame(10), np.arange(10.0), None) == (None, None)


def test_a_component_that_cannot_clear_the_floor_is_dropped_and_a_good_one_is_kept():
    """The drop rule is leaky RMSE / safety margin against the dummy floor."""
    y = np.zeros(50)
    good, hopeless = _Constant(0.5), _Constant(100.0)
    keep, dropped = leaky_rmse_keep_mask([good, hopeless], ["good", "hopeless"], _frame(), y, dummy_floor=1.0)
    assert keep == [True, False], f"expected the hopeless component to be dropped; got {keep}"
    assert dropped and dropped[0].startswith("hopeless")


def test_a_component_just_inside_the_safety_margin_survives():
    """The margin is generous by design: a component the honest gate might still pass is not pre-dropped."""
    y = np.zeros(50)
    borderline = _Constant(1.0 * PRESCREEN_SAFETY * 0.99)
    keep, dropped = leaky_rmse_keep_mask([borderline], ["borderline"], _frame(), y, dummy_floor=1.0)
    assert keep == [True] and not dropped


def test_a_component_whose_predict_raises_is_kept():
    """The screen saves refits; it must never decide a case it could not measure."""
    keep, dropped = leaky_rmse_keep_mask([_Broken()], ["broken"], _frame(), np.zeros(50), dummy_floor=1.0)
    assert keep == [True] and not dropped


def test_too_few_finite_rows_keeps_the_component():
    """Under ten jointly finite rows the leaky RMSE is noise, so the component survives to the honest gate."""
    y = np.full(50, np.nan)
    y[:5] = 0.0
    keep, dropped = leaky_rmse_keep_mask([_Constant(100.0)], ["sparse"], _frame(), y, dummy_floor=1.0)
    assert keep == [True] and not dropped


def test_the_floor_comes_from_the_strongest_dummy_baseline():
    """The floor is that dummy's primary-metric value; anything incomplete yields no floor and no screening."""
    metadata = {"dummy_baselines": {"TargetTypes.REGRESSION": {"y": {
        "strongest": "median", "primary_metric": "RMSE",
        "data": {"median": {"RMSE": 3.25}, "mean": {"RMSE": 4.0}},
    }}}}
    assert dummy_floor_from_metadata(metadata, "TargetTypes.REGRESSION", "y") == 3.25
    assert dummy_floor_from_metadata({}, "TargetTypes.REGRESSION", "y") is None
    metadata["dummy_baselines"]["TargetTypes.REGRESSION"]["y"]["data"]["median"]["RMSE"] = float("nan")
    assert dummy_floor_from_metadata(metadata, "TargetTypes.REGRESSION", "y") is None
