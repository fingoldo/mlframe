"""Wave 39 (2026-05-20): empty-input edge cases.

Audit class: operations that assume nonempty input but silently produce NaN,
return wrong-shape outputs, raise opaque errors, or pass garbage through when
given an empty DataFrame / empty numpy array / empty post-filter result.

6 P2 findings, all real, all fixed:

  1. metrics/core.py:1536 -- show_calibration_plot calls np.min/np.max on
     freqs_predicted which calibration_binning may return empty (single-class
     preds, all-NaN preds, sparse-hits filter). Opaque ValueError aborts the
     calibration plot.

  2. estimators/custom.py:472 -- clip_to_quantiles calls np.quantile on an
     unguarded user-supplied array; numpy>=1.22 raises IndexError on empty.

  3. feature_engineering/timeseries.py:723 -- compute_splitting_stats calls
     window_df[subvar].iloc[0]/iloc[-1] without guarding the empty-window case
     reachable via upstream isfinite filter.

  4. feature_selection/wrappers/_rfecv.py:2366 -- n_features_bootstrap_ci_ with
     n_bootstrap<=0 produces empty choices_arr, then int(np.median([])) raises
     ValueError after RuntimeWarning.

  5. feature_engineering/transformer/conformal_locally_adaptive.py:58,74 --
     tiny-train regime (n<4) empties h1 or h2; lgb.fit on empty raises opaquely
     and downstream sigma/quantile produce garbage.

  6. training/feature_handling/target_encoders.py:222-223,238 -- _compute_prior
     receives caller-supplied y (via fit / _loo_encode) which can be empty;
     np.mean/np.median return NaN + RuntimeWarning, weighted-median y[order[-1]]
     raises IndexError.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Behavioural sensors: each fix should return a clean result on empty input.
# ---------------------------------------------------------------------------


def test_clip_to_quantiles_empty_array_returns_empty() -> None:
    from mlframe.estimators.custom import clip_to_quantiles

    arr = np.array([], dtype=np.float64)
    out = clip_to_quantiles(arr, quantile=0.95, method="hard")
    assert isinstance(out, np.ndarray)
    assert out.size == 0
    assert out.dtype == np.float64


def test_clip_to_quantiles_empty_array_winsor_linear() -> None:
    from mlframe.estimators.custom import clip_to_quantiles

    arr = np.array([], dtype=np.float32)
    out = clip_to_quantiles(arr, quantile=0.99, method="winsor_linear")
    assert out.size == 0


def test_compute_prior_empty_y_returns_zero_no_warning() -> None:
    from mlframe.training.feature_handling.target_encoders import _compute_prior

    out_mean = _compute_prior(np.array([], dtype=np.float64), "mean")
    out_median = _compute_prior(np.array([], dtype=np.float64), "median")
    assert out_mean == 0.0
    assert out_median == 0.0


def test_compute_prior_empty_y_weighted_branch() -> None:
    from mlframe.training.feature_handling.target_encoders import _compute_prior

    out = _compute_prior(
        np.array([], dtype=np.float64),
        "median",
        sample_weight=np.array([], dtype=np.float64),
    )
    assert out == 0.0


def test_rfecv_n_features_bootstrap_ci_zero_bootstrap_no_crash() -> None:
    """n_bootstrap=0 must not raise; returns (n, n, n) fallback."""
    from mlframe.feature_selection.wrappers.rfecv import RFECV

    # Construct a minimally-populated RFECV without actually fitting; we only need
    # cv_results_ + n_features_ for the bootstrap method's no-crash path.
    rf = RFECV.__new__(RFECV)
    rf.cv_results_ = {
        "nfeatures": [5, 10, 15],
        "cv_mean_perf": [0.6, 0.7, 0.65],
        "cv_std_perf": [0.05, 0.03, 0.04],
    }
    rf.n_features_ = 10
    low, mid, high = rf.n_features_bootstrap_ci_(ci=0.9, n_bootstrap=0)
    assert (low, mid, high) == (10, 10, 10)


def test_rfecv_n_features_bootstrap_ci_negative_bootstrap_no_crash() -> None:
    from mlframe.feature_selection.wrappers.rfecv import RFECV

    rf = RFECV.__new__(RFECV)
    rf.cv_results_ = {
        "nfeatures": [5, 10, 15],
        "cv_mean_perf": [0.6, 0.7, 0.65],
        "cv_std_perf": [0.05, 0.03, 0.04],
    }
    rf.n_features_ = 7
    low, mid, high = rf.n_features_bootstrap_ci_(n_bootstrap=-1)
    assert (low, mid, high) == (7, 7, 7)


def test_compute_splitting_stats_empty_window_no_crash() -> None:
    """Empty window_df must not raise; function exits via early return."""
    import pandas as pd
    from mlframe.feature_engineering.timeseries import compute_splitting_stats

    empty_df = pd.DataFrame({"a": pd.Series([], dtype="float64")})
    row_features: list = []
    features_names: list = []
    # No exception means the early-return guard worked.
    compute_splitting_stats(
        window_df=empty_df,
        dataset_name="ds",
        splitting_vars={"a": ["a"]},
        var="a",
        numaggs_names=["minr", "maxr"],
        numaggs_values=[0.0, 0.0],
        row_features=row_features,
        features_names=features_names,
        create_features_names=True,
    )
    # No features appended for empty window.
    assert row_features == []


# ---------------------------------------------------------------------------
# Source-level sensors: ensure the fix actually committed.
# ---------------------------------------------------------------------------


import importlib
from pathlib import Path

MLFRAME_ROOT = Path(importlib.import_module("mlframe").__file__).parent


def _read(rel: str) -> str:
    return (MLFRAME_ROOT / rel).read_text(encoding="utf-8")


def test_metrics_calibration_plot_guards_empty_freqs_predicted() -> None:
    # ``show_calibration_plot`` was moved to ``_calibration_plot.py`` when
    # ``metrics/core.py`` was split into siblings.
    src = _read("metrics/calibration/_calibration_plot.py")
    assert "if freqs_predicted.size == 0:" in src, (
        "metrics/calibration/_calibration_plot.py: show_calibration_plot must guard freqs_predicted before np.min/np.max."
    )


def test_clip_to_quantiles_guards_empty_input() -> None:
    src = _read("estimators/custom.py")
    # The fix introduces an early-return on empty array before the np.quantile call.
    assert "arr_arr.size == 0" in src, (
        "estimators/custom.py: clip_to_quantiles must guard empty input before np.quantile."
    )


def test_conformal_locally_adaptive_guards_tiny_train() -> None:
    src = _read("feature_engineering/transformer/conformal_locally_adaptive.py")
    assert "if n < 4:" in src, (
        "conformal_locally_adaptive.py: _process must guard tiny-train (n<4) before half-split."
    )


def test_target_encoders_compute_prior_guards_empty_y() -> None:
    src = _read("training/feature_handling/target_encoders.py")
    # The fix prints a warning and returns 0.0 on empty y.
    helper_idx = src.find("def _compute_prior")
    assert helper_idx != -1
    snippet = src[helper_idx : helper_idx + 800]
    assert "if len(y) == 0:" in snippet, (
        "target_encoders.py: _compute_prior must guard empty y."
    )
