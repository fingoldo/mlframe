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

# ---------------------------------------------------------------------------
# Behavioural sensors: each fix should return a clean result on empty input.
# ---------------------------------------------------------------------------


def test_clip_to_quantiles_empty_array_returns_empty() -> None:
    """Clip to quantiles empty array returns empty."""
    from mlframe.estimators.custom import clip_to_quantiles

    arr = np.array([], dtype=np.float64)
    out = clip_to_quantiles(arr, quantile=0.95, method="hard")
    assert isinstance(out, np.ndarray)
    assert out.size == 0
    assert out.dtype == np.float64


def test_clip_to_quantiles_empty_array_winsor_linear() -> None:
    """Clip to quantiles empty array winsor linear."""
    from mlframe.estimators.custom import clip_to_quantiles

    arr = np.array([], dtype=np.float32)
    out = clip_to_quantiles(arr, quantile=0.99, method="winsor_linear")
    assert out.size == 0


def test_compute_prior_empty_y_returns_zero_no_warning() -> None:
    """Compute prior empty y returns zero no warning."""
    from mlframe.training.feature_handling.target_encoders import _compute_prior

    out_mean = _compute_prior(np.array([], dtype=np.float64), "mean")
    out_median = _compute_prior(np.array([], dtype=np.float64), "median")
    assert out_mean == 0.0
    assert out_median == 0.0


def test_compute_prior_empty_y_weighted_branch() -> None:
    """Compute prior empty y weighted branch."""
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
    """Rfecv n features bootstrap ci negative bootstrap no crash."""
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
# Guards on degenerate inputs
# ---------------------------------------------------------------------------


def test_metrics_calibration_plot_guards_empty_freqs_predicted(tmp_path, caplog) -> None:
    """A calibration plot over empty bin data is skipped with a warning: nothing is returned or written, and nothing raises."""
    import logging

    from mlframe.metrics.calibration._calibration_plot import show_calibration_plot

    plot_file = tmp_path / "calibration.png"
    empty = np.array([], dtype=np.float64)
    with caplog.at_level(logging.WARNING):
        result = show_calibration_plot(empty, empty, np.array([], dtype=np.int64), show_plots=False, plot_file=str(plot_file))
    assert result is None
    assert not plot_file.exists()
    assert any("no bin data available" in r.getMessage() for r in caplog.records)


def test_conformal_locally_adaptive_guards_tiny_train() -> None:
    """A train fold with fewer than 4 rows yields the all-zero feature block; a normal fold yields real features."""
    from mlframe.feature_engineering.transformer.conformal_locally_adaptive import compute_conformal_locally_adaptive_features

    rng = np.random.default_rng(0)
    X_query = rng.normal(size=(4, 2)).astype(np.float32)
    tiny = compute_conformal_locally_adaptive_features(rng.normal(size=(3, 2)).astype(np.float32), np.array([0.1, 0.5, 0.9]), X_query, seed=1)
    assert tiny.shape == (4, 5)
    assert not np.any(tiny.to_numpy())
    X = rng.normal(size=(80, 2)).astype(np.float32)
    normal = compute_conformal_locally_adaptive_features(X, X[:, 0] + 0.1 * rng.normal(size=80), X_query, seed=1)
    assert normal.shape == (4, 5)
    assert np.all(np.isfinite(normal.to_numpy()))
    assert np.any(normal.to_numpy() != 0.0)


def test_target_encoders_compute_prior_guards_empty_y() -> None:
    """An empty ``y`` yields a 0.0 no-evidence prior instead of NaN or IndexError.

    Exercised rather than grepped: np.mean/np.median of an empty array return NaN with a
    RuntimeWarning, and the weighted branch indexes into ``y`` and would raise IndexError, so calling
    it is both a stronger check and immune to the helper moving between modules.
    """
    import numpy as np

    from mlframe.training.feature_handling.target_encoders import _compute_prior

    empty = np.array([], dtype=np.float64)
    for prior_kind in ("mean", "median"):
        assert _compute_prior(empty, prior_kind) == 0.0
        assert _compute_prior(empty, prior_kind, sample_weight=empty) == 0.0
