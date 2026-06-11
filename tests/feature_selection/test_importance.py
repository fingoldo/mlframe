"""Unit and biz_value tests for mlframe.feature_selection.importance.

Public surface (per the U16 audit finding):
  - ``compute_permutation_importances``
  - ``plot_feature_importance``
  - ``show_shap_beeswarm_plot`` (smoke-test only — depends on SHAP + a model with TreeExplainer support)
  - ``explain_top_feature_importances``

Tests prefer behavioral assertions (per memory ``feedback_behavioral_tests``) — no
``inspect.getsource``-string checks.
"""
from __future__ import annotations

import os

import matplotlib

# Force Agg before any matplotlib pyplot import — must work headless on CI / Windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402

from sklearn.linear_model import LinearRegression  # noqa: E402

from mlframe.feature_selection.importance import (  # noqa: E402
    compute_permutation_importances,
    plot_feature_importance,
)

pytestmark = pytest.mark.uses_matplotlib


# ----------------------------------------------------------------------------
# Synthetic fixtures
# ----------------------------------------------------------------------------


def _build_signal_dataset(n: int = 300, n_noise: int = 4, seed: int = 0):
    """Build a regression dataset where column 0 is the SOLE signal."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 1 + n_noise))
    y = X[:, 0] * 2.0 + rng.normal(size=n) * 0.1
    cols = ["signal"] + [f"noise_{i}" for i in range(n_noise)]
    return X, y, cols


# ----------------------------------------------------------------------------
# compute_permutation_importances
# ----------------------------------------------------------------------------


def test_compute_permutation_returns_polars_dataframe():
    X, y, cols = _build_signal_dataset(n=200, n_noise=3)
    model = LinearRegression().fit(X, y)
    out = compute_permutation_importances(model, X, y, columns=cols, n_repeats=3, random_state=0)
    assert isinstance(out, pl.DataFrame), f"must return polars DataFrame; got {type(out).__name__}"
    expected_cols = {"importances_mean", "importances_std", "feature"}
    actual_cols = set(out.columns)
    assert expected_cols.issubset(actual_cols), \
        f"output must contain {expected_cols}; got {actual_cols}"


def test_compute_permutation_one_row_per_feature():
    X, y, cols = _build_signal_dataset(n=200, n_noise=4)
    model = LinearRegression().fit(X, y)
    out = compute_permutation_importances(model, X, y, columns=cols, n_repeats=3, random_state=0)
    assert out.shape[0] == len(cols), f"expected one row per feature; got {out.shape[0]} rows for {len(cols)} features"


def test_compute_permutation_biz_value_signal_ranks_top():
    """biz_value: signal column must rank #1 by importances_mean (post-sort).

    The function sorts by (importances_mean - 0.2 * importances_std) descending, so the
    top row is the strongest feature. This is the actual production contract."""
    X, y, cols = _build_signal_dataset(n=400, n_noise=5, seed=42)
    model = LinearRegression().fit(X, y)
    out = compute_permutation_importances(model, X, y, columns=cols, n_repeats=5, random_state=0)
    top_feature = out["feature"][0]
    assert top_feature == "signal", \
        f"signal column must rank #1; got {top_feature} (full importances_mean: {dict(zip(out['feature'].to_list(), out['importances_mean'].to_list()))})"
    # Tighter biz_value: signal importance must be MUCH larger than any noise importance
    sig_imp = out.filter(pl.col("feature") == "signal")["importances_mean"][0]
    noise_imps = [
        out.filter(pl.col("feature") == f"noise_{i}")["importances_mean"][0]
        for i in range(5)
    ]
    max_noise = max(noise_imps)
    assert sig_imp > max_noise * 3.0, \
        f"signal importance ({sig_imp:.3f}) must dominate noise (max {max_noise:.3f}) by >=3x"


def test_compute_permutation_deterministic_with_random_state():
    """Same random_state must produce identical importances."""
    X, y, cols = _build_signal_dataset(n=200, n_noise=3, seed=7)
    model = LinearRegression().fit(X, y)
    out1 = compute_permutation_importances(model, X, y, columns=cols, n_repeats=3, random_state=42)
    out2 = compute_permutation_importances(model, X, y, columns=cols, n_repeats=3, random_state=42)
    np.testing.assert_allclose(
        out1["importances_mean"].to_numpy(),
        out2["importances_mean"].to_numpy(),
        rtol=1e-12,
        err_msg="permutation importance must be deterministic with the same random_state",
    )


def test_compute_permutation_filters_zero_mean_zero_std():
    """Module-level filter ``~((mean == 0) & (std == 0))`` drops rows where the model
    learned the feature is purely useless. Verify the filter actually runs on a
    contrived case: a perfectly-constant column that no model gains from."""
    n = 200
    rng = np.random.default_rng(0)
    # Two-feature setup: x0 is the only signal, x1 is constant.
    X = np.column_stack([rng.normal(size=n), np.full(n, 7.0)])
    y = X[:, 0] * 2.0 + rng.normal(size=n) * 0.05
    model = LinearRegression().fit(X, y)
    out = compute_permutation_importances(model, X, y, columns=["signal", "constant"], n_repeats=3, random_state=0)
    feats = out["feature"].to_list()
    assert "signal" in feats, "signal feature must remain in output"
    # constant column has mean=0 std=0 importance — should be filtered out
    assert "constant" not in feats, \
        f"constant-importance feature must be filtered; got {feats}"


# ----------------------------------------------------------------------------
# plot_feature_importance
# ----------------------------------------------------------------------------


def test_plot_feature_importance_returns_sorted_pd_dataframe():
    """plot_feature_importance returns a pd.DataFrame indexed by feature name, with a
    single column ``fi`` sorted descending."""
    plt.close("all")
    fi = np.array([0.2, 0.7, 0.05, 0.4])
    cols = ["a", "b", "c", "d"]
    out = plot_feature_importance(fi, cols, kind="test", show_plots=False, plot_file="", log_fi=False)
    assert out.columns.tolist() == ["fi"], f"return DF must have a single 'fi' column; got {out.columns.tolist()}"
    # Sorted descending by fi
    fis = out["fi"].tolist()
    assert fis == sorted(fis, reverse=True), f"return DF must be sorted descending; got {fis}"
    # The top row's index must be the largest-fi feature
    assert out.index[0] == "b", f"expected 'b' (fi=0.7) as top; got {out.index[0]}"


def test_plot_feature_importance_positive_only_filter():
    plt.close("all")
    fi = np.array([0.5, -0.3, 0.2, -0.1])
    cols = ["pos1", "neg1", "pos2", "neg2"]
    out = plot_feature_importance(fi, cols, kind="t", positive_fi_only=True, show_plots=False, plot_file="", log_fi=False)
    # All retained values must be > 0 by contract
    assert (out["fi"] > 0).all(), f"positive_fi_only must drop non-positive rows; got {out['fi'].tolist()}"
    # Negative features must be dropped
    assert "neg1" not in out.index
    assert "neg2" not in out.index


def test_plot_feature_importance_empty_columns_uses_index():
    """When columns is empty, the function falls back to integer index."""
    plt.close("all")
    fi = np.array([0.3, 0.1, 0.5])
    out = plot_feature_importance(fi, columns=[], kind="t", show_plots=False, plot_file="", log_fi=False)
    assert out.shape[0] == 3, "must keep all features when no explicit columns"
    # Integer indices
    assert set(out.index.tolist()) == {0, 1, 2}, f"index must be 0..N-1; got {out.index.tolist()}"


def test_plot_feature_importance_saves_to_file(tmp_path):
    """When ``plot_file`` is given, the figure is saved to that path. Smoke-check the file is written and has non-trivial size."""
    plt.close("all")
    fi = np.array([0.3, 0.1, 0.5, 0.0])
    cols = ["a", "b", "c", "d"]
    fp = tmp_path / "fi_test.png"
    plot_feature_importance(fi, cols, kind="t", show_plots=False, plot_file=str(fp), log_fi=False)
    assert fp.exists(), f"plot_file must exist on disk after call; expected {fp}"
    assert fp.stat().st_size > 1024, f"saved PNG must be non-trivial (>1KB); got {fp.stat().st_size} bytes"


def test_plot_feature_importance_no_log_when_log_fi_false(caplog):
    """log_fi=False suppresses the text-log line even when log_top_n > 0."""
    import logging
    plt.close("all")
    fi = np.array([0.3, 0.1, 0.5])
    cols = ["a", "b", "c"]
    with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.importance"):
        plot_feature_importance(fi, cols, kind="t", show_plots=False, plot_file="", log_fi=False, log_top_n=10)
    fi_log_records = [r for r in caplog.records if "[FI top" in r.getMessage()]
    assert fi_log_records == [], f"log_fi=False must suppress FI text log; got {fi_log_records}"


# ----------------------------------------------------------------------------
# importances_std whiskers (INV-22): permutation dispersion must reach the bars
# ----------------------------------------------------------------------------


def _capture_barh_xerr(monkeypatch):
    """Spy on Axes.barh so we can inspect the xerr it actually receives.

    plot_feature_importance closes its figure before returning, so the rendered
    BarContainer is not retrievable afterwards; capturing the draw call is the
    behavioral proxy for "the std reached the bars as error whiskers".
    """
    import matplotlib.axes as _maxes
    real_barh = _maxes.Axes.barh
    seen = {}

    def _spy(self, *args, **kwargs):
        seen["xerr"] = kwargs.get("xerr")
        container = real_barh(self, *args, **kwargs)
        seen["has_errorbar"] = getattr(container, "errorbar", None) is not None
        return container

    monkeypatch.setattr(_maxes.Axes, "barh", _spy)
    return seen


def test_plot_feature_importance_renders_xerr_whiskers_when_std_given(monkeypatch, tmp_path):
    """When importances_std is supplied, the FI bars get an xerr error bar (whiskers).

    On pre-fix code plot_feature_importance has no importances_std parameter, so
    passing it raises TypeError; and even threaded, the barh call carried no xerr,
    so seen['xerr'] would be None. This test catches both."""
    plt.close("all")
    fi = np.array([0.2, 0.7, 0.05, 0.4, 0.1])
    std = np.array([0.05, 0.30, 0.01, 0.10, 0.02])
    cols = ["a", "b", "c", "d", "e"]
    seen = _capture_barh_xerr(monkeypatch)
    plot_feature_importance(
        fi, cols, kind="t", show_plots=False, plot_file=str(tmp_path / "fi.png"),
        log_fi=False, importances_std=std,
    )
    assert seen.get("xerr") is not None, "importances_std must reach ax.barh as xerr"
    assert seen["has_errorbar"], "barh BarContainer must carry an errorbar (whiskers)"
    # The whiskers must be aligned to the picked + signed-sorted bars and non-negative.
    xerr = np.asarray(seen["xerr"], dtype=float)
    assert xerr.shape[0] == len(cols)
    assert np.all(xerr >= 0.0)
    assert np.any(xerr > 0.0)


def test_plot_feature_importance_no_xerr_when_std_absent(monkeypatch, tmp_path):
    """Native tree-gain / coef importances pass no std -> no whiskers (xerr stays None)."""
    plt.close("all")
    fi = np.array([0.2, 0.7, 0.05, 0.4])
    cols = ["a", "b", "c", "d"]
    seen = _capture_barh_xerr(monkeypatch)
    plot_feature_importance(
        fi, cols, kind="t", show_plots=False, plot_file=str(tmp_path / "fi.png"), log_fi=False,
    )
    assert seen.get("xerr") is None, "no std -> barh must not receive xerr"
    assert not seen.get("has_errorbar", False)


def test_plot_feature_importance_ignores_misaligned_std(monkeypatch, tmp_path):
    """A std array whose shape does not match feature_importances is ignored, not crashed."""
    plt.close("all")
    fi = np.array([0.2, 0.7, 0.05, 0.4])
    bad_std = np.array([0.1, 0.2])  # wrong length
    cols = ["a", "b", "c", "d"]
    seen = _capture_barh_xerr(monkeypatch)
    plot_feature_importance(
        fi, cols, kind="t", show_plots=False, plot_file=str(tmp_path / "fi.png"),
        log_fi=False, importances_std=bad_std,
    )
    assert seen.get("xerr") is None, "misaligned std must be dropped, not forwarded"
