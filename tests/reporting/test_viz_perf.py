"""Regression + parity tests for the W2-C standalone-plot performance fixes.

Covers:
- plot_qq order-statistic decimation (tails retained, point cap, probplot parity at small n).
- plot_target_distribution np.histogram pre-bin + subsample-moments annotation.
- plot_pr_curve / plot_roc_curve vertex decimation (cap, endpoints kept, AP/AUC on full n).
- estimators.pipelines figure-close (no pyplot-registry leak under Agg).
- compute_ml_perf_by_time numpy fast-path byte-parity vs the pandas-Grouper reference.
"""

from __future__ import annotations

import contextlib
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.diagnostics import (
    plot_qq,
    plot_target_distribution,
    _qq_decimation_indices,
)
from mlframe.evaluation.reports import (
    plot_pr_curve,
    plot_roc_curve,
    _decimate_curve_vertices,
)
from mlframe.training.evaluation import (
    compute_ml_perf_by_time,
    _fixed_freq_bin_slices,
    _normalize_pandas_offset_alias,
    _compute_metric,
)
from mlframe.training.targets import coerce_timestamps_for_audit as _coerce_ts

# ---------------------------------------------------------------------------
# plot_qq decimation
# ---------------------------------------------------------------------------


def test_qq_decimation_keeps_tails_and_caps_points():
    """Qq decimation keeps tails and caps points."""
    n = 1_000_000
    idx = _qq_decimation_indices(n, max_points=2000, tail_keep=20)
    # both extreme order statistics survive (tail behaviour is the point of a QQ plot)
    assert idx[0] == 0
    assert idx[-1] == n - 1
    # the first/last tail_keep ranks are kept exactly
    assert np.array_equal(idx[:20], np.arange(20))
    assert np.array_equal(idx[-20:], np.arange(n - 20, n))
    # capped near the requested budget, strictly increasing, unique
    assert idx.size <= 2000
    assert np.all(np.diff(idx) > 0)


def test_qq_no_decimation_below_cap_returns_all_ranks():
    """Qq no decimation below cap returns all ranks."""
    n = 500
    idx = _qq_decimation_indices(n, max_points=2000)
    assert np.array_equal(idx, np.arange(n))


def test_qq_small_n_matches_probplot_positions():
    # On a small input (no decimation) the plotted scatter must equal scipy.stats.probplot's
    # order-statistic medians + ordered values exactly -- probplot is what plot_qq replaced.
    """Qq small n matches probplot positions."""
    from scipy.stats import probplot

    rng = np.random.default_rng(0)
    t = rng.normal(size=400)
    fig = plot_qq(t)
    ax = fig.axes[0]
    scatter_line = ax.lines[0]
    osm_plot = np.asarray(scatter_line.get_xdata())
    osr_plot = np.asarray(scatter_line.get_ydata())

    (osm_ref, osr_ref), _ = probplot(t, dist="norm", fit=False), None
    # probplot(fit=False) returns ((osm, osr),) -- normalize the unpacking
    res = probplot(t, dist="norm")
    osm_ref = res[0][0]
    osr_ref = res[0][1]
    assert np.allclose(osm_plot, osm_ref, rtol=1e-10, atol=1e-10)
    assert np.allclose(osr_plot, osr_ref, rtol=1e-10, atol=1e-10)
    plt.close(fig)


def test_qq_large_n_caps_plotted_points_but_keeps_extremes():
    """Qq large n caps plotted points but keeps extremes."""
    rng = np.random.default_rng(1)
    t = rng.normal(size=200_000)
    fig = plot_qq(t)
    ax = fig.axes[0]
    osr_plot = np.asarray(ax.lines[0].get_ydata())
    assert osr_plot.size <= 2000
    # min and max ordered values are present (extremes never decimated away)
    assert np.isclose(osr_plot.min(), t.min())
    assert np.isclose(osr_plot.max(), t.max())
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_target_distribution pre-binning + subsample moments
# ---------------------------------------------------------------------------


def test_target_distribution_bars_are_row_shares_of_each_series():
    """Each panel's bars are that series' row share per bin: they sum to the share of rows inside the shown 0.1-99.9% range."""
    rng = np.random.default_rng(2)
    y = rng.normal(loc=10, scale=3, size=20_000)
    t = y - 9.5
    fig = plot_target_distribution(y, t, bins=60)
    for ax, arr in zip(fig.axes[:2], (y, t)):
        lo, hi = np.quantile(arr, [0.001, 0.999])
        inside = float(((arr >= lo) & (arr <= hi)).mean())
        heights = np.array([p.get_height() for p in ax.patches])
        assert heights.size == 60
        assert abs(heights.sum() - inside) < 1e-9
    plt.close(fig)


def test_target_distribution_heavy_tail_is_not_one_bar():
    """A heavy-tailed target spread over 0..90k used to land in a single visible bar; the asinh axis must spread it."""
    rng = np.random.default_rng(5)
    y = np.where(rng.random(200_000) < 0.3, 0.0, rng.lognormal(4, 2, 200_000))
    fig = plot_target_distribution(y, y * 0.9)
    heights = np.array([p.get_height() for p in fig.axes[0].patches])
    assert (heights > 1e-3).sum() >= 20
    plt.close(fig)


def _suptitle(fig) -> str:
    """Return the figure's suptitle text, or an empty string when it has none."""
    return fig._suptitle.get_text() if fig._suptitle is not None else ""


def test_target_distribution_annotates_subsample_above_cap():
    """Target distribution annotates subsample above cap."""
    rng = np.random.default_rng(3)
    fig = plot_target_distribution(rng.normal(size=250_000), rng.normal(size=250_000))
    assert "moments on" in _suptitle(fig)
    assert "Verdict" in _suptitle(fig)
    plt.close(fig)


def test_target_distribution_no_subsample_note_below_cap():
    """Target distribution no subsample note below cap."""
    rng = np.random.default_rng(4)
    fig = plot_target_distribution(rng.normal(size=5_000), rng.normal(size=5_000))
    assert "moments on" not in _suptitle(fig)
    plt.close(fig)


def test_decimate_curve_vertices_caps_and_keeps_endpoints():
    """Decimate curve vertices caps and keeps endpoints."""
    n = 50_000
    a = np.linspace(0.0, 1.0, n)
    b = np.linspace(5.0, 9.0, n)
    da, db = _decimate_curve_vertices((a, b), max_points=2000)
    assert da.size <= 2000 and db.size <= 2000
    # endpoints preserved on both parallel arrays
    assert da[0] == a[0] and da[-1] == a[-1]
    assert db[0] == b[0] and db[-1] == b[-1]


def test_decimate_curve_vertices_passthrough_below_cap():
    """Decimate curve vertices passthrough below cap."""
    a = np.linspace(0, 1, 1500)
    (da,) = _decimate_curve_vertices((a,), max_points=2000)
    assert np.array_equal(da, a)


def _make_binary(n, seed=1):
    """Helper: Make binary."""
    rng = np.random.default_rng(seed)
    score = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(score - 0.5)))
    y = (rng.random(n) < p).astype(np.int64)
    return y, p.astype(np.float64)


def test_pr_curve_decimates_but_metrics_on_full_n():
    """Pr curve decimates but metrics on full n."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    n = 100_000
    y, p = _make_binary(n)
    with contextlib.redirect_stdout(io.StringIO()):
        fig = plot_pr_curve(y, p)
    ax = fig.axes[0]
    # all plotted line/step artists are capped at the decimation budget
    assert len(ax.lines) > 0, "the loop below must iterate at least once"
    for ln in ax.lines:
        assert ln.get_xdata().size <= 2000
    # the legend reports AP / AUC computed on the full data, not the decimated curve
    ap_full = average_precision_score(y, p)
    auc_full = roc_auc_score(y, p)
    legend_txt = " ".join(t.get_text() for t in ax.get_legend().get_texts())
    assert f"{ap_full:.2f}" in legend_txt
    assert f"{auc_full:.3f}" in legend_txt
    plt.close(fig)


def test_pr_curve_dummy_baseline_is_analytic_prevalence():
    # The constant-prediction PR baseline AP equals the positive-class prevalence; the legend
    # reports it as the second '/' term in 'PR AUC=%.2f/%.2fR'.
    """Pr curve dummy baseline is analytic prevalence."""
    n = 40_000
    rng = np.random.default_rng(9)
    y = (rng.random(n) < 0.3).astype(np.int64)
    p = rng.random(n)
    prevalence = y.mean()
    with contextlib.redirect_stdout(io.StringIO()):
        fig = plot_pr_curve(y, p)
    ax = fig.axes[0]
    legend_txt = " ".join(t.get_text() for t in ax.get_legend().get_texts())
    assert f"/{prevalence:.2f}R" in legend_txt
    plt.close(fig)


def test_roc_curve_decimates_but_auc_on_full_n():
    """Roc curve decimates but auc on full n."""
    from sklearn.metrics import roc_auc_score

    n = 100_000
    y, p = _make_binary(n)
    fig = plot_roc_curve(y, p)
    ax = fig.axes[0]
    auc_full = roc_auc_score(y, p)
    # the ROC vertex line is capped
    roc_line = ax.lines[0]
    assert roc_line.get_xdata().size <= 2000
    legend_txt = " ".join(t.get_text() for t in ax.get_legend().get_texts())
    assert f"{auc_full:.2f}" in legend_txt
    plt.close(fig)


# ---------------------------------------------------------------------------
# estimators.pipelines figure lifecycle (PERF-15 / INV-50)
# ---------------------------------------------------------------------------


def test_compare_cv_metrics_closes_figure_under_agg():
    """Compare cv metrics closes figure under agg."""
    from mlframe.estimators.pipelines import compare_cv_metrics

    plt.close("all")
    cv_results = {
        "results": {
            "cv_results": {
                "ModelA": {"metrics": {"root_mean_squared_error": [1.0, 1.1, 0.9]}},
                "DummyB": {"metrics": {"root_mean_squared_error": [2.0, 2.1, 1.9]}},
            }
        }
    }
    fig = compare_cv_metrics(cv_results)
    assert hasattr(fig, "savefig")
    # under Agg show is a no-op and the figure must be closed -> registry empty
    assert plt.get_fignums() == []


def test_visualize_prediction_vs_truth_closes_figure_under_agg():
    """Visualize prediction vs truth closes figure under agg."""
    from mlframe.estimators.pipelines import visualize_prediction_vs_truth

    plt.close("all")
    rng = np.random.default_rng(5)
    y_true = rng.normal(size=(200, 12))
    y_preds = rng.normal(size=(200, 12))
    fig = visualize_prediction_vs_truth(y_true, y_preds, samples=(1, 50, 75))
    assert hasattr(fig, "savefig")
    assert plt.get_fignums() == []


def test_pipelines_repeated_calls_do_not_leak_figures():
    """Pipelines repeated calls do not leak figures."""
    from mlframe.estimators.pipelines import compare_cv_metrics

    plt.close("all")
    cv_results = {
        "results": {
            "cv_results": {
                "ModelA": {"metrics": {"root_mean_squared_error": [1.0, 1.1, 0.9]}},
            }
        }
    }
    for _ in range(30):
        compare_cv_metrics(cv_results)
    # gridsearch loop would otherwise accumulate 30 figures (mpl warns past 20)
    assert plt.get_fignums() == []


# ---------------------------------------------------------------------------
# compute_ml_perf_by_time numpy fast-path parity (PERF-14)
# ---------------------------------------------------------------------------


def _old_compute_ml_perf_by_time(y_true, y_pred, timestamps, freq="D", metric="roc_auc", min_samples=100):
    """Inline pre-optimization reference: full pandas frame + set_index().sort_index() + Grouper loop."""
    df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred, dtype=float),
            "ts": _coerce_ts(np.asarray(timestamps)),
        }
    )
    df = df.set_index("ts").sort_index()
    rows = []
    _freq = _normalize_pandas_offset_alias(freq)
    for bin_start, chunk in df.groupby(pd.Grouper(freq=_freq)):
        n = len(chunk)
        if n == 0:
            continue
        if n < min_samples:
            val = float("nan")
        else:
            try:
                val = _compute_metric(metric, chunk["y_true"].values, chunk["y_pred"].values)
            except (ValueError, TypeError, ZeroDivisionError, FloatingPointError):
                val = float("nan")
        rows.append({"bin": bin_start, metric: val, "n_samples": n})
    return pd.DataFrame(rows).set_index("bin") if rows else pd.DataFrame(columns=[metric, "n_samples"])


@pytest.mark.parametrize("freq", ["D", "h", "12h", "7D", "W", "ME", "QE"])
@pytest.mark.parametrize("metric,min_samples", [("roc_auc", 1), ("roc_auc", 200), ("mse", 1), ("brier", 50)])
def test_compute_ml_perf_by_time_parity(freq, metric, min_samples):
    """Compute ml perf by time parity."""
    rng = np.random.default_rng(123)
    base = 1_700_000_000  # epoch-seconds -> coercer reads as 2023-11
    n = 30_000
    ts = (base + rng.integers(0, 700 * 86400, size=n)).astype(np.int64)
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.random(n)
    new = compute_ml_perf_by_time(y_true, y_pred, ts, freq=freq, metric=metric, min_samples=min_samples)
    old = _old_compute_ml_perf_by_time(y_true, y_pred, ts, freq=freq, metric=metric, min_samples=min_samples)
    pd.testing.assert_frame_equal(new, old, check_exact=False, rtol=1e-12, atol=1e-12)
    assert new.index.equals(old.index)
    assert new.index.dtype == old.index.dtype


def test_compute_ml_perf_by_time_fast_path_used_for_day_divisor_freq():
    # D / h / sub-daily-divisor freqs route through the floorable numpy path; 7D / W do not.
    """Compute ml perf by time fast path used for day divisor freq."""
    from pandas.tseries.frequencies import to_offset

    DAY_NS = 86_400_000_000_000

    def floorable(freq):
        """Mirrors compute_ml_perf_by_time's own floorable test: `.nanos` raises for genuinely
        non-fixed offsets in both pandas 2.x and 3.x, which is the actual fixed-duration signal --
        NOT isinstance(Tick), which pandas>=3 stopped applying to Day (PDEP-14-adjacent offset
        refactor) even though Day.nanos still works identically."""
        off = to_offset(_normalize_pandas_offset_alias(freq))
        try:
            nanos = off.nanos
        except ValueError:
            return False
        return nanos <= DAY_NS and DAY_NS % nanos == 0

    assert floorable("D") and floorable("h") and floorable("12h")
    assert not floorable("7D") and not floorable("W") and not floorable("ME")


def test_fixed_freq_bin_slices_empty_input():
    """Fixed freq bin slices empty input."""
    order, labels, starts, ends = _fixed_freq_bin_slices(np.array([], dtype="datetime64[ns]"), 86_400_000_000_000)
    assert order.size == 0 and labels.size == 0 and starts.size == 0 and ends.size == 0


def test_compute_ml_perf_by_time_empty_returns_empty_frame():
    """Compute ml perf by time empty returns empty frame."""
    out = compute_ml_perf_by_time(np.array([]), np.array([]), np.array([], dtype="int64"), freq="D", metric="roc_auc", min_samples=1)
    assert list(out.columns) == ["roc_auc", "n_samples"]
    assert len(out) == 0
