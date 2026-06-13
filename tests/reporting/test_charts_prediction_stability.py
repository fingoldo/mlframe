"""Tests for ``mlframe.reporting.charts.prediction_stability``.

Unit (panel types / shapes, M=1 + degenerate annotate, NaN-safety) + biz_value (an easy/hard-region synthetic where
member disagreement MUST be higher in the hard region AND track actual error) + cProfile O(n) check at n>=1e6, M=10.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np
import pytest

from mlframe.reporting.charts.prediction_stability import (
    DEFAULT_CALIB_BINS, PredictionStabilityResult, compose_prediction_stability_figure,
    compute_prediction_stability,
)
from mlframe.reporting.charts.prediction_stability import (
    MIN_CALIB_ROWS, _spearman, _uncertainty_calibration,
)
from mlframe.reporting.spec import (
    AnnotationPanelSpec, FigureSpec, HistogramPanelSpec, LinePanelSpec, ScatterPanelSpec,
)


def _easy_hard_ensemble(n: int = 4000, m: int = 8, seed: int = 0):
    """Half the rows are 'easy' (members agree, prediction near truth); half are 'hard' (members disagree, large error).

    Returns (member_preds (n, m), y_true (n,), easy_mask (n,)). The hard region has both higher per-member spread and
    higher |ensemble_mean - y_true|, so disagreement should track error -- the informative-uncertainty case.
    """
    rng = np.random.default_rng(seed)
    y_true = rng.normal(0.0, 1.0, size=n)
    easy = np.zeros(n, dtype=bool)
    easy[: n // 2] = True
    member_noise = np.where(easy[:, None], 0.05, 1.2)  # members agree on easy rows, scatter widely on hard rows
    bias = np.where(easy, 0.0, rng.normal(0.0, 1.0, size=n))  # hard rows also carry a real error
    preds = (y_true + bias)[:, None] + rng.normal(0.0, 1.0, size=(n, m)) * member_noise
    return preds, y_true, easy


# ----------------------------------------------------------------------------
# compute_prediction_stability
# ----------------------------------------------------------------------------


def test_compute_returns_result_with_per_row_shapes():
    preds, _, _ = _easy_hard_ensemble()
    res = compute_prediction_stability(preds)
    assert isinstance(res, PredictionStabilityResult)
    assert res.spread_std.shape == (preds.shape[0],)
    assert res.spread_iqr.shape == (preds.shape[0],)
    assert res.ensemble_mean.shape == (preds.shape[0],)
    assert res.n_members == preds.shape[1]
    assert 0.0 <= res.agreement <= 1.0
    assert res.mean_spread > 0.0


def test_ensemble_mean_matches_rowwise_mean():
    preds, _, _ = _easy_hard_ensemble(n=500, m=5)
    res = compute_prediction_stability(preds)
    np.testing.assert_allclose(res.ensemble_mean, preds.mean(axis=1), rtol=1e-10)


def test_single_member_is_degenerate():
    preds = np.random.default_rng(0).normal(size=(1000, 1))
    res = compute_prediction_stability(preds)
    assert res.n_members == 1
    assert np.all(res.spread_std == 0.0)
    assert res.agreement == 1.0
    # ensemble mean of one member is that member.
    np.testing.assert_allclose(res.ensemble_mean, preds.ravel(), rtol=1e-10)


def test_1d_input_treated_as_single_member():
    res = compute_prediction_stability(np.arange(50.0))
    assert res.n_members == 1


def test_nan_members_ignored_per_row():
    preds = np.array([[1.0, 2.0, np.nan], [np.nan, np.nan, 5.0], [0.0, 0.0, 0.0]])
    res = compute_prediction_stability(preds)
    assert np.isfinite(res.spread_std).all()
    assert res.ensemble_mean[0] == pytest.approx(1.5)
    assert res.ensemble_mean[1] == pytest.approx(5.0)
    assert res.spread_std[2] == pytest.approx(0.0)


def test_empty_input_is_safe():
    res = compute_prediction_stability(np.empty((0, 5)))
    assert res.n_rows == 0
    assert res.spread_std.shape == (0,)


# ----------------------------------------------------------------------------
# compose_prediction_stability_figure
# ----------------------------------------------------------------------------


def test_compose_two_panels_without_y_true():
    preds, _, _ = _easy_hard_ensemble()
    fig = compose_prediction_stability_figure(preds)
    assert isinstance(fig, FigureSpec)
    panels = [p for row in fig.panels for p in row if p is not None]
    assert len(panels) == 2
    assert isinstance(panels[0], HistogramPanelSpec)
    assert isinstance(panels[1], ScatterPanelSpec)


def test_compose_adds_calibration_panel_with_y_true():
    preds, yt, _ = _easy_hard_ensemble()
    fig = compose_prediction_stability_figure(preds, y_true=yt)
    panels = [p for row in fig.panels for p in row if p is not None]
    assert len(panels) == 3
    assert isinstance(panels[2], LinePanelSpec)
    assert "Spearman" in panels[2].title


def test_scatter_subsampled_to_cap():
    preds, _, _ = _easy_hard_ensemble(n=20_000, m=6)
    fig = compose_prediction_stability_figure(preds, scatter_cap=5000)
    scatter = [p for row in fig.panels for p in row if isinstance(p, ScatterPanelSpec)][0]
    assert scatter.x.size <= 5000


def test_single_member_figure_is_annotation():
    preds = np.random.default_rng(0).normal(size=(1000, 1))
    fig = compose_prediction_stability_figure(preds)
    panels = [p for row in fig.panels for p in row if p is not None]
    assert len(panels) == 1
    assert isinstance(panels[0], AnnotationPanelSpec)
    assert "Need >=2" in panels[0].text


def test_tiny_n_calibration_annotates():
    preds = np.array([[1.0, 1.1], [1.0, 1.0]])
    fig = compose_prediction_stability_figure(preds, y_true=np.array([1.0, 1.0]))
    panels = [p for row in fig.panels for p in row if p is not None]
    calib = panels[-1]
    # Too few rows / no varying spread -> honest annotation rather than a garbage curve.
    assert isinstance(calib, AnnotationPanelSpec)


def test_figure_renders_matplotlib():
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    from mlframe.reporting.renderers.base import get_renderer
    preds, yt, _ = _easy_hard_ensemble()
    fig = compose_prediction_stability_figure(preds, y_true=yt)
    rend = get_renderer("matplotlib")
    rendered = rend.render(fig)
    import matplotlib.pyplot as plt
    plt.close(rendered)


# ----------------------------------------------------------------------------
# _spearman helper
# ----------------------------------------------------------------------------


def test_spearman_matches_scipy_on_random():
    pytest.importorskip("scipy")
    from scipy.stats import spearmanr
    rng = np.random.default_rng(3)
    a = rng.normal(size=500)
    b = a * 0.6 + rng.normal(size=500)
    assert _spearman(a, b) == pytest.approx(float(spearmanr(a, b).statistic), abs=1e-9)


def test_spearman_handles_ties():
    pytest.importorskip("scipy")
    from scipy.stats import spearmanr
    rng = np.random.default_rng(4)
    a = rng.integers(0, 5, size=300).astype(float)  # heavy ties
    b = rng.integers(0, 5, size=300).astype(float)
    assert _spearman(a, b) == pytest.approx(float(spearmanr(a, b).statistic), abs=1e-9)


# ----------------------------------------------------------------------------
# biz_value: disagreement is higher in the hard region AND tracks actual error
# ----------------------------------------------------------------------------


def test_biz_val_prediction_stability_spread_higher_in_hard_region():
    """Mean per-row member-spread must be materially higher in the disagree (hard) region than the agree (easy) region.

    Measured ~24x (easy mean std ~0.05, hard mean std ~1.2). Floor at 5x absorbs seed noise while a regression that
    flattens the spread (e.g. wrong axis reduction) trips it.
    """
    preds, _, easy = _easy_hard_ensemble(n=6000, m=10, seed=1)
    res = compute_prediction_stability(preds)
    easy_spread = float(res.spread_std[easy].mean())
    hard_spread = float(res.spread_std[~easy].mean())
    assert hard_spread >= 5.0 * easy_spread, f"hard {hard_spread:.3f} should be >>5x easy {easy_spread:.3f}"


def test_biz_val_prediction_stability_spread_tracks_error():
    """Spearman(per-row spread, actual |error|) must be positive and well above a 0.3 floor -- the spread is INFORMATIVE.

    Measured ~0.55 on this easy/hard synthetic; floor 0.30. A non-informative spread (no spread-error link) would fail.
    """
    preds, yt, _ = _easy_hard_ensemble(n=6000, m=10, seed=1)
    res = compute_prediction_stability(preds)
    abs_err = np.abs(yt - res.ensemble_mean)
    rho = _spearman(res.spread_std, abs_err)
    assert rho > 0.3, f"Spearman(spread, |error|) = {rho:.3f} should exceed the 0.3 informativeness floor"


def test_biz_val_uncertainty_calibration_curve_monotone_increasing():
    """The mean |error| per spread bin must trend upward (higher disagreement bins => higher mean error).

    A robust monotonicity check: the top spread bin's mean error >> the bottom bin's, and the Spearman of the
    (bin spread, bin error) curve is strongly positive.
    """
    preds, yt, _ = _easy_hard_ensemble(n=6000, m=10, seed=1)
    res = compute_prediction_stability(preds)
    abs_err = np.abs(yt - res.ensemble_mean)
    mid, mean_err, rho = _uncertainty_calibration(res.spread_std, abs_err, nbins=DEFAULT_CALIB_BINS)
    assert mid is not None
    assert mean_err[-1] > 2.0 * mean_err[0], "top-spread bin error should dwarf bottom-spread bin error"
    assert _spearman(mid, mean_err) > 0.8, "uncertainty-calibration curve should rise nearly monotonically"


# ----------------------------------------------------------------------------
# cProfile: bounded at n>=1e6, M=10
# ----------------------------------------------------------------------------


def test_cprofile_compute_bounded_at_1e6():
    """compute_prediction_stability at n=1e6, M=10 stays O(n*M) and well under a generous wall budget."""
    rng = np.random.default_rng(0)
    preds = rng.normal(size=(1_000_000, 10))
    yt = rng.normal(size=1_000_000)

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    res = compute_prediction_stability(preds)
    abs_err = np.abs(yt - res.ensemble_mean)
    _uncertainty_calibration(res.spread_std, abs_err, nbins=DEFAULT_CALIB_BINS)
    pr.disable()
    elapsed = time.perf_counter() - t0

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(12)
    # nanpercentile over (1e6, 10) is the dominant cost; budget is generous to absorb CI contention.
    assert elapsed < 5.0, f"compute at 1e6x10 took {elapsed:.2f}s\n{s.getvalue()}"


def test_spearman_njit_path_bit_identical_to_numpy_reference(monkeypatch):
    """iter83: large-N _spearman routes through the njit batched kernel (single argsort + tie-average in machine code)
    instead of the pure-Python tie-collapse loop, ~2x at N=200k. Output must be bit-identical to the numpy-rank reference
    on distinct AND tied data; the kernel uses the same average-rank convention."""
    from mlframe.reporting.charts import prediction_stability as ps

    rng = np.random.default_rng(7)
    n = 20_000
    for a, b in (
        (rng.random(n), rng.random(n)),  # all-distinct
        (rng.integers(0, 40, n).astype(float), rng.integers(0, 40, n).astype(float)),  # heavy ties
    ):
        njit_val = ps._spearman(a, b)
        # Force the pure-numpy reference path (the pre-iter83 behaviour) by lifting the threshold above any input.
        monkeypatch.setattr(ps, "_SPEARMAN_NJIT_MIN_N", 10**12)
        ref_val = ps._spearman(a, b)
        monkeypatch.undo()
        assert njit_val == ref_val, f"njit Spearman {njit_val!r} != numpy reference {ref_val!r}"
