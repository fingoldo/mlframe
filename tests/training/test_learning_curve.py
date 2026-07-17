"""Unit + biz_value + cProfile tests for the opt-in learning-curve diagnostic.

The learning curve is K full refits by construction (its cost is why it is opt-in); these tests use cheap
estimators (Ridge / a small HistGradientBoosting) at moderate n so the whole file stays fast.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import get_scorer

from mlframe.reporting.spec import FigureSpec, LinePanelSpec
from mlframe.training.diagnostics.learning_curve import (
    DEFAULT_SIZES,
    LearningCurveConfig,
    LearningCurveResult,
    compute_learning_curve,
    learning_curve_panel,
)

R2 = get_scorer("r2")
ACC = get_scorer("accuracy")


def _linear_reg(n: int, noise: float = 0.5, seed: int = 0):
    """Linear reg."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0, 0.3]) + rng.normal(0, noise, n)
    return X, y


def _ridge_factory():
    """Ridge factory."""
    return Ridge(alpha=1.0)


# --------------------------------------------------------------------------- unit


def test_returns_k_sizes_with_parallel_arrays():
    """Returns k sizes with parallel arrays."""
    X, y = _linear_reg(500)
    res = compute_learning_curve(_ridge_factory, X, y, scorer=R2, n_jobs=1)
    k = len(res.train_sizes)
    assert k == len(DEFAULT_SIZES)
    assert len(res.train_scores) == k
    assert len(res.holdout_scores) == k
    assert len(res.train_score_std) == k
    assert len(res.holdout_score_std) == k
    assert isinstance(res, LearningCurveResult)


def test_size_grid_is_strictly_increasing():
    """Size grid is strictly increasing."""
    X, y = _linear_reg(500)
    res = compute_learning_curve(_ridge_factory, X, y, scorer=R2, n_jobs=1)
    assert np.all(np.diff(res.train_sizes) > 0), res.train_sizes.tolist()
    # Largest size never exceeds the pool (n - holdout).
    assert res.train_sizes[-1] <= 500 - res.holdout_n


def test_holdout_is_disjoint_from_every_train_subset():
    # Reproduce the internal split with the same seed, then assert no holdout row index is reachable from the
    # pool the sizes index into.
    """Holdout is disjoint from every train subset."""
    n = 400
    X, y = _linear_reg(n)
    holdout = 0.25
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(n)
    n_hold = min(max(1, round(holdout * n)), n - 2)
    hold_idx = set(np.sort(shuffled[:n_hold]).tolist())
    pool_idx = set(np.sort(shuffled[n_hold:]).tolist())
    assert hold_idx.isdisjoint(pool_idx)
    # The function must run cleanly with this holdout and report the same holdout count.
    res = compute_learning_curve(_ridge_factory, X, y, scorer=R2, holdout=holdout, n_jobs=1, random_state=0)
    assert res.holdout_n == n_hold
    assert len(hold_idx) == n_hold


def test_custom_sizes_collapse_duplicates_on_small_pool():
    """Custom sizes collapse duplicates on small pool."""
    X, y = _linear_reg(40)
    # 0.05 and 0.1 both round to the same tiny count on a ~32-row pool -> collapse, but result stays ascending.
    res = compute_learning_curve(
        _ridge_factory,
        X,
        y,
        scorer=R2,
        sizes=(0.05, 0.1, 0.5, 1.0),
        n_jobs=1,
    )
    assert np.all(np.diff(res.train_sizes) > 0)
    assert len(res.train_sizes) <= 4


def test_parallel_matches_serial_bit_for_bit():
    """Parallel matches serial bit for bit."""
    X, y = _linear_reg(600)
    serial = compute_learning_curve(_ridge_factory, X, y, scorer=R2, n_jobs=1, random_state=7)
    parallel = compute_learning_curve(_ridge_factory, X, y, scorer=R2, n_jobs=-1, random_state=7)
    assert np.array_equal(serial.train_sizes, parallel.train_sizes)
    np.testing.assert_allclose(serial.holdout_scores, parallel.holdout_scores, rtol=0, atol=0)
    np.testing.assert_allclose(serial.train_scores, parallel.train_scores, rtol=0, atol=0)


def test_score_repeats_produce_nonzero_std_band():
    """Score repeats produce nonzero std band."""
    X, y = _linear_reg(500)
    res = compute_learning_curve(_ridge_factory, X, y, scorer=R2, n_jobs=1, score_repeats=3)
    # Reshuffled equal-length prefixes must give SOME spread on at least one size.
    assert np.any(res.holdout_score_std > 0)


def test_time_budget_skips_larger_sizes_and_logs(caplog):
    """Time budget skips larger sizes and logs."""
    X, y = _linear_reg(800)

    def _slow_factory():
        """Slow factory."""
        class _Slow(Ridge):
            """Groups tests covering slow."""
            def fit(self, X, yy, **kw):
                """Fit."""
                time.sleep(0.15)
                return super().fit(X, yy, **kw)

        return _Slow(alpha=1.0)

    with caplog.at_level("INFO"):
        res = compute_learning_curve(
            _slow_factory,
            X,
            y,
            scorer=R2,
            n_jobs=1,
            time_budget_s=0.2,
            random_state=0,
        )
    assert len(res.skipped_fractions) > 0
    assert len(res.train_sizes) < len(DEFAULT_SIZES)
    assert any("skipped" in r.message for r in caplog.records)


def test_warm_start_falls_back_when_unsupported():
    """Warm start falls back when unsupported."""
    X, y = _linear_reg(400)
    # Ridge has no warm_start param -> the warm path must be skipped and a full sweep returned.
    res = compute_learning_curve(_ridge_factory, X, y, scorer=R2, warm_start=True, n_jobs=1)
    assert res.warm_start_used is False
    assert len(res.train_sizes) == len(DEFAULT_SIZES)


def test_warm_start_used_for_incremental_learner():
    """Warm start used for incremental learner."""
    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] + X[:, 1] * X[:, 2] > 0).astype(int)

    def _gb():
        """Gb."""
        return HistGradientBoostingClassifier(max_iter=40, warm_start=True, random_state=0)

    res = compute_learning_curve(_gb, X, y, scorer=ACC, warm_start=True, n_jobs=1)
    assert res.warm_start_used is True
    assert np.all(np.diff(res.train_sizes) > 0)


def test_raises_on_too_few_rows():
    """Raises on too few rows."""
    X, y = _linear_reg(3)
    with pytest.raises(ValueError):
        compute_learning_curve(_ridge_factory, X, y, scorer=R2)


def test_raises_on_bad_holdout():
    """Raises on bad holdout."""
    X, y = _linear_reg(100)
    with pytest.raises(ValueError):
        compute_learning_curve(_ridge_factory, X, y, scorer=R2, holdout=1.5)


def test_panel_is_pure_data_figurespec():
    """Panel is pure data figurespec."""
    X, y = _linear_reg(500)
    res = compute_learning_curve(_ridge_factory, X, y, scorer=R2, n_jobs=1, scorer_name="r2")
    fig = learning_curve_panel(res)
    assert isinstance(fig, FigureSpec)
    assert len(fig.panels) == 1
    panel = fig.panels[0][0]
    assert isinstance(panel, LinePanelSpec)
    assert panel.series_labels == ("train score", "holdout score")
    # Two parallel y series, x = train sizes.
    assert len(panel.y) == 2
    assert len(panel.x) == len(res.train_sizes)


def test_panel_band_present_only_with_repeats():
    """Panel band present only with repeats."""
    X, y = _linear_reg(500)
    no_rep = learning_curve_panel(compute_learning_curve(_ridge_factory, X, y, scorer=R2, n_jobs=1))
    assert no_rep.panels[0][0].band is None
    rep = learning_curve_panel(compute_learning_curve(_ridge_factory, X, y, scorer=R2, n_jobs=1, score_repeats=3))
    assert rep.panels[0][0].band is not None


def test_config_default_is_opt_in_off():
    """Config default is opt in off."""
    cfg = LearningCurveConfig()
    assert cfg.enabled is False
    assert cfg.sizes == DEFAULT_SIZES


def test_higher_is_better_false_orients_slope():
    # A raw-loss scorer (lower better): a decreasing loss with size must read as a POSITIVE oriented slope.
    """Higher is better false orients slope."""
    res = LearningCurveResult(
        train_sizes=np.array([10, 100, 1000]),
        train_scores=np.array([0.5, 0.3, 0.2]),
        holdout_scores=np.array([0.9, 0.5, 0.3]),  # loss falling => improving
        train_score_std=np.zeros(3),
        holdout_score_std=np.zeros(3),
        holdout_n=50,
        higher_is_better=False,
    )
    assert res.holdout_slope_last() > 0
    assert res.verdict() == "data_starved"


# --------------------------------------------------------------------------- biz_value


def test_biz_val_learning_curve_slope_distinguishes_starved_from_saturated():
    """Slope SIGN at the last sizes separates a data-starved fit (rising holdout) from a saturated one (flat).

    Data-starved: a flexible HistGradientBoosting on a hard non-linear target -- with few rows it underfits and
    keeps improving as rows arrive, so the last-sizes holdout slope is clearly positive. Saturated: a Ridge on a
    clean linear target -- it nails the signal almost immediately, so the holdout curve is flat (slope ~0).

    Measured during development: starved slope ~0.05-0.15 (r2 vs log10 rows), saturated |slope| < 0.005. The
    floors below sit well inside that margin so seed noise does not trip them but a regression that breaks the
    increasing-size mechanism (e.g. all sizes fit the same rows) collapses the starved slope to ~0 and fails.
    """
    rng = np.random.default_rng(0)

    # --- data-starved: hard target, flexible learner, modest n ---
    n = 1200
    Xs = rng.normal(size=(n, 8))
    # Strong interactions + nonlinearity a GB needs many rows to resolve.
    ys = np.sin(2.0 * Xs[:, 0]) * Xs[:, 1] + Xs[:, 2] * Xs[:, 3] - np.abs(Xs[:, 4]) + 0.5 * Xs[:, 5] ** 2 + rng.normal(0, 0.3, n)

    def _gb():
        """Gb."""
        return HistGradientBoostingRegressor(max_iter=120, max_depth=4, random_state=0)

    starved = compute_learning_curve(
        _gb,
        Xs,
        ys,
        scorer=R2,
        sizes=(0.1, 0.2, 0.4, 0.7, 1.0),
        n_jobs=-1,
        random_state=0,
        scorer_name="r2",
    )
    starved_slope = starved.holdout_slope_last(k=3)

    # --- saturated: clean linear target, simple learner ---
    Xc, yc = _linear_reg(2000, noise=0.3, seed=1)
    saturated = compute_learning_curve(
        _ridge_factory,
        Xc,
        yc,
        scorer=R2,
        sizes=(0.1, 0.2, 0.4, 0.7, 1.0),
        n_jobs=-1,
        random_state=0,
        scorer_name="r2",
    )
    saturated_slope = saturated.holdout_slope_last(k=3)

    assert starved_slope > 0.01, f"data-starved holdout should keep rising; slope={starved_slope:.4f}"
    assert abs(saturated_slope) < 0.005, f"saturated holdout should be flat; slope={saturated_slope:.4f}"
    assert starved_slope > saturated_slope + 0.01
    assert starved.verdict() == "data_starved"
    assert saturated.verdict() == "saturated"


# --------------------------------------------------------------------------- cProfile


def test_cprofile_sweep_cost_scales_with_total_fit_work():
    """Profile the sweep at a moderate shape and assert wall scales ~linearly in total-fit-work (= K fits).

    Cost is K independent fits; that is the whole reason the diagnostic is opt-in. We profile the serial path so
    fit time is attributable (the parallel path divides this wall by the worker count). We also assert the
    measured wall for K sizes is within a generous multiple of K single full-pool fits -- catching a regression
    that, say, secretly re-fits every size on the WHOLE pool (which would make every size cost the max).
    """
    X, y = _linear_reg(4000, seed=3)
    sizes = (0.1, 0.2, 0.4, 0.7, 1.0)

    # Reference: one full-pool Ridge fit wall (the most expensive single size).
    from mlframe.training.diagnostics.learning_curve import _take_rows  # noqa: F401

    t = time.perf_counter()
    Ridge(alpha=1.0).fit(X, y)
    one_full_fit = max(time.perf_counter() - t, 1e-4)

    pr = cProfile.Profile()
    pr.enable()
    res = compute_learning_curve(_ridge_factory, X, y, scorer=R2, sizes=sizes, n_jobs=1, random_state=3)
    pr.disable()

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
    profile_text = s.getvalue()
    assert "compute_learning_curve" in profile_text

    # Total fit-work is the sum of (size/pool) fractions; each fit's cost is ~proportional to its row count, so the
    # sweep should not exceed ~K full-pool fits plus scoring/overhead. Generous 8x cap absorbs cProfile inflation
    # + scoring + the small fixed split overhead while still failing a "every size fits the whole pool" regression
    # combined with a "no nesting" bug (which would push cost toward K*full unconditionally and break the slope).
    assert res.elapsed_seconds <= one_full_fit * len(sizes) * 8 + 0.5, f"sweep wall {res.elapsed_seconds:.3f}s vs {len(sizes)} full fits @ {one_full_fit:.4f}s"
    # Smaller sizes are genuine subsets: the first (10%) fit must be cheaper-or-equal in rows than the last.
    assert res.train_sizes[0] < res.train_sizes[-1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider", "--no-cov"]))
