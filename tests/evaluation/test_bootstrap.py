"""Sensor tests for ``mlframe.evaluation.bootstrap``.

Covers:
  - Bootstrap CI on synthetic AUC=0.85 problem: point ~0.85, CI contains true value.
  - Reproducibility under fixed seed.
  - Stratified resampling preserves class balance.
  - DeLong test returns small p when AUCs differ materially; large p when identical.
  - DeLong returns NaN p on degenerate input rather than crashing.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, mean_squared_error

from mlframe.evaluation.bootstrap import bootstrap_metric, bootstrap_metrics, delong_test

# Every test in this module exercises only synthetic data at n<=4000 with default bootstrap n=200-300; wall-time
# stays well under 2s per test. Marking as ``fast`` so the ``pytest -m fast`` smoke run keeps these in scope per the
# B2 #6 audit observation that ``@pytest.mark.fast`` had unrealistically narrow adoption.
pytestmark = [pytest.mark.fast]


def _make_binary_auc_data(n: int = 2000, separation: float = 1.6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate a binary classification problem with population AUC tunable by ``separation``."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    score = rng.normal(loc=separation * y, scale=1.0)
    return y, score


def test_bootstrap_metric_recovers_known_auc_point_and_ci():
    """At separation=1.6 the population AUC is ~0.87. Point/CI should bracket the true value."""
    y, score = _make_binary_auc_data(n=4000, separation=1.6, seed=42)
    true_auc = roc_auc_score(y, score)
    res = bootstrap_metric(
        y,
        score,
        metric_fn=lambda yt, yp: roc_auc_score(yt, yp),
        n_bootstrap=300,
        alpha=0.05,
        stratify=y,
        random_state=123,
    )
    assert "point" in res and "lo" in res and "hi" in res and "samples" in res
    assert abs(res["point"] - true_auc) < 1e-9, f"point estimate should equal full-sample metric ({true_auc:.6f}); got {res['point']:.6f}"
    assert res["lo"] < res["point"] < res["hi"], "CI must bracket the point estimate"
    assert res["lo"] <= true_auc <= res["hi"], f"95% CI [{res['lo']:.4f}, {res['hi']:.4f}] must contain true AUC {true_auc:.4f}"
    assert len(res["samples"]) == 300


def test_bootstrap_metric_reproducible_under_fixed_seed():
    """Two calls with the same seed must return bit-identical CI bounds + samples."""
    y, score = _make_binary_auc_data(n=1500, separation=1.0, seed=7)
    metric = lambda yt, yp: roc_auc_score(yt, yp)
    a = bootstrap_metric(y, score, metric_fn=metric, n_bootstrap=200, random_state=999)
    b = bootstrap_metric(y, score, metric_fn=metric, n_bootstrap=200, random_state=999)
    assert a["point"] == b["point"]
    assert a["lo"] == b["lo"]
    assert a["hi"] == b["hi"]
    np.testing.assert_array_equal(a["samples"], b["samples"])


def test_bootstrap_metric_regression_rmse():
    """Smoke test on a regression metric: bootstrap CI should bracket the in-sample RMSE."""
    rng = np.random.default_rng(13)
    y = rng.normal(0, 1, size=800)
    y_pred = y + rng.normal(0, 0.5, size=800)
    rmse = lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp)))
    res = bootstrap_metric(y, y_pred, metric_fn=rmse, n_bootstrap=200, random_state=2024)
    assert res["lo"] <= res["point"] <= res["hi"]
    # Population RMSE is 0.5; the bootstrap CI of in-sample RMSE on n=800 should comfortably contain it.
    assert res["lo"] <= 0.5 <= res["hi"]


def test_bootstrap_metric_rejects_mismatched_shapes():
    """Bootstrap metric rejects mismatched shapes."""
    with pytest.raises(ValueError, match="row counts diverge"):
        bootstrap_metric(np.zeros(10), np.zeros(8), metric_fn=lambda a, b: 0.0, n_bootstrap=10)


def test_bootstrap_metric_rejects_tiny_n():
    """Bootstrap metric rejects tiny n."""
    with pytest.raises(ValueError, match="need at least 2 samples"):
        bootstrap_metric(np.zeros(1), np.zeros(1), metric_fn=lambda a, b: 0.0, n_bootstrap=10)


def test_delong_detects_real_auc_difference():
    """When score_a is materially better than score_b on the same y, p_value should be small."""
    y, score_good = _make_binary_auc_data(n=2000, separation=2.0, seed=1)
    rng = np.random.default_rng(2)
    # Bad scorer: shuffled noisy signal -> AUC ~0.5
    score_bad = rng.normal(size=y.shape[0])
    res = delong_test(y, score_good, score_bad)
    assert res["auc_a"] > res["auc_b"]
    assert res["diff"] > 0.2, f"expected large AUC difference, got {res['diff']:.3f}"
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["p_value"] < 0.01, f"strong difference should yield p<<0.05; got p={res['p_value']:.4f}"


def test_delong_returns_high_p_when_scores_identical():
    """Comparing a scorer to itself should give exactly diff=0, p=1.0 (z=0)."""
    y, score = _make_binary_auc_data(n=1500, separation=1.2, seed=44)
    res = delong_test(y, score, score)
    assert res["diff"] == 0.0
    assert res["p_value"] == pytest.approx(1.0, abs=1e-9)


def test_delong_rejects_multiclass():
    """Delong rejects multiclass."""
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    score = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="binary 0/1"):
        delong_test(y, score, score + 1)


def _bm_metric_set():
    """Helper that bm metric set."""
    auc = lambda yy, pp: float(roc_auc_score(yy, pp))
    brier = lambda yy, pp: float(np.mean((yy - pp) ** 2))

    def ll(yy, pp):
        """Helper that ll."""
        pc = np.clip(pp, 1e-15, 1 - 1e-15)
        return float(-np.mean(yy * np.log(pc) + (1 - yy) * np.log(1 - pc)))

    return {"roc_auc": auc, "brier": brier, "log_loss": ll}


@pytest.mark.parametrize("stratified", [True, False])
def test_bootstrap_metrics_bit_identical_to_per_metric(stratified):
    """bootstrap_metrics sharing one resample loop must yield per-metric results
    bit-identical to calling bootstrap_metric once per metric with the same seed
    (identical index sequence; a metric raising never advances the RNG)."""
    y, score = _make_binary_auc_data(n=3000, seed=3)
    p = 1.0 / (1.0 + np.exp(-score))  # squash to [0,1] for brier/log_loss
    mfns = _bm_metric_set()
    strat = y if stratified else None
    shared = bootstrap_metrics(y, p, mfns, n_bootstrap=300, stratify=strat, random_state=4242)
    for name, fn in mfns.items():
        single = bootstrap_metric(y, p, metric_fn=fn, n_bootstrap=300, stratify=strat, random_state=4242)
        assert shared[name]["point"] == single["point"]
        assert shared[name]["lo"] == single["lo"]
        assert shared[name]["hi"] == single["hi"]
        assert np.array_equal(shared[name]["samples"], single["samples"])


def test_bootstrap_metric_method_bca_is_default():
    """The default method is BCa; an explicit method='percentile' yields the legacy interval."""
    y, score = _make_binary_auc_data(n=2000, separation=1.6, seed=8)
    metric = lambda yt, yp: roc_auc_score(yt, yp)
    default = bootstrap_metric(y, score, metric_fn=metric, n_bootstrap=300, random_state=11)
    bca = bootstrap_metric(y, score, metric_fn=metric, n_bootstrap=300, random_state=11, method="bca")
    pct = bootstrap_metric(y, score, metric_fn=metric, n_bootstrap=300, random_state=11, method="percentile")
    assert default["lo"] == bca["lo"] and default["hi"] == bca["hi"], "default must equal method='bca'"
    # On a skewed AUC the BCa cut-points shift away from the plain percentiles -> bounds differ.
    assert (default["lo"], default["hi"]) != (pct["lo"], pct["hi"]), "BCa and percentile should differ on skewed AUC"
    # Same resample distribution either way (method only changes the reduction, not the RNG draws).
    np.testing.assert_array_equal(default["samples"], pct["samples"])


def test_bootstrap_metric_bca_falls_back_to_percentile_when_no_skew_signal():
    """A constant metric (every resample identical) has no z0/acceleration signal -> BCa == percentile bounds."""
    y, score = _make_binary_auc_data(n=500, separation=1.0, seed=9)
    const_metric = lambda yt, yp: 0.5
    bca = bootstrap_metric(y, score, metric_fn=const_metric, n_bootstrap=100, random_state=3, method="bca")
    pct = bootstrap_metric(y, score, metric_fn=const_metric, n_bootstrap=100, random_state=3, method="percentile")
    assert bca["lo"] == pct["lo"] and bca["hi"] == pct["hi"]


def test_bootstrap_metrics_bca_matches_per_metric_bca():
    """bootstrap_metrics with default BCa must equal per-metric bootstrap_metric BCa (same seed/jackknife)."""
    y, score = _make_binary_auc_data(n=2500, seed=14)
    p = 1.0 / (1.0 + np.exp(-score))
    mfns = _bm_metric_set()
    shared = bootstrap_metrics(y, p, mfns, n_bootstrap=300, random_state=77)
    for name, fn in mfns.items():
        single = bootstrap_metric(y, p, metric_fn=fn, n_bootstrap=300, random_state=77)
        assert shared[name]["lo"] == single["lo"]
        assert shared[name]["hi"] == single["hi"]


def test_bootstrap_metrics_isolates_a_failing_metric():
    """A metric that raises on every resample gets an ``error`` entry; the other
    metrics still return valid CIs (one bad metric must not sink the batch)."""
    y, score = _make_binary_auc_data(n=1500, seed=5)
    p = 1.0 / (1.0 + np.exp(-score))

    def boom(yy, pp):
        """Helper that boom."""
        raise RuntimeError("always fails")

    res = bootstrap_metrics(
        y,
        p,
        {"good": lambda yy, pp: float(np.mean((yy - pp) ** 2)), "bad": boom},
        n_bootstrap=200,
        random_state=7,
    )
    assert "error" in res["bad"] and "point" not in res["bad"]
    assert "error" not in res["good"] and np.isfinite(res["good"]["point"])


def test_bootstrap_metrics_empty_and_shape_guard():
    """Bootstrap metrics empty and shape guard."""
    y, score = _make_binary_auc_data(n=500, seed=6)
    assert bootstrap_metrics(y, score, {}, n_bootstrap=50) == {}
    with pytest.raises(ValueError):
        bootstrap_metrics(np.zeros(10), np.zeros(8), {"m": lambda a, b: 0.0}, n_bootstrap=10)


def test_delong_degenerate_returns_nan_p():
    """Constant scores -> singular covariance -> p=nan, not a crash."""
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    flat = np.zeros(8)
    res = delong_test(y, flat, flat)
    assert np.isnan(res["p_value"]) or res["p_value"] == pytest.approx(1.0, abs=1e-9)


def test_jackknife_mean_metric_matches_gather_and_is_On():
    """The O(n) algebraic jackknife (_jackknife_mean_metric) must match the generic gather jackknife
    (_jackknife_metric) to sum-reduction-order FP noise for mean-decomposable metrics, and drive BCa CI bounds that
    match the generic path to ~1e-9. Regression sensor for the 517x bootstrap-jackknife optimization (log_loss/brier/
    rmse): re-gathering n-1 rows + recomputing the metric per leave-out point is replaced by LOO_i=(sum-row_i)/(n-1)."""
    import numpy as np
    from mlframe.evaluation.bootstrap import (
        _jackknife_metric,
        _jackknife_mean_metric,
        bootstrap_metric,
    )
    from mlframe.metrics.core import fast_log_loss, fast_brier_score_loss

    rng = np.random.default_rng(0)
    n = 40000
    y = rng.integers(0, 2, n).astype(np.float64)
    p = rng.uniform(0.001, 0.999, n)

    eps = np.finfo(p.dtype).eps
    pc = np.clip(p, eps, 1.0 - eps)
    ll_per_row = np.where(y == 1, -np.log(pc), -np.log(1.0 - pc))
    br_per_row = (p - y) ** 2

    # LOO arrays: algebraic vs gather (sorted; both skip the same single-class points for log_loss, none for brier).
    g_ll = np.sort(_jackknife_metric(y, p, lambda a, b: float(fast_log_loss(a, b))))
    a_ll = np.sort(_jackknife_mean_metric(y, ll_per_row, requires_both_classes=True))
    assert g_ll.shape == a_ll.shape and np.max(np.abs(g_ll - a_ll)) < 1e-11

    g_br = np.sort(_jackknife_metric(y, p, lambda a, b: float(fast_brier_score_loss(a, b))))
    a_br = np.sort(_jackknife_mean_metric(y, br_per_row, requires_both_classes=False))
    assert g_br.shape == a_br.shape and np.max(np.abs(g_br - a_br)) < 1e-11

    # End-to-end BCa CI: fast per-row path vs generic gather path agree to ~1e-9.
    _ll = lambda a, b: float(fast_log_loss(a, b))
    gen = bootstrap_metric(y, p, _ll, n_bootstrap=300, random_state=5)
    fast = bootstrap_metric(
        y,
        p,
        _ll,
        n_bootstrap=300,
        random_state=5,
        jackknife_per_row=(lambda yy, pp: np.where(yy == 1, -np.log(np.clip(pp, eps, 1 - eps)), -np.log(1 - np.clip(pp, eps, 1 - eps))), True, None),
    )
    assert gen["point"] == fast["point"]
    assert np.isclose(gen["lo"], fast["lo"], rtol=1e-9, atol=0.0)
    assert np.isclose(gen["hi"], fast["hi"], rtol=1e-9, atol=0.0)

    # Degenerate guard: non-finite per-row -> None (caller falls back to the exact gather path).
    bad = ll_per_row.copy()
    bad[0] = np.nan
    assert _jackknife_mean_metric(y, bad, requires_both_classes=True) is None


def test_jackknife_auc_matches_gather_bit_identical():
    """The placement-value ROC-AUC jackknife (_jackknife_auc) must be BIT-IDENTICAL to re-running the AUC on the n-1
    kept rows (the generic gather idx-jackknife), on both continuous and tied scores, and drive an identical BCa CI.
    Regression sensor for the 159x AUC-jackknife optimization (33.7s -> 0.21s at n=300k)."""
    import numpy as np
    from mlframe.evaluation.bootstrap import _jackknife_auc, _jackknife_metric_idx, bootstrap_metrics
    from mlframe.metrics.core import make_bootstrap_auc_resampler

    rng = np.random.default_rng(0)
    for scores_kind in ("continuous", "tied"):
        n = 30000
        y = rng.integers(0, 2, n).astype(np.float64)
        s = rng.uniform(0, 1, n) if scores_kind == "continuous" else np.round(rng.uniform(0, 1, n), 2)
        gen = np.sort(_jackknife_metric_idx(n, make_bootstrap_auc_resampler(y, s)))
        fast = np.sort(_jackknife_auc(y, s))
        assert gen.shape == fast.shape, scores_kind
        assert np.array_equal(gen, fast), (scores_kind, float(np.max(np.abs(gen - fast))))

    # End-to-end BCa CI must be bit-identical with vs without the fast AUC jackknife.
    n = 30000
    y = rng.integers(0, 2, n).astype(np.float64)
    s = rng.uniform(0, 1, n)
    gen = bootstrap_metrics(y, s, {}, metric_fns_idx={"roc_auc": make_bootstrap_auc_resampler(y, s)}, n_bootstrap=300, stratify=y, random_state=7)
    fast = bootstrap_metrics(
        y,
        s,
        {},
        metric_fns_idx={"roc_auc": make_bootstrap_auc_resampler(y, s)},
        n_bootstrap=300,
        stratify=y,
        random_state=7,
        jackknife_fns={"roc_auc": lambda yy, ss: _jackknife_auc(yy, ss)},
    )
    assert gen["roc_auc"]["point"] == fast["roc_auc"]["point"]
    assert gen["roc_auc"]["lo"] == fast["roc_auc"]["lo"]
    assert gen["roc_auc"]["hi"] == fast["roc_auc"]["hi"]

    # Degenerate: <2 of a class -> None (caller falls back to gather).
    y_deg = np.zeros(100)
    y_deg[0] = 1.0
    assert _jackknife_auc(y_deg, rng.uniform(0, 1, 100)) is None


def test_bootstrap_metric_per_row_resample_fastpath_matches_generic():
    """The per-row resample fast path (each resample = reduce_fn(mean(per_row[idx])), one gather + mean, vs gathering
    both y[idx]/p[idx] and recomputing the metric) must match the generic metric_fn resample path to sum-reduction-order
    FP noise on the CI bounds, for a mean-decomposable metric (RMSE). Regression sensor for the 3.1x RMSE-bootstrap
    optimization; also pins the both-classes gate (a requires_both_classes=True per-row spec must NOT engage the fast
    path, since it can't reproduce the metric's single-class NaN skip)."""
    import numpy as np
    from mlframe.evaluation.bootstrap import bootstrap_metric
    from mlframe.metrics.scoring import fast_rmse

    np.random.default_rng(0)
    _rmse_pr = lambda yy, pp: (np.asarray(yy, float) - np.asarray(pp, float)) ** 2
    for seed in range(8):
        r = np.random.default_rng(seed)
        n = int(r.integers(3000, 40000))
        y = r.standard_normal(n)
        p = y + r.standard_normal(n) * 0.5
        gen = bootstrap_metric(y, p, fast_rmse, n_bootstrap=400, random_state=seed)
        fast = bootstrap_metric(y, p, fast_rmse, n_bootstrap=400, random_state=seed, jackknife_per_row=(_rmse_pr, False, np.sqrt))
        assert gen["point"] == fast["point"]
        assert np.isclose(gen["lo"], fast["lo"], rtol=1e-9, atol=0.0)
        assert np.isclose(gen["hi"], fast["hi"], rtol=1e-9, atol=0.0)
