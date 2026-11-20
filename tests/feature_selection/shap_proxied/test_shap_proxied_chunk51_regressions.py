"""Regressions for defects found reading the ShapProxiedFS prefilter, preflight and revalidation modules."""

from __future__ import annotations

import itertools
import logging

import numpy as np
from scipy.optimize import linprog
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_prefilter as pf_mod
from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_prefilter_univariate as uni_mod
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_preflight import _cv_score
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _shap_proxy_loss as loss_mod
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_core_stability import _truncated_nucleolus
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_refine import revalidate_top_n, within_cluster_refine
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_sampling import _sample_anchor_subsets
from mlframe.utils.log_throttle import reset_throttle_counts


def _holdout_problem(n_search=300, n_hold=60, n_cols=4, seed=0):
    """Search rows with a real signal in column 0 and a SINGLE-CLASS holdout, so the AUC loss is undefined (NaN)."""
    rng = np.random.default_rng(seed)
    X_search = rng.normal(size=(n_search, n_cols))
    y_search = (X_search[:, 0] + 0.3 * rng.normal(size=n_search) > 0).astype(int)
    X_hold = rng.normal(size=(n_hold, n_cols))
    y_hold = np.ones(n_hold, dtype=int)
    return X_search, y_search, X_hold, y_hold


def test_preflight_cv_score_is_finite_for_a_multiclass_target():
    """A 3-class target scored with binary-only 'roc_auc' raised and became NaN, which the gate read as 'fallback'."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(450, 3))
    y = np.digitize(X[:, 0], [-0.5, 0.5])
    score = _cv_score(LogisticRegression(max_iter=200), X, y, True)
    assert np.isfinite(score) and score > 0.6


def test_preflight_cv_score_failure_is_logged_at_warning(caplog):
    """A failing CV fit returns NaN AND says so at WARNING (it was debug-only)."""
    reset_throttle_counts()

    class Boom(LogisticRegression):
        """Estimator whose fit always raises."""

        def fit(self, X, y, sample_weight=None):
            """Raise to simulate a degenerate fold."""
            raise RuntimeError("boom")

    X = np.random.default_rng(0).normal(size=(60, 2))
    y = np.arange(60) % 2
    with caplog.at_level(logging.WARNING):
        out = _cv_score(Boom(), X, y, True)
    assert np.isnan(out)
    assert any("_cv_score" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_anchor_sampler_zero_tail_fraction_is_pure_weighted():
    """uniform_tail_frac=0 documents 'pure weighted', but every anchor with k>=2 fell through to a uniform draw."""
    n_features = 30
    weights = np.r_[50.0, np.zeros(n_features - 1)]
    anchors = _sample_anchor_subsets(n_features, 20, np.random.default_rng(0), min_card=2, max_card=3, weights=weights, uniform_tail_frac=0.0)
    with_top = sum(1 for a in anchors if 0 in a)
    assert with_top >= 0.8 * len(anchors), (with_top, len(anchors))


def test_within_cluster_refine_keeps_columns_when_the_holdout_loss_is_undefined():
    """A single-class holdout makes the AUC loss NaN; NaN won argmin and passed the threshold test, collapsing the set to one column."""
    X_search, y_search, X_hold, y_hold = _holdout_problem()
    out = within_cluster_refine(
        [0, 1, 2, 3], LogisticRegression(max_iter=200), X_search, y_search, X_hold, y_hold,
        classification=True, metric="auc", n_jobs=1, refine_n_estimators=None,
    )
    assert out == [0, 1, 2, 3]


def test_revalidate_top_n_survives_an_undefined_honest_loss(caplog):
    """Every honest loss NaN used to raise 'min() arg is an empty sequence'; it now keeps the first candidate and warns."""
    reset_throttle_counts()
    X_search, y_search, X_hold, y_hold = _holdout_problem()
    candidates = [(0.1, (0, 1)), (0.2, (0,)), (0.3, (1, 2))]
    with caplog.at_level(logging.WARNING):
        best, ranked, _ = revalidate_top_n(
            candidates, LogisticRegression(max_iter=200), X_search, y_search, X_hold, y_hold,
            classification=True, metric="auc", n_jobs=1,
        )
    assert tuple(best) == (0, 1)
    assert len(ranked) == 3
    assert any("no candidate has a finite honest loss" in r.getMessage() for r in caplog.records)


def test_honest_loss_disk_key_separates_estimator_classes_with_equal_params():
    """Two estimator classes exposing identical parameter names and values must not share a cached-loss key."""

    class ModelA(LogisticRegression):
        """First estimator class."""

    class ModelB(LogisticRegression):
        """Second estimator class with the very same parameters."""

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = np.arange(40) % 2
    args = (X, y, X, y, [0, 1], True, "logloss", None, None, None)
    k_a = loss_mod._build_honest_loss_disk_key(ModelA(), *args)
    k_b = loss_mod._build_honest_loss_disk_key(ModelB(), *args)
    assert k_a is not None and k_b is not None and k_a != k_b
    assert loss_mod._build_perm_fit_disk_key(ModelA(), X, y, [0, 1], None, None) != loss_mod._build_perm_fit_disk_key(ModelB(), X, y, [0, 1], None, None)


def test_parallel_honest_losses_honours_negative_n_jobs_below_minus_one(monkeypatch):
    """n_jobs=-2 means 'all cores but one' in joblib; it was clamped to a single serial worker."""
    import joblib

    seen = {}

    class FakeParallel:
        """Records the requested worker count and runs the tasks inline."""

        def __init__(self, n_jobs=None, **kw):
            """Remember n_jobs."""
            seen["n_jobs"] = n_jobs

        def __call__(self, gen):
            """Run every delayed task serially."""
            return [f(*a, **k) for f, a, k in gen]

    monkeypatch.setattr(joblib, "Parallel", FakeParallel)
    monkeypatch.setattr(loss_mod, "_honest_loss", lambda *a, **k: 0.5)
    monkeypatch.setattr("os.cpu_count", lambda: 8)
    tasks = [([0], i) for i in range(6)]
    out = loss_mod._parallel_honest_losses(tasks, None, np.zeros((4, 2)), None, None, None, True, "brier", -2)
    assert out == [0.5] * 6
    assert seen["n_jobs"] == 6


def _random_game(k, rng):
    """A random transferable-utility game over ``k`` players."""
    base = rng.random(k)
    return {
        c: float(max(0.0, sum(base[j] for j in c) + rng.normal() * 0.3 * len(c) ** 0.5)) for r in range(1, k + 1) for c in itertools.combinations(range(k), r)
    }


def test_truncated_nucleolus_sharpens_beyond_the_least_core_point():
    """The sharpen pass pinned frozen rows with an eps term, so it returned the least-core point unchanged; it must lower the next-largest excess."""
    rng = np.random.default_rng(1)
    k = 4
    games = [_random_game(k, rng) for _ in range(2)]
    v = games[1]
    coal = [c for r in range(1, k) for c in itertools.combinations(range(k), r)]
    n = k + 1
    c_obj = np.zeros(n)
    c_obj[-1] = 1.0
    A = np.zeros((len(coal), n))
    b = np.zeros(len(coal))
    for r, c in enumerate(coal):
        A[r, list(c)] = -1.0
        A[r, -1] = -1.0
        b[r] = -v[c]
    A_eq = np.zeros((1, n))
    A_eq[0, :k] = 1.0
    grand = v[tuple(range(k))]
    res = linprog(c_obj, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=[grand], bounds=[(0, None)] * k + [(None, None)], method="highs")
    x0, eps0 = res.x[:k], float(res.x[-1])
    v_vals = {c: v[c] for c in coal}
    x1, _ = _truncated_nucleolus(coal, v_vals, grand, k, x0, eps0, max_iters=3)

    def excess(x):
        """Sorted-descending excess vector of an allocation."""
        return np.sort(np.array([v_vals[c] - x[list(c)].sum() for c in coal]))[::-1]

    e0, e1 = excess(x0), excess(x1)
    assert abs(e1[0] - e0[0]) < 1e-6
    assert e1[1] < e0[1] - 0.1
    assert abs(x1.sum() - grand) < 1e-6


def test_fast_model_logs_when_reduced_budget_params_are_rejected(caplog):
    """A template that rejects the reduced-budget params used to rank silently with the full booster."""
    reset_throttle_counts()

    class Picky(BaseEstimator):
        """Exposes max_depth but refuses to set any parameter."""

        def __init__(self, max_depth=None):
            """Accept max_depth."""
            self.max_depth = max_depth

        def set_params(self, **params):
            """Reject every parameter update."""
            raise ValueError("no")

        def fit(self, X, y):
            """Record a flat importance vector."""
            self.feature_importances_ = np.ones(X.shape[1])
            return self

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] > 0).astype(int)
    with caplog.at_level(logging.WARNING):
        pf_mod._rank_fast_model(Picky(), X, y, n_features=3)
    assert any("fast_model" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_stable_f_classif_skips_the_unused_sum_of_squares_pass(monkeypatch):
    """The stable path recentres the chunk itself, so the raw per-class sum of squares is dead work."""
    calls = []
    real = uni_mod._per_class_sums_legacy

    def spy(chunk, masks, acc_dtype, K, b, **kw):
        """Record whether the sum-of-squares pass was requested."""
        calls.append(kw.get("with_sumsq", True))
        return real(chunk, masks, acc_dtype, K, b, **kw)

    monkeypatch.setattr(uni_mod, "_per_class_sums_legacy", spy)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5))
    y = np.arange(120) % 3
    stable = uni_mod.f_classif_chunked(X, y, use_gemm=False, stable=True)
    assert calls and not any(calls)
    calls.clear()
    raw = uni_mod.f_classif_chunked(X, y, use_gemm=False, stable=False)
    assert calls and all(calls)
    np.testing.assert_allclose(stable, raw, rtol=1e-8)


def test_univariate_rank_accepts_pandas_nullable_columns_with_na():
    """A Float64 column holding pd.NA became an object array and raised TypeError inside the float cast."""
    import pandas as pd

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc")).astype("Float64")
    X.iloc[3, 0] = pd.NA
    y = (X["b"].to_numpy(dtype=float) > 0).astype(int)
    scores = pf_mod._rank_univariate(X, y, classification=True)
    assert scores.shape == (3,) and scores[1] == scores.max()
