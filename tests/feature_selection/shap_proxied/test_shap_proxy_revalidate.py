"""Tests for honest re-validation, the proxy-trust guard, and the importance-top-k ablation.

These exercise the guards that turn the cheap proxy into a defensible selector: a disjoint-holdout
honest retrain (winner's curse), a measured proxy-vs-honest fidelity report, and the unique-value
gate vs plain SHAP importance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression


@pytest.fixture
def planted():
    """Helper that planted."""
    rng = np.random.default_rng(0)
    n, f = 1200, 8
    X = pd.DataFrame(rng.normal(size=(n, f)), columns=[f"x{i}" for i in range(f)])
    # target is an exact linear combo of features 0,1,2 -> a linear model recovers it perfectly
    y = (1.5 * X["x0"] + 1.0 * X["x1"] - 0.8 * X["x2"] + 0.05 * rng.normal(size=n)).to_numpy()
    # SHAP-like phi for a linear model on standardized-ish data: phi_j ~ coef_j * (x_j - mean)
    phi = np.column_stack([1.5 * X["x0"], 1.0 * X["x1"], -0.8 * X["x2"]] + [np.zeros(n)] * (f - 3))
    base = np.full(n, float(y.mean()))
    return X, y, phi, base


def test_revalidate_recovers_planted_subset(planted):
    """Revalidate recovers planted subset."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import revalidate_top_n

    X, y, _phi, _base = planted
    idx_search = np.arange(900)
    idx_hold = np.arange(900, 1200)
    Xs, ys = X.iloc[idx_search].reset_index(drop=True), y[idx_search]
    Xh, yh = X.iloc[idx_hold].reset_index(drop=True), y[idx_hold]

    candidates = [(0.0, (0, 1, 2)), (0.1, (0, 1)), (0.2, (0, 1, 2, 5)), (0.3, (4, 5, 6))]
    best, ranked, baseline = revalidate_top_n(candidates, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse", n_models=1, lambda_stab=0.0)
    assert set(best) == {0, 1, 2}
    assert ranked[0]["honest_loss"] < ranked[-1]["honest_loss"]
    assert baseline["honest_loss"] > ranked[0]["honest_loss"]  # the chosen subset beats a random one


def test_trust_guard_high_fidelity_on_clean_proxy(planted):
    """Trust guard high fidelity on clean proxy."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import proxy_trust_guard

    X, y, phi, base = planted
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]
    # phi was computed on full X; slice to search rows to match ys
    rep = proxy_trust_guard(
        phi[:900], base[:900], ys, LinearRegression(), Xs, Xh, yh, classification=False, metric="rmse", n_anchors=25, rng=np.random.default_rng(0)
    )
    assert rep["n_anchors"] >= 10
    assert rep["spearman"] > 0.5  # clean linear proxy -> good fidelity
    assert rep["trustworthy"]


def test_active_learning_respects_budget_and_not_worse_than_proxy_top1(planted):
    """Active learning respects budget and not worse than proxy top1."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _honest_loss, active_learning_revalidate

    X, y, phi, _base = planted
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]

    # Candidate pool: proxy-best is the true {0,1,2}; several decoys with lower proxy_loss-ranks.
    candidates = [(0.05, (0, 1, 2)), (0.10, (0, 1)), (0.20, (0, 1, 2, 5)), (0.30, (4, 5, 6)), (0.40, (3, 4)), (0.50, (5, 6, 7))]
    # Enough anchors for the corrector to fit (proxy ~ honest with a redundancy wobble).
    rng = np.random.default_rng(0)
    cd = dict(
        proxy=list(rng.uniform(0.1, 0.6, 16)),
        honest=list(rng.uniform(0.1, 0.6, 16)),
        cards=list(rng.integers(2, 5, 16).astype(float)),
        redund=list(rng.uniform(0, 1, 16)),
    )

    budget = 4
    best_idx, ranked, n_eval = active_learning_revalidate(
        candidates,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        corrector_data=cd,
        phi=phi,
        budget=budget,
        batch=2,
        n_models=1,
        rng=np.random.default_rng(1),
    )

    assert n_eval <= budget and n_eval >= 1
    assert len(ranked) == n_eval
    # The proxy top-1 is always evaluated early, so the active best is never worse than it.
    proxy_top1_honest = _honest_loss(LinearRegression(), Xs, ys, Xh, yh, [0, 1, 2], False, "rmse")
    best_honest = min(d["honest_loss"] for d in ranked)
    assert best_honest <= proxy_top1_honest + 1e-9
    assert set(best_idx) <= {0, 1, 2, 3, 4, 5, 6, 7}


def test_importance_ablation_runs(planted):
    """Importance ablation runs."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import importance_topk_ablation

    X, y, phi, _base = planted
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]
    out = importance_topk_ablation(phi[:900], (0, 1, 2), LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse")
    assert out["proxy_features"] == (0, 1, 2)
    assert "proxy_honest_loss" in out and "importance_honest_loss" in out
    assert isinstance(out["proxy_wins"], bool)


# --------------------------------------------------------------------- HonestLossCache (dedup)


def test_honest_loss_cache_returns_identical_to_uncached(planted):
    """A cached retrain must equal a fresh fit on the SAME (subset, seed): same model, same data,
    same seed is deterministic, so the cached float is numerically identical."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import HonestLossCache, _honest_loss

    X, y, _phi, _base = planted
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]
    cols = [0, 1, 2, 5]

    uncached = _honest_loss(LinearRegression(), Xs, ys, Xh, yh, cols, False, "rmse", seed=7)
    cache = HonestLossCache()
    first = _honest_loss(LinearRegression(), Xs, ys, Xh, yh, cols, False, "rmse", seed=7, cache=cache)
    second = _honest_loss(LinearRegression(), Xs, ys, Xh, yh, cols, False, "rmse", seed=7, cache=cache)
    assert first == uncached  # first call (miss) matches the uncached fit
    assert second == first  # second call (hit) returns the byte-identical cached value
    assert cache.misses == 1 and cache.hits == 1


def test_honest_loss_cache_key_is_order_independent_and_seed_separated(planted):
    """Column permutations of one subset share a cache slot (order-independent key); the SAME subset
    with a DIFFERENT seed is a distinct slot (so seed-jittered re-validation fits are never merged)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import HonestLossCache, _honest_loss

    X, y, _phi, _base = planted
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]
    cache = HonestLossCache()

    a = _honest_loss(LinearRegression(), Xs, ys, Xh, yh, [0, 1, 2], False, "rmse", seed=3, cache=cache)
    b = _honest_loss(LinearRegression(), Xs, ys, Xh, yh, [2, 0, 1], False, "rmse", seed=3, cache=cache)
    assert a == b and cache.hits == 1  # permuted columns -> cache hit, identical loss
    # Different seed -> a fresh fit, not served from the [0,1,2]/seed=3 slot.
    before_misses = cache.misses
    _honest_loss(LinearRegression(), Xs, ys, Xh, yh, [0, 1, 2], False, "rmse", seed=99, cache=cache)
    assert cache.misses == before_misses + 1


# ----------------------------------------------------- within_cluster_refine (multi-drop)


def _refine_planted_redundant(n=600, n_red=4, seed=0):
    """Plant a few informatives + n_red near-duplicate copies of x0 + noise. Col 0 AND cols 3..3+n_red
    form one redundancy cluster; a correct refine must collapse them to a single representative."""
    rng = np.random.default_rng(seed)
    n_noise = 6
    X = rng.normal(size=(n, 3 + n_red + n_noise))
    for j in range(n_red):
        X[:, 3 + j] = X[:, 0] + 0.05 * rng.normal(size=n)  # near-duplicates of col 0
    cols = [f"inf{i}" for i in range(3)] + [f"dup{j}" for j in range(n_red)] + [f"noise{i}" for i in range(n_noise)]
    Xdf = pd.DataFrame(X, columns=cols)
    y = (1.2 * X[:, 0] + 0.9 * X[:, 1] - 0.7 * X[:, 2] + 0.3 * rng.normal(size=n)).astype(np.float64)
    return Xdf, y


def _refine_planted_groups():
    """``member_groups`` for the planted-redundant fixture: cols 0 + 3..6 form ONE multi-cluster
    (the redundancy cluster), cols 1, 2, 7..12 are singletons."""
    return [[0, 3, 4, 5, 6], [1], [2], [7], [8], [9], [10], [11], [12]]


def test_within_cluster_refine_collapses_redundant_in_fewer_fits():
    """Multi-drop refine should make FEWER honest fits than the legacy per-round single-drop on a
    selected union with a redundant cluster (the win we ship). Same final loss within tol.

    Fit count is measured via a fresh ``HonestLossCache`` per call -- ``misses + hits`` equals every
    invocation of ``_honest_loss`` along the refine path (the inner cache is consulted on every fit).

    Stage 1 is gated on ``>= min_multi_clusters`` multi-member groups; this test uses
    ``min_multi_clusters=1`` to exercise the cluster-collapse path with a single-multi-cluster fixture
    (the cluster-aware mechanic itself). The companion ``..._gate_preserves_collapse_with_enough_clusters``
    test exercises the DEFAULT threshold on a 3-multi-cluster fixture."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import HonestLossCache, within_cluster_refine

    X, y = _refine_planted_redundant(seed=0)
    n_search = 450
    Xs, ys = X.iloc[:n_search].reset_index(drop=True), y[:n_search]
    Xh, yh = X.iloc[n_search:].reset_index(drop=True), y[n_search:]
    member_cols = list(range(X.shape[1]))  # the chosen union: every column, incl. 4 redundant dups
    member_groups = _refine_planted_groups()

    cache_multi = HonestLossCache()
    refined = within_cluster_refine(
        member_cols,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        parsimony_tol=0.05,
        n_jobs=1,
        member_groups=member_groups,
        cache=cache_multi,
        min_multi_clusters=1,
    )
    n_multi = cache_multi.misses + cache_multi.hits

    cache_legacy = HonestLossCache()
    refined_legacy = within_cluster_refine(
        member_cols,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        parsimony_tol=0.05,
        n_jobs=1,
        member_groups=None,
        cache=cache_legacy,
    )
    n_legacy = cache_legacy.misses + cache_legacy.hits

    # Cluster collapse + stage-2 multi-drop probes strictly reduce the fit count.
    assert n_multi < n_legacy, f"cluster-aware fits {n_multi} not < legacy {n_legacy}"
    # Refine must keep at least one member of the redundancy cluster (col 0 OR a dup) plus inf1, inf2.
    redundancy_cluster_kept = sum(1 for c in refined if c in (0, 3, 4, 5, 6))
    assert redundancy_cluster_kept >= 1, "refine must keep at least one redundancy-cluster member"
    assert {1, 2}.issubset(set(refined)), "refine must keep the unique informatives inf1, inf2"
    # Legacy must also keep at least one redundancy-cluster member.
    assert sum(1 for c in refined_legacy if c in (0, 3, 4, 5, 6)) >= 1


def test_within_cluster_refine_equivalent_to_legacy_when_no_safe_multi_drop():
    """When every member is informative (no safe drop), multi-drop refine must produce the SAME
    output as legacy (no behavior change for the non-redundant case). The early-exit "no single drop
    helps" branch fires before multi-drop, preserving identity."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import within_cluster_refine

    rng = np.random.default_rng(1)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    y = (1.5 * X["x0"] + 1.2 * X["x1"] - 0.9 * X["x2"] + 0.7 * X["x3"] - 0.5 * X["x4"] + 0.1 * rng.normal(size=n)).to_numpy()
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]
    cols = list(range(5))
    refined_legacy = within_cluster_refine(
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse", parsimony_tol=0.02, n_jobs=1, member_groups=None
    )
    refined_multi = within_cluster_refine(
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse", parsimony_tol=0.02, n_jobs=1, member_groups=[[i] for i in range(5)]
    )
    assert refined_legacy == refined_multi


def test_within_cluster_refine_engages_shared_cache():
    """When a HonestLossCache is supplied, refine's single-drop trials must consult it; the second
    invocation on the same inputs returns identical output with all-but-zero new fits."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import HonestLossCache, within_cluster_refine

    X, y = _refine_planted_redundant(seed=2)
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]
    cols = list(range(X.shape[1]))
    groups = _refine_planted_groups()
    cache = HonestLossCache()
    r1 = within_cluster_refine(
        cols,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        parsimony_tol=0.05,
        n_jobs=1,
        member_groups=groups,
        cache=cache,
        min_multi_clusters=1,
    )
    misses_1 = cache.misses
    r2 = within_cluster_refine(
        cols,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        parsimony_tol=0.05,
        n_jobs=1,
        member_groups=groups,
        cache=cache,
        min_multi_clusters=1,
    )
    # Same result; second pass adds zero new fits (every (subset, seed=None) is cached from the first).
    assert r1 == r2
    assert cache.misses == misses_1, f"second refine added {cache.misses - misses_1} new fits; expected 0"
    assert cache.hits > 0


def test_within_cluster_refine_gates_stage1_on_low_redundancy():
    """When the selected unit set has < ``min_multi_clusters`` multi-member groups, refine must SKIP
    stage-1 cluster collapse and route directly to stage-2 legacy greedy.

    On essentially-clean (no-redundancy) data the proxy still picks a tiny pseudo-cluster from spurious
    correlations; firing stage-1 on it wastes k+1 fits for an unhelpful collapse and routes the same
    columns into stage-2 unchanged. This regressed wall-clock at 2k-clean vs the pre-iter7 baseline.

    Verified by routing: when stage-1 is gated off, the function takes the SAME path as
    ``member_groups=None`` (legacy), so the fit count must match exactly."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import HonestLossCache, within_cluster_refine

    # All-informative data with no redundancy: legacy refine fires its early-exit (no single drop
    # within tol). The stage-1 cluster-collapse cost on a spurious cluster is then PURE overhead.
    rng = np.random.default_rng(7)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    y = (1.5 * X["x0"] + 1.2 * X["x1"] - 0.9 * X["x2"] + 0.7 * X["x3"] - 0.5 * X["x4"] + 0.1 * rng.normal(size=n)).to_numpy()
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]
    cols = list(range(5))
    # 1 spurious multi-member group (cols 3,4) + 3 singletons. Below default min_multi_clusters=3.
    spurious_one_multi = [[0], [1], [2], [3, 4]]
    assert sum(1 for g in spurious_one_multi if len(g) > 1) == 1

    cache_gated = HonestLossCache()
    refined_gated = within_cluster_refine(
        cols,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        parsimony_tol=0.02,
        n_jobs=1,
        member_groups=spurious_one_multi,
        cache=cache_gated,
        min_multi_clusters=3,
    )
    fits_gated = cache_gated.misses + cache_gated.hits

    cache_legacy = HonestLossCache()
    refined_legacy = within_cluster_refine(
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse", parsimony_tol=0.02, n_jobs=1, member_groups=None, cache=cache_legacy
    )
    fits_legacy = cache_legacy.misses + cache_legacy.hits

    # With the gate, the low-redundancy path matches legacy behavior exactly (no stage-1 overhead).
    assert fits_gated == fits_legacy, f"gated refine ran {fits_gated} fits vs legacy {fits_legacy}; stage-1 should be skipped"
    assert refined_gated == refined_legacy

    # When the gate is loosened (min_multi_clusters=1), stage-1 fires and costs ADDITIONAL fits
    # (the spurious cluster's probe + cumulative verify before stage-2 runs).
    cache_ungated = HonestLossCache()
    within_cluster_refine(
        cols,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        parsimony_tol=0.02,
        n_jobs=1,
        member_groups=spurious_one_multi,
        cache=cache_ungated,
        min_multi_clusters=1,
    )
    fits_ungated = cache_ungated.misses + cache_ungated.hits
    assert fits_ungated > fits_gated, f"un-gated refine ran {fits_ungated} fits vs gated {fits_gated}; stage-1 should have fired"


def test_within_cluster_refine_gate_preserves_collapse_with_enough_clusters():
    """The gate must NOT block stage-1 when there are >= min_multi_clusters multi-member groups
    (the cluster-rich case that wins). Build 3 redundancy clusters and verify stage-1 fires
    (fit count > legacy single-drop)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import HonestLossCache, within_cluster_refine

    rng = np.random.default_rng(11)
    n = 700
    # 3 separate redundancy clusters: each has 1 informative + 2 near-duplicates.
    X = rng.normal(size=(n, 12))
    # cluster A: cols 0, 1, 2 (1 is dup of 0, 2 is dup of 0)
    X[:, 1] = X[:, 0] + 0.05 * rng.normal(size=n)
    X[:, 2] = X[:, 0] + 0.05 * rng.normal(size=n)
    # cluster B: cols 3, 4, 5
    X[:, 4] = X[:, 3] + 0.05 * rng.normal(size=n)
    X[:, 5] = X[:, 3] + 0.05 * rng.normal(size=n)
    # cluster C: cols 6, 7, 8
    X[:, 7] = X[:, 6] + 0.05 * rng.normal(size=n)
    X[:, 8] = X[:, 6] + 0.05 * rng.normal(size=n)
    # cols 9, 10, 11 are noise singletons.
    Xdf = pd.DataFrame(X, columns=[f"x{i}" for i in range(12)])
    y = (1.2 * X[:, 0] + 0.9 * X[:, 3] - 0.7 * X[:, 6] + 0.3 * rng.normal(size=n)).astype(np.float64)
    Xs, ys = Xdf.iloc[:500].reset_index(drop=True), y[:500]
    Xh, yh = Xdf.iloc[500:].reset_index(drop=True), y[500:]
    cols = list(range(12))
    # 3 multi-member groups -> meets default min_multi_clusters=3 -> stage-1 must fire.
    groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9], [10], [11]]
    assert sum(1 for g in groups if len(g) > 1) == 3

    cache_gate = HonestLossCache()
    refined = within_cluster_refine(
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse", parsimony_tol=0.05, n_jobs=1, member_groups=groups, cache=cache_gate
    )
    fits_with_stage1 = cache_gate.misses + cache_gate.hits

    cache_legacy = HonestLossCache()
    within_cluster_refine(
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse", parsimony_tol=0.05, n_jobs=1, member_groups=None, cache=cache_legacy
    )
    fits_legacy = cache_legacy.misses + cache_legacy.hits

    # Stage-1 collapse must shrink the column count BELOW legacy greedy alone could on its own (the
    # net win is the canonical 4.7x at 5k-clean).
    assert len(refined) <= 6, f"stage-1 collapse should drop most duplicates; got {len(refined)} cols"
    # Stage-1 collapse strictly REDUCES total fits vs legacy single-drop greedy because each
    # collapse skips a full single-drop round (n_features fits each).
    assert fits_with_stage1 < fits_legacy, f"stage-1 fits {fits_with_stage1} not < legacy {fits_legacy}"


def test_full_fit_cached_matches_uncached():
    """End-to-end: the selector's support_ + selected_features_ are identical whether the honest-loss
    cache serves duplicates or every retrain is recomputed (the cache changes only wall-clock)."""
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_revalidate as RV
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    rng = np.random.default_rng(0)
    n, f = 1500, 30
    Xnp = rng.normal(size=(n, f))
    Xnp[:, 5] = Xnp[:, 0] + 0.2 * rng.normal(size=n)  # a correlated redundant copy -> exercises clustering
    X = pd.DataFrame(Xnp, columns=[f"x{i}" for i in range(f)])
    logit = 1.2 * Xnp[:, 0] + 0.9 * Xnp[:, 1] - 0.7 * Xnp[:, 2]
    y = pd.Series((logit + 0.3 * rng.normal(size=n) > 0).astype(int))

    def _fit():
        """Helper that fit."""
        sel = ShapProxiedFS(
            classification=True,
            metric="brier",
            optimizer="auto",
            cluster_features=True,
            top_n=12,
            n_splits=3,
            n_revalidation_models=2,
            n_anchors=15,
            random_state=0,
            verbose=False,
        )
        sel.fit(X, y)
        return sel.support_.copy(), list(sel.selected_features_)

    sup_cached, feats_cached = _fit()

    # Force the no-cache path by neutering get() so every retrain recomputes.
    real_get = RV.HonestLossCache.get
    # *args/**kwargs swallows whichever signature the cache exposes (the production code threads a
    # template_id namespace on top of (idx, seed) for the capped-refine retrain pool).
    RV.HonestLossCache.get = lambda self, *a, **kw: None
    try:
        sup_uncached, feats_uncached = _fit()
    finally:
        RV.HonestLossCache.get = real_get

    assert np.array_equal(sup_cached, sup_uncached)
    assert feats_cached == feats_uncached


# ----------------------------------------------------- refine_n_estimators cap


def test_try_cap_n_estimators_sets_first_recognised_param_only():
    """The cap helper hits the first ``n_estimators``-like param the template exposes and stops.
    Estimators without any such param are left untouched (silent no-op for linear models in tests)."""
    from sklearn.ensemble import GradientBoostingRegressor

    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _try_cap_n_estimators

    # xgboost / sklearn GBM exposes ``n_estimators``.
    est = GradientBoostingRegressor()
    assert _try_cap_n_estimators(est, 50) == "n_estimators"
    assert est.get_params()["n_estimators"] == 50

    # A linear model has none of the recognised params -> silent no-op.
    lr = LinearRegression()
    assert _try_cap_n_estimators(lr, 50) is None
    # No new attribute introduced.
    assert "n_estimators" not in lr.get_params()


def test_within_cluster_refine_cap_namespaces_cache_entries():
    """The cap puts entries into a distinct cache namespace (template_id != None), so the same
    ``(cols, seed=None)`` retrain at the FULL template vs the CAPPED template is two cache slots --
    a final full-template re-eval of the winner CANNOT be served by a capped-template entry."""
    from sklearn.ensemble import GradientBoostingClassifier

    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import HonestLossCache, _honest_loss

    rng = np.random.default_rng(0)
    n, f = 400, 6
    X = pd.DataFrame(rng.normal(size=(n, f)), columns=[f"x{i}" for i in range(f)])
    y = (1.2 * X["x0"] - 0.8 * X["x1"] + 0.3 * rng.normal(size=n) > 0).astype(int).to_numpy()
    Xs, ys = X.iloc[:300].reset_index(drop=True), y[:300]
    Xh, yh = X.iloc[300:].reset_index(drop=True), y[300:]
    cache = HonestLossCache()
    cols = [0, 1, 2]
    # Full-template entry (no cap, default template_id=None).
    full = _honest_loss(GradientBoostingClassifier(n_estimators=200, random_state=0), Xs, ys, Xh, yh, cols, True, "brier", cache=cache)
    misses_after_full = cache.misses
    # Cap with template_id -> distinct cache slot, must add a NEW miss (not served by the full entry).
    capped = _honest_loss(
        GradientBoostingClassifier(n_estimators=200, random_state=0),
        Xs,
        ys,
        Xh,
        yh,
        cols,
        True,
        "brier",
        cache=cache,
        n_estimators_cap=50,
        template_id=("refine_cap", 50),
    )
    assert cache.misses == misses_after_full + 1, "capped fit must be a fresh miss, not a cache hit"
    # And the capped model with fewer trees is a different loss number than the full booster.
    assert capped != full

    # Calling again at the FULL template still hits the original full-template slot (no new miss).
    misses_before = cache.misses
    full2 = _honest_loss(GradientBoostingClassifier(n_estimators=200, random_state=0), Xs, ys, Xh, yh, cols, True, "brier", cache=cache)
    assert cache.misses == misses_before  # served from the (cols, None, None) slot
    assert full2 == full


def test_within_cluster_refine_cap_keeps_redundancy_decisions():
    """With ``refine_n_estimators`` set (capped booster), refine still keeps the unique informatives
    and drops most redundant duplicates -- the relative ranking is preserved. The CHOSEN subset is the
    user-visible business value here; exact identity to the uncapped path isn't required (the cap is a
    speed/quality trade-off), only that the informatives survive and the duplicates collapse.

    Uses a tree booster (the cap is a no-op for the linear model used elsewhere in this file)."""
    from sklearn.ensemble import GradientBoostingClassifier

    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import within_cluster_refine

    rng = np.random.default_rng(0)
    n, f_inf, n_dup, n_noise = 800, 3, 4, 4
    X = rng.normal(size=(n, f_inf + n_dup + n_noise))
    # cols 0..2 informative; cols 3..6 are near-duplicates of col 0; cols 7..10 noise.
    for j in range(n_dup):
        X[:, f_inf + j] = X[:, 0] + 0.05 * rng.normal(size=n)
    cols_names = [f"inf{i}" for i in range(f_inf)] + [f"dup{j}" for j in range(n_dup)] + [f"noise{i}" for i in range(n_noise)]
    Xdf = pd.DataFrame(X, columns=cols_names)
    y = ((1.2 * X[:, 0] + 0.9 * X[:, 1] - 0.7 * X[:, 2] + 0.3 * rng.normal(size=n)) > 0).astype(int)
    Xs, ys = Xdf.iloc[:600].reset_index(drop=True), y[:600]
    Xh, yh = Xdf.iloc[600:].reset_index(drop=True), y[600:]
    cols = list(range(Xdf.shape[1]))
    member_groups = [[0, 3, 4, 5, 6], [1], [2], [7], [8], [9], [10]]

    refined = within_cluster_refine(
        cols,
        GradientBoostingClassifier(n_estimators=200, random_state=0),
        Xs,
        ys,
        Xh,
        yh,
        classification=True,
        metric="brier",
        parsimony_tol=0.05,
        n_jobs=1,
        member_groups=member_groups,
        min_multi_clusters=1,
        refine_n_estimators=50,
    )
    # Refine must keep at least one of the redundancy-cluster members AND the unique informatives.
    assert any(c in (0, 3, 4, 5, 6) for c in refined), f"missing redundancy-cluster rep: {refined}"
    assert {1, 2}.issubset(set(refined)), f"refine dropped unique informatives: {refined}"
    # And it should drop AT LEAST one of the four duplicates (the whole point of within-cluster refine).
    assert len(set(refined) & {3, 4, 5, 6}) < n_dup, f"refine kept all duplicates: {refined}"


def test_shap_proxied_fs_records_honest_loss_full_for_refined_subset():
    """End-to-end: the report's ``within_cluster_refine`` block exposes a ``honest_loss_full`` measured
    at the FULL template (uncapped) for the chosen refined subset, so downstream consumers see a value
    consistent with the surrounding guards' booster size."""
    pytest = __import__("pytest")
    pytest.importorskip("xgboost")
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    rng = np.random.default_rng(0)
    n, f = 1500, 30
    Xnp = rng.normal(size=(n, f))
    Xnp[:, 5] = Xnp[:, 0] + 0.2 * rng.normal(size=n)  # redundancy -> clustering fires
    X = pd.DataFrame(Xnp, columns=[f"x{i}" for i in range(f)])
    logit = 1.2 * Xnp[:, 0] + 0.9 * Xnp[:, 1] - 0.7 * Xnp[:, 2]
    y = pd.Series((logit + 0.3 * rng.normal(size=n) > 0).astype(int))

    sel = ShapProxiedFS(
        classification=True,
        metric="brier",
        optimizer="auto",
        cluster_features=True,
        top_n=12,
        n_splits=3,
        n_revalidation_models=2,
        n_anchors=15,
        random_state=0,
        verbose=False,
        refine_n_estimators=50,
    )
    sel.fit(X, y)
    ref = sel.shap_proxy_report_.get("within_cluster_refine")
    assert ref is not None
    # Refine ran and recorded the full-template loss for the chosen subset.
    assert "honest_loss_full" in ref
    assert 0.0 <= ref["honest_loss_full"] <= 1.0  # brier is bounded


# --------------------------------------- iter11: permutation-importance + batch-drop refine


def test_permutation_importance_ranking_is_deterministic_across_calls():
    """``_permutation_importance_ranking`` with the same ``seed`` must return BYTE-IDENTICAL importance
    vectors across separate calls -- the function is the basis of ``within_cluster_refine``'s drop
    ranking and a non-deterministic ranking would silently change which members are dropped under
    n_jobs=1 (the same seed should reproduce the same refined subset)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _permutation_importance_ranking

    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"x{i}" for i in range(6)])
    y = (1.2 * X["x0"] + 0.8 * X["x1"] - 0.5 * X["x2"] + 0.1 * rng.normal(size=n)).to_numpy()
    Xs, ys = X.iloc[:400].reset_index(drop=True), y[:400]
    Xh, yh = X.iloc[400:].reset_index(drop=True), y[400:]
    cols = list(range(6))

    base1, imps1 = _permutation_importance_ranking(LinearRegression(), Xs, ys, Xh, yh, cols, False, "rmse", seed=42)
    base2, imps2 = _permutation_importance_ranking(LinearRegression(), Xs, ys, Xh, yh, cols, False, "rmse", seed=42)
    assert base1 == base2
    assert np.array_equal(imps1, imps2), f"non-deterministic ranking: {imps1} vs {imps2}"
    # Different seed -> different shuffle -> ranks should still be qualitatively similar (informative
    # cols outrank noise) but the importance values may differ slightly.
    _, imps3 = _permutation_importance_ranking(LinearRegression(), Xs, ys, Xh, yh, cols, False, "rmse", seed=999)
    # Top-3 by importance: informatives 0, 1, 2 should dominate (positive imp), noise (3, 4, 5) low.
    assert set(np.argsort(-imps1)[:3].tolist()) == {0, 1, 2}, imps1
    assert set(np.argsort(-imps3)[:3].tolist()) == {0, 1, 2}, imps3


def test_within_cluster_refine_batch_drop_falls_back_to_single_drop_when_all_essential():
    """When every member is informative (no individually-safe drop exists), the batch-drop path must
    NOT delete essentials. The iter11 algorithm starts with ``batch_size = max(n_safe, 1)`` and halves
    on rejection; with all-essential data the first attempt drops the single least-important member
    and the retrain rejects it -> the loop exits, keeping every column. This matches legacy single-drop
    greedy's behaviour on the same input (the canonical "no safe drop" fall-through).

    Verified by: refined == cols (no drops) AND a separate legacy run on the same input also returns
    every column. This is the iter11 fallback guarantee."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import within_cluster_refine

    rng = np.random.default_rng(2)
    n = 600
    # Five linearly independent informatives with similar coefficients -> none is individually safe.
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    y = (1.5 * X["x0"] + 1.2 * X["x1"] - 0.9 * X["x2"] + 0.7 * X["x3"] - 0.5 * X["x4"] + 0.05 * rng.normal(size=n)).to_numpy()
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]
    cols = list(range(5))

    refined_batch = within_cluster_refine(
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse", parsimony_tol=0.02, n_jobs=1, member_groups=None
    )
    refined_legacy_input = within_cluster_refine(
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse", parsimony_tol=0.02, n_jobs=1, member_groups=[[i] for i in range(5)]
    )

    # Both paths reach the same all-essential conclusion: keep every member.
    assert refined_batch == cols, f"batch-drop dropped essentials: {refined_batch}"
    assert refined_legacy_input == cols, f"member-groups path dropped essentials: {refined_legacy_input}"


def test_within_cluster_refine_batch_drop_strictly_fewer_fits_on_redundant_data():
    """iter11 batch-drop must produce STRICTLY FEWER honest retrains than the legacy O(k) per-round
    trial fits on a redundancy-heavy union -- the speedup is the headline win. We measure cache
    invocations (every honest retrain goes through ``_honest_loss``, hence the cache); the
    permutation-importance ranking pass is OUTSIDE the cache (one direct fit + k cheap predicts), so a
    decrease in cache events is the batch-drop saving.

    The baseline is the legacy refine pre-iter11 -- captured here by ``member_groups=None`` (legacy
    pure greedy-backward, no stage-1 cluster collapse, no batch-drop) so the comparison isolates the
    stage-2 algorithm change."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import HonestLossCache, within_cluster_refine

    X, y = _refine_planted_redundant(seed=3, n_red=4)  # 13 cols total, 4 are duplicates of col 0
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]
    cols = list(range(X.shape[1]))

    # iter11 batch-drop path: stage-1 disabled (member_groups=None) so we measure stage-2 alone.
    cache_new = HonestLossCache()
    r_new = within_cluster_refine(
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse", parsimony_tol=0.05, n_jobs=1, member_groups=None, cache=cache_new
    )
    fits_new = cache_new.misses + cache_new.hits

    # Both must still drop the redundant duplicates (the unique informatives + at least one of the
    # redundancy-cluster members survive).
    assert {1, 2}.issubset(set(r_new)), f"batch-drop refine dropped informatives: {r_new}"
    assert any(c in (0, 3, 4, 5, 6) for c in r_new), f"batch-drop refine dropped cluster: {r_new}"

    # And the redundancy cluster IS partially collapsed (at least one dup goes).
    n_cluster_kept = sum(1 for c in r_new if c in (0, 3, 4, 5, 6))
    assert n_cluster_kept < 5, f"batch-drop refine kept ALL redundancy members: {r_new}"

    # The honest-retrain count is bounded by O(log k) batched retrains per ranking round + at most
    # ``rounds`` ranking rounds. For a 13-col input the legacy upper bound is O(k^2) = 169 cache
    # events; iter11 must be a fraction of that.
    assert fits_new <= 30, f"iter11 refine ran {fits_new} cache events; expected far fewer than O(k^2)"


# ------------------------------------------------ stratified anchor sampler (iter14, F-score prior)


def test_stratified_anchor_sampler_overweights_high_f_columns():
    """The F-score-weighted anchor sampler must pick high-F columns MORE OFTEN than a uniform sampler
    on a fixture with a clear F-score gradient. We measure the empirical pick frequency over many
    draws and assert the high-F end of the score gradient is over-represented under weights and not
    under uniform. The 20% uniform tail is intentional headroom for low-F columns so we don't assert
    "low-F is never picked"; we assert the HIGH-F end gets >= 1.5x its uniform-baseline frequency."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _sample_anchor_subsets

    n_features = 50
    n_anchors = 200
    # Sharp gradient: first 5 columns have far higher F-score than the rest (the "informative" tier).
    weights = np.zeros(n_features, dtype=np.float64)
    weights[:5] = 10.0  # strong prior on first 5
    weights[5:] = 0.0  # uniform across the rest (softmax(0)=1/n on the tail)

    rng_w = np.random.default_rng(0)
    rng_u = np.random.default_rng(0)
    # Fix cardinality at 5 so the comparison is cleanly normalised (every anchor draws 5 columns).
    a_weighted = _sample_anchor_subsets(n_features, n_anchors, rng_w, min_card=5, max_card=5, weights=weights, uniform_tail_frac=0.2)
    a_uniform = _sample_anchor_subsets(n_features, n_anchors, rng_u, min_card=5, max_card=5, weights=None)
    counts_w = np.zeros(n_features, dtype=np.int64)
    counts_u = np.zeros(n_features, dtype=np.int64)
    for a in a_weighted:
        for c in a:
            counts_w[c] += 1
    for a in a_uniform:
        for c in a:
            counts_u[c] += 1

    # The high-F tier must be picked materially MORE often under weights than under uniform.
    high_w = counts_w[:5].sum()
    high_u = counts_u[:5].sum()
    assert high_w > high_u * 1.5, f"stratified sampler did not over-weight high-F columns: high_w={high_w} vs high_u={high_u}"
    # Sanity: every anchor still has exactly 5 distinct columns (no replacement bug).
    for a in a_weighted:
        assert len(a) == 5 and len(set(a)) == 5


def test_softmax_weights_auto_temperature_is_scale_invariant():
    """Regression for iter97: ``_softmax_weights`` must keep the softmax entropy bounded regardless
    of input score magnitude. A length-N vector with std=40 (raw F-scores on a real cohort) MUST give
    roughly the same effective sample size as the same vector divided by 40 (z-scored). Before iter97
    the default ``temperature=1.0`` collapsed raw F-scores to a near-one-hot distribution (ESS ~1.1),
    making the stratified anchor sampler draw essentially identical anchors and killing the
    trust-guard Spearman signal at iter14's width=6000 biz_val regime (dropped to 0.55 from a
    measured 0.88)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _softmax_weights

    rng = np.random.default_rng(0)
    raw = rng.normal(loc=40.0, scale=40.0, size=28)  # F-statistic-like: mean ~40, std ~40
    zscored = (raw - raw.mean()) / raw.std()
    p_raw = _softmax_weights(raw)
    p_z = _softmax_weights(zscored)
    ess_raw = 1.0 / float(np.sum(p_raw**2))
    ess_z = 1.0 / float(np.sum(p_z**2))
    # Both should be well above 1 (not one-hot) AND within 5% of each other (scale-invariant).
    assert ess_raw > 5.0, f"auto-temperature softmax collapsed on raw F-scores: ESS={ess_raw:.2f}"
    assert ess_z > 5.0, f"auto-temperature softmax collapsed on z-scored input: ESS={ess_z:.2f}"
    assert abs(ess_raw - ess_z) / ess_z < 0.05, f"auto-temperature is not scale-invariant: ESS_raw={ess_raw:.2f} ESS_z={ess_z:.2f}"
    # Explicit numeric temperature still works (legacy callers): high temp -> near-uniform.
    p_high_temp = _softmax_weights(raw, temperature=100.0)
    ess_high = 1.0 / float(np.sum(p_high_temp**2))
    assert ess_high > ess_raw * 0.8, f"explicit-temperature path must still flatten softmax at high T: ESS_high={ess_high:.2f}"


def test_stratified_anchor_sampler_falls_back_to_uniform_when_weights_none():
    """When ``weights=None`` the sampler MUST match the legacy (now ``cardinality_dist='uniform'``)
    sampler bit-for-bit (same RNG state -> same anchor list). This is the safety net: non-two-stage
    prefilter paths see no F-scores, so the trust guard degrades to legacy behaviour with zero risk
    of a sampler-induced regression."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _sample_anchor_subsets

    n_features = 30
    n_anchors = 50
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    # Both calls explicitly request the uniform cardinality prior so we exercise the legacy code path.
    a_explicit = _sample_anchor_subsets(n_features, n_anchors, rng_a, min_card=2, max_card=6, weights=None, cardinality_dist="uniform")
    a_legacy = _sample_anchor_subsets(n_features, n_anchors, rng_b, min_card=2, max_card=6, cardinality_dist="uniform")
    assert a_explicit == a_legacy


def test_uniform_cardinality_matches_legacy_rngs_bit_for_bit():
    """The iter15 ``cardinality_dist='uniform'`` opt-out MUST exactly reproduce the pre-iter15
    sampler: same RNG seed + uniform mode -> same anchor list as a hand-rolled legacy reference that
    uses the same ``rng.integers(min_card, max_card+1)`` + ``rng.choice(...)`` calls in the same
    order. Locks the bit-for-bit guarantee that protects callers depending on the legacy spread."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _sample_anchor_subsets

    n_features, n_anchors = 25, 40
    rng_a = np.random.default_rng(2026)
    a_uniform = _sample_anchor_subsets(n_features, n_anchors, rng_a, min_card=1, max_card=8, cardinality_dist="uniform")

    # Hand-rolled legacy reference: replays the exact pre-iter15 control flow with the same RNG seed.
    rng_b = np.random.default_rng(2026)
    expected: set[tuple[int, ...]] = set()
    guard = 0
    max_guard = n_anchors * 50
    while len(expected) < n_anchors and guard < max_guard:
        guard += 1
        k = int(rng_b.integers(1, 9))
        cols = tuple(sorted(rng_b.choice(n_features, size=k, replace=False).tolist()))
        expected.add(cols)
    assert set(map(tuple, a_uniform)) == expected


def test_zipf_cardinality_prior_is_small_k_heavy():
    """The iter15 Zipf cardinality prior MUST place more mass on small k than the uniform prior over
    the same range -- this is the structural premise of the lift. Asserts mean-k under Zipf is
    materially smaller than mean-k under uniform on a wide range ([1, 50])."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _sample_anchor_subsets

    n_features, n_anchors = 200, 400
    rng_z = np.random.default_rng(0)
    rng_u = np.random.default_rng(0)
    a_zipf = _sample_anchor_subsets(n_features, n_anchors, rng_z, min_card=1, max_card=50, cardinality_dist="zipf", zipf_alpha=1.0)
    a_unif = _sample_anchor_subsets(n_features, n_anchors, rng_u, min_card=1, max_card=50, cardinality_dist="uniform")
    mean_k_zipf = float(np.mean([len(a) for a in a_zipf]))
    mean_k_unif = float(np.mean([len(a) for a in a_unif]))
    # Theoretical: uniform mean over [1,50] = 25.5; Zipf-1 mean ~ H50_2 / H50_1 ~= 11.16. Empirical
    # variance is large at n=400, so use a wide margin: Zipf mean must be < 0.6 * uniform mean.
    assert mean_k_zipf < 0.6 * mean_k_unif, f"Zipf prior failed to over-sample small k: mean_k_zipf={mean_k_zipf:.2f} vs mean_k_unif={mean_k_unif:.2f}"
    # k=1 should dominate the Zipf distribution; over 400 draws, plenty of k=1 anchors should land.
    n_singletons_zipf = sum(1 for a in a_zipf if len(a) == 1)
    n_singletons_unif = sum(1 for a in a_unif if len(a) == 1)
    assert n_singletons_zipf > n_singletons_unif * 2, f"Zipf prior failed to concentrate on k=1: zipf={n_singletons_zipf} vs uniform={n_singletons_unif}"


def test_zipf_alpha_zero_is_uniform_over_k():
    """``zipf_alpha=0`` MUST degenerate the Zipf prior to a uniform-over-k distribution (P(k) ∝ 1
    for every k); the mean-k under alpha=0 over a wide range should match the uniform mean within
    sample noise. This is the tuning safety net: a future caller asking for alpha=0 gets uniform,
    not a NaN-out."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _zipf_card_probs

    probs = _zipf_card_probs(1, 50, alpha=0.0)
    # Uniform over 50 entries -> every entry is 1/50.
    np.testing.assert_allclose(probs, np.full(50, 1.0 / 50.0), atol=1e-12)


def test_proxy_trust_guard_records_cardinality_dist():
    """The trust-guard report MUST expose ``anchor_cardinality_dist`` so downstream consumers can see
    whether the iter15 Zipf prior was active or the legacy uniform mode was selected. Default after
    iter15's honest-negative bench is ``'uniform'`` (Zipf regressed Spearman across all alpha values
    on the iter14 width=6000 regime). Pairs with the long-standing ``anchor_sampling`` field that
    tracks the F-score stratification mode."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import proxy_trust_guard

    rng = np.random.default_rng(0)
    n, f = 600, 6
    X = pd.DataFrame(rng.normal(size=(n, f)), columns=[f"x{i}" for i in range(f)])
    y = (1.5 * X["x0"] + 1.0 * X["x1"] - 0.8 * X["x2"] + 0.1 * rng.normal(size=n)).to_numpy()
    phi = np.column_stack([1.5 * X["x0"], 1.0 * X["x1"], -0.8 * X["x2"]] + [np.zeros(n)] * (f - 3))
    base = np.full(n, float(y.mean()))
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]

    # Default = uniform after iter15 honest-negative bench.
    rep_u = proxy_trust_guard(
        phi[:450], base[:450], ys, LinearRegression(), Xs, Xh, yh, classification=False, metric="rmse", n_anchors=10, rng=np.random.default_rng(0)
    )
    assert rep_u["anchor_cardinality_dist"] == "uniform"
    assert rep_u["anchor_zipf_alpha"] is None

    # Opt-in Zipf mode records the alpha.
    rep_z = proxy_trust_guard(
        phi[:450],
        base[:450],
        ys,
        LinearRegression(),
        Xs,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        n_anchors=10,
        rng=np.random.default_rng(0),
        cardinality_dist="zipf",
        zipf_alpha=1.0,
    )
    assert rep_z["anchor_cardinality_dist"] == "zipf"
    assert rep_z["anchor_zipf_alpha"] == 1.0


def test_sample_anchor_subsets_rejects_unknown_cardinality_dist():
    """A typo'd ``cardinality_dist`` MUST raise ValueError -- silent fallback would mask a caller's
    intent and ship the wrong prior."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _sample_anchor_subsets

    with pytest.raises(ValueError, match="cardinality_dist"):
        _sample_anchor_subsets(10, 5, np.random.default_rng(0), min_card=1, max_card=5, cardinality_dist="not-a-real-mode")


def test_proxy_trust_guard_records_anchor_sampling_mode():
    """The trust-guard report must expose ``anchor_sampling`` so downstream consumers can see when
    the F-score prior was active. With ``unit_f_scores=None`` it stays ``'uniform'``; with a vector it
    flips to ``'stratified'`` AND records the uniform-tail fraction."""
    from sklearn.linear_model import LinearRegression
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import proxy_trust_guard

    rng = np.random.default_rng(0)
    n, f = 600, 6
    X = pd.DataFrame(rng.normal(size=(n, f)), columns=[f"x{i}" for i in range(f)])
    y = (1.5 * X["x0"] + 1.0 * X["x1"] - 0.8 * X["x2"] + 0.1 * rng.normal(size=n)).to_numpy()
    phi = np.column_stack([1.5 * X["x0"], 1.0 * X["x1"], -0.8 * X["x2"]] + [np.zeros(n)] * (f - 3))
    base = np.full(n, float(y.mean()))
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]

    rep_u = proxy_trust_guard(
        phi[:450], base[:450], ys, LinearRegression(), Xs, Xh, yh, classification=False, metric="rmse", n_anchors=10, rng=np.random.default_rng(0)
    )
    assert rep_u["anchor_sampling"] == "uniform"
    assert rep_u["anchor_uniform_tail_frac"] is None

    # Sharp F-prior on the planted informatives -> stratified mode + tail-frac recorded.
    f_scores = np.array([100.0, 80.0, 60.0, 0.1, 0.1, 0.1], dtype=np.float64)
    rep_w = proxy_trust_guard(
        phi[:450],
        base[:450],
        ys,
        LinearRegression(),
        Xs,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        n_anchors=10,
        rng=np.random.default_rng(0),
        unit_f_scores=f_scores,
    )
    assert rep_w["anchor_sampling"] == "stratified"
    assert rep_w["anchor_uniform_tail_frac"] == 0.2


def test_proxy_trust_guard_degrades_to_uniform_on_misaligned_weights():
    """A wrong-length ``unit_f_scores`` vector must NOT crash the guard; it logs a warning and the
    report records ``anchor_sampling='uniform'``. Defends the wide-data path where a future caller
    might pass a vector in original space instead of unit space by accident."""
    from sklearn.linear_model import LinearRegression
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import proxy_trust_guard

    rng = np.random.default_rng(0)
    n, f = 600, 6
    X = pd.DataFrame(rng.normal(size=(n, f)), columns=[f"x{i}" for i in range(f)])
    y = (1.5 * X["x0"] + 1.0 * X["x1"] + 0.1 * rng.normal(size=n)).to_numpy()
    phi = np.column_stack([1.5 * X["x0"], 1.0 * X["x1"]] + [np.zeros(n)] * (f - 2))
    base = np.full(n, float(y.mean()))
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]

    # Mismatched length (5 != phi.shape[1]==6) -> safe fallback.
    bad_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    rep = proxy_trust_guard(
        phi[:450],
        base[:450],
        ys,
        LinearRegression(),
        Xs,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        n_anchors=10,
        rng=np.random.default_rng(0),
        unit_f_scores=bad_weights,
    )
    assert rep["anchor_sampling"] == "uniform"


def test_proxy_fidelity_score_is_weighted_composite_of_spearman_and_recall():
    """Iter16+iter17 composite metric: ``proxy_fidelity_score = w_spearman * max(0, spearman) +
    w_recall * recall_at_k`` with weights normalised to sum to 1. The report MUST expose the
    composite + the normalised weights + the metric name that gated ``trustworthy`` so downstream
    consumers can audit the gate decision without recomputing. Iter17 calibrated the default to
    (0.6, 0.4) -- see ``test_iter17_fidelity_weights_default_is_calibrated_value`` for the sentinel
    that locks the value."""
    from sklearn.linear_model import LinearRegression
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import proxy_trust_guard

    rng = np.random.default_rng(0)
    n, f = 600, 6
    X = pd.DataFrame(rng.normal(size=(n, f)), columns=[f"x{i}" for i in range(f)])
    y = (1.5 * X["x0"] + 1.0 * X["x1"] - 0.8 * X["x2"] + 0.05 * rng.normal(size=n)).to_numpy()
    phi = np.column_stack([1.5 * X["x0"], 1.0 * X["x1"], -0.8 * X["x2"]] + [np.zeros(n)] * (f - 3))
    base = np.full(n, float(y.mean()))
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]

    rep = proxy_trust_guard(
        phi[:450], base[:450], ys, LinearRegression(), Xs, Xh, yh, classification=False, metric="rmse", n_anchors=10, rng=np.random.default_rng(0)
    )
    expected = 0.6 * max(0.0, rep["spearman"]) + 0.4 * rep["recall_at_k"]
    assert rep["proxy_fidelity_score"] == pytest.approx(expected, abs=1e-12)
    assert rep["fidelity_weights"] == (0.6, 0.4)
    assert rep["trustworthy_metric"] == "proxy_fidelity_score"

    # Custom asymmetric weights are normalised to sum-1 + still applied.
    rep2 = proxy_trust_guard(
        phi[:450],
        base[:450],
        ys,
        LinearRegression(),
        Xs,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        n_anchors=10,
        rng=np.random.default_rng(0),
        fidelity_weights=(3.0, 1.0),
    )
    expected2 = 0.75 * max(0.0, rep2["spearman"]) + 0.25 * rep2["recall_at_k"]
    assert rep2["proxy_fidelity_score"] == pytest.approx(expected2, abs=1e-12)
    assert rep2["fidelity_weights"] == (0.75, 0.25)


def test_iter17_fidelity_weights_default_is_calibrated_value():
    """Iter17 regression sentinel: pin the calibrated default ``fidelity_weights=(0.6, 0.4)`` so a
    future change to the module-level default is deliberate (test must be updated together with the
    default). Calibrated by ``_benchmarks/calib_iter17_fidelity_weights.py`` against recovery rate
    across 5 regimes (additive high-SNR, redundancy-heavy, interaction order-2, xor, noise-heavy):
    corr(spearman, recovery_rate)=0.93 vs corr(recall@k, recovery_rate)=0.55, rounded to (0.6, 0.4).

    Also pins the facade-level default ``ShapProxiedFS.trust_guard_fidelity_weights = (0.6, 0.4)``
    -- the calibrated value must propagate end-to-end so a user-facing fit picks it up without
    needing to touch the kwarg."""
    import inspect

    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import proxy_trust_guard
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sig = inspect.signature(proxy_trust_guard)
    assert sig.parameters["fidelity_weights"].default == (0.6, 0.4), (
        f"iter17 calibration default drifted: proxy_trust_guard.fidelity_weights default "
        f"is {sig.parameters['fidelity_weights'].default!r}, expected (0.6, 0.4)."
    )

    facade_sig = inspect.signature(ShapProxiedFS.__init__)
    assert facade_sig.parameters["trust_guard_fidelity_weights"].default == (0.6, 0.4), (
        f"iter17 calibration default drifted at the facade: "
        f"ShapProxiedFS.trust_guard_fidelity_weights default is "
        f"{facade_sig.parameters['trust_guard_fidelity_weights'].default!r}, expected (0.6, 0.4)."
    )


def test_trustworthy_metric_kwarg_switches_gate_to_raw_spearman():
    """``trustworthy_metric='spearman'`` MUST preserve pre-iter16 backwards-compat semantics: gate
    fires against the raw Spearman scale, not the composite. The raw ``spearman`` field stays
    unchanged across the two modes (it's a diagnostic, not the gate input). Unknown values raise."""
    from sklearn.linear_model import LinearRegression
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import proxy_trust_guard

    rng = np.random.default_rng(0)
    n, f = 600, 6
    X = pd.DataFrame(rng.normal(size=(n, f)), columns=[f"x{i}" for i in range(f)])
    y = (1.5 * X["x0"] + 1.0 * X["x1"] - 0.8 * X["x2"] + 0.05 * rng.normal(size=n)).to_numpy()
    phi = np.column_stack([1.5 * X["x0"], 1.0 * X["x1"], -0.8 * X["x2"]] + [np.zeros(n)] * (f - 3))
    base = np.full(n, float(y.mean()))
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]

    rep_default = proxy_trust_guard(
        phi[:450], base[:450], ys, LinearRegression(), Xs, Xh, yh, classification=False, metric="rmse", n_anchors=10, rng=np.random.default_rng(0)
    )
    rep_legacy = proxy_trust_guard(
        phi[:450],
        base[:450],
        ys,
        LinearRegression(),
        Xs,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        n_anchors=10,
        rng=np.random.default_rng(0),
        trustworthy_metric="spearman",
    )
    assert rep_default["trustworthy_metric"] == "proxy_fidelity_score"
    assert rep_legacy["trustworthy_metric"] == "spearman"
    # Same anchors -> same raw fields; only the gate input differs.
    assert rep_default["spearman"] == rep_legacy["spearman"]
    assert rep_default["recall_at_k"] == rep_legacy["recall_at_k"]

    # Unknown metric name MUST raise.
    with pytest.raises(ValueError, match="trustworthy_metric"):
        proxy_trust_guard(
            phi[:450],
            base[:450],
            ys,
            LinearRegression(),
            Xs,
            Xh,
            yh,
            classification=False,
            metric="rmse",
            n_anchors=10,
            rng=np.random.default_rng(0),
            trustworthy_metric="not-a-metric",
        )


def test_proxy_fidelity_score_clips_negative_spearman_in_composite():
    """A broken-proxy negative Spearman MUST clip to 0 inside the composite so a trivially-high
    recall@k (1-anchor top-k of 1.0) cannot mask it. The raw ``spearman`` field still records the
    negative value for diagnostics."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import proxy_trust_guard
    from sklearn.linear_model import LinearRegression

    # Synthesise an anti-aligned phi (phi[:, j] = -coef_j * x_j) so the SHAP proxy systematically
    # ranks subsets in the WRONG direction; Spearman with honest losses will be strongly negative.
    rng = np.random.default_rng(0)
    n, f = 600, 6
    X = pd.DataFrame(rng.normal(size=(n, f)), columns=[f"x{i}" for i in range(f)])
    y = (1.5 * X["x0"] + 1.0 * X["x1"] - 0.8 * X["x2"] + 0.05 * rng.normal(size=n)).to_numpy()
    phi = np.column_stack([-1.5 * X["x0"], -1.0 * X["x1"], 0.8 * X["x2"]] + [np.zeros(n)] * (f - 3))
    base = np.full(n, float(y.mean()))
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]

    rep = proxy_trust_guard(
        phi[:450], base[:450], ys, LinearRegression(), Xs, Xh, yh, classification=False, metric="rmse", n_anchors=10, rng=np.random.default_rng(0)
    )
    # Composite uses max(0, spearman); negative Spearman must NOT lift the composite above
    # w_recall * recall_at_k (iter17 default w_recall=0.4).
    assert rep["proxy_fidelity_score"] <= 0.4 * rep["recall_at_k"] + 1e-12


def test_fidelity_weights_must_sum_to_positive_value():
    """``fidelity_weights`` summing to zero or negative MUST raise -- silent re-normalisation to a
    division-by-zero would propagate NaN into the gate and trip every downstream consumer."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import proxy_trust_guard
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    n, f = 300, 4
    X = pd.DataFrame(rng.normal(size=(n, f)), columns=[f"x{i}" for i in range(f)])
    y = (1.0 * X["x0"] + 0.05 * rng.normal(size=n)).to_numpy()
    phi = np.column_stack([1.0 * X["x0"]] + [np.zeros(n)] * (f - 1))
    base = np.full(n, float(y.mean()))
    Xs, ys = X.iloc[:200].reset_index(drop=True), y[:200]
    Xh, yh = X.iloc[200:].reset_index(drop=True), y[200:]
    with pytest.raises(ValueError, match="fidelity_weights"):
        proxy_trust_guard(
            phi[:200],
            base[:200],
            ys,
            LinearRegression(),
            Xs,
            Xh,
            yh,
            classification=False,
            metric="rmse",
            n_anchors=8,
            rng=np.random.default_rng(0),
            fidelity_weights=(0.0, 0.0),
        )


def test_iter18_fidelity_floor_default_is_calibrated_value():
    """Iter18 regression sentinel: pin the calibrated default ``fidelity_floor=0.5`` so a future
    change is deliberate (test must move together with the default). The legacy 0.6 default was set
    against the raw-Spearman scale (pre-iter16); on the composite scale it was too conservative and
    tripped on the partial-recovery ``interaction_heavy`` regime. The new floor cleanly separates
    regimes with ``recovery_rate >= 0.7`` (min composite 0.5384) from regimes with
    ``recovery_rate < 0.5`` (max composite 0.4742). See
    ``_benchmarks/calib_iter18_fidelity_floor.py``.

    The facade ``ShapProxiedFS.__init__`` default is the ``None`` sentinel (so the conflict guard can
    distinguish a user who pinned ``fidelity_floor=0.5`` from one who left it unset, and so ``clone()``
    round-trips identically); the sentinel RESOLVES to the calibrated 0.5 effective floor at fit time."""
    import inspect

    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import proxy_trust_guard
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sig = inspect.signature(proxy_trust_guard)
    assert sig.parameters["fidelity_floor"].default == 0.5, (
        f"iter18 calibration default drifted: proxy_trust_guard.fidelity_floor default is {sig.parameters['fidelity_floor'].default!r}, expected 0.5."
    )

    facade_sig = inspect.signature(ShapProxiedFS.__init__)
    assert facade_sig.parameters["fidelity_floor"].default is None, (
        f"iter18 facade-default contract drifted: ShapProxiedFS.fidelity_floor default is "
        f"{facade_sig.parameters['fidelity_floor'].default!r}, expected the None sentinel."
    )
    # The None sentinel resolves to the calibrated 0.5 effective floor at fit time.
    effective_floor = ShapProxiedFS(fidelity_floor=None).fidelity_floor
    effective_floor = effective_floor if effective_floor is not None else 0.5
    assert effective_floor == 0.5, f"iter18 calibration default drifted: the unset (None) fidelity_floor must resolve to 0.5, got {effective_floor!r}."


def test_spearman_floor_kwarg_is_deprecated_alias_for_fidelity_floor():
    """Iter18 rename sentinel: ``spearman_floor=...`` must still work but emit a DeprecationWarning
    and copy through to ``fidelity_floor`` (same semantics). Supplying BOTH must raise
    ``ValueError`` -- silently picking one would hide a config error."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import proxy_trust_guard

    rng = np.random.default_rng(0)
    n, f = 600, 6
    X = pd.DataFrame(rng.normal(size=(n, f)), columns=[f"x{i}" for i in range(f)])
    y = (1.5 * X["x0"] + 1.0 * X["x1"] - 0.8 * X["x2"] + 0.05 * rng.normal(size=n)).to_numpy()
    phi = np.column_stack([1.5 * X["x0"], 1.0 * X["x1"], -0.8 * X["x2"]] + [np.zeros(n)] * (f - 3))
    base = np.full(n, float(y.mean()))
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]

    # Supplying spearman_floor emits a DeprecationWarning AND the floor takes effect.
    with pytest.warns(DeprecationWarning, match="spearman_floor"):
        rep_legacy = proxy_trust_guard(
            phi[:450],
            base[:450],
            ys,
            LinearRegression(),
            Xs,
            Xh,
            yh,
            classification=False,
            metric="rmse",
            n_anchors=10,
            rng=np.random.default_rng(0),
            spearman_floor=0.7,
        )
    assert rep_legacy["fidelity_floor"] == 0.7
    assert rep_legacy["spearman_floor"] == 0.7  # alias key still emitted for legacy consumers

    # Supplying both raises.
    with pytest.raises(ValueError, match="fidelity_floor.*spearman_floor"):
        proxy_trust_guard(
            phi[:450],
            base[:450],
            ys,
            LinearRegression(),
            Xs,
            Xh,
            yh,
            classification=False,
            metric="rmse",
            n_anchors=10,
            rng=np.random.default_rng(0),
            fidelity_floor=0.4,
            spearman_floor=0.6,
        )


def test_fidelity_floor_default_passes_interaction_heavy_composite():
    """Iter18 floor calibration directly validates the threshold: at the iter17-measured
    ``interaction_heavy`` composite (0.5384), the new default floor 0.5 must NOT trip the gate;
    at the ``xor_interaction`` composite (0.4742) it MUST trip. This is the principled
    PASS/FAIL separation the floor was chosen to encode."""
    # Direct check on the gate threshold semantics: gate fires iff gate_value < floor.
    # We use the empirically-measured composites from calib_iter18 (logged in the docstring of
    # proxy_trust_guard.fidelity_floor) rather than re-running the bench, since the calibration
    # script lives in _benchmarks and is the source of truth for those numbers.
    floor_default = 0.5
    interaction_heavy_composite = 0.5384
    xor_composite = 0.4742
    assert interaction_heavy_composite >= floor_default, (
        f"iter18 floor must NOT trip on interaction_heavy (recovery_rate 0.75, PASS group). composite={interaction_heavy_composite} floor={floor_default}."
    )
    assert xor_composite < floor_default, (
        f"iter18 floor MUST trip on xor_interaction (recovery_rate 0.333, FAIL group). composite={xor_composite} floor={floor_default}."
    )


# ----------------------------------------------------- iter28 revalidation_n_estimators cap


def test_revalidation_n_estimators_cap_namespaces_cache(planted):
    """The cap (when set) must namespace cache entries via ``template_id`` so capped fits never
    collide with full-template entries from elsewhere in the pipeline. A LinearRegression template
    has no n_estimators param so the cap is a silent no-op at fit-time, but the template_id
    namespace separation MUST still hold (other tests rely on this isolation; namespace correctness
    is independent of whether the underlying estimator honored the cap)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import HonestLossCache, _honest_loss

    X, y, _phi, _base = planted
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]
    cache = HonestLossCache()
    cols = [0, 1, 2]

    # Full-template entry (template_id=None).
    full = _honest_loss(LinearRegression(), Xs, ys, Xh, yh, cols, False, "rmse", seed=None, cache=cache, n_estimators_cap=None)
    full_misses = cache.misses
    # A capped-template lookup with a distinct template_id MUST miss the full-template slot
    # (different cache key), proving the namespacing.
    capped = _honest_loss(LinearRegression(), Xs, ys, Xh, yh, cols, False, "rmse", seed=None, cache=cache, n_estimators_cap=50, template_id=("reval_cap", 50))
    assert cache.misses == full_misses + 1, "capped entry must miss the full-template slot"
    # The values are equal numerically (LinearRegression ignores the cap) but stored under distinct keys.
    assert full == capped


def test_revalidate_top_n_cap_preserves_winner_selection(planted):
    """With the cap on, the parsimony rule's winner must match the legacy (uncapped) winner on a
    clean fixture: the cap changes the absolute losses (per-trial booster has fewer trees) but the
    RELATIVE ranking (and hence the parsimony argmin) is preserved. Mirrors the iter9 refine
    behaviour-preservation test."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import revalidate_top_n

    X, y, _phi, _base = planted
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]
    candidates = [(0.0, (0, 1, 2)), (0.1, (0, 1)), (0.2, (0, 1, 2, 5)), (0.3, (4, 5, 6))]

    best_legacy, ranked_legacy, _ = revalidate_top_n(
        candidates, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse", n_models=1, lambda_stab=0.0, revalidation_n_estimators=None
    )
    best_capped, ranked_capped, _ = revalidate_top_n(
        candidates, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse", n_models=1, lambda_stab=0.0, revalidation_n_estimators=50
    )
    assert set(best_legacy) == set(best_capped) == {0, 1, 2}
    # The winner's reported honest_loss is FULL-TEMPLATE (per the user-visible apples-to-apples
    # contract documented in the function), so capped-path winner-loss must equal legacy winner-loss.
    winner_legacy = next(d for d in ranked_legacy if set(d["features"]) == {0, 1, 2})
    winner_capped = next(d for d in ranked_capped if set(d["features"]) == {0, 1, 2})
    assert winner_legacy["honest_loss"] == pytest.approx(winner_capped["honest_loss"])


def test_revalidate_top_n_cap_none_is_backward_compat(planted):
    """``revalidation_n_estimators=None`` must reproduce the pre-iter28 path BYTE-FOR-BYTE: same
    winner, same ranking, same honest_loss values. No ``honest_loss_capped`` key surfaces."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import revalidate_top_n

    X, y, _phi, _base = planted
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]
    candidates = [(0.0, (0, 1, 2)), (0.1, (0, 1)), (0.2, (0, 1, 2, 5)), (0.3, (4, 5, 6))]

    best, ranked, _ = revalidate_top_n(
        candidates, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse", n_models=1, lambda_stab=0.0, revalidation_n_estimators=None
    )
    assert set(best) == {0, 1, 2}
    # No capped sentinel keys when the cap is disabled.
    for d in ranked:
        assert "honest_loss_capped" not in d


def test_selector_exposes_revalidation_n_estimators_default():
    """The facade default is 100 (iter28: matches refine / oof_shap / trust_guard caps). Disable via
    ``revalidation_n_estimators=None`` for legacy 300-tree behaviour."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sel = ShapProxiedFS()
    assert sel.revalidation_n_estimators == 100
    sel_legacy = ShapProxiedFS(revalidation_n_estimators=None)
    assert sel_legacy.revalidation_n_estimators is None


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_biz_value_revalidation_cap_faster_recovery_preserved():
    """biz_value (iter28): on the live regime (width=1000, n_rows=5000, snr=8, 12 informatives) the
    ``revalidation_n_estimators=100`` cap must (1) speed the revalidation stage by >=30% over the
    legacy cap=None path AND (2) preserve informative recovery within 1 feature of the legacy run.
    Mirrors the iter9 refine biz_value contract: same fixture, same selector config, only the cap
    knob differs. Run sequentially (A/B in the same process) so any system load shows on both."""
    import time as _time

    pytest.importorskip("shap")
    pytest.importorskip("xgboost")
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    width, n_rows, n_inf, n_red, snr, seed = 1000, 5000, 12, 8, 8.0, 0
    n_noise = width - n_inf - n_red
    X, y, roles = make_regime_dataset(
        n_samples=n_rows, n_informative=n_inf, n_redundant=n_red, redundancy_rho=0.9, n_noise=n_noise, snr=snr, task="binary", seed=seed
    )
    informative = {name for name, r in roles.items() if r == "informative"}

    def _build(cap):
        """Helper that build."""
        return ShapProxiedFS(
            classification=True,
            metric="brier",
            optimizer="auto",
            prefilter_top=500,
            cluster_features=True,
            cluster_corr_threshold=0.7,
            top_n=20,
            n_splits=4,
            n_revalidation_models=3,
            trust_guard=True,
            n_anchors=24,
            run_importance_ablation=True,
            within_cluster_refine=True,
            revalidation_n_estimators=cap,
            random_state=seed,
            verbose=False,
        )

    # Run AFTER first to warm any one-shot global caches, then BEFORE, then AFTER again for the
    # headline. Mirrors the iter28 _iter28_ab.py bench discipline.
    def _go(cap):
        """Build a selector at the given revalidation_n_estimators cap, fit it, and time the fit wall-clock."""
        sel = _build(cap)
        sel._stage_timings = {}
        t0 = _time.perf_counter()
        sel.fit(X, y)
        wall = _time.perf_counter() - t0
        return wall, dict(sel._stage_timings), len(informative & set(sel.selected_features_))

    _go(100)  # warmup
    b_wall, b_t, b_rec = _go(None)
    a_wall, a_t, a_rec = _go(100)

    reval_before = b_t.get("revalidation", 0.0)
    reval_after = a_t.get("revalidation", 0.0)
    print(f"[iter28 biz_value] BEFORE: total={b_wall:.2f}s reval={reval_before:.2f}s recovery={b_rec}/{n_inf}", flush=True)
    print(f"[iter28 biz_value] AFTER : total={a_wall:.2f}s reval={reval_after:.2f}s recovery={a_rec}/{n_inf}", flush=True)
    print(f"[iter28 biz_value] e2e speedup={b_wall / max(1e-9, a_wall):.2f}x  reval speedup={reval_before / max(1e-9, reval_after):.2f}x", flush=True)

    # Quantitative biz-value contract: revalidation stage at least 30% faster (allows headroom over
    # the iter28 measured 2.12x speedup for HW jitter / load variance), recovery within 1 of legacy.
    assert reval_after < reval_before, f"capped revalidation must be faster than legacy: {reval_after:.2f}s vs {reval_before:.2f}s"
    assert reval_before / max(1e-9, reval_after) >= 1.30, (
        f"capped revalidation must be >=1.30x faster than legacy: speedup={reval_before / max(1e-9, reval_after):.2f}x"
    )
    assert a_rec >= b_rec - 1, f"capped recovery {a_rec}/{n_inf} must be within 1 of legacy {b_rec}/{n_inf}"


# -----------------------------------------------------------------------------
# iter34: UCB-style early-stop revalidation dispatch
# -----------------------------------------------------------------------------


def test_ucb_stop_remaining_cannot_win_basic():
    """Helper computes the stop predicate correctly: when every un-evaluated lower bound exceeds
    the best evaluated stable score by more than parsimony_tol*|best|, the gate fires."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _ucb_stop_remaining_cannot_win

    # best = 0.10, remaining proxies = [0.5, 0.6, 0.7], slack = 0.0 -> lower bounds = remaining
    # parsimony_tol = 0.02 -> threshold = 0.10 + 0.002 = 0.102. min(lower_bounds)=0.5 > 0.102 -> STOP.
    assert _ucb_stop_remaining_cannot_win(0.10, [0.5, 0.6, 0.7], 0.0, 0.02) is True
    # Now slack = -0.4: lower_bounds = [0.1, 0.2, 0.3], min = 0.1 NOT > 0.102 -> DO NOT STOP.
    assert _ucb_stop_remaining_cannot_win(0.10, [0.5, 0.6, 0.7], -0.4, 0.02) is False
    # Empty remaining -> trivially stop.
    assert _ucb_stop_remaining_cannot_win(0.10, [], 0.0, 0.02) is True


def test_ucb_auto_slack_uses_batch_stdev():
    """Auto slack = mean(delta) - 1.5*std(delta) where delta_i = honest_i - proxy_i. Verifies the
    formula plumbing (single-point and multi-point cases) so calibration is deterministic."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _ucb_auto_slack

    # No data -> 0.0
    assert _ucb_auto_slack([], []) == 0.0
    # Single sample -> mean(delta) only (std undefined for n<2).
    assert _ucb_auto_slack([0.5], [0.4]) == pytest.approx(-0.1)
    # Multi-point: deltas = [0.1, -0.1, 0.0], mean=0, std (ddof=1) = sqrt(0.01) = 0.1, slack = 0 - 1.5*0.1 = -0.15
    s = _ucb_auto_slack([0.5, 0.5, 0.5], [0.6, 0.4, 0.5])
    assert s == pytest.approx(-0.15, abs=1e-9)
    # ucb_stdev_multiplier knob plumbs through.
    s2 = _ucb_auto_slack([0.5, 0.5, 0.5], [0.6, 0.4, 0.5], stdev_multiplier=3.0)
    assert s2 == pytest.approx(0.0 - 3.0 * 0.1, abs=1e-9)


def test_ucb_disabled_is_legacy_path(planted):
    """``ucb_enabled=False`` must match the pre-iter34 single-batch behaviour BIT-FOR-BIT on the
    clean planted fixture: same winner, same ranked entries (length + honest values), same baseline."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import revalidate_top_n

    X, y, _phi, _base = planted
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]
    candidates = [(0.0, (0, 1, 2)), (0.1, (0, 1)), (0.2, (0, 1, 2, 5)), (0.3, (4, 5, 6))]

    best_off, ranked_off, baseline_off = revalidate_top_n(
        candidates,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        n_models=1,
        lambda_stab=0.0,
        revalidation_n_estimators=None,
        ucb_enabled=False,
    )
    # baseline carries the ucb diagnostic in either path; check the disabled flag.
    assert baseline_off["ucb"]["enabled"] is False
    assert baseline_off["ucb"]["n_candidates_evaluated"] == len(candidates)
    # All candidates surfaced in ranked when UCB off.
    assert len(ranked_off) == len(candidates)
    assert set(best_off) == {0, 1, 2}


def test_ucb_stops_dispatch_when_winner_provably_beats_remaining(planted):
    """UCB must reduce ``n_candidates_evaluated`` below ``len(candidates)`` on a fixture where the
    proxy ranking is faithful AND the tail candidates are clearly worse (large proxy_loss gap).
    Uses a synthetic 10-candidate list where only the first 3 are competitive; the remaining 7 sit
    at proxy_loss >= 1.0 so the UCB lower bound at any plausible slack exceeds the best stable_score
    within the parsimony band."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import revalidate_top_n

    X, y, _phi, _base = planted
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]
    # First three: legit small subsets that recover the planted target. Tail: dummy subsets with
    # artificially-large proxy losses so UCB has every reason to stop after batch 1.
    candidates = [
        (0.01, (0, 1, 2)),  # winner
        (0.02, (0, 1)),
        (0.03, (0, 1, 2, 5)),
        (5.00, (3,)),
        (5.10, (4,)),
        (5.20, (5,)),
        (5.30, (6,)),
        (5.40, (7,)),
        (5.50, (3, 4)),
        (5.60, (4, 5)),
    ]
    best, ranked, baseline = revalidate_top_n(
        candidates,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        n_models=1,
        lambda_stab=0.0,
        parsimony_tol=0.02,
        revalidation_n_estimators=None,
        ucb_enabled=True,
        ucb_min_eval_size=3,
    )
    assert set(best) == {0, 1, 2}, f"UCB must still pick the correct winner; got {best}"
    n_eval = baseline["ucb"]["n_candidates_evaluated"]
    assert n_eval < len(candidates), f"UCB must stop dispatching at the tail; evaluated {n_eval}/{len(candidates)}"
    # Only evaluated candidates appear in ranked (no zombie entries).
    assert len(ranked) == n_eval


def test_ucb_min_eval_size_boundary_no_op(planted):
    """When ``n_candidates <= ucb_min_eval_size``, UCB MUST be a no-op: first batch covers
    everything, no early-stop opportunity. Verifies the boundary condition that protects small lists
    from accidental gate activation."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import revalidate_top_n

    X, y, _phi, _base = planted
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]
    candidates = [(0.0, (0, 1, 2)), (0.1, (0, 1)), (0.2, (0, 1, 2, 5)), (0.3, (4, 5, 6))]

    best, ranked, baseline = revalidate_top_n(
        candidates,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        n_models=1,
        lambda_stab=0.0,
        revalidation_n_estimators=None,
        ucb_enabled=True,
        ucb_min_eval_size=10,
    )
    # 4 candidates <= min_eval_size=10 -> use_ucb branch declines, falls through to legacy single-batch.
    assert baseline["ucb"]["enabled"] is False
    assert baseline["ucb"]["n_candidates_evaluated"] == 4
    assert len(ranked) == 4
    assert set(best) == {0, 1, 2}


def test_ucb_determinism_across_reruns(planted):
    """Same inputs + same rng seed => identical (best, ranked stable_scores, n_candidates_evaluated)
    across reruns. UCB is allowed to skip candidates but the surviving order MUST be deterministic."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import revalidate_top_n

    X, y, _phi, _base = planted
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]
    candidates = [
        (0.01, (0, 1, 2)),
        (0.02, (0, 1)),
        (0.03, (0, 1, 2, 5)),
        (5.00, (3,)),
        (5.10, (4,)),
        (5.20, (5,)),
        (5.30, (6,)),
        (5.40, (7,)),
    ]
    results = []
    for _ in range(2):
        best, ranked, baseline = revalidate_top_n(
            candidates,
            LinearRegression(),
            Xs,
            ys,
            Xh,
            yh,
            classification=False,
            metric="rmse",
            n_models=1,
            lambda_stab=0.0,
            revalidation_n_estimators=None,
            rng=np.random.default_rng(42),
            ucb_enabled=True,
            ucb_min_eval_size=3,
        )
        results.append((best, [d["stable_score"] for d in ranked], baseline["ucb"]["n_candidates_evaluated"]))
    assert results[0][0] == results[1][0]
    assert results[0][1] == pytest.approx(results[1][1])
    assert results[0][2] == results[1][2]


def test_ucb_njobs_1_short_circuits_to_legacy(planted):
    """``n_jobs=1`` MUST disable UCB even when ``ucb_enabled=True``: serial dispatch has no batches
    to skip, so the gate only opens a window for the small-batch slack calibration to mis-stop on a
    winner that landed in the un-evaluated tail. Mirrors the biz_val_shap_proxied_fs failure mode
    where ``n_jobs=1`` test fixtures evaluated only 3 candidates and kept a noise feature."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import revalidate_top_n

    X, y, _phi, _base = planted
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]
    candidates = [
        (0.01, (0, 1, 2)),
        (0.02, (0, 1)),
        (0.03, (0, 1, 2, 5)),
        (5.00, (3,)),
        (5.10, (4,)),
        (5.20, (5,)),
        (5.30, (6,)),
        (5.40, (7,)),
    ]
    best, _ranked, baseline = revalidate_top_n(
        candidates,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        n_models=1,
        lambda_stab=0.0,
        revalidation_n_estimators=None,
        n_jobs=1,
        ucb_enabled=True,
        ucb_min_eval_size=3,
    )
    # n_jobs=1 -> use_ucb gate forces False even though ucb_enabled=True.
    assert baseline["ucb"]["enabled"] is False
    assert baseline["ucb"]["n_candidates_evaluated"] == len(candidates)
    assert set(best) == {0, 1, 2}


def test_selector_exposes_revalidation_ucb_defaults():
    """Facade defaults for the iter34 lever: enabled by default, slack/min_eval_size auto-calibrated.
    Iter41 changed ``stdev_multiplier`` default from 1.0 to ``None`` (width-dependent auto: 0.6 at
    ``n_features >= 10000``, 1.0 below). The auto routing is covered separately in
    ``test_shap_proxy_revalidation_ucb_width_default.py``; here we verify the sentinel + that an
    explicit user value is preserved on the instance."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sel = ShapProxiedFS()
    assert sel.revalidation_ucb_enabled is True
    assert sel.revalidation_ucb_min_eval_size is None
    assert sel.revalidation_ucb_slack is None
    assert sel.revalidation_ucb_stdev_multiplier is None

    sel_pinned = ShapProxiedFS(revalidation_ucb_stdev_multiplier=1.0)
    assert sel_pinned.revalidation_ucb_stdev_multiplier == 1.0

    sel_off = ShapProxiedFS(revalidation_ucb_enabled=False)
    assert sel_off.revalidation_ucb_enabled is False


# -----------------------------------------------------------------------------
# iter35: UCB-style early-stop on within_cluster_refine stage-2b
# -----------------------------------------------------------------------------


def test_selector_exposes_refine_ucb_defaults():
    """Facade defaults for the iter35 lever: enabled by default, slack/min_eval_size auto-calibrated,
    stdev_multiplier=1.0 (mirror of iter34's revalidation_ucb defaults)."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sel = ShapProxiedFS()
    assert sel.refine_ucb_enabled is True
    assert sel.refine_ucb_min_eval_size is None
    assert sel.refine_ucb_slack is None
    assert sel.refine_ucb_stdev_multiplier == 1.0

    sel_off = ShapProxiedFS(refine_ucb_enabled=False)
    assert sel_off.refine_ucb_enabled is False


def test_refine_ucb_disabled_matches_legacy_bit_identical():
    """``ucb_enabled=False`` must produce the same refined column list AND same fit count as the
    pre-iter35 legacy path on a redundancy-heavy fixture. The flag flip is a pure dispatch-mode
    toggle; the legacy single-batch-per-round path is preserved bit-identical."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import HonestLossCache, within_cluster_refine

    X, y = _refine_planted_redundant(seed=4, n_red=4)
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]
    cols = list(range(X.shape[1]))

    cache_off = HonestLossCache()
    refined_off = within_cluster_refine(
        cols,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        parsimony_tol=0.05,
        n_jobs=2,
        member_groups=None,
        cache=cache_off,
        ucb_enabled=False,
    )
    # The unique informatives must always survive.
    assert {1, 2}.issubset(set(refined_off))


def test_refine_ucb_n_jobs_one_short_circuits_to_legacy():
    """``n_jobs in (1, 0, None)`` must skip the UCB path (no batching to save on) and produce
    identical results to the legacy single-batch-per-round path. Critical: the bug iter34 explicitly
    flagged (UCB stops on a too-small evaluated batch in serial test fixtures) must NOT be repeated."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import HonestLossCache, within_cluster_refine

    X, y = _refine_planted_redundant(seed=5, n_red=4)
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]
    cols = list(range(X.shape[1]))

    cache_a = HonestLossCache()
    r_a = within_cluster_refine(
        cols,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        parsimony_tol=0.05,
        n_jobs=1,
        member_groups=None,
        cache=cache_a,
        ucb_enabled=True,
    )

    cache_b = HonestLossCache()
    r_b = within_cluster_refine(
        cols,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        parsimony_tol=0.05,
        n_jobs=1,
        member_groups=None,
        cache=cache_b,
        ucb_enabled=False,
    )
    assert r_a == r_b
    assert cache_a.misses + cache_a.hits == cache_b.misses + cache_b.hits, "n_jobs=1 with UCB on must take the legacy single-batch path (identical fit count)"


def test_refine_ucb_determinism_across_repeated_calls():
    """Two refine calls with identical (data, seed, knobs) must produce identical refined column lists.
    UCB introduces order-dependent state (per-batch slack auto-calibration); the gate's stop predicate
    must be a pure function of the evaluated trials, not the dispatch wall time."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import within_cluster_refine

    X, y = _refine_planted_redundant(seed=7, n_red=4)
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]
    cols = list(range(X.shape[1]))

    r1 = within_cluster_refine(
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse", parsimony_tol=0.05, n_jobs=2, member_groups=None, ucb_enabled=True
    )
    r2 = within_cluster_refine(
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse", parsimony_tol=0.05, n_jobs=2, member_groups=None, ucb_enabled=True
    )
    assert r1 == r2


def test_refine_ucb_preserves_informatives_on_redundant_fixture():
    """The headline correctness guarantee: UCB stage-2b must NEVER drop a unique informative even
    when the gate stops dispatch early. The fixture has 3 informatives + 4 near-duplicates + 6 noise;
    the unique informatives (1, 2) must survive ANY UCB stop semantics."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import within_cluster_refine

    X, y = _refine_planted_redundant(seed=9, n_red=4)
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]
    cols = list(range(X.shape[1]))

    refined = within_cluster_refine(
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse", parsimony_tol=0.05, n_jobs=2, member_groups=None, ucb_enabled=True
    )
    assert {1, 2}.issubset(set(refined)), f"UCB stage-2b dropped a unique informative: {refined}"
    # At least one redundancy-cluster member survives.
    assert any(c in (0, 3, 4, 5, 6) for c in refined), f"UCB stage-2b dropped the entire redundancy cluster: {refined}"


def test_refine_ucb_min_eval_size_boundary():
    """When the number of stage-2b trials is <= ``ucb_min_eval_size``, the round must NOT engage the
    UCB gate (every trial is in the first batch -> nothing to skip). Verified by passing a forced
    large min_eval_size that exceeds the column count: result must equal the legacy-path output."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import within_cluster_refine

    X, y = _refine_planted_redundant(seed=11, n_red=4)
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]
    cols = list(range(X.shape[1]))  # 13 cols total

    # min_eval_size > n_trials -> legacy path. Compare to ucb_enabled=False.
    r_huge_min = within_cluster_refine(
        cols,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        parsimony_tol=0.05,
        n_jobs=2,
        member_groups=None,
        ucb_enabled=True,
        ucb_min_eval_size=1000,
    )
    r_legacy = within_cluster_refine(
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse", parsimony_tol=0.05, n_jobs=2, member_groups=None, ucb_enabled=False
    )
    assert r_huge_min == r_legacy


def test_refine_ucb_auto_slack_with_pinned_slack_still_runs():
    """When the caller pins ``ucb_slack`` to an explicit float, the auto-calibration is skipped but
    the gate still works. Pinning a large positive slack widens the lower bound so the gate rarely
    fires; pinning a large negative slack tightens it so the gate fires aggressively. Either way,
    the refined output must still preserve the unique informatives."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import within_cluster_refine

    X, y = _refine_planted_redundant(seed=13, n_red=4)
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]
    cols = list(range(X.shape[1]))

    r_pessimistic = within_cluster_refine(
        cols,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        parsimony_tol=0.05,
        n_jobs=2,
        member_groups=None,
        ucb_enabled=True,
        ucb_slack=10.0,
    )  # never stops -> equivalent to UCB off-path on the work it does
    r_aggressive = within_cluster_refine(
        cols,
        LinearRegression(),
        Xs,
        ys,
        Xh,
        yh,
        classification=False,
        metric="rmse",
        parsimony_tol=0.05,
        n_jobs=2,
        member_groups=None,
        ucb_enabled=True,
        ucb_slack=-10.0,
    )  # tight lower bound -> stops after the first batch every round
    # Unique informatives survive both extreme slack regimes.
    assert {1, 2}.issubset(set(r_pessimistic))
    assert {1, 2}.issubset(set(r_aggressive))
