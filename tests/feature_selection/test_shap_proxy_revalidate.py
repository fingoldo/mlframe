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
    from mlframe.feature_selection._shap_proxy_revalidate import revalidate_top_n

    X, y, phi, base = planted
    idx_search = np.arange(900)
    idx_hold = np.arange(900, 1200)
    Xs, ys = X.iloc[idx_search].reset_index(drop=True), y[idx_search]
    Xh, yh = X.iloc[idx_hold].reset_index(drop=True), y[idx_hold]

    candidates = [(0.0, (0, 1, 2)), (0.1, (0, 1)), (0.2, (0, 1, 2, 5)), (0.3, (4, 5, 6))]
    best, ranked, baseline = revalidate_top_n(
        candidates, LinearRegression(), Xs, ys, Xh, yh,
        classification=False, metric="rmse", n_models=1, lambda_stab=0.0)
    assert set(best) == {0, 1, 2}
    assert ranked[0]["honest_loss"] < ranked[-1]["honest_loss"]
    assert baseline["honest_loss"] > ranked[0]["honest_loss"]  # the chosen subset beats a random one


def test_trust_guard_high_fidelity_on_clean_proxy(planted):
    from mlframe.feature_selection._shap_proxy_revalidate import proxy_trust_guard

    X, y, phi, base = planted
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]
    # phi was computed on full X; slice to search rows to match ys
    rep = proxy_trust_guard(phi[:900], base[:900], ys, LinearRegression(), Xs, Xh, yh,
                            classification=False, metric="rmse", n_anchors=25,
                            rng=np.random.default_rng(0))
    assert rep["n_anchors"] >= 10
    assert rep["spearman"] > 0.5  # clean linear proxy -> good fidelity
    assert rep["trustworthy"]


def test_active_learning_respects_budget_and_not_worse_than_proxy_top1(planted):
    from mlframe.feature_selection._shap_proxy_revalidate import _honest_loss, active_learning_revalidate

    X, y, phi, base = planted
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]

    # Candidate pool: proxy-best is the true {0,1,2}; several decoys with lower proxy_loss-ranks.
    candidates = [(0.05, (0, 1, 2)), (0.10, (0, 1)), (0.20, (0, 1, 2, 5)),
                  (0.30, (4, 5, 6)), (0.40, (3, 4)), (0.50, (5, 6, 7))]
    # Enough anchors for the corrector to fit (proxy ~ honest with a redundancy wobble).
    rng = np.random.default_rng(0)
    cd = dict(proxy=list(rng.uniform(0.1, 0.6, 16)), honest=list(rng.uniform(0.1, 0.6, 16)),
              cards=list(rng.integers(2, 5, 16).astype(float)), redund=list(rng.uniform(0, 1, 16)))

    budget = 4
    best_idx, ranked, n_eval = active_learning_revalidate(
        candidates, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse",
        corrector_data=cd, phi=phi, budget=budget, batch=2, n_models=1, rng=np.random.default_rng(1))

    assert n_eval <= budget and n_eval >= 1
    assert len(ranked) == n_eval
    # The proxy top-1 is always evaluated early, so the active best is never worse than it.
    proxy_top1_honest = _honest_loss(LinearRegression(), Xs, ys, Xh, yh, [0, 1, 2], False, "rmse")
    best_honest = min(d["honest_loss"] for d in ranked)
    assert best_honest <= proxy_top1_honest + 1e-9
    assert set(best_idx) <= {0, 1, 2, 3, 4, 5, 6, 7}


def test_importance_ablation_runs(planted):
    from mlframe.feature_selection._shap_proxy_revalidate import importance_topk_ablation

    X, y, phi, base = planted
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]
    out = importance_topk_ablation(phi[:900], (0, 1, 2), LinearRegression(), Xs, ys, Xh, yh,
                                   classification=False, metric="rmse")
    assert out["proxy_features"] == (0, 1, 2)
    assert "proxy_honest_loss" in out and "importance_honest_loss" in out
    assert isinstance(out["proxy_wins"], bool)


# --------------------------------------------------------------------- HonestLossCache (dedup)


def test_honest_loss_cache_returns_identical_to_uncached(planted):
    """A cached retrain must equal a fresh fit on the SAME (subset, seed): same model, same data,
    same seed is deterministic, so the cached float is numerically identical."""
    from mlframe.feature_selection._shap_proxy_revalidate import HonestLossCache, _honest_loss

    X, y, phi, base = planted
    Xs, ys = X.iloc[:900].reset_index(drop=True), y[:900]
    Xh, yh = X.iloc[900:].reset_index(drop=True), y[900:]
    cols = [0, 1, 2, 5]

    uncached = _honest_loss(LinearRegression(), Xs, ys, Xh, yh, cols, False, "rmse", seed=7)
    cache = HonestLossCache()
    first = _honest_loss(LinearRegression(), Xs, ys, Xh, yh, cols, False, "rmse", seed=7, cache=cache)
    second = _honest_loss(LinearRegression(), Xs, ys, Xh, yh, cols, False, "rmse", seed=7, cache=cache)
    assert first == uncached  # first call (miss) matches the uncached fit
    assert second == first    # second call (hit) returns the byte-identical cached value
    assert cache.misses == 1 and cache.hits == 1


def test_honest_loss_cache_key_is_order_independent_and_seed_separated(planted):
    """Column permutations of one subset share a cache slot (order-independent key); the SAME subset
    with a DIFFERENT seed is a distinct slot (so seed-jittered re-validation fits are never merged)."""
    from mlframe.feature_selection._shap_proxy_revalidate import HonestLossCache, _honest_loss

    X, y, phi, base = planted
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
    cols = ([f"inf{i}" for i in range(3)]
            + [f"dup{j}" for j in range(n_red)]
            + [f"noise{i}" for i in range(n_noise)])
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
    from mlframe.feature_selection._shap_proxy_revalidate import HonestLossCache, within_cluster_refine

    X, y = _refine_planted_redundant(seed=0)
    n_search = 450
    Xs, ys = X.iloc[:n_search].reset_index(drop=True), y[:n_search]
    Xh, yh = X.iloc[n_search:].reset_index(drop=True), y[n_search:]
    member_cols = list(range(X.shape[1]))  # the chosen union: every column, incl. 4 redundant dups
    member_groups = _refine_planted_groups()

    cache_multi = HonestLossCache()
    refined = within_cluster_refine(
        member_cols, LinearRegression(), Xs, ys, Xh, yh, classification=False,
        metric="rmse", parsimony_tol=0.05, n_jobs=1, member_groups=member_groups, cache=cache_multi,
        min_multi_clusters=1)
    n_multi = cache_multi.misses + cache_multi.hits

    cache_legacy = HonestLossCache()
    refined_legacy = within_cluster_refine(
        member_cols, LinearRegression(), Xs, ys, Xh, yh, classification=False,
        metric="rmse", parsimony_tol=0.05, n_jobs=1, member_groups=None, cache=cache_legacy)
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
    from mlframe.feature_selection._shap_proxy_revalidate import within_cluster_refine

    rng = np.random.default_rng(1)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    y = (1.5 * X["x0"] + 1.2 * X["x1"] - 0.9 * X["x2"] + 0.7 * X["x3"] - 0.5 * X["x4"]
         + 0.1 * rng.normal(size=n)).to_numpy()
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]
    cols = list(range(5))
    refined_legacy = within_cluster_refine(
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse",
        parsimony_tol=0.02, n_jobs=1, member_groups=None)
    refined_multi = within_cluster_refine(
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse",
        parsimony_tol=0.02, n_jobs=1, member_groups=[[i] for i in range(5)])
    assert refined_legacy == refined_multi


def test_within_cluster_refine_engages_shared_cache():
    """When a HonestLossCache is supplied, refine's single-drop trials must consult it; the second
    invocation on the same inputs returns identical output with all-but-zero new fits."""
    from mlframe.feature_selection._shap_proxy_revalidate import HonestLossCache, within_cluster_refine

    X, y = _refine_planted_redundant(seed=2)
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]
    cols = list(range(X.shape[1]))
    groups = _refine_planted_groups()
    cache = HonestLossCache()
    r1 = within_cluster_refine(
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse",
        parsimony_tol=0.05, n_jobs=1, member_groups=groups, cache=cache, min_multi_clusters=1)
    misses_1 = cache.misses
    r2 = within_cluster_refine(
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse",
        parsimony_tol=0.05, n_jobs=1, member_groups=groups, cache=cache, min_multi_clusters=1)
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
    from mlframe.feature_selection._shap_proxy_revalidate import HonestLossCache, within_cluster_refine

    # All-informative data with no redundancy: legacy refine fires its early-exit (no single drop
    # within tol). The stage-1 cluster-collapse cost on a spurious cluster is then PURE overhead.
    rng = np.random.default_rng(7)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    y = (1.5 * X["x0"] + 1.2 * X["x1"] - 0.9 * X["x2"] + 0.7 * X["x3"] - 0.5 * X["x4"]
         + 0.1 * rng.normal(size=n)).to_numpy()
    Xs, ys = X.iloc[:450].reset_index(drop=True), y[:450]
    Xh, yh = X.iloc[450:].reset_index(drop=True), y[450:]
    cols = list(range(5))
    # 1 spurious multi-member group (cols 3,4) + 3 singletons. Below default min_multi_clusters=3.
    spurious_one_multi = [[0], [1], [2], [3, 4]]
    assert sum(1 for g in spurious_one_multi if len(g) > 1) == 1

    cache_gated = HonestLossCache()
    refined_gated = within_cluster_refine(
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse",
        parsimony_tol=0.02, n_jobs=1, member_groups=spurious_one_multi, cache=cache_gated,
        min_multi_clusters=3)
    fits_gated = cache_gated.misses + cache_gated.hits

    cache_legacy = HonestLossCache()
    refined_legacy = within_cluster_refine(
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse",
        parsimony_tol=0.02, n_jobs=1, member_groups=None, cache=cache_legacy)
    fits_legacy = cache_legacy.misses + cache_legacy.hits

    # With the gate, the low-redundancy path matches legacy behavior exactly (no stage-1 overhead).
    assert fits_gated == fits_legacy, (
        f"gated refine ran {fits_gated} fits vs legacy {fits_legacy}; stage-1 should be skipped")
    assert refined_gated == refined_legacy

    # When the gate is loosened (min_multi_clusters=1), stage-1 fires and costs ADDITIONAL fits
    # (the spurious cluster's probe + cumulative verify before stage-2 runs).
    cache_ungated = HonestLossCache()
    within_cluster_refine(
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse",
        parsimony_tol=0.02, n_jobs=1, member_groups=spurious_one_multi, cache=cache_ungated,
        min_multi_clusters=1)
    fits_ungated = cache_ungated.misses + cache_ungated.hits
    assert fits_ungated > fits_gated, (
        f"un-gated refine ran {fits_ungated} fits vs gated {fits_gated}; stage-1 should have fired")


def test_within_cluster_refine_gate_preserves_collapse_with_enough_clusters():
    """The gate must NOT block stage-1 when there are >= min_multi_clusters multi-member groups
    (the cluster-rich case that wins). Build 3 redundancy clusters and verify stage-1 fires
    (fit count > legacy single-drop)."""
    from mlframe.feature_selection._shap_proxy_revalidate import HonestLossCache, within_cluster_refine

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
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse",
        parsimony_tol=0.05, n_jobs=1, member_groups=groups, cache=cache_gate)
    fits_with_stage1 = cache_gate.misses + cache_gate.hits

    cache_legacy = HonestLossCache()
    within_cluster_refine(
        cols, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse",
        parsimony_tol=0.05, n_jobs=1, member_groups=None, cache=cache_legacy)
    fits_legacy = cache_legacy.misses + cache_legacy.hits

    # Stage-1 collapse must shrink the column count BELOW legacy greedy alone could on its own (the
    # net win is the canonical 4.7x at 5k-clean).
    assert len(refined) <= 6, f"stage-1 collapse should drop most duplicates; got {len(refined)} cols"
    # Stage-1 collapse strictly REDUCES total fits vs legacy single-drop greedy because each
    # collapse skips a full single-drop round (n_features fits each).
    assert fits_with_stage1 < fits_legacy, (
        f"stage-1 fits {fits_with_stage1} not < legacy {fits_legacy}")


def test_full_fit_cached_matches_uncached():
    """End-to-end: the selector's support_ + selected_features_ are identical whether the honest-loss
    cache serves duplicates or every retrain is recomputed (the cache changes only wall-clock)."""
    from mlframe.feature_selection import _shap_proxy_revalidate as RV
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    rng = np.random.default_rng(0)
    n, f = 1500, 30
    Xnp = rng.normal(size=(n, f))
    Xnp[:, 5] = Xnp[:, 0] + 0.2 * rng.normal(size=n)  # a correlated redundant copy -> exercises clustering
    X = pd.DataFrame(Xnp, columns=[f"x{i}" for i in range(f)])
    logit = 1.2 * Xnp[:, 0] + 0.9 * Xnp[:, 1] - 0.7 * Xnp[:, 2]
    y = pd.Series((logit + 0.3 * rng.normal(size=n) > 0).astype(int))

    def _fit():
        sel = ShapProxiedFS(classification=True, metric="brier", optimizer="auto",
                            cluster_features=True, top_n=12, n_splits=3, n_revalidation_models=2,
                            n_anchors=15, random_state=0, verbose=False)
        sel.fit(X, y)
        return sel.support_.copy(), list(sel.selected_features_)

    sup_cached, feats_cached = _fit()

    # Force the no-cache path by neutering get() so every retrain recomputes.
    real_get = RV.HonestLossCache.get
    RV.HonestLossCache.get = lambda self, idx, seed: None
    try:
        sup_uncached, feats_uncached = _fit()
    finally:
        RV.HonestLossCache.get = real_get

    assert np.array_equal(sup_cached, sup_uncached)
    assert feats_cached == feats_uncached
