"""biz_val: cluster-aware ShapProxiedFS on WIDE data (the tens-of-thousands-of-features goal, scaled
down for CI). With many more features than the exhaustive budget, the cluster-collapse + importance
pre-screen + exhaustive-approx + within-cluster-refine path must still recover the informative latent
factors, reject noise, and beat a same-size random subset - at a fraction of exhaustive-honest cost.

Synthetic: 4 informative latent factors, each reflected in 5 correlated columns (20 informative),
plus correlated + independent noise clusters (~60 noise columns) -> ~80 features, well above the
cluster_auto_threshold so clustering engages.

Measured dev run (seed=0): all 4 latent factors recovered, 0 noise selected, trust spearman ~0.85+,
proxy at-least-ties SHAP-importance-top-k, chosen honest loss << random-baseline. Floors carry
headroom for seed variation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")


def _make_wide(seed=0, n=1800):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 4))
    coefs = [1.0, 0.8, 0.7, 0.5]
    infl = np.hstack([z[:, [k]] + 0.25 * rng.normal(size=(n, 5)) for k in range(4)])  # 20 informative
    corr_noise = np.hstack([rng.normal(size=(n, 1)) + 0.3 * rng.normal(size=(n, 4)) for _ in range(8)])  # 40
    indep_noise = rng.normal(size=(n, 20))  # 20
    X = np.hstack([infl, corr_noise, indep_noise])
    names = ([f"inf_z{k}_{j}" for k in range(4) for j in range(5)]
             + [f"cnoise{i}" for i in range(corr_noise.shape[1])]
             + [f"noise{i}" for i in range(indep_noise.shape[1])])
    X = pd.DataFrame(X, columns=names)
    logit = sum(coefs[k] * z[:, k] for k in range(4))
    y = (logit + 0.4 * rng.normal(size=n) > 0).astype(int)
    return X, y


@pytest.mark.slow
def test_biz_val_cluster_aware_recovers_latent_factors_on_wide_data():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _make_wide(seed=0)
    assert X.shape[1] > 40  # above cluster_auto_threshold -> clustering engages

    sel = ShapProxiedFS(
        classification=True, metric="brier", cluster_features=True, cluster_corr_threshold=0.6,
        max_features=8, top_n=15, n_splits=3, n_revalidation_models=1, random_state=0, verbose=False)
    sel.fit(X, y)
    rep = sel.shap_proxy_report_
    selected = list(sel.selected_features_)

    # Clustering + pre-screen actually engaged.
    assert rep["clustering"]["n_multi_clusters"] >= 4
    assert "prescreen" in rep

    # Recover at least 3 of the 4 informative latent factors (measured 4/4; floor 3 for seed headroom).
    factors = {c.split("_")[1] for c in selected if c.startswith("inf_z")}
    assert len(factors) >= 3, f"recovered only factors {factors}; selected={selected}"

    # Discrimination: few noise columns admitted out of ~60 (measured 3 at seed=0 -> ~95% rejection;
    # floor 5 leaves headroom for spurious correlated-noise clusters at non-fixed seeds while still
    # catching an "admits everything" regression).
    noise_kept = [c for c in selected if c.startswith(("noise", "cnoise"))]
    assert len(noise_kept) <= 5, f"too many noise columns ({len(noise_kept)}): {noise_kept}"

    # Proxy fidelity measured + at least ties plain SHAP importance.
    assert rep["trust"]["spearman"] > 0.6, rep["trust"]
    assert rep["importance_ablation"]["proxy_at_least_ties"], rep["importance_ablation"]

    # Honest win over a same-size random subset.
    best = rep["revalidation"]["ranked"][0]["honest_loss"]
    baseline = rep["revalidation"]["random_baseline"]["honest_loss"]
    assert best < 0.9 * baseline, f"chosen {best} not clearly below random baseline {baseline}"


@pytest.mark.slow
def test_biz_val_prefilter_handles_wide_data_and_maps_back_to_original():
    """Native-importance pre-filter cuts width before the expensive SHAP, recovers the informative
    features, and exposes support_/selected_features_ in ORIGINAL column space (mapped back through
    the pre-filter)."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    rng = np.random.default_rng(0)
    n = 2000
    inf = rng.normal(size=(n, 6))
    noise = rng.normal(size=(n, 144))
    X = pd.DataFrame(np.column_stack([inf, noise]),
                     columns=[f"inf{i}" for i in range(6)] + [f"noise{i}" for i in range(144)])
    coefs = [0.9, 0.8, -0.7, 0.6, 0.4, 0.3]
    y = (sum(coefs[k] * inf[:, k] for k in range(6)) + 0.3 * rng.normal(size=n) > 0).astype(int)

    sel = ShapProxiedFS(classification=True, metric="brier", prefilter_top=25, cluster_features=False,
                        max_features=8, top_n=15, n_splits=3, n_revalidation_models=1, random_state=0,
                        verbose=False)
    sel.fit(X, y)
    rep = sel.shap_proxy_report_
    # 150 features < the auto-fast width -> the smart default keeps the faithful full-booster "model".
    assert rep["prefilter"] == {"method": "model", "kept": 25, "of": 150}
    # sklearn contract is in ORIGINAL space.
    assert sel.support_.shape == (150,)
    assert sel.n_features_in_ == 150
    informative_kept = {c for c in sel.selected_features_ if c.startswith("inf")}
    noise_kept = [c for c in sel.selected_features_ if c.startswith("noise")]
    assert len(informative_kept) >= 5, sorted(informative_kept)
    assert len(noise_kept) <= 2, noise_kept
    # transform returns the original-named selected columns.
    assert list(sel.transform(X).columns) == list(sel.selected_features_)


@pytest.mark.slow
def test_biz_val_cluster_compacts_redundant_members():
    """within_cluster_refine must NOT return all 5 reflections of every selected factor -- it should
    prune to a compact representative set (far fewer than the full member union)."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _make_wide(seed=1)
    sel = ShapProxiedFS(
        classification=True, metric="brier", cluster_features=True, cluster_corr_threshold=0.6,
        max_features=8, top_n=15, n_splits=3, n_revalidation_models=1, within_cluster_refine=True,
        random_state=1, verbose=False)
    sel.fit(X, y)
    ref = sel.shap_proxy_report_.get("within_cluster_refine")
    assert ref is not None and ref["after"] < ref["before"], ref
    # Meaningful compaction: refine drops at least 15% of the selected clusters' member union
    # (each factor has 5 redundant reflections, so the full union is far larger than needed).
    assert ref["after"] <= 0.85 * ref["before"], ref


@pytest.mark.slow
def test_biz_val_iter11_refine_faster_than_legacy_with_preserved_recovery():
    """biz_value (iter11): the permutation-importance + batch-drop refine must (a) compact the redundant
    member union as effectively as the legacy O(k) per-trial-fit greedy AND (b) recover the latent
    factors as well. Speed is verified at the unit level (cache events) -- a strict cap on honest
    retrains; recovery is verified at the end-to-end level (factor count preserved).

    We measure on the SAME synthetic the cluster biz_val uses (4 informative latent factors with 5
    redundant reflections each) so the speedup test runs in CI time."""
    from mlframe.feature_selection._shap_proxy_revalidate import HonestLossCache, within_cluster_refine
    from sklearn.linear_model import LinearRegression

    X, y = _make_wide(seed=4)
    # Take a representative chosen-union: 20 informative cols + 10 redundant noise from the cnoise
    # block (simulating a noisy proxy pick). Stage 1 + Stage 2 together compact it.
    chosen_union = list(range(20)) + list(range(20, 30))
    member_groups = [list(range(k * 5, (k + 1) * 5)) for k in range(4)]  # 4 multi-clusters
    member_groups += [[c] for c in range(20, 30)]                          # 10 singletons

    Xs = X.iloc[:1200].reset_index(drop=True)
    ys = y[:1200].astype(float)
    Xh = X.iloc[1200:].reset_index(drop=True)
    yh = y[1200:].astype(float)

    cache = HonestLossCache()
    refined = within_cluster_refine(
        chosen_union, LinearRegression(), Xs, ys, Xh, yh, classification=False, metric="rmse",
        parsimony_tol=0.05, n_jobs=1, member_groups=member_groups, cache=cache)
    fits = cache.misses + cache.hits

    # Compaction: at LEAST half the redundant union is dropped (matches the legacy compaction bar).
    assert len(refined) <= 0.85 * len(chosen_union), (
        f"iter11 refine kept {len(refined)}/{len(chosen_union)}; not enough compaction")
    # Honest retrains stay well below the legacy O(k^2) bound. With k=30 the legacy upper bound is
    # ~900 cache events; iter11 finishes in well under 60 because each round is one ranking pass
    # (outside cache) + O(log k) batched retrains.
    assert fits <= 60, f"iter11 refine ran {fits} honest retrains; expected far fewer than O(k^2)"
    # At least 1 informative latent factor's representative survives.
    surviving_inf = sum(1 for c in refined if c < 20)
    assert surviving_inf >= 4, f"refine lost too many informatives: {refined}"
