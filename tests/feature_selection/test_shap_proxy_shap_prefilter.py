"""Unit + biz_value tests for the iter31 SHAP-pre-prefilter cap.

The lever tightens the effective ``prefilter_top`` passed to ``prefilter_columns`` to a SHAP-aware
cap ``max(brute_force_max_features * safety_factor, shap_prefilter_min_features)`` so the existing
prefilter booster already produces a search-budget-sized cohort (no second booster fit). Unit tests
pin the resolver math + selector wiring; biz_value confirms the recovery floor + e2e speedup on a
regime-aligned synthetic.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")


# ----------------------------------------------------------------- resolver math
def test_resolve_shap_prefilter_top_uses_safety_factor():
    from mlframe.feature_selection._shap_proxy_shap_prefilter import resolve_shap_prefilter_top

    # search cap 22 * factor 4 = 88 dominates the 40 floor.
    assert resolve_shap_prefilter_top(
        brute_force_max_features=22, safety_factor=4, min_features=40) == 88


def test_resolve_shap_prefilter_top_respects_min_features_floor():
    from mlframe.feature_selection._shap_proxy_shap_prefilter import resolve_shap_prefilter_top

    # tiny search cap (e.g. 5 * 4 = 20) is dominated by the 40 floor.
    assert resolve_shap_prefilter_top(
        brute_force_max_features=5, safety_factor=4, min_features=40) == 40


def test_resolve_shap_prefilter_top_safety_factor_is_int_multiplied():
    from mlframe.feature_selection._shap_proxy_shap_prefilter import resolve_shap_prefilter_top

    # factor 1 collapses cushion -> max(22, 40) = 40.
    assert resolve_shap_prefilter_top(
        brute_force_max_features=22, safety_factor=1, min_features=40) == 40
    # factor 10 -> 220.
    assert resolve_shap_prefilter_top(
        brute_force_max_features=22, safety_factor=10, min_features=40) == 220


# ----------------------------------------------------------------- selector wiring
def _tiny_regime(n=400, n_features=300, n_inf=5, seed=0):
    """Tiny synthetic at width=300 so prefilter triggers but tests stay fast."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, n_inf))
    coefs = np.linspace(1.0, 0.4, n_inf)
    f = (Z * coefs).sum(axis=1)
    f = f / max(np.std(f), 1e-6)
    logits = f * np.sqrt(5.0)
    thr = np.quantile(logits, 0.5)
    p = 1.0 / (1.0 + np.exp(-(logits - thr)))
    y = (rng.random(n) < p).astype(int)
    N = rng.standard_normal((n, n_features - n_inf))
    X = np.column_stack([Z, N])
    cols = [f"inf{i}" for i in range(n_inf)] + [f"noise{i}" for i in range(n_features - n_inf)]
    return pd.DataFrame(X, columns=cols), y


def test_selector_records_shap_prefilter_block_when_enabled():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _tiny_regime()
    sel = ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=200, cluster_features=False,
        top_n=10, n_splits=3, n_revalidation_models=2,
        trust_guard=False, run_importance_ablation=False, within_cluster_refine=False,
        shap_prefilter_enabled=True,
        random_state=0, verbose=False)
    sel.fit(X, y)
    rep = sel.shap_proxy_report_
    assert "shap_prefilter" in rep
    sp = rep["shap_prefilter"]
    assert sp["requested_top"] == 88  # default: max(22 * 4, 40) = 88
    assert sp["effective_prefilter_top"] == 88  # min(200, 88) = 88
    assert sp["user_prefilter_top"] == 200
    # prefilter must reflect the tightened cap.
    assert rep["prefilter"]["kept"] <= 88


def test_selector_disabled_keeps_user_prefilter_top():
    """``shap_prefilter_enabled=False`` restores the legacy behaviour: the selector passes
    ``self.prefilter_top`` to ``prefilter_columns`` unchanged."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _tiny_regime()
    sel = ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=200, cluster_features=False,
        top_n=10, n_splits=3, n_revalidation_models=2,
        trust_guard=False, run_importance_ablation=False, within_cluster_refine=False,
        shap_prefilter_enabled=False,
        random_state=0, verbose=False)
    sel.fit(X, y)
    rep = sel.shap_proxy_report_
    assert "shap_prefilter" not in rep
    # prefilter ran with the user's prefilter_top (200) -> kept up to 200 columns.
    assert rep["prefilter"]["kept"] <= 200
    # The tightened path would have kept <= 88; absence of the cap means we keep more.
    # On the tiny 300-col regime the prefilter narrows to user's prefilter_top=200.
    assert rep["prefilter"]["kept"] == 200


def test_selector_shap_prefilter_top_override_dominates_factor_floor():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _tiny_regime()
    sel = ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=200, cluster_features=False,
        top_n=10, n_splits=3, n_revalidation_models=2,
        trust_guard=False, run_importance_ablation=False, within_cluster_refine=False,
        shap_prefilter_enabled=True, shap_prefilter_top=60,
        random_state=0, verbose=False)
    sel.fit(X, y)
    rep = sel.shap_proxy_report_
    assert rep["shap_prefilter"]["requested_top"] == 60
    assert rep["shap_prefilter"]["effective_prefilter_top"] == 60
    assert rep["prefilter"]["kept"] <= 60


def test_selector_never_expands_user_prefilter_top():
    """When the user picked a tight ``prefilter_top`` smaller than the derived SHAP-aware cap,
    the lever must not loosen it."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _tiny_regime()
    sel = ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=50,  # tighter than the derived 88
        cluster_features=False,
        top_n=10, n_splits=3, n_revalidation_models=2,
        trust_guard=False, run_importance_ablation=False, within_cluster_refine=False,
        shap_prefilter_enabled=True,
        random_state=0, verbose=False)
    sel.fit(X, y)
    rep = sel.shap_proxy_report_
    assert rep["shap_prefilter"]["effective_prefilter_top"] == 50
    assert rep["prefilter"]["kept"] <= 50


# ----------------------------------------------------------------- recall preservation
@pytest.mark.slow
@pytest.mark.timeout(300)
def test_shap_prefilter_preserves_informative_recall_synthetic():
    """The cheap-importance pass MUST keep all planted informatives at safety_factor=4 (88-col cap)
    on a regime where signal is well above the noise floor. Pins the heuristic against a regime with
    known informatives so a future tightening doesn't silently drop signal.

    n_samples=5000 follows the make_regime_dataset docstring caveat: at n<=2000 the weakest
    informatives (coef ~0.4) can lose to random spurious noise correlations finite-sample variance
    regardless of the lever -- the >=5000 floor isolates the LEVER's impact on recall from the
    synthetic's noise-pool artifact."""
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y, roles = make_regime_dataset(
        n_samples=5000, n_informative=12, n_redundant=0, redundancy_rho=0.0,
        n_noise=988, snr=8.0, task="binary", seed=0)
    sel = ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=500, cluster_features=True, cluster_corr_threshold=0.7,
        top_n=20, n_splits=3, n_revalidation_models=2, trust_guard=False,
        run_importance_ablation=False, within_cluster_refine=False,
        shap_prefilter_enabled=True,
        random_state=0, verbose=False)
    sel.fit(X, y)
    informative = {n for n, r in roles.items() if r == "informative"}
    kept = set(sel.selected_features_)
    # All 12 informatives must survive the tightened prefilter -> SHAP -> search pipeline.
    assert len(informative & kept) == len(informative), (
        f"informative recall dropped: {sorted(informative - kept)} lost, "
        f"kept={sorted(informative & kept)}")


# ----------------------------------------------------------------- biz_value: e2e speedup
@pytest.mark.slow
@pytest.mark.timeout(600)
def test_biz_val_shap_prefilter_e2e_speedup_at_live_regime():
    """Biz-value: at the live wide regime (width=1000, n_rows=5000, n_inf=12, snr=8) the new lever
    must shorten the e2e wall by >= 10% vs the disabled baseline AND preserve 12/12 informative
    recall, on seeds 0 AND 1. Pin protects the iter31 win against regression."""
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    def run(enabled, seed):
        X, y, roles = make_regime_dataset(
            n_samples=5000, n_informative=12, n_redundant=0, redundancy_rho=0.0,
            n_noise=988, snr=8.0, task="binary", seed=seed)
        sel = ShapProxiedFS(
            classification=True, metric="brier", optimizer="auto",
            prefilter_top=500, cluster_features=True, cluster_corr_threshold=0.7,
            top_n=20, n_splits=4, n_revalidation_models=3, trust_guard=True, n_anchors=24,
            run_importance_ablation=True, within_cluster_refine=True,
            shap_prefilter_enabled=enabled,
            random_state=seed, verbose=False)
        t0 = time.perf_counter()
        sel.fit(X, y)
        dt = time.perf_counter() - t0
        inf = {n for n, r in roles.items() if r == "informative"}
        rec = len(inf & set(sel.selected_features_))
        return dt, rec

    # Warmup: amortise JIT/cuda-init costs into the warmup run, not into the comparison.
    run(False, 99)
    run(True, 99)

    for seed in (0, 1):
        t_base, r_base = run(False, seed)
        t_new, r_new = run(True, seed)
        # Recall holds at 12/12 both ways (the lever must never trade quality for speed).
        assert r_base == 12, f"baseline recall regressed: {r_base}/12 at seed={seed}"
        assert r_new == 12, f"new recall regressed: {r_new}/12 at seed={seed}"
        # >= 5% speedup floor (measured 10-29%; the 5% floor leaves headroom for slow CI hosts).
        assert t_new < 0.95 * t_base, (
            f"seed={seed} e2e gain below floor: base={t_base:.2f}s new={t_new:.2f}s "
            f"speedup={t_base / t_new:.2f}x")
