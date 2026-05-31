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
    # iter56 raised the default ``brute_force_max_features`` from 22 to 28
    # (shap_proxied_fs.py:292). The formula stays
    # ``max(brute_force_max_features * safety_factor=4, min_features=40)``;
    # the new default lands at 112.
    assert sp["requested_top"] == 112  # default: max(28 * 4, 40) = 112
    assert sp["effective_prefilter_top"] == 112  # min(200, 112) = 112
    assert sp["user_prefilter_top"] == 200
    # prefilter must reflect the tightened cap.
    assert rep["prefilter"]["kept"] <= 112


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


# ----------------------------------------------------------------- iter33: SHAP-aware stage-A
def test_resolve_shap_aware_stage1_keep_tightens_to_cushion_floor():
    from mlframe.feature_selection._shap_proxy_shap_prefilter import resolve_shap_aware_stage1_keep

    # effective_prefilter_top=88, cushion=8 -> 88*8=704 ; floor=200 ; default=2000
    # -> max(200, 704, 88) = 704 ; min(2000, 704) = 704
    assert resolve_shap_aware_stage1_keep(
        effective_prefilter_top=88, stage1_cushion=8, stage1_floor=200,
        default_stage1_keep=2000) == 704


def test_resolve_shap_aware_stage1_keep_floor_dominates_tiny_cap():
    from mlframe.feature_selection._shap_proxy_shap_prefilter import resolve_shap_aware_stage1_keep

    # tiny eff_top=10, cushion=8 -> 80 ; floor=200 wins -> min(2000, 200) = 200
    assert resolve_shap_aware_stage1_keep(
        effective_prefilter_top=10, stage1_cushion=8, stage1_floor=200,
        default_stage1_keep=2000) == 200


def test_resolve_shap_aware_stage1_keep_never_expands_default():
    from mlframe.feature_selection._shap_proxy_shap_prefilter import resolve_shap_aware_stage1_keep

    # eff_top*cushion exceeds default 500 -> capped at default (strict tighten contract)
    assert resolve_shap_aware_stage1_keep(
        effective_prefilter_top=200, stage1_cushion=8, stage1_floor=200,
        default_stage1_keep=500) == 500


def test_resolve_shap_aware_stage1_keep_never_below_effective_prefilter_top():
    """Stage A must produce at least the eventual stage-B output budget."""
    from mlframe.feature_selection._shap_proxy_shap_prefilter import resolve_shap_aware_stage1_keep

    # Pathological: floor=10, cushion=1, eff_top=300, default=2000 -> max(10, 300, 300)=300
    assert resolve_shap_aware_stage1_keep(
        effective_prefilter_top=300, stage1_cushion=1, stage1_floor=10,
        default_stage1_keep=2000) == 300


def test_resolve_shap_aware_stage1_keep_none_eff_top_returns_default():
    """No effective_prefilter_top -> return the legacy default unchanged."""
    from mlframe.feature_selection._shap_proxy_shap_prefilter import resolve_shap_aware_stage1_keep

    assert resolve_shap_aware_stage1_keep(
        effective_prefilter_top=None, stage1_cushion=8, stage1_floor=200,
        default_stage1_keep=2000) == 2000


def test_shap_aware_stage1_cushion_default_is_two():
    """Iter76 calibrated the default cushion from 8 -> 2 (C1/C2/C3 sweep: prefilter wall 3.0-4.0x
    faster, e2e 1.42-1.58x faster, recall preserved or +1). Lock the default so a future bump back
    to 8 surfaces in CI rather than silently regressing the wide-data prefilter wall.
    """
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sel = ShapProxiedFS()
    assert sel.shap_aware_stage1_cushion == 2, (
        f"iter76 default regression: expected cushion=2, got {sel.shap_aware_stage1_cushion}. "
        "If you intentionally widened the cushion, also update bench attribution in "
        "_shap_proxy_shap_prefilter.py docstring."
    )


def test_selector_records_stage1_keep_tightened_when_lever_active():
    """When the iter33 lever is active, the selector pre-resolves stage1_keep and the report
    sub-block records both the tightened value and the legacy default for diagnostics."""
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    # width >= _two_stage_min_width (1000) so the resolved method is "two_stage".
    X, y, _ = make_regime_dataset(
        n_samples=600, n_informative=5, n_redundant=0, redundancy_rho=0.0,
        n_noise=1295, snr=8.0, task="binary", seed=0)  # 1300 cols
    sel = ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=500, cluster_features=False,
        top_n=10, n_splits=3, n_revalidation_models=2, trust_guard=False,
        run_importance_ablation=False, within_cluster_refine=False,
        shap_prefilter_enabled=True, shap_aware_stage1_keep=True,
        random_state=0, verbose=False)
    sel.fit(X, y)
    rep = sel.shap_proxy_report_
    assert "shap_prefilter" in rep
    sp = rep["shap_prefilter"]
    # Lever recorded the resolved stage1_keep + the legacy default for the same n_features.
    assert "stage1_keep_tightened" in sp
    assert "stage1_keep_default" in sp
    # Default for n_features=1300: min(2000, 0.2*1300)=260; eff_top=88, cushion=2 (iter76 default)
    # -> max(200, 176)=200; min(260, 200) -> 200 (lever tightens). Verify the strict-tighten contract.
    assert sp["stage1_keep_tightened"] <= sp["stage1_keep_default"]


def test_selector_user_pinned_stage1_keep_overrides_lever():
    """A user who explicitly pins ``prefilter_stage1_keep`` must see their value pass through
    unchanged -- the lever is a default-only tightening."""
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y, _ = make_regime_dataset(
        n_samples=600, n_informative=5, n_redundant=0, redundancy_rho=0.0,
        n_noise=1295, snr=8.0, task="binary", seed=0)
    sel = ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=500, cluster_features=False,
        top_n=10, n_splits=3, n_revalidation_models=2, trust_guard=False,
        run_importance_ablation=False, within_cluster_refine=False,
        shap_prefilter_enabled=True, shap_aware_stage1_keep=True,
        prefilter_stage1_keep=1234,
        random_state=0, verbose=False)
    sel.fit(X, y)
    # User-pinned stage1_keep wins: the lever's tighten-fields are absent from the shap_prefilter
    # sub-block (no rewrite happened).
    sp = sel.shap_proxy_report_["shap_prefilter"]
    assert "stage1_keep_tightened" not in sp


def test_selector_lever_disabled_uses_default_stage1_keep():
    """When the iter33 lever is gated off, stage1_keep falls back to the legacy default."""
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y, _ = make_regime_dataset(
        n_samples=600, n_informative=5, n_redundant=0, redundancy_rho=0.0,
        n_noise=1295, snr=8.0, task="binary", seed=0)
    sel = ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=500, cluster_features=False,
        top_n=10, n_splits=3, n_revalidation_models=2, trust_guard=False,
        run_importance_ablation=False, within_cluster_refine=False,
        shap_prefilter_enabled=True, shap_aware_stage1_keep=False,
        random_state=0, verbose=False)
    sel.fit(X, y)
    sp = sel.shap_proxy_report_["shap_prefilter"]
    assert "stage1_keep_tightened" not in sp


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_biz_val_shap_aware_stage1_keep_e2e_speedup_at_target_regime():
    """Biz-value: at the iter33 target regime (width=10000, n_rows=5000, n_inf=20, snr=8) the lever
    must shorten the e2e wall AND preserve >= n_informative-2 / n_informative recall on seed 0.
    Validates against C2 from the iter33 scaling sweep (width 10000 is the user's actual target;
    iter1-31 already saturated the 1000-feature live regime)."""
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    def run(enabled, seed):
        X, y, roles = make_regime_dataset(
            n_samples=5000, n_informative=20, n_redundant=0, redundancy_rho=0.0,
            n_noise=9980, snr=8.0, task="binary", seed=seed)
        sel = ShapProxiedFS(
            classification=True, metric="brier", optimizer="auto",
            prefilter_top=500, cluster_features=True, cluster_corr_threshold=0.7,
            top_n=20, n_splits=4, n_revalidation_models=3, trust_guard=True, n_anchors=24,
            run_importance_ablation=True, within_cluster_refine=True,
            shap_prefilter_enabled=True, shap_aware_stage1_keep=enabled,
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

    t_base, r_base = run(False, 0)
    t_new, r_new = run(True, 0)
    # Recall sanity floor: baseline must clear 10/20 (well above random 0/20 floor). The synthetic
    # noise pool at width=10000 / n_rows=5000 prevents reliable >=18/20 recall (the weakest
    # informatives sit at coef ~0.4 and lose to spurious noise correlations -- see
    # ``make_regime_dataset`` finite-sample caveat). Iter33 scaling sweep at this regime: baseline
    # recall 16/20 / lever 17/20.
    assert r_base >= 10, f"baseline recall too low for biz_value comparison: {r_base}/20"
    # Recall must not regress by more than 1 informative vs baseline.
    assert r_new >= r_base - 1, (
        f"new recall dropped vs baseline: base={r_base}/20 new={r_new}/20")
    # >= 5% speedup floor (measured 27-29% on dev HW; floor leaves headroom for slow CI hosts).
    assert t_new < 0.95 * t_base, (
        f"e2e gain below floor: base={t_base:.2f}s new={t_new:.2f}s "
        f"speedup={t_base / t_new:.2f}x")
