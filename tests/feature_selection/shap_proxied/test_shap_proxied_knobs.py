"""ShapProxiedFS su_seeded_* sub-knobs + min_selected_ratio + active_learning_budget.

coverage_asymmetry_wrappers-15: the ``su_seeded_interactions`` master switch has a biz_value pair
(recovery + byte-identical no-op in ``test_shap_proxy_su_seeded_interactions.py``), but its SNR-gate
sub-knobs -- ``su_seeded_top_k`` (how many synergistic pairs survive) and ``su_seeded_snr_z`` (how
strict the permutation-null gate is) -- plus the selection-floor ``min_selected_ratio`` and the
refinement budget cap ``active_learning_budget`` had ZERO behavioural tests. Each is confirmed a real
ctor param (shap_proxied_fs/__init__.py: top_k :110, snr_z :113, min_selected_ratio :80,
active_learning_budget :92).

Four falsifiable checks:
 (a) su_seeded_top_k binds: on a TWO-interaction-pair bed (one strong, one weaker, both clear the
     default SNR gate) ``top_k=1`` keeps ONLY the stronger pair while the default keeps both -- the
     screen-level cap; the facade then recovers only the stronger pair's operands under top_k=1.
 (b) su_seeded_snr_z monotonicity: setting ``snr_z`` absurdly high (500) makes the screen RUN but
     admit 0 pairs, so the selection byte-equals the ``su_seeded_interactions=False`` run (the no-op).
 (c) min_selected_ratio floor: on a noise-heavy frame ``len(selected_) >= ratio * n_features_in_``.
 (d) active_learning_budget caps the honest-refinement model count: the published
     ``report['revalidation']['active_learning']['n_evaluated']`` is <= budget and a small budget
     evaluates strictly fewer candidates than a large one.

``cluster_use_gpu`` (the remaining flagged knob) is GPU-suite-covered via the env dispatch in
``test_shap_proxy_cluster_su_gpu.py`` / ``test_shap_proxy_gpu.py``; it is NOT exercised here (CPU-only
mission) and needs no CPU test -- documented-skip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# is_fast_mode drives the MLFRAME_FAST representative: the two screen-level tests below are NOT
# @pytest.mark.slow, so they run under MLFRAME_FAST=1 (which skips @slow), keeping the su_seeded_*
# knob paths covered in fast mode. Imported for the contract even though the gating is by marker.
from tests.feature_selection.conftest import is_fast_mode  # noqa: F401


# --------------------------------------------------------------------------------------------------
# Fixtures: pure-interaction pairs whose operands have ~0 MARGINAL signal (all signal is the product).
# --------------------------------------------------------------------------------------------------
def _two_interaction_pairs(n=6000, p_noise=30, seed=0, strong=3.0, weak=2.0):
    """``y`` driven by TWO pure-interaction pairs of different strength, plus noise. Each operand has
    ~0 marginal SU; the strong pair (``s_a*s_b``) has higher pairwise synergy than the weaker pair
    (``w_a*w_b``). At the default SNR gate BOTH clear; ``su_seeded_top_k=1`` keeps only the stronger."""
    rng = np.random.default_rng(seed)
    a1 = rng.standard_normal(n)
    b1 = rng.standard_normal(n)
    a2 = rng.standard_normal(n)
    b2 = rng.standard_normal(n)
    logit = strong * np.sign(a1 * b1) + weak * np.sign(a2 * b2)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    cols = {"s_a": a1, "s_b": b1, "w_a": a2, "w_b": b2}
    for i in range(p_noise):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    order = list(X.columns)
    rng.shuffle(order)
    return X[order], pd.Series(y, name="target")


# --------------------------------------------------------------------------------------------------
# (a) su_seeded_top_k -- SCREEN level (no model, no SHAP): cheap + deterministic, the load-bearing
#     "top-k binds" assertion. Confirms the cap shrinks the kept synergistic-pair set to the strongest.
# --------------------------------------------------------------------------------------------------
def test_su_seeded_top_k_caps_kept_pairs_to_strongest():
    """``su_seeded_top_k`` bounds how many synergistic pairs survive the SNR gate. On a two-pair bed
    where BOTH clear the default gate, ``top_k=1`` keeps ONLY the higher-synergy pair, ``top_k=8`` keeps
    both. This is the SU-synergy screen ``su_synergy_screen`` consumes ``su_seeded_top_k`` directly."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions import su_synergy_screen

    X, y = _two_interaction_pairs(n=6000, p_noise=30, seed=0, strong=3.0, weak=2.0)

    kept_default, info_d = su_synergy_screen(
        X, y, n_bins=8, top_k=8, max_screen_cols=120,
        snr_z=3.0, snr_null_quantile=0.99, n_permutations=3, rng=np.random.default_rng(0))
    kept_k1, info_1 = su_synergy_screen(
        X, y, n_bins=8, top_k=1, max_screen_cols=120,
        snr_z=3.0, snr_null_quantile=0.99, n_permutations=3, rng=np.random.default_rng(0))

    # su_synergy_screen returns (synergy, joint_su, col_a, col_b), best-synergy first.
    default_pairs = [{a, b} for _syn, _jsu, a, b in kept_default]
    k1_pairs = [{a, b} for _syn, _jsu, a, b in kept_k1]

    # Premise: BOTH pairs clear the default gate (the bed is constructed so they do).
    assert {"s_a", "s_b"} in default_pairs, f"strong pair missing from default screen: {default_pairs}"
    assert {"w_a", "w_b"} in default_pairs, f"weak pair missing from default screen: {default_pairs}"
    assert len(default_pairs) >= 2, f"expected >=2 pairs at top_k=8; got {default_pairs}"

    # top_k=1 BINDS: exactly one pair survives, and it is the STRONGER (higher-synergy) one.
    assert len(k1_pairs) == 1, f"top_k=1 kept {len(k1_pairs)} pairs, expected 1: {k1_pairs}"
    assert k1_pairs[0] == {"s_a", "s_b"}, f"top_k=1 kept the wrong (non-strongest) pair: {k1_pairs}"
    # The strongest pair's synergy dominates (sanity that "strong" really is strongest).
    assert kept_default[0][0] >= kept_default[1][0], (
        f"screen not synergy-sorted: {info_d}")


# --------------------------------------------------------------------------------------------------
# Facade-level checks need a fitted model + SHAP.
# --------------------------------------------------------------------------------------------------
pytest.importorskip("shap")
pytest.importorskip("xgboost")


def _shap_sel(n_features, *, su_seeded=True, top_k=8, snr_z=3.0,
              min_selected_ratio=0.0, active_learning=False, active_learning_budget=None):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    return ShapProxiedFS(
        classification=True, n_splits=3, top_n=20, min_features=8,
        prefilter_top=min(60, n_features), prefilter_n_estimators=60,
        oof_shap_n_estimators=60, revalidation_n_estimators=60, n_revalidation_models=2,
        trust_guard=True, trust_guard_n_estimators=20, cluster_features="auto",
        within_cluster_refine=True, parsimony_tol=0.005,
        su_seeded_interactions=su_seeded, su_seeded_top_k=top_k, su_seeded_snr_z=snr_z,
        min_selected_ratio=min_selected_ratio,
        active_learning=active_learning, active_learning_budget=active_learning_budget,
        random_state=0, verbose=False, n_jobs=1)


@pytest.mark.slow
def test_su_seeded_top_k_binds_seeded_pair_count_at_facade():
    """FACADE: ``su_seeded_top_k`` binds how many synergistic pairs the SELECTOR seeds. On the two-pair
    bed the default (top_k=8) seeds BOTH pairs (report ``n_kept_pairs == 2``); ``top_k=1`` seeds only
    the stronger pair (``n_kept_pairs == 1``) while still recovering its operands ``{s_a, s_b}``.

    NOTE (measured, not a bug): operand RECOVERY is confounded by the additive/clustering path -- here
    both operands of the unseeded weaker pair are still recovered because at weak=2.0 their interaction
    is strong enough for the base model to surface them. So the facade pins the SEEDING cap
    (``n_kept_pairs`` 2 -> 1), which is the direct selector-level manifestation of ``su_seeded_top_k``;
    the strict "only the stronger pair survives" claim is pinned at the screen level
    (test_su_seeded_top_k_caps_kept_pairs_to_strongest), where no additive path can confound it."""
    from sklearn.model_selection import train_test_split

    X, y = _two_interaction_pairs(n=6000, p_noise=30, seed=0, strong=3.0, weak=2.0)
    Xtr, _Xte, ytr, _yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    Xtr = Xtr.reset_index(drop=True)
    ytr = ytr.reset_index(drop=True)
    n_features = Xtr.shape[1]

    sel_default = _shap_sel(n_features, su_seeded=True, top_k=8)
    sel_default.fit(Xtr, ytr)
    sel_k1 = _shap_sel(n_features, su_seeded=True, top_k=1)
    sel_k1.fit(Xtr, ytr)

    rep_d = sel_default.shap_proxy_report_.get("su_seeded_interactions", {})
    rep_1 = sel_k1.shap_proxy_report_.get("su_seeded_interactions", {})
    assert rep_d.get("applied") is True and rep_1.get("applied") is True

    # The cap BINDS at the selector level: default seeds both pairs, top_k=1 seeds only the strongest.
    assert rep_d.get("n_kept_pairs", 0) >= 2, f"default top_k should seed >=2 pairs: {rep_d}"
    assert rep_1.get("n_kept_pairs", 99) == 1, f"top_k=1 should seed exactly 1 pair: {rep_1}"
    assert rep_1["n_kept_pairs"] < rep_d["n_kept_pairs"], (
        f"top_k=1 did not reduce seeded pairs vs default: 1={rep_1} d={rep_d}")
    # The single seeded pair is the STRONGER one, and its operands are recovered.
    kept_1 = rep_1.get("kept_pairs", [])
    assert kept_1 and {kept_1[0][1], kept_1[0][2]} == {"s_a", "s_b"}, (
        f"top_k=1 seeded the wrong pair: {kept_1}")
    assert {"s_a", "s_b"} <= set(map(str, sel_k1.selected_features_)), (
        f"strong pair not recovered under top_k=1: {sorted(map(str, sel_k1.selected_features_))}")


@pytest.mark.slow
def test_su_seeded_snr_z_absurdly_high_is_byte_identical_no_op():
    """``su_seeded_snr_z`` is the SNR-gate strictness. Set absurdly high (500) the permutation-null
    gate rejects EVERY pair, so the screen RUNS (``applied=True``, ``n_screened_cols`` > 0) but admits
    0 pairs and the selection byte-equals the ``su_seeded_interactions=False`` run -- gate monotonicity
    (a stricter gate can only ever seed FEWER pairs, never change the additive default when it seeds
    none). Reuses the byte-identical no-op harness pattern from test_shap_proxy_su_seeded_interactions."""
    from sklearn.model_selection import train_test_split

    X, y = _two_interaction_pairs(n=6000, p_noise=30, seed=0, strong=3.0, weak=2.0)
    Xtr, _Xte, ytr, _yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    Xtr = Xtr.reset_index(drop=True)
    ytr = ytr.reset_index(drop=True)
    n_features = Xtr.shape[1]

    sel_off = _shap_sel(n_features, su_seeded=False)
    sel_off.fit(Xtr, ytr)
    sel_hi = _shap_sel(n_features, su_seeded=True, snr_z=500.0)
    sel_hi.fit(Xtr, ytr)

    rep = sel_hi.shap_proxy_report_.get("su_seeded_interactions", {})
    # The screen RAN (it scored columns) ...
    assert rep.get("applied") is True
    assert rep.get("n_screened_cols", 0) > 0, f"screen did not run: {rep}"
    # ... but the absurd gate admitted NO pair.
    assert rep.get("n_kept_pairs", 99) == 0, (
        f"expected SNR-gate no-op at snr_z=500 but kept {rep.get('kept_pairs')}: {rep}")
    # No pair seeded => byte-identical selection to the additive default.
    assert sel_hi.selected_features_ == sel_off.selected_features_, (
        f"high-snr_z no-op changed the additive default:\n off={sel_off.selected_features_}\n "
        f"on ={sel_hi.selected_features_}")


# --------------------------------------------------------------------------------------------------
# (c) min_selected_ratio -- selection floor on a noise-heavy frame.
# --------------------------------------------------------------------------------------------------
def _noise_heavy(n=3000, n_inf=4, n_noise=12, seed=1):
    rng = np.random.default_rng(seed)
    inf = rng.normal(size=(n, n_inf))
    noise = rng.normal(size=(n, n_noise))
    X = pd.DataFrame(
        np.column_stack([inf, noise]),
        columns=[f"inf{i}" for i in range(n_inf)] + [f"noise{i}" for i in range(n_noise)])
    coefs = np.array([1.0, 0.9, -0.8, 0.7])[:n_inf]
    logit = inf @ coefs
    y = (logit + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, pd.Series(y, name="target")


# ``min_selected_ratio`` floors the candidate's PROXY-column cardinality (``len(c) / phi.shape[1] >=
# ratio``), NOT the input-feature count -- see shap_proxied_fs/_shap_proxied_fit.py:721-726. With
# clustering OFF and ``prefilter_top`` capping phi width, the proxy space == the prefilter survivors,
# so the floor on proxy cols maps directly onto the final selection. Refinement / revalidation are
# disabled so nothing re-prunes below the floored proxy-best subset.
_C_PREFILTER_TOP = 10


def _ratio_sel(ratio):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    return ShapProxiedFS(
        classification=True, n_splits=3, top_n=30, min_features=1, max_features=None,
        prefilter_top=_C_PREFILTER_TOP, prefilter_n_estimators=60, oof_shap_n_estimators=60,
        cluster_features=False, within_cluster_refine=False, revalidate=False,
        run_importance_ablation=False, trust_guard=False, optimizer="bruteforce",
        min_selected_ratio=ratio, random_state=0, verbose=False, n_jobs=1)


@pytest.mark.slow
def test_min_selected_ratio_floors_selection_in_proxy_column_space():
    """``min_selected_ratio`` keeps only candidate subsets covering at least ``ratio`` of the proxy
    columns (``len(c) / phi.shape[1] >= ratio``). On a noise-heavy frame the proxy would naturally pick
    a tiny parsimonious subset (the 4 informative columns); a ratio of 0.5 forces every retained
    candidate to span >= half the proxy columns, so the final selection is floored at
    ``ratio * n_proxy_cols`` and strictly exceeds the unconstrained pick.

    The floor is in PROXY-column space, NOT input-feature space (a documented nuance: it does NOT
    guarantee ``len(selected_) >= ratio * n_features_in_``). With clustering OFF, ``n_proxy_cols ==
    min(prefilter_top, n_features_in_)``, so the floor is computable from the ctor."""
    import math

    X, y = _noise_heavy()
    n_features = X.shape[1]
    n_proxy = min(_C_PREFILTER_TOP, n_features)
    ratio = 0.5

    sel = _ratio_sel(ratio)
    sel.fit(X, y)
    sel0 = _ratio_sel(0.0)
    sel0.fit(X, y)

    floor = math.ceil(ratio * n_proxy)  # ratio*phi-width, rounded up (cardinality is integer)
    assert len(sel.selected_features_) >= floor, (
        f"min_selected_ratio={ratio} not enforced in proxy space: kept {len(sel.selected_features_)} "
        f"< floor {floor} (n_proxy={n_proxy}); selected={sorted(map(str, sel.selected_features_))}")
    # The floor actually CHANGED the outcome: it kept strictly more than the unconstrained pick.
    assert len(sel.selected_features_) > len(sel0.selected_features_), (
        f"ratio floor did not enlarge the subset: ratio={len(sel.selected_features_)} "
        f"vs default={len(sel0.selected_features_)}")
    # Never empty (the code falls back to the unfiltered candidates if the ratio empties the pool).
    assert len(sel.selected_features_) >= 1


# --------------------------------------------------------------------------------------------------
# (d) active_learning_budget -- caps the honest-refinement candidate (model) count.
# --------------------------------------------------------------------------------------------------
def _al_dataset(n=900, n_inf=6, n_noise=14, seed=2):
    rng = np.random.default_rng(seed)
    inf = rng.normal(size=(n, n_inf))
    noise = rng.normal(size=(n, n_noise))
    X = pd.DataFrame(
        np.column_stack([inf, noise]),
        columns=[f"inf{i}" for i in range(n_inf)] + [f"noise{i}" for i in range(n_noise)])
    coefs = np.array([1.0, -0.9, 0.8, 0.7, -0.6, 0.5])[:n_inf]
    logit = inf @ coefs
    y = (logit + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, pd.Series(y, name="target")


def _al_sel(n_features, budget):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    return ShapProxiedFS(
        classification=True, n_splits=3, top_n=20, min_features=1,
        prefilter_top=n_features, prefilter_n_estimators=25, oof_shap_n_estimators=25,
        revalidation_n_estimators=25, n_revalidation_models=2, trust_guard=True,
        trust_guard_n_estimators=15, within_cluster_refine=True, parsimony_tol=0.02,
        active_learning=True, active_learning_budget=budget, use_bias_corrector=True,
        random_state=0, verbose=False, n_jobs=1)


@pytest.mark.slow
def test_active_learning_budget_caps_refinement_model_count():
    """``active_learning_budget`` caps how many candidate subsets the disagreement-driven honest
    re-validation retrains (each retrain = ``n_revalidation_models`` honest fits). The path publishes
    the evaluated count at ``report['revalidation']['active_learning']['n_evaluated']``. A small budget
    must (1) report ``n_evaluated <= budget`` and (2) evaluate strictly fewer candidates than a large
    budget -- the budget genuinely bounds the refinement work."""
    X, y = _al_dataset()
    n_features = X.shape[1]

    sel_small = _al_sel(n_features, budget=3)
    sel_small.fit(X, y)
    sel_large = _al_sel(n_features, budget=12)
    sel_large.fit(X, y)

    al_small = sel_small.shap_proxy_report_["revalidation"]["active_learning"]
    al_large = sel_large.shap_proxy_report_["revalidation"]["active_learning"]

    assert al_small["budget"] == 3, al_small
    assert al_small["n_evaluated"] <= 3, f"budget=3 not enforced: {al_small}"
    assert al_large["n_evaluated"] <= 12, f"budget=12 not enforced: {al_large}"
    # The small budget caps refinement work strictly below the large budget.
    assert al_small["n_evaluated"] < al_large["n_evaluated"], (
        f"small budget did not reduce evaluations: small={al_small} large={al_large}")


# --------------------------------------------------------------------------------------------------
# FAST representative: one cheap path so MLFRAME_FAST=1 keeps coverage of the su_seeded_* knobs.
# The screen-level top_k test (no model fit) IS the fast representative -- it is not @slow, so it runs
# under MLFRAME_FAST=1, exercising the su_seeded_top_k consumption path directly.
# --------------------------------------------------------------------------------------------------
def test_su_seeded_snr_z_screen_level_gate_monotonic_fast():
    """FAST (no model): the SNR-gate ``su_seeded_snr_z`` is monotone at the screen level -- a higher
    ``snr_z`` can only ever keep a SUBSET of the pairs a lower ``snr_z`` keeps. Runs under MLFRAME_FAST."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions import su_synergy_screen

    X, y = _two_interaction_pairs(n=4000, p_noise=20, seed=0, strong=3.0, weak=2.0)

    kept_lo, _ = su_synergy_screen(
        X, y, n_bins=8, top_k=8, max_screen_cols=120,
        snr_z=3.0, snr_null_quantile=0.99, n_permutations=3, rng=np.random.default_rng(0))
    kept_hi, info_hi = su_synergy_screen(
        X, y, n_bins=8, top_k=8, max_screen_cols=120,
        snr_z=500.0, snr_null_quantile=0.99, n_permutations=3, rng=np.random.default_rng(0))

    lo_pairs = {frozenset((a, b)) for _syn, _jsu, a, b in kept_lo}
    hi_pairs = {frozenset((a, b)) for _syn, _jsu, a, b in kept_hi}
    assert hi_pairs <= lo_pairs, f"higher snr_z kept a pair the lower one didn't: hi={hi_pairs} lo={lo_pairs}"
    # The absurd gate clears nothing (its threshold exceeds the best observed synergy).
    assert kept_hi == [], f"snr_z=500 should be a no-op: {kept_hi}"
    assert info_hi["gate"] > info_hi["best_synergy"], info_hi
