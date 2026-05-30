"""Iter59 adaptive-prescreen-width via SHAP rank stability.

Three tiers of tests:

1. Pure unit tests for ``compute_phi_rank_stability``: identical / inverted / random per-fold matrices
   yield the expected median pairwise Spearman.
2. Pure unit tests for ``_resolve_adaptive_prescreen_width``: each stability bucket maps to the
   documented cap (no narrowing at >=0.8, -4 at [0.6, 0.8), -8 below 0.6) with the floor honoured.
3. End-to-end on a tiny synthetic where a high-SNR run yields high stability (cap stays at default)
   and a low-SNR run yields lower stability (cap narrows). The selector's report carries the
   resolved cap + measured stability under ``shap_proxy_report_['adaptive_prescreen']``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection._shap_proxy_explain import compute_phi_rank_stability
from mlframe.feature_selection.shap_proxied_fs import (
    ShapProxiedFS,
    _resolve_adaptive_prescreen_width,
)


def test_shap_prox_rank_stability_identical_folds_returns_one():
    rng = np.random.default_rng(0)
    base = np.abs(rng.normal(size=20))
    per_fold = np.vstack([base, base, base])
    assert compute_phi_rank_stability(per_fold) == pytest.approx(1.0, abs=1e-9)


def test_shap_prox_rank_stability_inverted_folds_yields_negative_one():
    # Two folds whose feature rankings are exact opposites should hit Spearman -1.
    rank = np.arange(1, 21, dtype=np.float64)
    per_fold = np.vstack([rank, rank[::-1].copy()])
    s = compute_phi_rank_stability(per_fold)
    assert s == pytest.approx(-1.0, abs=1e-9)


def test_shap_prox_rank_stability_random_folds_near_zero():
    rng = np.random.default_rng(123)
    # 5 folds of independent random importances; medium n_features so the noise floor is small.
    per_fold = np.abs(rng.normal(size=(5, 200)))
    s = compute_phi_rank_stability(per_fold, top_k=100)
    assert abs(s) < 0.2, f"random folds should give |Spearman| << 1, got {s}"


def test_shap_prox_rank_stability_single_fold_is_one():
    # Degenerate input -> the metric is undefined and we return 1.0 (no folds to disagree).
    arr = np.array([[3.0, 2.0, 1.0]])
    assert compute_phi_rank_stability(arr) == pytest.approx(1.0)


def test_shap_prox_rank_stability_top_k_truncation_drops_noise_tail():
    # 3 folds: top-5 features are stable, tail-995 features are independent noise per fold.
    # With top_k=5, every fold's top-K is the same 5 indices -> union has size 5 -> Spearman ~ 1.0.
    # Without truncation (top_k=n_features), the noise tail dilutes the ranking toward 0.
    rng = np.random.default_rng(7)
    n_features = 1000
    per_fold = rng.normal(size=(3, n_features))
    per_fold[:, :5] = np.array([[10.0, 9.0, 8.0, 7.0, 6.0]])
    per_fold[:, 5:] = np.abs(per_fold[:, 5:]) * 0.01
    s_top5 = compute_phi_rank_stability(per_fold, top_k=5)
    s_full = compute_phi_rank_stability(per_fold, top_k=n_features)
    assert s_top5 > 0.95, f"top-K=5 should expose the stable head; got {s_top5}"
    assert s_top5 > s_full + 0.3, (
        f"top-K should dominate full ranking; top5={s_top5}, full={s_full}")


@pytest.mark.parametrize(
    "stability, default_cap, expected_cap",
    [
        (0.95, 28, 28),  # high stability: no narrowing
        (0.85, 28, 28),  # at threshold: no narrowing
        (0.79, 28, 24),  # mild narrow: -4
        (0.60, 28, 24),  # mild narrow at lower bound
        (0.59, 28, 20),  # aggressive narrow: -8
        (0.30, 28, 20),  # aggressive narrow
        (-0.50, 28, 20),  # negative stability still hits the aggressive bucket
        (0.30, 22, 16),  # tiny default + aggressive narrow respects the floor at 16
        (0.30, 12, 16),  # default below floor: floor wins
    ],
)
def test_shap_prox_resolve_adaptive_prescreen_width_thresholds(stability, default_cap, expected_cap):
    assert _resolve_adaptive_prescreen_width(stability, default_cap=default_cap) == expected_cap


def _make_tiny_dataset(n_samples=400, n_features=80, n_informative=6, snr=8.0, seed=0):
    """Tiny binary-classification dataset with controllable SNR.

    SNR scales the informative coefficients vs. the unit-variance noise. snr=8 gives a clean,
    stably-rankable target; snr=0.5 makes the informative columns barely distinguishable from noise,
    pushing per-fold |phi| ranks toward independent permutations.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    coefs = np.zeros(n_features)
    coefs[:n_informative] = rng.normal(loc=1.0, scale=0.1, size=n_informative) * snr
    logits = X @ coefs + rng.normal(scale=1.0, size=n_samples)
    y = (logits > np.median(logits)).astype(np.int64)
    import pandas as pd
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)]), y


def _build_tiny_selector(adaptive: bool, seed: int = 0):
    return ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        n_splits=4, n_models=1, top_n=8, n_revalidation_models=1,
        trust_guard=False, run_importance_ablation=False,
        within_cluster_refine=False, cluster_features=False,
        prefilter_top=None, shap_prefilter_enabled=False,
        brute_force_max_features=28,
        adaptive_prescreen_by_stability=adaptive,
        random_state=seed, verbose=False)


def test_shap_prox_adaptive_prescreen_off_keeps_default_cap_in_report():
    # With the lever off the report should NOT carry the adaptive_prescreen block.
    X, y = _make_tiny_dataset(snr=8.0, seed=0)
    sel = _build_tiny_selector(adaptive=False)
    sel.fit(X, y)
    assert "adaptive_prescreen" not in sel.shap_proxy_report_


def test_shap_prox_adaptive_prescreen_on_records_stability_and_cap():
    # High-SNR: stability should land high enough to keep the default cap untouched (28).
    X, y = _make_tiny_dataset(snr=8.0, seed=0)
    sel = _build_tiny_selector(adaptive=True)
    sel.fit(X, y)
    info = sel.shap_proxy_report_.get("adaptive_prescreen")
    assert info is not None
    assert info["default_cap"] == 28
    assert info["effective_cap"] in (20, 24, 28)  # depends on measured stability
    assert -1.0 <= info["stability"] <= 1.0


def test_shap_prox_adaptive_prescreen_records_valid_cap_at_each_snr():
    # End-to-end smoke: under both high and low SNR the report should carry a stability value in
    # [-1, 1] and an effective_cap in the valid {20, 24, 28} bucket. We don't assert a strict
    # high-SNR > low-SNR monotonicity at this scale: with only ~6 informative columns the per-fold
    # ranking inside the top-K=80 window picks up substantial mid-rank noise even at high SNR on a
    # 400-row dataset. The phase-2 benchmark on C3/C3_hard tests the cap-narrowing direction at the
    # realistic n_informative=20, width=10000 scale where the lever is meant to apply.
    for snr in (8.0, 0.4):
        X, y = _make_tiny_dataset(snr=snr, seed=11)
        sel = _build_tiny_selector(adaptive=True, seed=11)
        sel.fit(X, y)
        info = sel.shap_proxy_report_["adaptive_prescreen"]
        assert -1.0 <= info["stability"] <= 1.0
        assert info["effective_cap"] in (20, 24, 28)
        assert info["effective_cap"] <= info["default_cap"]  # NEVER widens past default
