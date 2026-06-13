"""Unit + biz_value tests for the two ShapProxiedFS adaptivity guards.

1. ADAPTIVE TRUST-GUARD ANCHOR BUDGET (``n_anchors="auto"``):
   ``_resolve_adaptive_n_anchors`` = clip(round(6*sqrt(p)), 10, 100). Unit: monotone in p, clamps,
   ~30 at the small-frame calibration point, 100 ceiling on wide. biz_value: on a WIDE frame the
   auto budget lifts the trust-guard fidelity over the legacy fixed-30 (the guard is tighter where
   the fixed budget was sparse).

2. SELF-TUNING (KNEE) PRESCREEN LADDER (``prescreen_ladder_mode="knee"``):
   ``_resolve_knee_prescreen_cap`` keeps the full cap on a dense importance curve and narrows toward
   the knee on a sparse one. Unit: dense keeps, sparse narrows, floor honoured. The default is
   ``"hardcoded"`` (knee bench-rejected as default -- loses held-out AUC on wide-dense); these tests
   pin the OPT-IN behaviour so the recoverable path cannot silently break.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.shap_proxied_fs import (
    ShapProxiedFS,
    _resolve_adaptive_n_anchors,
    _resolve_knee_prescreen_cap,
)


# --------------------------------------------------------------------------------------------------
# Lever 1: adaptive anchor budget -- unit
# --------------------------------------------------------------------------------------------------
def test_anchors_calibration_point_is_about_30():
    # 6*sqrt(25) = 30 -- reproduces the legacy fixed default at the small-frame regime it was tuned on.
    assert _resolve_adaptive_n_anchors(25) == 30


def test_anchors_clamps_low_and_high():
    assert _resolve_adaptive_n_anchors(1) == 10        # floor
    assert _resolve_adaptive_n_anchors(0) == 10        # degenerate -> floor
    assert _resolve_adaptive_n_anchors(10_000) == 100  # ceiling
    assert _resolve_adaptive_n_anchors(2000) == 100    # already ceiling by p~=278


def test_anchors_monotone_in_width():
    vals = [_resolve_adaptive_n_anchors(p) for p in (20, 100, 400, 1000)]
    assert vals == sorted(vals)
    assert vals[0] < vals[-1]


def test_anchors_override_params_via_kwargs():
    # Custom c/lo/hi tune the curve (mirrors the kernel_tuning_cache override path).
    assert _resolve_adaptive_n_anchors(100, c=2.0, lo=5, hi=50) == 20  # 2*sqrt(100)=20
    assert _resolve_adaptive_n_anchors(100, c=2.0, lo=5, hi=15) == 15  # clamped to hi


# --------------------------------------------------------------------------------------------------
# Lever 2: knee prescreen ladder -- unit
# --------------------------------------------------------------------------------------------------
def test_knee_keeps_full_cap_on_dense_signal():
    # Uniform importance -> dense -> no narrowing.
    cap, info = _resolve_knee_prescreen_cap(np.ones(40), default_cap=28)
    assert cap == 28
    assert info["mode"] == "knee"


def test_knee_keeps_full_cap_on_mild_slope():
    # A gentle linear decay is still "dense enough" -> keep full cap.
    cap, _ = _resolve_knee_prescreen_cap(np.linspace(1.0, 0.5, 40), default_cap=28)
    assert cap == 28


def test_knee_narrows_on_sparse_signal():
    # A few dominant columns + long noise tail -> narrow toward the knee, down to the floor.
    imp = np.array([10.0, 9.0, 8.0, 1.0] + [0.01] * 36)
    cap, info = _resolve_knee_prescreen_cap(imp, default_cap=28)
    assert cap < 28
    assert cap >= 16  # floor honoured
    assert info["sparsity"] > 0.10


def test_knee_floor_honoured_on_extreme_sparsity():
    imp = np.array([100.0] + [1e-6] * 39)
    cap, _ = _resolve_knee_prescreen_cap(imp, default_cap=28, floor=16)
    assert cap == 16


def test_knee_handles_degenerate_short_input():
    cap, info = _resolve_knee_prescreen_cap(np.array([1.0, 0.0]), default_cap=28)
    assert cap == 28
    assert info["knee"] is None


# --------------------------------------------------------------------------------------------------
# Constructor defaults
# --------------------------------------------------------------------------------------------------
def test_defaults_anchors_auto_ladder_hardcoded():
    s = ShapProxiedFS()
    assert s.n_anchors == "auto"               # lever 1 flipped to adaptive
    assert s.prescreen_ladder_mode == "hardcoded"  # lever 2 rejected as default


def test_explicit_int_anchors_pins_legacy():
    s = ShapProxiedFS(n_anchors=30)
    assert s.n_anchors == 30


# --------------------------------------------------------------------------------------------------
# biz_value: adaptive anchors tighten the guard on WIDE data
# --------------------------------------------------------------------------------------------------
pytest.importorskip("shap")
pytest.importorskip("xgboost")


def _make_wide(seed, width=2000, n_inf=4, n_red=4, snr=2.5, rho=0.85):
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    X, y, _ = make_regime_dataset(
        n_samples=4000, n_informative=n_inf, n_redundant=n_red, redundancy_rho=rho,
        n_noise=width - n_inf - n_red, snr=snr, task="binary", seed=seed)
    return X, pd.Series(y)


def _fit_fidelity(X, y, n_anchors, seed):
    s = ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto", top_n=12, n_splits=3,
        n_revalidation_models=2, n_anchors=n_anchors, prescreen_ladder_mode="off",
        prefilter_top=2000, random_state=seed, verbose=False, n_jobs=1)
    s.fit(X, y)
    rep = s.shap_proxy_report_
    return rep["trust"]["proxy_fidelity_score"], rep["trust_n_anchors"]["resolved"]


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_biz_val_adaptive_anchors_tighten_guard_on_wide():
    """On a WIDE frame (p=2000) ``n_anchors='auto'`` resolves to the 100 ceiling and the trust-guard
    fidelity is >= the legacy fixed-30 by a measured margin. Bench measured auto>=fixed on 5/6
    seeds x widths; this pins one wide seed where auto strictly wins (fid 0.957 vs 0.828 at seed=2)
    with generous headroom for seed noise."""
    X, y = _make_wide(seed=2)
    fid_auto, n_auto = _fit_fidelity(X, y, "auto", seed=2)
    fid_fix, _ = _fit_fidelity(X, y, 30, seed=2)
    assert n_auto == 100, f"auto budget should hit the ceiling on wide data, got {n_auto}"
    # Measured 0.957 (auto) vs 0.828 (fixed); floor the win at +0.05 for seed headroom.
    assert fid_auto >= fid_fix + 0.05, f"auto fidelity {fid_auto:.4f} not clearly above fixed {fid_fix:.4f}"


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_biz_val_knee_ge_off_on_sparse_holdout():
    """The knee ladder (opt-in) must not HURT a sparse-signal frame: held-out AUC of a refit on the
    knee-selected features >= the no-narrowing selection (it ties on sparse, the regime knee targets).
    Pins the recoverable path so a future change can't quietly regress it."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    import xgboost as xgb

    X, y = _make_wide(seed=0, n_inf=4, n_red=4, snr=2.5)

    def sel_feats(ladder):
        s = ShapProxiedFS(
            classification=True, metric="brier", optimizer="auto", top_n=12, n_splits=3,
            n_revalidation_models=2, n_anchors="auto", prescreen_ladder_mode=ladder,
            prefilter_top=2000, random_state=0, verbose=False, n_jobs=1)
        s.fit(X, y)
        return list(s.selected_features_)

    def holdout_auc(cols):
        Xtr, Xte, ytr, yte = train_test_split(X[cols], y, test_size=0.3, random_state=0, stratify=y)
        m = xgb.XGBClassifier(n_estimators=120, max_depth=4, n_jobs=1, random_state=0,
                              tree_method="hist", verbosity=0)
        m.fit(Xtr, ytr)
        return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])

    auc_knee = holdout_auc(sel_feats("knee"))
    auc_off = holdout_auc(sel_feats("off"))
    assert auc_knee >= auc_off - 0.01, f"knee AUC {auc_knee:.4f} regressed vs off {auc_off:.4f}"
