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
from mlframe.feature_selection.shap_proxied_fs._shap_proxied_resolvers import noise_floor_rescue_keep_set


# --------------------------------------------------------------------------------------------------
# Lever 1: adaptive anchor budget -- unit
# --------------------------------------------------------------------------------------------------
def test_anchors_calibration_point_is_about_30():
    # 6*sqrt(25) = 30 -- reproduces the legacy fixed default at the small-frame regime it was tuned on.
    """Anchors calibration point is about 30."""
    assert _resolve_adaptive_n_anchors(25) == 30


def test_anchors_clamps_low_and_high():
    """Anchors clamps low and high."""
    assert _resolve_adaptive_n_anchors(1) == 10  # floor
    assert _resolve_adaptive_n_anchors(0) == 10  # degenerate -> floor
    assert _resolve_adaptive_n_anchors(10_000) == 100  # ceiling
    assert _resolve_adaptive_n_anchors(2000) == 100  # already ceiling by p~=278


def test_anchors_monotone_in_width():
    """Anchors monotone in width."""
    vals = [_resolve_adaptive_n_anchors(p) for p in (20, 100, 400, 1000)]
    assert vals == sorted(vals)
    assert vals[0] < vals[-1]


def test_anchors_override_params_via_kwargs():
    # Custom c/lo/hi tune the curve (mirrors the kernel_tuning_cache override path).
    """Anchors override params via kwargs."""
    assert _resolve_adaptive_n_anchors(100, c=2.0, lo=5, hi=50) == 20  # 2*sqrt(100)=20
    assert _resolve_adaptive_n_anchors(100, c=2.0, lo=5, hi=15) == 15  # clamped to hi


# --------------------------------------------------------------------------------------------------
# Lever 2: knee prescreen ladder -- unit
# --------------------------------------------------------------------------------------------------
def test_knee_keeps_full_cap_on_dense_signal():
    # Uniform importance -> dense -> no narrowing.
    """Knee keeps full cap on dense signal."""
    cap, info = _resolve_knee_prescreen_cap(np.ones(40), default_cap=28)
    assert cap == 28
    assert info["mode"] == "knee"


def test_knee_keeps_full_cap_on_mild_slope():
    # A gentle linear decay is still "dense enough" -> keep full cap.
    """Knee keeps full cap on mild slope."""
    cap, _ = _resolve_knee_prescreen_cap(np.linspace(1.0, 0.5, 40), default_cap=28)
    assert cap == 28


def test_knee_narrows_on_sparse_signal():
    # A few dominant columns + long noise tail -> narrow toward the knee, down to the floor.
    """Knee narrows on sparse signal."""
    imp = np.array([10.0, 9.0, 8.0, 1.0] + [0.01] * 36)
    cap, info = _resolve_knee_prescreen_cap(imp, default_cap=28)
    assert cap < 28
    assert cap >= 16  # floor honoured
    assert info["sparsity"] > 0.10


def test_knee_floor_honoured_on_extreme_sparsity():
    """Knee floor honoured on extreme sparsity."""
    imp = np.array([100.0] + [1e-6] * 39)
    cap, _ = _resolve_knee_prescreen_cap(imp, default_cap=28, floor=16)
    assert cap == 16


def test_knee_handles_degenerate_short_input():
    """Knee handles degenerate short input."""
    cap, info = _resolve_knee_prescreen_cap(np.array([1.0, 0.0]), default_cap=28)
    assert cap == 28
    assert info["knee"] is None


# --------------------------------------------------------------------------------------------------
# Lever 2 bug fix: noise-floor rescue -- the knee only reads the shape of the top-`default_cap`
# window, so a few dominant (strong) columns make ANY frame look front-loaded even when the tail past
# the knee carries real, weaker-but-genuine signal (not noise). Without the rescue this silently drops
# real features whenever the pipeline opts into the knee ladder. Confirmed empirically in-session: on
# a synthetic 8-strong/8-weak/2984-noise importance vector, the pre-fix cap narrowed to 16 (dropping
# all 8 weak features at ranks 8-15); the noise-floor rescue widens back to 28 (covers all 8).
# --------------------------------------------------------------------------------------------------
def test_knee_rescue_recovers_weak_signal_past_the_knee():
    """Knee rescue recovers weak signal past the knee."""
    rng = np.random.default_rng(0)
    strong = rng.uniform(2.0, 3.0, 8)
    weak = rng.uniform(0.15, 0.3, 8)
    noise = np.abs(rng.normal(0, 0.01, 2984))
    importance = np.concatenate([strong, weak, noise])

    cap, info = _resolve_knee_prescreen_cap(importance, default_cap=28)

    order = np.argsort(-importance)
    weak_ranks = sorted(int(np.where(order == i)[0][0]) for i in range(8, 16))
    assert all(r < cap for r in weak_ranks), f"weak feature ranks {weak_ranks} not all covered by cap={cap}"
    assert info["rescued"] > 0
    assert info["noise_floor"] > 0


def test_knee_rescue_is_a_pure_widening_never_narrows_below_unrescued_cap():
    # A frame with NO weak tail (just strong + noise, no rescue candidates): rescue must not
    # perturb the plain knee-narrowed cap.
    """Knee rescue is a pure widening never narrows below unrescued cap."""
    imp = np.array([10.0, 9.0, 8.0, 1.0] + [0.001] * 36)
    cap, info = _resolve_knee_prescreen_cap(imp, default_cap=28)
    assert info["rescued"] == 0
    assert cap < 28  # still narrows -- the rescue found nothing to widen for


def test_knee_rescue_noise_floor_uses_full_tail_not_just_head():
    # Regression pin: noise_floor must be derived from the FULL importance vector's tail, not just
    # the top-default_cap head -- otherwise a wide frame with many noise columns beyond the head
    # would compute an inflated (head-only) floor and under-rescue.
    """Knee rescue noise floor uses full tail not just head."""
    rng = np.random.default_rng(1)
    strong = rng.uniform(2.0, 3.0, 4)
    weak = rng.uniform(0.1, 0.2, 4)
    noise_small = np.abs(rng.normal(0, 0.005, 20))  # inside the default_cap head window
    noise_large_tail = np.abs(rng.normal(0, 0.005, 5000))  # outside the head window
    importance = np.concatenate([strong, weak, noise_small, noise_large_tail])

    cap, _info = _resolve_knee_prescreen_cap(importance, default_cap=28)
    order = np.argsort(-importance)
    weak_ranks = sorted(int(np.where(order == i)[0][0]) for i in range(4, 8))
    assert all(r < cap for r in weak_ranks), f"weak feature ranks {weak_ranks} not covered by cap={cap}"


# --------------------------------------------------------------------------------------------------
# Shared noise-floor rescue primitive (finding 5: flat default-path prescreen, _shap_proxied_fit.py)
# --------------------------------------------------------------------------------------------------
def test_noise_floor_rescue_keep_set_recovers_weak_signal_past_the_cap():
    # Properly stressing fixture: 8 weak features deliberately ranked at positions 33-40, past a
    # cap of 28, among 25 spuriously-elevated noise columns that outrank them by chance. Confirmed
    # empirically in-session: naive top-28 cut covers only 5/8 weak features; the rescue covers 8/8.
    """Noise floor rescue keep set recovers weak signal past the cap."""
    rng = np.random.default_rng(0)
    strong = rng.uniform(2.0, 3.0, 8)
    noise_between = np.abs(rng.normal(0, 0.5, 25))
    weak = rng.uniform(0.15, 0.3, 8)
    noise_rest = np.abs(rng.normal(0, 0.01, 71))
    importance = np.concatenate([strong, noise_between, weak, noise_rest])
    weak_idx = set(range(33, 41))

    top_keep = np.argsort(-importance)[:28]
    pre_rescue_covered = len(weak_idx & set(int(i) for i in top_keep))
    keep = noise_floor_rescue_keep_set(importance, top_keep)
    post_rescue_covered = len(weak_idx & keep)

    assert pre_rescue_covered < 8, "fixture should reproduce the pre-fix bug (a real recall gap)"
    assert post_rescue_covered == 8, f"rescue should recover all weak features, got {post_rescue_covered}/8"


def test_noise_floor_rescue_keep_set_never_drops_original_keep():
    """Noise floor rescue keep set never drops original keep."""
    rng = np.random.default_rng(1)
    importance = np.abs(rng.normal(0, 1, 50))
    top_keep = np.argsort(-importance)[:10]
    keep = noise_floor_rescue_keep_set(importance, top_keep)
    assert set(int(i) for i in top_keep) <= keep


def test_noise_floor_rescue_keep_set_is_noop_when_nothing_clears_the_floor():
    # A frame with a hard cliff (top-K all strong, rest pure near-zero noise): rescue adds nothing.
    """Noise floor rescue keep set is noop when nothing clears the floor."""
    imp = np.array([10.0, 9.0, 8.0] + [1e-6] * 47)
    top_keep = np.argsort(-imp)[:3]
    keep = noise_floor_rescue_keep_set(imp, top_keep)
    assert keep == set(int(i) for i in top_keep)


def test_noise_floor_rescue_keep_set_handles_empty_and_degenerate_input():
    """Noise floor rescue keep set handles empty and degenerate input."""
    keep = noise_floor_rescue_keep_set(np.array([]), np.array([], dtype=np.int64))
    assert keep == set()
    keep = noise_floor_rescue_keep_set(np.array([np.nan, np.nan]), np.array([0], dtype=np.int64))
    assert keep == {0}


# --------------------------------------------------------------------------------------------------
# Constructor defaults
# --------------------------------------------------------------------------------------------------
def test_defaults_anchors_auto_ladder_hardcoded():
    """Defaults anchors auto ladder hardcoded."""
    s = ShapProxiedFS()
    assert s.n_anchors == "auto"  # lever 1 flipped to adaptive
    assert s.prescreen_ladder_mode == "hardcoded"  # lever 2 rejected as default


def test_explicit_int_anchors_pins_legacy():
    """Explicit int anchors pins legacy."""
    s = ShapProxiedFS(n_anchors=30)
    assert s.n_anchors == 30


# --------------------------------------------------------------------------------------------------
# biz_value: adaptive anchors tighten the guard on WIDE data
# --------------------------------------------------------------------------------------------------
pytest.importorskip("shap")
pytest.importorskip("xgboost")


def _make_wide(seed, width=2000, n_inf=4, n_red=4, snr=2.5, rho=0.85):
    """Make wide."""
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    X, y, _ = make_regime_dataset(
        n_samples=4000, n_informative=n_inf, n_redundant=n_red, redundancy_rho=rho, n_noise=width - n_inf - n_red, snr=snr, task="binary", seed=seed
    )
    return X, pd.Series(y)


def _fit_fidelity(X, y, n_anchors, seed):
    """Fit fidelity."""
    s = ShapProxiedFS(
        classification=True,
        metric="brier",
        optimizer="auto",
        top_n=12,
        n_splits=3,
        n_revalidation_models=2,
        n_anchors=n_anchors,
        prescreen_ladder_mode="off",
        prefilter_top=2000,
        random_state=seed,
        verbose=False,
        n_jobs=1,
    )
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
        """Sel feats."""
        s = ShapProxiedFS(
            classification=True,
            metric="brier",
            optimizer="auto",
            top_n=12,
            n_splits=3,
            n_revalidation_models=2,
            n_anchors="auto",
            prescreen_ladder_mode=ladder,
            prefilter_top=2000,
            random_state=0,
            verbose=False,
            n_jobs=1,
        )
        s.fit(X, y)
        return list(s.selected_features_)

    def holdout_auc(cols):
        """Holdout auc."""
        Xtr, Xte, ytr, yte = train_test_split(X[cols], y, test_size=0.3, random_state=0, stratify=y)
        m = xgb.XGBClassifier(n_estimators=120, max_depth=4, n_jobs=1, random_state=0, tree_method="hist", verbosity=0)
        m.fit(Xtr, ytr)
        return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])

    auc_knee = holdout_auc(sel_feats("knee"))
    auc_off = holdout_auc(sel_feats("off"))
    assert auc_knee >= auc_off - 0.01, f"knee AUC {auc_knee:.4f} regressed vs off {auc_off:.4f}"
