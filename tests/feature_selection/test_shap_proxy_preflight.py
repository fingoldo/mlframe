"""biz_val for the pre-flight decision rule (Batch C): it must correctly distinguish the regime where
ShapProxiedFS shines (additive, high-SNR, well-fit) from where it struggles (pure interaction / XOR),
from cheap statistics alone -- so a user can decide whether/how to run it before paying the full cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")


def _additive(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    inf = rng.normal(size=(n, 5))
    noise = rng.normal(size=(n, 5))
    X = pd.DataFrame(np.column_stack([inf, noise]), columns=[f"f{i}" for i in range(10)])
    logit = 1.0 * inf[:, 0] + 0.9 * inf[:, 1] - 0.8 * inf[:, 2] + 0.6 * inf[:, 3] + 0.4 * inf[:, 4]
    y = (logit > 0).astype(int)  # high-SNR additive -> the favourable regime
    return X, y


def test_preflight_corr_gate_bit_identical_under_iter25_cap():
    """Iter25 booster cap (max_rows + n_estimators) must NOT alter ``max_abs_corr`` -- the
    redundancy gate is driven by an independent ``max_rows_corr`` knob (default 5000, matches the
    legacy implementation's row sample) so the corr column sample's rng draw is bit-for-bit identical
    to the pre-iter25 path. Fast unit test (small synthetic; xgboost ranking call dominates but on
    n=300 / f=20 runs <5s).
    """
    from mlframe.feature_selection._shap_proxy_preflight import preflight

    rng = np.random.default_rng(0)
    n = 300
    z = rng.normal(size=(n, 3))
    # 9 redundant copies + 8 noise -> max|corr| well above 0.7 floor.
    refl = np.hstack([z[:, [k]] + 0.05 * rng.normal(size=(n, 3)) for k in range(3)])
    noise = rng.normal(size=(n, 8))
    X = pd.DataFrame(np.column_stack([refl, noise]),
                     columns=[f"f{i}" for i in range(refl.shape[1] + noise.shape[1])])
    y = (z[:, 0] + 0.5 * z[:, 1] + 0.3 * rng.normal(size=n) > 0).astype(int)

    legacy = preflight(X, y, classification=True, random_state=0,
                      max_rows=5000, n_estimators=150)
    capped = preflight(X, y, classification=True, random_state=0,
                      max_rows=2000, n_estimators=100)
    # Redundancy gate output bit-for-bit identical (corr-pass determinism preserved).
    assert capped["diagnostics"]["max_abs_corr"] == legacy["diagnostics"]["max_abs_corr"]
    # Recommendation invariant on this small synthetic.
    assert capped["recommendation"] == legacy["recommendation"]


def _xor(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8))
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(int)  # pure interaction
    return pd.DataFrame(x, columns=[f"f{i}" for i in range(8)]), y


@pytest.mark.slow
def test_biz_val_preflight_favours_additive_flags_interaction():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    add = ShapProxiedFS.preflight(*_additive(), classification=True, random_state=0)
    xor = ShapProxiedFS.preflight(*_xor(), classification=True, random_state=0)

    # Additive high-SNR -> high additive ratio, recommended to run.
    assert add["diagnostics"]["additive_ratio"] > 0.7, add["diagnostics"]
    assert add["recommendation"] == "run", add

    # XOR -> low additive ratio (a depth-1 stump can't model it), flagged interaction-heavy.
    assert xor["diagnostics"]["additive_ratio"] < 0.6, xor["diagnostics"]
    assert xor["recommendation"] in ("caution", "fallback")
    assert any("interaction" in r for r in xor["reasons"]), xor["reasons"]
    assert "enable interaction_aware=True" in xor["suggestions"]


@pytest.mark.slow
def test_biz_val_preflight_flags_redundancy_and_width():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    rng = np.random.default_rng(0)
    n = 2000
    z = rng.normal(size=(n, 3))
    refl = np.hstack([z[:, [k]] + 0.1 * rng.normal(size=(n, 20)) for k in range(3)])  # 60 redundant
    X = pd.DataFrame(refl, columns=[f"f{i}" for i in range(refl.shape[1])])
    y = (z[:, 0] + 0.8 * z[:, 1] - 0.7 * z[:, 2] + 0.3 * rng.normal(size=n) > 0).astype(int)
    rep = ShapProxiedFS.preflight(X, y, classification=True, random_state=0)
    assert rep["diagnostics"]["max_abs_corr"] >= 0.7
    assert "enable cluster_features=True" in rep["suggestions"]
    assert rep["diagnostics"]["n_features"] > 40


# iter25 cap: the preflight booster CV calls were dominating the gate's wall-clock (live test
# 2026-05-28 width=1000 / n_rows=5000 measured preflight=77s vs full fit=86s). Capping
# ``n_estimators=100`` + booster row subsample to 2000 cuts ~30% off the gate on the same regime
# without flipping the recommendation -- this test pins both invariants.

# Match the iter17 calibration regimes (additive/redundancy/interaction/xor/noise_heavy) -- these are
# the gate's calibration anchor and must produce identical recommendations under the cap.
_ITER17_REGIMES = (
    dict(name="additive_highSNR",
         kwargs=dict(n_samples=2000, n_informative=8, n_redundant=0, n_noise=400,
                     interaction_order=0, interaction_strength=0.0, snr=5.0,
                     task="binary", seed=0)),
    dict(name="redundancy_heavy",
         kwargs=dict(n_samples=1200, n_informative=8, n_redundant=24, redundancy_rho=0.9,
                     n_noise=400, snr=2.5, task="binary", seed=1)),
    dict(name="interaction_heavy",
         kwargs=dict(n_samples=1200, n_informative=8, n_redundant=0, n_noise=400,
                     interaction_order=2, interaction_strength=0.7, snr=3.0,
                     task="binary", seed=2)),
    dict(name="xor_interaction",
         kwargs=dict(n_samples=1500, n_informative=6, n_redundant=0, n_noise=400,
                     interaction_order="xor", interaction_strength=0.9, snr=3.0,
                     task="binary", seed=4)),
    dict(name="noise_heavy",
         kwargs=dict(n_samples=1500, n_informative=8, n_redundant=0, n_noise=1200,
                     snr=2.0, task="binary", seed=3)),
)


@pytest.mark.slow
@pytest.mark.parametrize("regime", _ITER17_REGIMES, ids=lambda r: r["name"])
def test_preflight_capped_recommendation_matches_legacy_iter17(regime):
    """Iter25 cap (max_rows=2000, n_estimators=100) must produce identical recommendation to the
    legacy params (max_rows=5000, n_estimators=150) across the iter17 calibration regimes. The
    ``additive_ratio`` and ``full_model_fit`` are allowed to differ slightly (smaller cap = slightly
    lower CV score on average) but the gate decision and the suggestion set must be stable.
    """
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y, _ = make_regime_dataset(**regime["kwargs"])
    legacy = ShapProxiedFS.preflight(X, y, classification=True, random_state=0,
                                     max_rows=5000, n_estimators=150)
    capped = ShapProxiedFS.preflight(X, y, classification=True, random_state=0,
                                     max_rows=2000, n_estimators=100)

    assert capped["recommendation"] == legacy["recommendation"], (
        f"regime {regime['name']}: cap flipped recommendation "
        f"{legacy['recommendation']!r} -> {capped['recommendation']!r}; "
        f"ratios legacy={legacy['diagnostics']['additive_ratio']:.3f} "
        f"capped={capped['diagnostics']['additive_ratio']:.3f}")
    assert sorted(capped["suggestions"]) == sorted(legacy["suggestions"]), (
        f"regime {regime['name']}: suggestion set changed under cap "
        f"legacy={legacy['suggestions']} capped={capped['suggestions']}")
    # max_abs_corr uses an independent ``max_rows_corr`` knob (default 5000) so the redundancy gate
    # remains bit-for-bit identical to the legacy pre-iter25 implementation.
    assert capped["diagnostics"]["max_abs_corr"] == legacy["diagnostics"]["max_abs_corr"]


@pytest.mark.slow
def test_biz_val_preflight_under_30s_at_width_1000():
    """Live-test regime (2026-05-28): preflight at width=1000 / n_rows=5000 must complete under 30s.

    Pre-iter25 the gate took 77s on the user's machine vs the 86s full fit it was supposed to gate
    cheaply; iter25 caps the booster CV calls (n_estimators=150 -> 100, max_rows=5000 -> 2000) which
    restores the cheap-check semantics. The 30s gate is conservative: typical worktree measurement
    is 20-22s; budget headroom accounts for slower CI machines.
    """
    import time
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y, _ = make_regime_dataset(
        n_samples=5000, n_informative=12, n_redundant=8, n_noise=980,
        snr=8.0, task="binary", seed=0,
    )
    t0 = time.time()
    rep = ShapProxiedFS.preflight(X, y, classification=True, random_state=0)
    elapsed = time.time() - t0
    # Recommendation should still be "run" on this favourable wide regime (high SNR, low informative
    # density relative to noise but additive signal dominates).
    assert rep["recommendation"] in ("run", "caution"), rep
    assert elapsed < 30.0, (
        f"preflight at width=1000 took {elapsed:.1f}s, exceeds 30s budget; "
        f"check booster cap (n_estimators) is in effect.")
