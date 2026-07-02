"""Unit + biz_value tests for the near-copy increment-learnability precheck in ``_auto_base``.

A base whose ``|corr(base, y)| > base_max_abs_corr_with_y`` (~1.0) is normally dropped: its additive
inverse ``y = T_hat + base`` is carried entirely by ``base`` and extrapolates badly under group shift.
But such a near-copy earns its keep as a COMPOSITE when the residual ``y - linfit(base)`` still carries
LEARNABLE signal from the OTHER features -- then ``y_hat = T_hat + base`` genuinely beats feeding ``base``
as a plain feature. The precheck measures max_j MI(x_j, residual) on a capped screening sample and exempts
the base when it clears a small threshold; a pure-noise residual is dropped as before. Provenance-causal
bases stay exempt regardless of the precheck.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.training.composite.discovery._auto_base import _auto_base
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _make_self(config, X: np.ndarray, feature_names: list[str], target_col: str | None = None):
    name_to_col = {n: i for i, n in enumerate(feature_names)}

    def _build_feature_matrix(df, cols, idx):
        if not cols:
            return np.zeros((idx.size, 0), dtype=np.float64)
        return np.column_stack([X[idx, name_to_col[c]] for c in cols])

    return SimpleNamespace(
        config=config,
        _build_feature_matrix=_build_feature_matrix,
        _hint_strengths_pct=None,
        _target_col=target_col,
    )


def _cfg(**overrides):
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True,
        auto_base_top_k=10,
        auto_base_structural_boost=False,
        auto_base_demote_time_index=False,
        auto_base_demote_spatial_coords=False,
        auto_base_null_perms=0,
        auto_base_dedup_corr_threshold=1.0,
        mi_estimator="bin",
        mi_nbins=16,
        mi_sample_n=None,
        random_state=0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _dominant_base(n, rng, scale=5.0):
    """A latent whose variance dwarfs any additive signal so ``|corr(base, y)| > 0.9995``."""
    return (scale * np.cumsum(rng.standard_normal(n))).astype(np.float64)


def test_near_copy_with_learnable_residual_is_exempted():
    rng = np.random.default_rng(0)
    n = 5000
    base = _dominant_base(n, rng)
    s = rng.standard_normal(n).astype(np.float64)
    y = base + 3.0 * s + rng.standard_normal(n) * 0.2  # residual y-base = 3*s+noise is learnable from ``s``
    feats = ["s", "genuine_base", "noise0", "noise1"]
    X = np.column_stack([s, base, rng.standard_normal(n), rng.standard_normal(n)])
    obj = _make_self(_cfg(), X, feats)
    top = _auto_base(obj, df=None, usable_features=feats, y_train=y, train_idx=np.arange(n))
    assert "genuine_base" in top, f"near-copy with learnable residual must be kept, got {top}"


def test_near_copy_with_noise_residual_is_dropped():
    rng = np.random.default_rng(1)
    n = 5000
    base = _dominant_base(n, rng)
    y = base + rng.standard_normal(n) * 0.2  # residual y-base ~ pure noise; composite adds nothing
    feats = ["copy_base", "noise0", "noise1", "noise2"]
    X = np.column_stack([base, rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)])
    obj = _make_self(_cfg(), X, feats)
    top = _auto_base(obj, df=None, usable_features=feats, y_train=y, train_idx=np.arange(n))
    assert "copy_base" not in top, f"noise-residual near-copy must be dropped, got {top}"


def test_precheck_off_drops_all_near_copies():
    rng = np.random.default_rng(0)
    n = 5000
    base = _dominant_base(n, rng)
    s = rng.standard_normal(n).astype(np.float64)
    y = base + 3.0 * s + rng.standard_normal(n) * 0.2
    feats = ["s", "genuine_base", "noise0", "noise1"]
    X = np.column_stack([s, base, rng.standard_normal(n), rng.standard_normal(n)])
    obj = _make_self(_cfg(near_copy_increment_learnability_precheck=False), X, feats)
    top = _auto_base(obj, df=None, usable_features=feats, y_train=y, train_idx=np.arange(n))
    assert "genuine_base" not in top, f"precheck OFF must drop the learnable near-copy too, got {top}"


def test_causal_provenance_near_copy_stays_exempt_regardless():
    rng = np.random.default_rng(2)
    n = 5000
    base = _dominant_base(n, rng)
    y = base + rng.standard_normal(n) * 0.2  # noise residual: precheck alone would DROP it
    feats = ["y_prev", "noise0", "noise1"]  # ``{target}_prev`` -> provenance-causal, exempt by name
    X = np.column_stack([base, rng.standard_normal(n), rng.standard_normal(n)])
    obj = _make_self(_cfg(), X, feats, target_col="y")
    top = _auto_base(obj, df=None, usable_features=feats, y_train=y, train_idx=np.arange(n))
    assert "y_prev" in top, f"provenance-causal near-copy must stay exempt regardless of precheck, got {top}"


def test_degenerate_no_other_features_drops_near_copy():
    rng = np.random.default_rng(3)
    n = 300  # tiny n + only the base column present -> nothing to learn the residual from
    base = _dominant_base(n, rng)
    y = base + rng.standard_normal(n) * 0.2
    feats = ["only_base"]
    X = base.reshape(-1, 1)
    obj = _make_self(_cfg(), X, feats)
    top = _auto_base(obj, df=None, usable_features=feats, y_train=y, train_idx=np.arange(n))
    assert "only_base" not in top, f"near-copy with no other features to learn from must be dropped, got {top}"


def test_biz_val_learnable_composite_reconstruction_beats_raw():
    """biz_value: the genuine composite (near-copy base + learnable residual) is EXEMPTED and its
    reconstruction ``base + fit(other_features -> y-base)`` beats a raw-y-mean model by a wide margin,
    while a noise-residual near-copy is dropped. Quantitative floor: composite RMSE <= 0.35 * raw RMSE."""
    rng = np.random.default_rng(11)
    n = 5000
    base = _dominant_base(n, rng)
    s = rng.standard_normal(n).astype(np.float64)
    y = base + 3.0 * s + rng.standard_normal(n) * 0.2

    # 1. The genuine composite base survives the near-copy drop.
    feats = ["s", "genuine_base", "noise0", "noise1"]
    X = np.column_stack([s, base, rng.standard_normal(n), rng.standard_normal(n)])
    obj = _make_self(_cfg(), X, feats)
    top = _auto_base(obj, df=None, usable_features=feats, y_train=y, train_idx=np.arange(n))
    assert "genuine_base" in top

    # 2. The composite reconstruction genuinely beats a raw-y model on a holdout.
    tr, te = np.arange(n // 2), np.arange(n // 2, n)
    resid = y - base
    a = float(np.cov(s[tr], resid[tr])[0, 1] / s[tr].var())
    b = float(resid[tr].mean() - a * s[tr].mean())
    y_hat_composite = base[te] + (a * s[te] + b)
    rmse_composite = float(np.sqrt(np.mean((y[te] - y_hat_composite) ** 2)))
    rmse_raw = float(np.sqrt(np.mean((y[te] - y[tr].mean()) ** 2)))
    assert rmse_composite <= 0.35 * rmse_raw, (
        f"composite RMSE {rmse_composite:.3f} should beat raw {rmse_raw:.3f} by a wide margin"
    )

    # 3. A noise-residual near-copy is still dropped (the precheck does not admit noise).
    y_noise = base + rng.standard_normal(n) * 0.2
    feats2 = ["copy_base", "noise0", "noise1"]
    X2 = np.column_stack([base, rng.standard_normal(n), rng.standard_normal(n)])
    obj2 = _make_self(_cfg(), X2, feats2)
    top2 = _auto_base(obj2, df=None, usable_features=feats2, y_train=y_noise, train_idx=np.arange(n))
    assert "copy_base" not in top2
