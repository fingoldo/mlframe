"""biz_value: WAIC transform-validation tie-break in CompositeTargetDiscovery.

The ``transform_waic_validation_enabled`` flag folds an out-of-fold information-criterion score into the tiny-model
rerank: within a relative-RMSE noise band, the transform whose tiny-CV residuals generalise better (higher WAIC) is
ranked above an overfit competitor that tiny-CV RMSE alone cannot separate. These tests pin (a) the wiring -- the score
map is populated only when the flag is on -- and (b) the discrimination -- a genuinely-generalising transform earns a
higher WAIC than a deliberately-overfit one on the same screen sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.composite.discovery._eval_waic import compute_transform_waic
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _make_config(**overrides):
    defaults = dict(
        enabled=True,
        base_candidates=["base"],
        transforms=("diff", "ratio", "logratio", "linear_residual"),
        top_k_after_mi=4,
        top_m_after_tiny=2,
        mi_sample_n=2000,
        tiny_model_sample_n=2000,
        eps_mi_gain=0.001,
        random_state=42,
        require_beats_raw_baseline=False,
        fail_on_no_gain="fallback_raw",
    )
    defaults.update(overrides)
    return CompositeTargetDiscoveryConfig(**defaults)


def _linear_residual_dataset(seed: int = 0, n: int = 2000):
    """y is base + a feature-driven residual: ``linear_residual`` is the generalising target transform."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(2.0, 6.0, n)
    other = rng.standard_normal(n)
    y = 1.5 * base + 0.8 * other + rng.standard_normal(n) * 0.25
    return pd.DataFrame({"base": base, "other": other, "y": y})


def _fit(df, config):
    n = len(df)
    train_idx = np.arange(0, int(0.8 * n))
    val_idx = np.arange(int(0.8 * n), n)
    disc = CompositeTargetDiscovery(config)
    disc.fit(df, target_col="y", feature_cols=["base", "other"], train_idx=train_idx, val_idx=val_idx)
    return disc


def test_waic_scores_populated_only_when_flag_enabled():
    df = _linear_residual_dataset()
    on = _fit(df, _make_config(transform_waic_validation_enabled=True))
    off = _fit(df, _make_config(transform_waic_validation_enabled=False))

    on_scores = getattr(on, "_tiny_rerank_waic_scores", {}) or {}
    off_scores = getattr(off, "_tiny_rerank_waic_scores", {}) or {}
    assert on_scores, "flag ON must populate _tiny_rerank_waic_scores for the reranked specs"
    assert all(np.isfinite(v) for v in on_scores.values()), f"WAIC scores must be finite: {on_scores}"
    assert not off_scores, f"flag OFF must not compute WAIC; got {off_scores}"


def test_waic_prefers_generalising_over_overfit_transform():
    """A genuinely-generalising target (a smooth feature-linear signal) earns a higher WAIC than a target the same
    features can only memorise (pure i.i.d. noise of matched scale): the noise target's out-of-fold density collapses
    and its effective-complexity penalty grows, so WAIC ranks it strictly below the signal target on identical X."""
    rng = np.random.default_rng(7)
    n = 2000
    x = np.column_stack([rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)])
    signal = x @ np.array([1.3, -0.7, 0.5]) + rng.standard_normal(n) * 0.2
    noise = rng.standard_normal(n) * float(np.std(signal))  # matched-scale, feature-independent -> only memorisable

    w_signal = compute_transform_waic(signal, x, n_folds=4, random_state=0)
    w_noise = compute_transform_waic(noise, x, n_folds=4, random_state=0)
    assert w_signal.valid and w_noise.valid
    assert w_signal.waic > w_noise.waic, (
        f"generalising signal target should out-WAIC the memorise-only noise target; "
        f"signal={w_signal.waic:.3f} noise={w_noise.waic:.3f}"
    )
