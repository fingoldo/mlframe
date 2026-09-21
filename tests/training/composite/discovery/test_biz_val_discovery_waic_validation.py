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
    """Make config."""
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
    """Fit."""
    n = len(df)
    train_idx = np.arange(0, int(0.8 * n))
    val_idx = np.arange(int(0.8 * n), n)
    disc = CompositeTargetDiscovery(config)
    disc.fit(df, target_col="y", feature_cols=["base", "other"], train_idx=train_idx, val_idx=val_idx)
    return disc


_WIDE = dict(top_m_after_tiny=10, top_k_after_mi=20, transforms=("diff", "ratio", "logratio", "linear_residual", "additive_residual", "monotonic_residual"))


def test_waic_scores_populated_only_when_flag_enabled():
    """With the flag on, the specs sitting in an RMSE tie near the top get a finite WAIC; with it off, none do.

    WAIC only re-orders specs inside a noise band that reaches the top-m window, so those are the only ones scored:
    ``diff`` and ``additive_residual`` tie here, and a spread-out spec's score could never change the result.
    """
    df = _linear_residual_dataset()
    on = _fit(df, _make_config(transform_waic_validation_enabled=True, **_WIDE))
    off = _fit(df, _make_config(transform_waic_validation_enabled=False, **_WIDE))

    on_scores = getattr(on, "_tiny_rerank_waic_scores", {}) or {}
    off_scores = getattr(off, "_tiny_rerank_waic_scores", {}) or {}
    assert on_scores, "flag ON must score the specs tied near the top of the rerank"
    assert all(np.isfinite(v) for v in on_scores.values()), f"WAIC scores must be finite: {on_scores}"
    assert not off_scores, f"flag OFF must not compute WAIC; got {off_scores}"


def test_only_tied_specs_near_the_top_are_scored():
    """Every scored spec has a partner within the 2% band, and the untied leaders are left unscored."""
    df = _linear_residual_dataset()
    on = _fit(df, _make_config(transform_waic_validation_enabled=True, **_WIDE))
    scores = on.tiny_rerank_scores_ or {}
    scored = set(on._tiny_rerank_waic_scores)
    for name in scored:
        partners = [o for o in scores if o != name and abs(scores[o] - scores[name]) <= 0.02 * abs(min(scores[o], scores[name]))]
        assert partners, f"{name} was WAIC-scored without being tied to any other spec"
    assert len(scored) < len(scores), "specs outside every tie must not be scored"


def test_rmse_bands_group_consecutive_ties():
    """Bands are runs within 2% of their first member's RMSE; a non-finite score always stands alone."""
    from mlframe.training.composite.discovery._tiny_rerank_waic import rmse_bands

    scores = [1.00, 1.01, 1.05, 1.06, 2.0, float("inf")]
    assert rmse_bands(list(range(6)), scores, 0.02) == [[0, 1], [2, 3], [4], [5]]


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
    assert (
        w_signal.waic > w_noise.waic
    ), f"generalising signal target should out-WAIC the memorise-only noise target; signal={w_signal.waic:.3f} noise={w_noise.waic:.3f}"
