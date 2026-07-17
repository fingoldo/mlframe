"""Regression test for the TVT-Identity-MLP collapse cascade.

Production TVT run, 2026-05-22 (4M rows, 25 features, group-aware
split on 771 wells, Identity-activation MLP downstream): MLP collapsed
to R^2=-326 on test while Ridge nailed R^2=1.00 on the same data.
Root cause: Identity-MLP extrapolates linearly into unseen groups,
and the composite-discovery default gates were filtering out the
only composite (pure-lag residualisation ``y - alpha*lag_y``) that
would have bounded the MLP's prediction range to ~residual_noise.

Six defaults were flipped to let pure-lag composites survive to the
per-target training phase:

* ``composite_skip_when_raw_dominates_ratio: 0.03 -> 0.0``
* ``composite_skip_when_ablation_delta_pct: 500.0 -> 0.0``
* ``eps_mi_gain: -0.5 -> -10.0``
* ``top_k_after_mi: 8 -> 32``
* ``require_beats_raw_baseline: True -> False``
* ``per_bin_n_bins: 5 -> 0``

This test synthesises a strict AR(1) regime (the simplest stand-in for
the TVT regression target) and verifies the discovery yields a
pure-lag composite end-to-end with DEFAULTS ONLY (no gate overrides).
Reverting any of the six defaults could individually or jointly
re-introduce the production MLP-collapse failure on TVT-shaped data;
the per-default unit-assertion tests in
``test_composite_skip_when_raw_dominates.py`` complement this
integration-level check.

Why a single end-to-end test instead of per-gate inversion tests:
the gate-trigger conditions on synthetic data don't fully reproduce
the production TVT mechanic (which involves a 25-feature pipeline,
LGBM-proxy folds on real heavy-tail residuals, and per-bin variance
collapse). The per-default unit tests in the sibling file already
lock the constants; this test locks the joint behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _tvt_like_synthetic(n: int = 2000, seed: int = 0):
    """Build (df, target_col, feature_cols, train_idx).

    AR(1) target with autocorr ~ 0.999 (matches TVT lag1_corr=1.0000
    per-group), plus a noisy "global feature" loosely correlated
    with y and a pure-noise feature. The lag feature DOMINATES the
    raw model -- exactly the regime where pure-lag residualisation
    is the right move for downstream models with unbounded outputs.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float64)
    y[0] = rng.normal(0, 10)
    for i in range(1, n):
        y[i] = 0.999 * y[i - 1] + rng.normal(scale=1.0)
    lag_y = np.r_[y[0], y[:-1]].astype(np.float64)
    x_corr = (y + rng.normal(scale=5.0, size=n)).astype(np.float64)
    x_noise = rng.normal(size=n).astype(np.float64)
    df = pd.DataFrame(
        {
            "lag_y": lag_y,
            "x_corr": x_corr,
            "x_noise": x_noise,
            "y": y,
        }
    )
    feature_cols = ["lag_y", "x_corr", "x_noise"]
    train_idx = np.arange(int(0.8 * n))
    return df, "y", feature_cols, train_idx


def _has_pure_lag_spec(disc) -> bool:
    """True iff at least one spec uses ``lag_y`` as base AND
    a residualisation transform (no Y-only unary transform)."""
    pure_lag_transforms = {
        "linear_residual",
        "monotonic_residual",
        "diff",
        "chain_linres_cbrt",
        "chain_linres_yj",
        "chain_monres_cbrt",
        "chain_monres_yj",
    }
    for s in disc.specs_:
        if getattr(s, "base_column", None) == "lag_y" and getattr(s, "transform_name", None) in pure_lag_transforms:
            return True
    return False


class TestTVTPureLagSurvivesDefaults:
    """Load-bearing end-to-end test. Runs the FULL discovery pipeline
    (including Phase B tiny-rerank) on AR(1) synthetic data with
    DEFAULT GATE THRESHOLDS. The six default flips applied 2026-05-22
    must collectively let the pure-lag composite reach the final
    ``specs_`` list."""

    def test_six_post_fix_defaults_are_in_place(self) -> None:
        """Snapshot the six gate defaults so a silent revert of any
        single one trips the assertion before the integration test
        below has to run."""
        cfg = CompositeTargetDiscoveryConfig()
        assert cfg.composite_skip_when_raw_dominates_ratio == 0.0
        assert cfg.composite_skip_when_ablation_delta_pct == 0.0
        assert cfg.eps_mi_gain == -10.0
        assert cfg.top_k_after_mi == 32
        assert cfg.require_beats_raw_baseline is False
        assert cfg.per_bin_n_bins == 0

    def test_defaults_yield_pure_lag_composite(self) -> None:
        """Defaults yield pure lag composite."""
        df, tgt, feats, train_idx = _tvt_like_synthetic()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True,
            base_candidates=["lag_y"],
            transforms=["linear_residual", "monotonic_residual", "diff"],
            screening="hybrid",
            mi_sample_n=1500,
            tiny_model_sample_n=1200,
            tiny_model_n_estimators=40,
            tiny_model_cv_folds=3,
            top_m_after_tiny=3,
            random_state=0,
            use_baseline_diagnostics_hint=False,
            auto_base_null_perms=0,
            auto_base_dedup_corr_threshold=1.0,
            auto_base_demote_time_index=False,
            auto_base_demote_spatial_coords=False,
            collapse_linear_residual_alpha_eps=0.0,
            cross_target_ensemble_strategy="off",
            detect_linear_residual_alpha_drift=False,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col=tgt, feature_cols=feats, train_idx=train_idx)
        assert _has_pure_lag_spec(disc), (
            "TVT-regression cement: defaults must yield at least one "
            "pure-lag composite (linres/monres/diff on the AR feature) "
            "so linear-only downstream models (Ridge, Identity-MLP) "
            "have a bounded-residual target to fit instead of "
            "extrapolating on raw y. Got "
            f"specs_={[s.name for s in disc.specs_]}."
        )

    def test_pure_lag_spec_has_bounded_residual_target(self) -> None:
        """Beyond surviving discovery, the pure-lag composite's residual
        target must actually be small-variance (the whole point of the
        composite is to limit MLP extrapolation damage). The Identity-
        MLP's predictions in T-space are bounded to a tiny range, so
        even worst-case extrapolation on inverse-transform y_hat =
        T_hat + alpha*lag_y stays close to alpha*lag_y, which is the
        AR(1) prediction baseline. Confirms the composite delivers the
        invariant that motivated the fix."""
        df, tgt, feats, train_idx = _tvt_like_synthetic()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True,
            base_candidates=["lag_y"],
            transforms=["linear_residual", "diff"],
            screening="mi",
            mi_sample_n=1500,
            random_state=0,
            use_baseline_diagnostics_hint=False,
            auto_base_null_perms=0,
            auto_base_dedup_corr_threshold=1.0,
            auto_base_demote_time_index=False,
            auto_base_demote_spatial_coords=False,
            collapse_linear_residual_alpha_eps=0.0,
            cross_target_ensemble_strategy="off",
            detect_linear_residual_alpha_drift=False,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col=tgt, feature_cols=feats, train_idx=train_idx)
        pure_lag_specs = [
            s for s in disc.specs_ if getattr(s, "base_column", None) == "lag_y" and getattr(s, "transform_name", None) in {"linear_residual", "diff"}
        ]
        assert pure_lag_specs, f"Expected pure-lag spec in specs_={[s.name for s in disc.specs_]}"
        # The residual T = y - alpha*lag_y on AR(1) noise=1.0 data has
        # std ~ 1.0 (matches the AR innovation std). y itself has std
        # ~ 30 (free-drift AR1). The composite should compress target
        # variance by at least 10x; pick a conservative 5x guard so
        # tiny-MI-screening noise doesn't trip the test.
        y_full = df["y"].to_numpy()
        from mlframe.training.composite.transforms import get_transform

        spec = pure_lag_specs[0]
        transform = get_transform(spec.transform_name)
        lag_full = df["lag_y"].to_numpy()
        t = transform.forward(y_full, lag_full, spec.fitted_params)
        compression = float(np.std(y_full) / max(np.std(t), 1e-9))
        assert compression > 5.0, (
            f"pure-lag composite '{spec.name}' should compress target "
            f"std by at least 5x on AR(1) data (got {compression:.1f}x); "
            f"y_std={np.std(y_full):.3f}, t_std={np.std(t):.3f}. The "
            "compression IS the invariant that bounds MLP extrapolation."
        )
