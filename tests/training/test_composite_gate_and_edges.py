"""Edge case tests for composite-target discovery, focused on the
production failure mode that motivated the 2026-05-10 hotfix.

Layered by what is being exercised:

A. **Raw-y baseline gate** -- the architectural safeguard that catches
   "wrong base" cases where MI-gain passes but the composite is actually
   harder for the model to predict than y itself. Production case:
   spatial coordinates X/Y/Z had pairwise MI(y, x) ~ 2 due to global
   spatial trend, but ``T = y - X`` added pure noise to the target.
   Without the gate, MI screening kept these and the resulting models
   had worse RMSE than raw.

B. **corr-threshold semantics** -- the filter that previously dropped
   legitimate autoregressive lags (TVT_prev). Default raised
   0.999 -> 0.99999 in the same hotfix.

C. **Auto-base ranking determinism / edge inputs**.

D. **NaN / heavy-tail / domain-validity** corner cases.

E. **Multi-target type / Polars** integration touchpoints that the
   gate must respect.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.configs import CompositeTargetDiscoveryConfig


# ----------------------------------------------------------------------
# Synthetic fixtures targeted at specific failure modes
# ----------------------------------------------------------------------


def _spatial_trend_y_uninformative_base(n: int = 1500, seed: int = 0):
    """y has a spatial trend (correlated with X/Y/Z coords) but the
    coords themselves carry NO structural residual signal -- they only
    reflect the global trend. Subtracting them from y just adds noise.
    Composite ``T = y - X`` should fail the raw-y baseline gate.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=n)
    Y = rng.normal(size=n)
    Z = rng.normal(size=n)
    # Useful structural feature (no relation to coords).
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    # y depends mostly on f1, f2 (the predictable structure) plus a
    # weak spatial trend through coords. Coords have non-zero MI with
    # y (~0.05-0.2 typical) but ZERO marginal predictive lift over f1+f2.
    y = (1.5 * f1 - 0.8 * f2
         + 0.3 * X + 0.2 * Y + 0.15 * Z
         + rng.normal(scale=0.2, size=n))
    return pd.DataFrame({
        "X": X, "Y": Y, "Z": Z, "f1": f1, "f2": f2, "y": y,
    })


def _slow_ar1_dominant_lag(n: int = 1500, seed: int = 0,
                           autocorr: float = 0.999):
    """AR(1) with very high autocorrelation. lag-1 (``y_prev``) carries
    almost all the structure; corr(y_prev, y) ~ autocorr. Pre-hotfix
    the default 0.999 threshold dropped y_prev as "leakage"; new
    0.99999 default keeps it.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    y[0] = rng.normal()
    for i in range(1, n):
        y[i] = autocorr * y[i - 1] + rng.normal(scale=0.1)
    y_prev = np.r_[y[0], y[:-1]]
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    return pd.DataFrame({"y_prev": y_prev, "f1": f1, "f2": f2, "y": y})


# ----------------------------------------------------------------------
# A. Raw-y baseline gate
# ----------------------------------------------------------------------


class TestRawYBaselineGate:
    """The safeguard that stops MI-only-passing junk from reaching the
    final model loop."""

    def test_spatial_coord_base_rejected_by_gate(self) -> None:
        """The production failure case: subtracting a spatial coordinate
        from y produces a target with EQUAL OR WORSE predictability than
        raw y, because the coord carries only global trend, not residual
        signal. Gate must reject; specs_ either empty or contain only
        composites whose base provides genuine residual lift."""
        df = _spatial_trend_y_uninformative_base()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True,
            screening="hybrid",  # required to enable Phase B + gate
            mi_sample_n=1000,
            tiny_model_sample_n=800,
            tiny_model_n_estimators=30,
            tiny_model_cv_folds=3,
            base_candidates=["X", "Y", "Z"],
            transforms=["diff", "linear_residual"],
            eps_mi_gain=-1.0,  # force candidates through MI to gate
            require_beats_raw_baseline=True,
            raw_baseline_tolerance=1.0,  # strict
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["X", "Y", "Z", "f1", "f2"],
                 train_idx=np.arange(1200))
        # Either all composites rejected -> empty specs, or some kept
        # but each kept one's tiny RMSE < raw_y_baseline_rmse_ * 1.0.
        if disc.specs_:
            for s in disc.specs_:
                composite_rmse = disc.tiny_rerank_scores_[s.name]
                assert composite_rmse < disc.raw_y_baseline_rmse_, (
                    f"Gate failed: {s.name} kept with RMSE "
                    f"{composite_rmse:.4f} >= raw {disc.raw_y_baseline_rmse_:.4f}"
                )

    def test_dominant_lag_passes_gate(self) -> None:
        """Sanity counterpart: when the base actually dominates (slow
        AR(1) lag-1), composite SHOULD beat raw-y baseline. If this
        regresses, the gate has become too aggressive."""
        df = _slow_ar1_dominant_lag()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="hybrid",
            mi_sample_n=1000, tiny_model_sample_n=800,
            tiny_model_n_estimators=30, tiny_model_cv_folds=3,
            base_candidates=["y_prev"],
            transforms=["diff", "linear_residual"],
            eps_mi_gain=-1.0,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["y_prev", "f1", "f2"],
                 train_idx=np.arange(1200))
        assert len(disc.specs_) >= 1
        # At least one survivor has tiny RMSE strictly < raw baseline.
        assert any(
            disc.tiny_rerank_scores_[s.name] < disc.raw_y_baseline_rmse_
            for s in disc.specs_
        )

    def test_gate_disabled_keeps_mi_survivors(self) -> None:
        """When ``require_beats_raw_baseline=False``, gate is skipped
        and tiny rerank just sorts by RMSE, keeping top-M."""
        df = _spatial_trend_y_uninformative_base()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="hybrid",
            mi_sample_n=1000, tiny_model_sample_n=800,
            tiny_model_n_estimators=30, tiny_model_cv_folds=3,
            base_candidates=["X", "Y"],
            transforms=["diff"],
            eps_mi_gain=-1.0,
            require_beats_raw_baseline=False,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["X", "Y", "f1", "f2"],
                 train_idx=np.arange(1200))
        # Gate didn't run -> baseline NaN.
        import math
        assert math.isnan(disc.raw_y_baseline_rmse_)
        # And specs survived MI / rerank since no gate veto.
        assert len(disc.specs_) >= 1

    def test_gate_off_in_mi_only_screening(self) -> None:
        """``screening='mi'`` skips Phase B entirely so the gate cannot
        run regardless of ``require_beats_raw_baseline``. Verifies the
        gate doesn't fire in the MI-only legacy path."""
        df = _slow_ar1_dominant_lag()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi",  # explicit legacy mode
            mi_sample_n=1000,
            base_candidates=["y_prev"],
            transforms=["diff"], eps_mi_gain=-1.0,
            require_beats_raw_baseline=True,  # ignored under "mi"
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["y_prev", "f1", "f2"],
                 train_idx=np.arange(1200))
        # No tiny rerank -> empty score map, NaN baseline.
        import math
        assert disc.tiny_rerank_scores_ == {}
        assert math.isnan(disc.raw_y_baseline_rmse_)

    def test_gate_tolerance_loose_keeps_marginal(self) -> None:
        """Tolerance > 1.0 admits composites that are slightly worse on
        the screening sample. Verifies the threshold = baseline * tol
        arithmetic by setting an absurdly loose tolerance."""
        df = _spatial_trend_y_uninformative_base()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="hybrid",
            mi_sample_n=1000, tiny_model_sample_n=800,
            tiny_model_n_estimators=30, tiny_model_cv_folds=3,
            base_candidates=["X"], transforms=["diff"],
            eps_mi_gain=-1.0,
            require_beats_raw_baseline=True,
            raw_baseline_tolerance=1000.0,  # admits everything finite
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["X", "f1", "f2"],
                 train_idx=np.arange(1200))
        # With absurd tolerance, surviving spec's RMSE may exceed raw
        # but stays < raw * 1000.
        if disc.specs_:
            for s in disc.specs_:
                cr = disc.tiny_rerank_scores_[s.name]
                assert cr < disc.raw_y_baseline_rmse_ * 1000.0

    def test_all_rejected_returns_empty_with_warning(self, caplog) -> None:
        """When the gate kills every composite, discovery returns no
        specs and logs a WARNING explaining the fall-back."""
        df = _spatial_trend_y_uninformative_base()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="hybrid",
            mi_sample_n=1000, tiny_model_sample_n=800,
            tiny_model_n_estimators=30, tiny_model_cv_folds=3,
            base_candidates=["X", "Y", "Z"],
            transforms=["diff"],  # only "diff" to keep all junk
            eps_mi_gain=-1.0,
            require_beats_raw_baseline=True,
            raw_baseline_tolerance=0.5,  # impossibly strict (<<1)
            fail_on_no_gain="fallback_raw",
        )
        disc = CompositeTargetDiscovery(cfg)
        with caplog.at_level(logging.WARNING):
            disc.fit(df, target_col="y",
                     feature_cols=["X", "Y", "Z", "f1", "f2"],
                     train_idx=np.arange(1200))
        # All rejected by gate.
        assert disc.specs_ == []
        # And the WARNING was logged with the gate-rejection signal.
        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING
                    and "raw-y baseline gate" in r.getMessage()]
        # Either the gate-rejection warning OR the fail_on_no_gain
        # warning must surface. Don't pin the exact wording -- just
        # that the user gets actionable signal.
        assert warnings or any(
            "no candidate cleared" in r.getMessage()
            for r in caplog.records if r.levelno == logging.WARNING
        )

    def test_tiny_rerank_scores_keyed_by_spec_name(self) -> None:
        df = _slow_ar1_dominant_lag()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="hybrid",
            mi_sample_n=1000, tiny_model_sample_n=800,
            tiny_model_n_estimators=30, tiny_model_cv_folds=3,
            base_candidates=["y_prev"],
            transforms=["diff", "linear_residual"],
            eps_mi_gain=-1.0,
            require_beats_raw_baseline=True,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["y_prev", "f1", "f2"],
                 train_idx=np.arange(1200))
        scores = disc.tiny_rerank_scores_
        assert isinstance(scores, dict)
        for s in disc.specs_:
            assert s.name in scores, (
                f"missing tiny RMSE for kept spec {s.name}")
            assert np.isfinite(scores[s.name])
        # Returned object is a copy, mutations don't poison state.
        scores["__corruption__"] = 99.0
        assert "__corruption__" not in disc.tiny_rerank_scores_


# ----------------------------------------------------------------------
# B. corr-threshold filter semantics
# ----------------------------------------------------------------------


class TestCorrThresholdEdges:
    def test_negative_corr_legitimate_lag_is_kept(self) -> None:
        """An AR(1) with negative coefficient (corr ~ -0.999) is also a
        legitimate strong predictor. ``abs(corr)`` is what gets compared,
        but the test verifies the new default keeps it."""
        rng = np.random.default_rng(0)
        n = 2000
        y = np.zeros(n)
        y[0] = rng.normal()
        # Negative AR(1): y[t] = -0.999 * y[t-1] + noise
        for i in range(1, n):
            y[i] = -0.999 * y[i - 1] + rng.normal(scale=0.05)
        y_prev = np.r_[y[0], y[:-1]]
        df = pd.DataFrame({"y_prev": y_prev, "f1": rng.normal(size=n), "y": y})
        c = float(np.corrcoef(y_prev[:1500], y[:1500])[0, 1])
        assert c < -0.9, f"fixture too weak: corr={c}"
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi",  # bypass tiny-model cost
            mi_sample_n=600, eps_mi_gain=-1.0,
            base_candidates=["y_prev"], transforms=["diff"],
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["y_prev", "f1"],
                 train_idx=np.arange(1500))
        bases = {s.base_column for s in disc.specs_}
        assert "y_prev" in bases

    def test_corr_exactly_at_threshold_drops(self) -> None:
        """Boundary: corr >= threshold drops. Construct a column with
        controlled corr and verify the boundary."""
        rng = np.random.default_rng(0)
        n = 1500
        y = rng.normal(size=n)
        # Manufactured corr > 0.99999 (above default threshold).
        # Build x = y + epsilon * noise so corr(x, y) ~ 1 - eps^2/2 for
        # small eps. eps=0.001 gives corr ~ 0.9999995.
        x = y + 0.001 * rng.normal(size=n)
        c = abs(np.corrcoef(x[:1200], y[:1200])[0, 1])
        # Sanity: we are above default threshold.
        assert c >= 0.99999, f"fixture wrong: corr={c}"
        df = pd.DataFrame({"x_near_y": x, "f1": rng.normal(size=n), "y": y})
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi",
            mi_sample_n=600,
            base_candidates=["x_near_y"], transforms=["diff"],
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["x_near_y", "f1"],
                 train_idx=np.arange(1200))
        # x_near_y dropped by corr threshold.
        drops = disc.filter_drops()
        assert any(d["name"] == "x_near_y"
                   and d["reason"] == "forbidden_base_corr_threshold"
                   for d in drops)
        # And specs_ has nothing using x_near_y.
        assert all(s.base_column != "x_near_y" for s in disc.specs_)

    def test_corr_one_dot_zero_filtered(self) -> None:
        """Literal y-copy must always be filtered (corr = 1.0)."""
        df = _slow_ar1_dominant_lag()
        df["y_copy"] = df["y"]  # exact copy
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi", mi_sample_n=600,
            base_candidates=["y_copy", "y_prev"],
            transforms=["diff"],
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["y_copy", "y_prev", "f1", "f2"],
                 train_idx=np.arange(1200))
        # y_copy MUST be in filter_drops with corr-reason.
        corr_drops = [d for d in disc.filter_drops()
                      if d["reason"] == "forbidden_base_corr_threshold"]
        assert any(d["name"] == "y_copy" for d in corr_drops)

    def test_filter_drops_records_insufficient_finite_rows(self) -> None:
        """A column where ~all train rows are NaN gets dropped with the
        ``insufficient_finite_rows`` reason, with the count surfaced."""
        df = _slow_ar1_dominant_lag()
        df["mostly_nan"] = np.nan
        df.loc[df.index[:5], "mostly_nan"] = 1.0
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi", mi_sample_n=600,
            base_candidates=["mostly_nan", "y_prev"],
            transforms=["diff"],
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["mostly_nan", "y_prev", "f1", "f2"],
                 train_idx=np.arange(1200))
        drops = [d for d in disc.filter_drops()
                 if d["name"] == "mostly_nan"]
        assert len(drops) == 1
        assert drops[0]["reason"] == "insufficient_finite_rows"
        assert drops[0]["n_finite"] <= 5


# ----------------------------------------------------------------------
# C. Auto-base ranking edges
# ----------------------------------------------------------------------


class TestAutoBaseEdges:
    def test_seed_determinism(self) -> None:
        """Same seed -> same auto-base picks. Without this we can't
        reproduce production runs."""
        df = _slow_ar1_dominant_lag()
        common = dict(
            enabled=True, screening="mi", mi_sample_n=800,
            top_k_after_mi=4, eps_mi_gain=-1.0, auto_base_top_k=2,
            random_state=42,
        )
        d1 = CompositeTargetDiscovery(
            CompositeTargetDiscoveryConfig(**common)
        ).fit(df, target_col="y",
              feature_cols=["y_prev", "f1", "f2"],
              train_idx=np.arange(1200))
        d2 = CompositeTargetDiscovery(
            CompositeTargetDiscoveryConfig(**common)
        ).fit(df, target_col="y",
              feature_cols=["y_prev", "f1", "f2"],
              train_idx=np.arange(1200))
        assert ([s.name for s in d1.specs_] == [s.name for s in d2.specs_])

    def test_all_features_filtered_returns_empty(self) -> None:
        """Every feature gets dropped (forbidden + non-numeric) ->
        discovery yields no specs, doesn't crash."""
        rng = np.random.default_rng(0)
        n = 1500
        y = rng.normal(size=n)
        df = pd.DataFrame({
            "target_enc_a": rng.normal(size=n),  # forbidden pattern
            "y_smooth_b": rng.normal(size=n),    # forbidden pattern
            "obj_c": ["x"] * n,                   # non-numeric
            "y": y,
        })
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi", mi_sample_n=600,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["target_enc_a", "y_smooth_b", "obj_c"],
                 train_idx=np.arange(1200))
        assert disc.specs_ == []
        # And the drops are accountable.
        reasons = {d["reason"] for d in disc.filter_drops()}
        assert reasons.issuperset(
            {"forbidden_pattern", "non_numeric"}
        )

    def test_auto_base_top_k_larger_than_features(self) -> None:
        """``auto_base_top_k=10`` with only 3 usable features returns
        all of them (no IndexError)."""
        df = _slow_ar1_dominant_lag()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi", mi_sample_n=600,
            auto_base_top_k=10,  # >> 3 features
            transforms=["diff"], eps_mi_gain=-10.0,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["y_prev", "f1", "f2"],
                 train_idx=np.arange(1200))
        bases = {s.base_column for s in disc.specs_}
        assert bases.issubset({"y_prev", "f1", "f2"})


# ----------------------------------------------------------------------
# D. NaN / heavy-tail / domain-validity
# ----------------------------------------------------------------------


class TestDataQualityEdges:
    def test_y_with_partial_nans_handled(self) -> None:
        """A few NaN rows in y on TRAIN don't crash discovery."""
        df = _slow_ar1_dominant_lag()
        df.loc[df.index[100:110], "y"] = np.nan
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi", mi_sample_n=600,
            base_candidates=["y_prev"], transforms=["diff"],
            eps_mi_gain=-1.0,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["y_prev", "f1", "f2"],
                 train_idx=np.arange(1200))
        # Discovery either kept y_prev or fell back gracefully -- it
        # MUST NOT crash.
        assert isinstance(disc.specs_, list)

    def test_base_with_partial_nans_kept_if_finite_above_threshold(self) -> None:
        """Base with 5% NaN train rows: enough finite (95%) so still
        passes the insufficient_finite_rows check (>=50 finite)."""
        df = _slow_ar1_dominant_lag()
        rng = np.random.default_rng(0)
        nan_mask = rng.random(len(df)) < 0.05
        df.loc[df.index[nan_mask], "y_prev"] = np.nan
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi", mi_sample_n=600,
            base_candidates=["y_prev"], transforms=["diff"],
            eps_mi_gain=-1.0,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["y_prev", "f1", "f2"],
                 train_idx=np.arange(1200))
        drops = [d for d in disc.filter_drops()
                 if d["name"] == "y_prev"]
        # y_prev should NOT have been dropped for insufficient finite rows.
        assert all(d["reason"] != "insufficient_finite_rows" for d in drops)

    def test_heavy_tail_y_distribution(self) -> None:
        """Heavy-tailed y (Cauchy-ish) doesn't crash MI estimation."""
        rng = np.random.default_rng(0)
        n = 1500
        y = rng.standard_cauchy(size=n)  # heavy tails
        # Clip to avoid extreme outliers crashing tiny-model fit.
        y = np.clip(y, -100, 100)
        f1 = rng.normal(size=n)
        f2 = rng.normal(size=n)
        df = pd.DataFrame({"f1": f1, "f2": f2, "y": y})
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi", mi_sample_n=800,
            base_candidates=["f1"], transforms=["diff"],
            eps_mi_gain=-10.0,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["f1", "f2"],
                 train_idx=np.arange(1200))
        # Just verify no crash; correctness on Cauchy is undefined.
        assert isinstance(disc.specs_, list)


# ----------------------------------------------------------------------
# E. Integration / Polars
# ----------------------------------------------------------------------


class TestIntegrationEdges:
    def test_default_screening_is_hybrid(self) -> None:
        """Default flipped from 'mi' to 'hybrid' in 2026-05-10. If a
        downstream consumer relies on 'mi' default, this breaks them
        loudly rather than silently."""
        cfg = CompositeTargetDiscoveryConfig()
        assert cfg.screening == "hybrid", (
            "Default screening changed -- update consumers")

    def test_default_corr_threshold_is_99999(self) -> None:
        """Default raised from 0.999 to 0.99999."""
        cfg = CompositeTargetDiscoveryConfig()
        assert cfg.forbidden_base_corr_threshold == 0.99999

    def test_polars_frame_with_gate(self) -> None:
        """Polars input: gate path must use the same column-extraction
        helpers as the pandas path, end-to-end."""
        df_pd = _slow_ar1_dominant_lag()
        df_pl = pl.from_pandas(df_pd)
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="hybrid",
            mi_sample_n=800, tiny_model_sample_n=600,
            tiny_model_n_estimators=20, tiny_model_cv_folds=3,
            base_candidates=["y_prev"], transforms=["diff"],
            eps_mi_gain=-1.0, require_beats_raw_baseline=True,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df_pl, target_col="y",
                 feature_cols=["y_prev", "f1", "f2"],
                 train_idx=np.arange(1200))
        # Either kept (most likely on dominant lag) or empty;
        # never raises.
        assert isinstance(disc.specs_, list)

    def test_disabled_config_skips_gate_entirely(self) -> None:
        """``enabled=False`` short-circuits before any rerank / gate."""
        df = _slow_ar1_dominant_lag()
        cfg = CompositeTargetDiscoveryConfig(enabled=False)
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["y_prev", "f1", "f2"],
                 train_idx=np.arange(1200))
        assert disc.specs_ == []
        # No rerank ran -> empty score map, NaN baseline.
        assert disc.tiny_rerank_scores_ == {}
        import math
        assert math.isnan(disc.raw_y_baseline_rmse_)
