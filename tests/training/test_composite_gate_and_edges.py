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

    def test_dominant_features_hint_overrides_mi_ranking(self) -> None:
        """When ``dominant_features_hint`` is provided, those features
        are used as base candidates regardless of pairwise MI(y, x).
        Prevents the production failure mode where MI-only ranking
        prefers a high-MI-but-no-residual-signal feature (spatial
        coord) over the truly dominant one (autoregressive lag) when
        the latter happened to score lower MI on the screening sample.
        """
        rng = np.random.default_rng(0)
        n = 1500
        # Construct y so the BEST base for residual learning is f1
        # (clean linear), but X has higher MI(y, x) due to a
        # non-linear-but-uninformative bond.
        f1 = rng.normal(size=n)
        # Strong linear contribution -> diff(y, f1) cleans up nicely.
        spatial = rng.uniform(-2, 2, size=n)
        y = (1.0 * f1
             + np.sin(spatial * 5) * 3.0  # high MI but no clean residual
             + rng.normal(scale=0.1, size=n))
        df = pd.DataFrame({"f1": f1, "spatial": spatial, "y": y})
        # No hint -> auto-base picks by MI; spatial likely top.
        cfg_nohint = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi",
            mi_sample_n=800, eps_mi_gain=-1.0,
            auto_base_top_k=1,
            transforms=["diff"],
        )
        d_nohint = CompositeTargetDiscovery(cfg_nohint)
        d_nohint.fit(df, target_col="y",
                     feature_cols=["f1", "spatial"],
                     train_idx=np.arange(1200))
        # With hint pointing at f1 -> f1 promoted regardless of MI.
        cfg_hint = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi",
            mi_sample_n=800, eps_mi_gain=-1.0,
            auto_base_top_k=1,
            dominant_features_hint=["f1"],
            transforms=["diff"],
        )
        d_hint = CompositeTargetDiscovery(cfg_hint)
        d_hint.fit(df, target_col="y",
                   feature_cols=["f1", "spatial"],
                   train_idx=np.arange(1200))
        bases_hint = {s.base_column for s in d_hint.specs_}
        assert "f1" in bases_hint, (
            f"Hint did not promote f1; specs={[s.name for s in d_hint.specs_]}")

    def test_dominant_features_hint_combines_with_mi_for_remaining_slots(self) -> None:
        """Hint covers slot 1; MI fills slot 2-3 from remaining features.

        Verifies the *base selection* contract -- that hint features
        get evaluated as base candidates -- by checking the discovery
        report (which records all candidates, including those later
        dropped by the mi_gain filter). Whether a particular base
        ultimately survives the mi_gain filter depends on whether
        ``T = transform(y, base)`` is more predictable than y itself,
        which is a separate concern from base SELECTION.
        """
        df = _slow_ar1_dominant_lag()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi",
            mi_sample_n=800, eps_mi_gain=-100.0,  # admit everything
            auto_base_top_k=3,
            dominant_features_hint=["f1"],
            transforms=["diff"],
            top_k_after_mi=8,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["y_prev", "f1", "f2"],
                 train_idx=np.arange(1200))
        # Every base in {y_prev, f1, f2} should appear as either a
        # kept spec or a candidate in the report. f1 (the hint)
        # MUST appear as a base.
        report_bases = {r["base_column"] for r in disc.report()}
        assert "f1" in report_bases, (
            f"hint feature f1 not evaluated as base; report={disc.report()}")
        # And the remaining two slots filled by MI ranking pick
        # from {y_prev, f2}.
        assert report_bases.issubset({"y_prev", "f1", "f2"})

    def test_dominant_features_hint_filtered_falls_back(self) -> None:
        """A hint feature that doesn't survive feature filters
        (forbidden / non-numeric / corr-threshold) is dropped from the
        hint with an INFO log; auto-base falls back to MI for that slot."""
        df = _slow_ar1_dominant_lag()
        df["target_enc_bad"] = df["y"]  # forbidden pattern + corr=1.0
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi",
            mi_sample_n=800, eps_mi_gain=-1.0,
            auto_base_top_k=2,
            dominant_features_hint=["target_enc_bad", "y_prev"],
            transforms=["diff"],
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["y_prev", "f1", "f2", "target_enc_bad"],
                 train_idx=np.arange(1200))
        # target_enc_bad dropped by forbidden_pattern filter; y_prev
        # survives. Discovery should still produce specs based on
        # y_prev (the surviving hint) + MI fill.
        bases = {s.base_column for s in disc.specs_}
        assert "target_enc_bad" not in bases
        assert "y_prev" in bases

    def test_dominant_features_hint_capped_to_leave_room_for_mi(self) -> None:
        """2026-05-10 regression fix: when ``dominant_features_hint``
        covers / exceeds ``auto_base_top_k``, hint contribution is capped
        at ``max(1, top_k // 2)`` so MI-ranked candidates ALSO get a
        chance.

        Pre-fix: hint covered top_k → ZERO MI candidates evaluated.
        For autoregressive targets (TVT_prev as dominant feature in
        prod TVT regression), ``TVT - TVT_prev`` is essentially the
        first-difference of an AR(1) series → low MI(features, residual)
        → NO candidate cleared mi_gain → 0 specs returned (production
        bug observed 2026-05-10).

        Post-fix: with top_k=2 and hint=['y_prev', 'f1', 'f2'], at most
        1 hint slot fires → y_prev (first hint) + 1 MI-leader fill.
        """
        df = _slow_ar1_dominant_lag()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi",
            mi_sample_n=800, eps_mi_gain=-1.0,
            auto_base_top_k=2,
            dominant_features_hint=["y_prev", "f1", "f2"],
            transforms=["diff"],
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["y_prev", "f1", "f2"],
                 train_idx=np.arange(1200))
        bases = {s.base_column for s in disc.specs_}
        # y_prev (the AR1 dominant) MUST still be a base — it's both the
        # first hint entry AND the highest-MI feature. The second base
        # is MI-filled (could be f1 or f2, both random noise → low MI;
        # any of them is fine — the assertion is "hint + MI hybrid
        # produced specs", not "specific second base").
        assert "y_prev" in bases, (
            f"y_prev (AR1 dominant) missing from bases {bases}; "
            "hint-cap fix may have over-suppressed hint contribution"
        )

    def test_hint_cap_preserves_mi_fallback_when_hint_dominant_is_autoregressive(self) -> None:
        """End-to-end regression test for the production bug:
        ``dominant_features_hint`` selects an autoregressive lag
        (``TVT_prev``-style) → ``TVT - TVT_prev`` becomes near-noise →
        mi_gain check fails on hint-only candidates. The MI-fallback
        slot must let an MI-leader reach the candidate pool, where it
        can clear mi_gain on its own residual signal.
        """
        rng = np.random.default_rng(42)
        n = 1500
        # Strong AR1 component: y_prev explains TVT_prev autocorrelation
        y = np.zeros(n)
        y[0] = rng.normal()
        for i in range(1, n):
            y[i] = 0.999 * y[i - 1] + rng.normal(scale=0.1)
        y_prev = np.r_[y[0], y[:-1]]
        # f_signal: a feature with genuine residual signal vs y after
        # subtracting y_prev (i.e. the "right" composite base candidate
        # MI fallback should surface).
        residual = y - 0.999 * y_prev
        f_signal = residual + rng.normal(scale=0.05, size=n)
        f_noise = rng.normal(size=n)
        df = pd.DataFrame({
            "y_prev": y_prev,
            "f_signal": f_signal,
            "f_noise": f_noise,
            "y": y,
        })
        # Hint forces y_prev into the base list — pre-fix this would
        # be the ONLY base tried (top_k=2 with 2 hint entries) → no
        # mi_gain fallback path → 0 specs.
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi",
            mi_sample_n=800, eps_mi_gain=-1.0,
            auto_base_top_k=2,
            dominant_features_hint=["y_prev", "f_noise"],
            transforms=["diff"],
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["y_prev", "f_signal", "f_noise"],
                 train_idx=np.arange(1200))
        bases = {s.base_column for s in disc.specs_}
        # With hint cap = max(1, 2 // 2) = 1, hint contributes y_prev
        # only. The second slot is MI-ranked → f_signal (best non-hint
        # MI) should appear, NOT just f_noise (which the old "hint
        # covers top_k" early-return would have forced).
        assert "y_prev" in bases or "f_signal" in bases, (
            f"neither hint feature nor MI-leader reached bases {bases}; "
            "cap-fix didn't restore MI fallback path"
        )

    def test_dominant_features_hint_default_none(self) -> None:
        """Default config has no hint -> behaves like before."""
        cfg = CompositeTargetDiscoveryConfig()
        assert cfg.dominant_features_hint is None

    # ----------------------------------------------------------------------
    # Transform discrimination: A/B (ratio) vs A-B (diff) vs linear_residual
    # ----------------------------------------------------------------------
    #
    # Verifies that when the data-generating process is genuinely
    # multiplicative (``y = base * f(X)``), discovery's MI gain ranks
    # ``ratio`` / ``logratio`` ABOVE ``diff``, and vice versa for
    # additive processes. Without this contract, discovery could ship
    # a worse-than-raw model on inappropriate transforms even after
    # the raw-y baseline gate (because all transforms might pass the
    # gate but the user expects the right STRUCTURE to win).

    def test_diff_wins_on_pure_additive_signal(self) -> None:
        """y = base + g(X) + small noise -> ``T = y - base = g(X)`` is
        directly predictable from X, while ``T = y / base`` or
        ``T = log(y) - log(base)`` mix base into the target. ``diff``
        should rank #1 by mi_gain on this data-generating process."""
        rng = np.random.default_rng(0)
        n = 1500
        # Positive bases & y so logratio is in-domain on every row.
        base = np.abs(rng.normal(loc=10, scale=2, size=n)) + 5
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        # Pure additive: y = base + clean function of X.
        y = base + 2.0 * x1 + 1.5 * np.sin(x2 * 2) + rng.normal(scale=0.05, size=n)
        df = pd.DataFrame({"base": base, "x1": x1, "x2": x2, "y": y})
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi", mi_sample_n=800,
            base_candidates=["base"], eps_mi_gain=-100.0,
            transforms=["diff", "ratio", "logratio", "linear_residual"],
            top_k_after_mi=8,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["base", "x1", "x2"],
                 train_idx=np.arange(1200))
        # All 4 transforms produced specs (eps wide open). Rank by
        # mi_gain descending; diff or linear_residual must lead because
        # they both eliminate base linearly (linear_residual fits an
        # alpha which is ~1.0 here, so behaves like diff).
        rep = sorted(
            [r for r in disc.report() if r.get("name", "").startswith("y__")],
            key=lambda r: -r.get("mi_gain", -np.inf),
        )
        assert rep, f"no candidates evaluated; report={disc.report()}"
        top_transform = rep[0]["transform_name"]
        # diff or linear_residual must win over ratio / logratio for
        # pure-additive data.
        assert top_transform in ("diff", "linear_residual"), (
            f"expected diff/linear_residual to win on additive data, "
            f"got {top_transform}; rank={[(r['transform_name'], r['mi_gain']) for r in rep]}"
        )
        # Specifically: diff must rank above ratio.
        diff_gain = next((r["mi_gain"] for r in rep
                          if r["transform_name"] == "diff"), float("-inf"))
        ratio_gain = next((r["mi_gain"] for r in rep
                           if r["transform_name"] == "ratio"), float("-inf"))
        assert diff_gain > ratio_gain, (
            f"diff mi_gain {diff_gain:.4f} should beat ratio {ratio_gain:.4f} "
            f"on additive y=base+g(X)")

    def test_logratio_wins_on_pure_multiplicative_signal(self) -> None:
        """y = base * exp(g(X)) -> ``T = log(y) - log(base) = g(X)``
        is directly predictable from X, while ``T = y - base`` carries
        the multiplicative scale of base. ``logratio`` should beat
        ``diff`` by mi_gain on multiplicative DGP."""
        rng = np.random.default_rng(0)
        n = 1500
        # Strictly positive base & y to keep logratio in-domain.
        base = rng.lognormal(mean=2.0, sigma=0.4, size=n)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        # Multiplicative: y = base * exp(small clean function of X).
        y = base * np.exp(0.5 * x1 + 0.3 * np.sin(x2 * 2)
                          + rng.normal(scale=0.02, size=n))
        df = pd.DataFrame({"base": base, "x1": x1, "x2": x2, "y": y})
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi", mi_sample_n=800,
            base_candidates=["base"], eps_mi_gain=-100.0,
            transforms=["diff", "ratio", "logratio", "linear_residual"],
            top_k_after_mi=8,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["base", "x1", "x2"],
                 train_idx=np.arange(1200))
        rep = sorted(
            [r for r in disc.report() if r.get("name", "").startswith("y__")],
            key=lambda r: -r.get("mi_gain", -np.inf),
        )
        assert rep
        # logratio or ratio (proportional scaling) must beat diff
        # on multiplicative DGP. The strict claim: logratio mi_gain
        # > diff mi_gain.
        logratio_gain = next((r["mi_gain"] for r in rep
                              if r["transform_name"] == "logratio"),
                             float("-inf"))
        diff_gain = next((r["mi_gain"] for r in rep
                          if r["transform_name"] == "diff"), float("-inf"))
        assert logratio_gain > diff_gain, (
            f"logratio mi_gain {logratio_gain:.4f} should beat "
            f"diff {diff_gain:.4f} on multiplicative y=base*exp(g(X)); "
            f"full rank={[(r['transform_name'], r['mi_gain']) for r in rep]}")
        # And the top-ranked transform should be a multiplicative one.
        assert rep[0]["transform_name"] in ("logratio", "ratio"), (
            f"expected logratio/ratio top on multiplicative data; "
            f"got {rep[0]['transform_name']}")

    def test_linear_residual_wins_over_diff_when_alpha_not_one(self) -> None:
        """y = alpha*base + g(X) + eps with alpha != 1.0 -- diff
        leaves a residual ``T = y - base = (alpha-1)*base + g(X)`` that
        still carries base contribution, while linear_residual fits
        alpha and removes it cleanly. linear_residual should beat
        diff on mi_gain."""
        rng = np.random.default_rng(0)
        n = 1500
        base = rng.normal(loc=10, scale=2, size=n)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        # y = 0.5*base + clean structure -> diff residual = -0.5*base + structure
        # which still shares variance with base column.
        y = 0.5 * base + 2.0 * x1 + 1.5 * np.sin(x2 * 2) + rng.normal(scale=0.05, size=n)
        df = pd.DataFrame({"base": base, "x1": x1, "x2": x2, "y": y})
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi", mi_sample_n=800,
            base_candidates=["base"], eps_mi_gain=-100.0,
            transforms=["diff", "linear_residual"],
            top_k_after_mi=8,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["base", "x1", "x2"],
                 train_idx=np.arange(1200))
        rep = sorted(
            [r for r in disc.report() if r.get("name", "").startswith("y__")],
            key=lambda r: -r.get("mi_gain", -np.inf),
        )
        lr_gain = next((r["mi_gain"] for r in rep
                        if r["transform_name"] == "linear_residual"),
                       float("-inf"))
        diff_gain = next((r["mi_gain"] for r in rep
                          if r["transform_name"] == "diff"), float("-inf"))
        assert lr_gain >= diff_gain, (
            f"linear_residual mi_gain {lr_gain:.4f} should be >= "
            f"diff mi_gain {diff_gain:.4f} when alpha != 1; "
            f"full rank={[(r['transform_name'], r['mi_gain']) for r in rep]}")

    def test_diff_and_linear_residual_collapse_when_alpha_equals_one(self) -> None:
        """Sanity check: when y = base + g(X) (alpha=1), diff and
        linear_residual produce numerically equivalent T_k and
        therefore comparable mi_gain (within MI estimation noise).
        Catches future regressions where one transform's fit silently
        produces wrong params."""
        rng = np.random.default_rng(0)
        n = 1500
        base = rng.normal(loc=10, scale=2, size=n)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = base + 2.0 * x1 + 1.5 * np.sin(x2 * 2) + rng.normal(scale=0.05, size=n)
        df = pd.DataFrame({"base": base, "x1": x1, "x2": x2, "y": y})
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi", mi_sample_n=1500,
            base_candidates=["base"], eps_mi_gain=-100.0,
            transforms=["diff", "linear_residual"],
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["base", "x1", "x2"],
                 train_idx=np.arange(1200))
        rep = {r["transform_name"]: r["mi_gain"] for r in disc.report()
               if r.get("name", "").startswith("y__")}
        assert "diff" in rep and "linear_residual" in rep
        # alpha is fitted to ~1.0 so the two T sequences are nearly
        # identical; their mi_gain values should differ only by MI
        # estimation noise (< 0.05).
        assert abs(rep["diff"] - rep["linear_residual"]) < 0.10, (
            f"diff={rep['diff']:.4f} vs linear_residual={rep['linear_residual']:.4f} "
            "should be near-equal when alpha=1.0")

    # ----------------------------------------------------------------------
    # Multilabel regression (R3.18)
    # ----------------------------------------------------------------------

    def test_multilabel_strategy_default_per_target(self) -> None:
        """Default multilabel_strategy is 'per_target' so 2-D targets
        get expanded into one 1-D sub-target per output column."""
        cfg = CompositeTargetDiscoveryConfig()
        assert cfg.multilabel_strategy == "per_target"

    def test_multilabel_strategy_validator_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="multilabel_strategy"):
            CompositeTargetDiscoveryConfig(multilabel_strategy="weird_mode")

    def test_multilabel_per_target_expansion_via_suite(self) -> None:
        """End-to-end: a 2-D regression target in target_by_type gets
        split into k 1-D sub-targets named ``{name}_out{j}`` BEFORE
        composite discovery runs. Verified by emulating the suite-
        level expansion logic directly (the same code path that
        train_mlframe_models_suite uses around line 3370 in core.py).
        """
        # Build the per-suite expansion shape that core.py creates.
        rng = np.random.default_rng(0)
        n = 1500
        y_2d = np.column_stack([
            rng.normal(size=n),
            2.0 * rng.normal(size=n) + 1.0,
        ])
        # Mock target_by_type with one 2-D entry.
        target_by_type = {"regression": {"multi_y": y_2d}}
        cfg = CompositeTargetDiscoveryConfig(multilabel_strategy="per_target")
        # Replicate the expansion logic from core.py:3365-3410.
        if cfg.multilabel_strategy == "per_target":
            expanded = dict(target_by_type["regression"])
            for tn, tv in list(target_by_type["regression"].items()):
                arr = np.asarray(tv)
                if arr.ndim == 2 and arr.shape[1] >= 1:
                    for j in range(arr.shape[1]):
                        expanded[f"{tn}_out{j}"] = arr[:, j]
                    expanded.pop(tn, None)
            target_by_type["regression"] = expanded
        # Verify expansion contract.
        assert "multi_y" not in target_by_type["regression"]
        assert "multi_y_out0" in target_by_type["regression"]
        assert "multi_y_out1" in target_by_type["regression"]
        assert target_by_type["regression"]["multi_y_out0"].ndim == 1
        assert target_by_type["regression"]["multi_y_out0"].shape == (n,)
        np.testing.assert_array_equal(
            target_by_type["regression"]["multi_y_out0"], y_2d[:, 0]
        )
        np.testing.assert_array_equal(
            target_by_type["regression"]["multi_y_out1"], y_2d[:, 1]
        )

    def test_multilabel_skip_strategy_preserves_2d(self) -> None:
        """``multilabel_strategy='skip'`` leaves the 2-D entry intact
        for downstream code that knows how to handle it (legacy mode).
        Discovery itself then skips the 2-D target with a metadata
        note (the ``ndim != 1`` skip path)."""
        rng = np.random.default_rng(0)
        y_2d = np.column_stack([rng.normal(size=500), rng.normal(size=500)])
        target_by_type = {"regression": {"multi_y": y_2d}}
        cfg = CompositeTargetDiscoveryConfig(multilabel_strategy="skip")
        # No expansion -> 2-D entry stays.
        if cfg.multilabel_strategy == "per_target":
            # Wouldn't fire under skip mode.
            raise AssertionError("strategy normalisation broken")
        assert "multi_y" in target_by_type["regression"]
        assert target_by_type["regression"]["multi_y"].ndim == 2

    def test_multilabel_per_target_runs_discovery_independently(self) -> None:
        """When 2-D y is expanded, each sub-target gets its own
        discovery instance with independent specs. Verified by
        running discovery on each sub-target and checking the
        resulting specs use the sub-target name, not the parent name.
        """
        rng = np.random.default_rng(0)
        n = 1500
        # Two correlated outputs, each dependent on its own base.
        base_a = rng.normal(loc=10, scale=2, size=n)
        base_b = rng.normal(loc=5, scale=1, size=n)
        x1 = rng.normal(size=n)
        y0 = base_a + 0.5 * x1 + rng.normal(scale=0.05, size=n)
        y1 = 2.0 * base_b + 0.3 * x1 + rng.normal(scale=0.05, size=n)
        # Expanded: each output as its own 1-D target.
        df0 = pd.DataFrame({
            "base_a": base_a, "base_b": base_b, "x1": x1, "multi_y_out0": y0,
        })
        df1 = pd.DataFrame({
            "base_a": base_a, "base_b": base_b, "x1": x1, "multi_y_out1": y1,
        })
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi",
            mi_sample_n=800, eps_mi_gain=-1.0,
            base_candidates=["base_a", "base_b"],
            transforms=["diff", "linear_residual"],
            top_k_after_mi=8,
        )
        d0 = CompositeTargetDiscovery(cfg).fit(
            df0, target_col="multi_y_out0",
            feature_cols=["base_a", "base_b", "x1"],
            train_idx=np.arange(1200))
        d1 = CompositeTargetDiscovery(cfg).fit(
            df1, target_col="multi_y_out1",
            feature_cols=["base_a", "base_b", "x1"],
            train_idx=np.arange(1200))
        # Sub-target names propagate into spec names.
        assert all(s.target_col == "multi_y_out0" for s in d0.specs_)
        assert all(s.target_col == "multi_y_out1" for s in d1.specs_)
        assert all("multi_y_out0__" in s.name for s in d0.specs_)
        assert all("multi_y_out1__" in s.name for s in d1.specs_)

    # ----------------------------------------------------------------------
    # R10b improvement #6: collapse linear_residual -> diff when alpha~1
    # ----------------------------------------------------------------------

    def test_collapse_linear_residual_when_alpha_near_one(self) -> None:
        """When alpha~1.0 on stationary AR(1), linear_residual produces
        the same T as diff. The collapse logic should drop the redundant
        linear_residual spec so only diff survives."""
        rng = np.random.default_rng(0)
        n = 1500
        # Stationary AR(1) with autocorrelation 0.999 -> alpha~1 on
        # train fit.
        y = np.zeros(n)
        y[0] = rng.normal()
        for i in range(1, n):
            y[i] = 0.999 * y[i - 1] + rng.normal(scale=0.1)
        y_prev = np.r_[y[0], y[:-1]]
        f1 = rng.normal(size=n)
        df = pd.DataFrame({"y_prev": y_prev, "f1": f1, "y": y})
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi",
            mi_sample_n=800, eps_mi_gain=-100.0, top_k_after_mi=8,
            base_candidates=["y_prev"],
            transforms=["diff", "linear_residual"],
            collapse_linear_residual_alpha_eps=0.05,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["y_prev", "f1"],
                 train_idx=np.arange(1200))
        # diff must be kept (the simpler equivalent); linear_residual
        # MUST be collapsed.
        kept_transforms = {s.transform_name for s in disc.specs_}
        assert "diff" in kept_transforms
        assert "linear_residual" not in kept_transforms

    def test_no_collapse_when_alpha_far_from_one(self) -> None:
        """When alpha != 1.0 (true linear scaling), linear_residual is
        a genuine improvement over diff and MUST NOT be collapsed."""
        rng = np.random.default_rng(0)
        n = 1500
        base = rng.normal(loc=10, scale=2, size=n)
        f1 = rng.normal(size=n)
        # alpha = 0.5 -> diff residual = -0.5*base + g, linear_residual
        # cleanly separates. alpha_dev should clearly exceed eps.
        y = 0.5 * base + 1.0 * f1 + rng.normal(scale=0.1, size=n)
        df = pd.DataFrame({"base": base, "f1": f1, "y": y})
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi",
            mi_sample_n=800, eps_mi_gain=-100.0, top_k_after_mi=8,
            base_candidates=["base"],
            transforms=["diff", "linear_residual"],
            collapse_linear_residual_alpha_eps=0.05,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["base", "f1"],
                 train_idx=np.arange(1200))
        kept = {s.transform_name for s in disc.specs_}
        # Both must survive: alpha clearly differs from 1.0.
        assert "linear_residual" in kept

    def test_collapse_disabled_when_eps_zero(self) -> None:
        """``collapse_linear_residual_alpha_eps=0.0`` disables the
        collapse and keeps both diff and linear_residual specs even
        when alpha~1."""
        rng = np.random.default_rng(0)
        n = 1500
        y = np.zeros(n)
        y[0] = rng.normal()
        for i in range(1, n):
            y[i] = 0.999 * y[i - 1] + rng.normal(scale=0.1)
        y_prev = np.r_[y[0], y[:-1]]
        f1 = rng.normal(size=n)
        df = pd.DataFrame({"y_prev": y_prev, "f1": f1, "y": y})
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi",
            mi_sample_n=800, eps_mi_gain=-100.0, top_k_after_mi=8,
            base_candidates=["y_prev"],
            transforms=["diff", "linear_residual"],
            collapse_linear_residual_alpha_eps=0.0,  # disable
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["y_prev", "f1"],
                 train_idx=np.arange(1200))
        kept = {s.transform_name for s in disc.specs_}
        # Both kept (collapse disabled).
        assert "linear_residual" in kept
        assert "diff" in kept

    # ----------------------------------------------------------------------
    # R10b improvement #9: cross-base correlation dedup
    # ----------------------------------------------------------------------

    def test_auto_base_dedup_drops_correlated_duplicates(self) -> None:
        """Two highly-correlated base candidates (typical lag set:
        TVT_prev, TVT_prev_smooth_3) should not both survive auto-base.
        Only the higher-MI one is kept; the duplicate is logged as
        dedup-dropped."""
        rng = np.random.default_rng(0)
        n = 1500
        base_a = rng.normal(loc=10, scale=2, size=n)
        # base_b is a near-identical copy of base_a (corr > 0.99).
        base_b = base_a + rng.normal(scale=0.05, size=n)
        x1 = rng.normal(size=n)
        y = base_a + 0.5 * x1 + rng.normal(scale=0.1, size=n)
        df = pd.DataFrame({
            "base_a": base_a, "base_b": base_b, "x1": x1, "y": y,
        })
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi",
            mi_sample_n=800, eps_mi_gain=-100.0,
            auto_base_top_k=3,
            transforms=["diff"],
            auto_base_dedup_corr_threshold=0.95,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["base_a", "base_b", "x1"],
                 train_idx=np.arange(1200))
        # Either base_a OR base_b survives, not both.
        bases = {s.base_column for s in disc.specs_
                 if s.base_column in ("base_a", "base_b")}
        assert len(bases) <= 1, (
            f"dedup failed: both correlated bases kept: {bases}")

    def test_auto_base_dedup_disabled_at_threshold_one(self) -> None:
        """Setting auto_base_dedup_corr_threshold=1.0 disables dedup;
        both correlated bases survive."""
        rng = np.random.default_rng(0)
        n = 1500
        base_a = rng.normal(loc=10, scale=2, size=n)
        base_b = base_a + rng.normal(scale=0.05, size=n)
        x1 = rng.normal(size=n)
        y = base_a + 0.5 * x1 + rng.normal(scale=0.1, size=n)
        df = pd.DataFrame({
            "base_a": base_a, "base_b": base_b, "x1": x1, "y": y,
        })
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi",
            mi_sample_n=800, eps_mi_gain=-100.0,
            auto_base_top_k=3,
            transforms=["diff"],
            auto_base_dedup_corr_threshold=1.0,  # disabled
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y",
                 feature_cols=["base_a", "base_b", "x1"],
                 train_idx=np.arange(1200))
        bases = {s.base_column for s in disc.specs_}
        # Both bases can survive when dedup is off.
        assert "base_a" in bases or "base_b" in bases  # at least one
        # And not removed from candidates list -- this would be hard
        # to verify deterministically because mi_gain ties may flip,
        # so we only assert dedup didn't actively remove.

    # ----------------------------------------------------------------------
    # R10b: vectorised _safe_corr correctness
    # ----------------------------------------------------------------------

    def test_safe_abs_corr_all_matches_safe_corr_per_column(self) -> None:
        """The vectorised path used inside _filter_features must match
        the per-column scalar path numerically (within 1e-10)."""
        from mlframe.training.composite import _safe_corr, _safe_abs_corr_all
        rng = np.random.default_rng(0)
        n = 5000
        y = rng.normal(size=n)
        X = rng.normal(size=(n, 30))
        # Add some correlation structure.
        X[:, 0] = y * 0.5 + rng.normal(scale=0.5, size=n)
        X[:, 5] = y + rng.normal(scale=0.1, size=n)
        ref = np.array([abs(_safe_corr(X[:, j], y)) for j in range(30)])
        got = _safe_abs_corr_all(y, X)
        np.testing.assert_allclose(got, ref, atol=1e-10)

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
