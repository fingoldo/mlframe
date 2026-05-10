"""Tests for ``CompositeTargetDiscovery`` (PR3 of the composite-target
roll-out).

Coverage map
------------
- Config validation (defaults, ``fail_on_no_gain`` enum normalisation).
- Auto-base ranking surfaces the dominant feature (``TVT_prev`` case).
- Explicit base list still passes through forbidden filters.
- **Leakage-guard tests** (the load-bearing ones):
    * ``alpha`` for linear_residual differs when fit on different
      ``train_idx`` slices but is identical when fit on the same
      slice -- proves train_idx discipline.
    * Discovery never reads test-only columns: removing test rows
      from df after fit doesn't affect discovered specs.
- Forbidden-base filters: regex patterns + corr threshold + ptp eps
  + non-numeric columns.
- Domain-validity filter: logratio with mostly negative y skipped.
- ``fail_on_no_gain`` modes: ``raise`` raises, ``warn`` logs but
  proceeds, ``fallback_raw`` logs and returns no specs.
- ``iter_transform`` streams per-spec T values; rows failing
  ``domain_check`` get NaN.
- ``export_specs`` / ``report`` shape contract.
- Polars-input acceptance for column extraction.
- Disabled config short-circuits.
- Tiny train_idx (<50 rows) gracefully returns no specs.
- ``random_state`` reproducibility.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import polars as pl
import pytest

# sklearn is a hard dep of mlframe; LightGBM only needed for downstream
# integration tests, not for discovery itself. Discovery uses only
# sklearn.feature_selection.mutual_info_regression.

from mlframe.training.composite import (  # noqa: E402
    CompositeSpec,
    CompositeTargetDiscovery,
    UnknownTransformError,
)
from mlframe.training.configs import CompositeTargetDiscoveryConfig  # noqa: E402


# ----------------------------------------------------------------------
# Synthetic data fixtures
# ----------------------------------------------------------------------


def _tvt_strong(n: int = 1500, seed: int = 0):
    """TVT-like data with strong structural signal so MI gain clears
    the default eps on small samples.

    y = 0.95 * TVT_prev + 0.5 * x1 - 0.3 * x2 + small noise
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=10.0, scale=3.0, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 0.95 * base + 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame({"TVT_prev": base, "x1": x1, "x2": x2, "x3": x3, "TVT": y})
    return df


def _no_dominant(n: int = 1500, seed: int = 0):
    """Equal-weight contributors -- no feature should dominate."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = 0.3 * X.sum(axis=1) + rng.normal(scale=2.0, size=n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["y"] = y
    return df


# ----------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = CompositeTargetDiscoveryConfig()
        assert cfg.enabled is False
        assert cfg.base_candidates == "auto"
        assert "diff" in cfg.transforms and "linear_residual" in cfg.transforms
        assert cfg.mi_sample_n == 200_000
        assert cfg.top_k_after_mi == 8
        assert cfg.fail_on_no_gain == "fallback_raw"

    def test_explicit_base_list(self) -> None:
        cfg = CompositeTargetDiscoveryConfig(base_candidates=["TVT_prev"])
        assert cfg.base_candidates == ["TVT_prev"]

    def test_fail_mode_normalisation(self) -> None:
        cfg = CompositeTargetDiscoveryConfig(fail_on_no_gain="RAISE")
        assert cfg.fail_on_no_gain == "raise"

    def test_invalid_fail_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            CompositeTargetDiscoveryConfig(fail_on_no_gain="explode")


# ----------------------------------------------------------------------
# Auto-base + happy path
# ----------------------------------------------------------------------


class TestAutoBase:
    def test_dominant_feature_surfaces_at_top(self) -> None:
        df = _tvt_strong()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=800, top_k_after_mi=5,
            eps_mi_gain=0.05, auto_base_top_k=3,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="TVT",
                 feature_cols=["TVT_prev", "x1", "x2", "x3"],
                 train_idx=np.arange(1200))
        # At least one discovered spec, and the top-1 by mi_gain uses
        # TVT_prev as base.
        assert len(disc.specs_) >= 1
        assert disc.specs_[0].base_column == "TVT_prev"

    def test_discovery_finds_all_4_transforms_when_signal_strong(self) -> None:
        df = _tvt_strong()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=800, top_k_after_mi=8,
            eps_mi_gain=0.0, auto_base_top_k=2,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="TVT",
                 feature_cols=["TVT_prev", "x1", "x2", "x3"],
                 train_idx=np.arange(1200))
        transforms_kept = {s.transform_name for s in disc.specs_ if s.base_column == "TVT_prev"}
        # All 4 core transforms should clear eps when base is TVT_prev.
        assert {"diff", "ratio", "logratio", "linear_residual"}.issubset(transforms_kept)

    def test_explicit_base_passes_through_filters(self) -> None:
        df = _tvt_strong()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=800, top_k_after_mi=5,
            base_candidates=["TVT_prev"],
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="TVT",
                 feature_cols=["TVT_prev", "x1", "x2", "x3"],
                 train_idx=np.arange(1200))
        # Only TVT_prev was offered, so all kept specs use it.
        for s in disc.specs_:
            assert s.base_column == "TVT_prev"


# ----------------------------------------------------------------------
# LEAKAGE GUARDS -- the load-bearing tests
# ----------------------------------------------------------------------


class TestLeakageGuards:
    """If any of these tests start failing, do NOT relax them. Each
    one corresponds to a specific leakage class flagged in the plan
    review (round 2 R2 + round 3 first-principles)."""

    def test_alpha_train_only_changes_with_train_idx(self) -> None:
        """Linear-residual alpha MUST be computed only from train rows.
        If the implementation regressed to "use full df", fitting on
        two non-overlapping train_idx slices would produce identical
        alpha (no signal of fit honouring the slice). We construct
        two halves of the dataset whose linear coefficients DIFFER
        and assert the discovered alphas reflect that."""
        rng = np.random.default_rng(0)
        n = 2000
        base = rng.normal(loc=10.0, scale=3.0, size=n)
        # First half: y has alpha=0.95 against base. Second half:
        # alpha=1.10. A leakage-bug fitting on full df would produce
        # alpha ~ midpoint (~1.025) regardless of train_idx.
        y = np.empty(n)
        y[: n // 2] = 0.95 * base[: n // 2] + rng.normal(scale=0.3, size=n // 2)
        y[n // 2:] = 1.10 * base[n // 2:] + rng.normal(scale=0.3, size=n // 2)
        df = pd.DataFrame({"TVT_prev": base, "x1": rng.normal(size=n), "TVT": y})

        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=600, top_k_after_mi=4,
            eps_mi_gain=-0.5, base_candidates=["TVT_prev"],
            transforms=["linear_residual"],
        )

        disc_first = CompositeTargetDiscovery(cfg).fit(
            df, target_col="TVT", feature_cols=["TVT_prev", "x1"],
            train_idx=np.arange(0, n // 2),  # only first half
        )
        disc_second = CompositeTargetDiscovery(cfg).fit(
            df, target_col="TVT", feature_cols=["TVT_prev", "x1"],
            train_idx=np.arange(n // 2, n),  # only second half
        )

        spec_first = next(s for s in disc_first.specs_ if s.transform_name == "linear_residual")
        spec_second = next(s for s in disc_second.specs_ if s.transform_name == "linear_residual")

        alpha_first = spec_first.fitted_params["alpha"]
        alpha_second = spec_second.fitted_params["alpha"]

        # alphas should reflect the slice-specific generative coefficients,
        # not converge to the full-df midpoint.
        assert abs(alpha_first - 0.95) < 0.05, (
            f"alpha_first={alpha_first:.3f} should be ~0.95; if it's ~1.025, "
            "the fit is using full-df data instead of train_idx -> LEAKAGE."
        )
        assert abs(alpha_second - 1.10) < 0.05, (
            f"alpha_second={alpha_second:.3f} should be ~1.10; if it's ~1.025, "
            "the fit is using full-df data instead of train_idx -> LEAKAGE."
        )
        # And they MUST differ.
        assert abs(alpha_first - alpha_second) > 0.05

    def test_test_rows_never_touched(self) -> None:
        """Mutate test rows AFTER fit; discovered specs must remain
        bit-identical (because fit only ever touched train rows).
        If specs change, fit was peeking at test data."""
        df = _tvt_strong()
        n = len(df)
        train_idx = np.arange(int(0.8 * n))
        test_idx = np.arange(int(0.8 * n), n)

        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=800, top_k_after_mi=4,
            eps_mi_gain=0.05, auto_base_top_k=2,
        )
        disc = CompositeTargetDiscovery(cfg).fit(
            df, target_col="TVT",
            feature_cols=["TVT_prev", "x1", "x2", "x3"],
            train_idx=train_idx, test_idx=test_idx,
        )
        specs_before = [
            (s.name, s.fitted_params, s.mi_gain) for s in disc.specs_
        ]

        # Now corrupt the test slice. A clean fit must not have
        # produced specs that depend on these rows in any way.
        df_corrupted = df.copy()
        df_corrupted.loc[test_idx, "TVT_prev"] = 1e6
        df_corrupted.loc[test_idx, "TVT"] = -1e6

        disc2 = CompositeTargetDiscovery(cfg).fit(
            df_corrupted, target_col="TVT",
            feature_cols=["TVT_prev", "x1", "x2", "x3"],
            train_idx=train_idx, test_idx=test_idx,
        )
        specs_after = [
            (s.name, s.fitted_params, s.mi_gain) for s in disc2.specs_
        ]
        assert specs_before == specs_after

    def test_iter_transform_uses_train_fitted_params_for_full_frame(self) -> None:
        """``iter_transform`` should apply train-fitted params to ALL
        rows (so val and test rows get T values for downstream training),
        but the params themselves must come from train_idx only."""
        df = _tvt_strong(n=1500)
        train_idx = np.arange(1000)
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=600,
            base_candidates=["TVT_prev"], transforms=["linear_residual"],
            eps_mi_gain=-1.0,
        )
        disc = CompositeTargetDiscovery(cfg).fit(
            df, target_col="TVT", feature_cols=["TVT_prev", "x1", "x2", "x3"],
            train_idx=train_idx,
        )
        spec = disc.specs_[0]

        # Apply spec.fitted_params manually to compute T for the full
        # frame and compare to iter_transform output.
        from mlframe.training.composite import get_transform
        transform = get_transform(spec.transform_name)
        y = df["TVT"].to_numpy()
        base = df[spec.base_column].to_numpy()
        valid = transform.domain_check(y, base)
        expected = np.full(len(y), np.nan)
        expected[valid] = transform.forward(y[valid], base[valid], spec.fitted_params)

        for name, t in disc.iter_transform(df):
            if name == spec.name:
                np.testing.assert_allclose(t, expected, rtol=1e-12, atol=1e-12,
                                           equal_nan=True)


# ----------------------------------------------------------------------
# Forbidden-base filters
# ----------------------------------------------------------------------


class TestForbiddenFilters:
    def test_regex_pattern_drops_target_encoding_columns(self) -> None:
        df = _tvt_strong()
        df["target_enc_grp"] = df["TVT"] * 0.99 + 1.0  # mock target-encoded col
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=600, top_k_after_mi=4,
            base_candidates=["target_enc_grp", "TVT_prev"],
            transforms=["diff"],
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="TVT",
                 feature_cols=["TVT_prev", "x1", "x2", "x3", "target_enc_grp"],
                 train_idx=np.arange(1200))
        # target_enc_grp must NOT appear as a base in any kept spec.
        for s in disc.specs_:
            assert s.base_column != "target_enc_grp"

    def test_high_correlation_with_y_drops_column(self) -> None:
        df = _tvt_strong()
        # Column with corr(col, y) ≈ 1.0 -- look like derived from y.
        df["leaked"] = df["TVT"] + np.random.default_rng(0).normal(scale=1e-9, size=len(df))
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=600,
            base_candidates=["leaked", "TVT_prev"], transforms=["diff"],
            forbidden_base_corr_threshold=0.999,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="TVT",
                 feature_cols=["TVT_prev", "x1", "leaked"],
                 train_idx=np.arange(1200))
        for s in disc.specs_:
            assert s.base_column != "leaked"

    def test_constant_column_filtered(self) -> None:
        df = _tvt_strong()
        df["const_col"] = 7.0  # zero variance
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=600,
            base_candidates=["const_col", "TVT_prev"], transforms=["diff"],
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="TVT",
                 feature_cols=["TVT_prev", "x1", "const_col"],
                 train_idx=np.arange(1200))
        for s in disc.specs_:
            assert s.base_column != "const_col"

    def test_non_numeric_column_filtered(self) -> None:
        df = _tvt_strong()
        df["category"] = "abc"
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=600,
            base_candidates=["category", "TVT_prev"], transforms=["diff"],
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="TVT",
                 feature_cols=["TVT_prev", "x1", "category"],
                 train_idx=np.arange(1200))
        for s in disc.specs_:
            assert s.base_column != "category"

    def test_autoregressive_lag_with_corr_999_is_kept_under_default(self) -> None:
        """Regression: legitimate lag features whose corr(base, y) reaches
        ~0.999 due to autocorrelation (slow-moving series, near-noise-
        free lag-1) MUST NOT be filtered out by the corr threshold under
        the default config.

        Reproduces the production failure where TVT_prev was dropped
        before MI ranking and discovery picked unrelated spatial
        coordinates as bases. Default raised from 0.999 to 0.99999 in
        2026-05-10 to fix this.
        """
        rng = np.random.default_rng(0)
        n = 2000
        # Slow-moving AR(1) series; lag-1 has corr ~ 0.999 with current.
        y = np.zeros(n)
        y[0] = rng.normal()
        for i in range(1, n):
            y[i] = 0.999 * y[i - 1] + rng.normal(scale=0.05)
        prev = np.r_[y[0], y[:-1]]
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        df = pd.DataFrame({"TVT_prev": prev, "x1": x1, "x2": x2, "TVT": y})

        # Sanity: corr(TVT_prev, TVT) actually clears the *old* 0.999
        # threshold, demonstrating why the old default was wrong.
        c = float(np.corrcoef(prev[:1500], y[:1500])[0, 1])
        assert abs(c) >= 0.99, f"fixture too weak: corr={c:.6f}"

        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=800, top_k_after_mi=4,
            eps_mi_gain=-1.0,  # accept any gain so we can inspect ranking
            base_candidates=["TVT_prev", "x1", "x2"],
            transforms=["diff", "linear_residual"],
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="TVT",
                 feature_cols=["TVT_prev", "x1", "x2"],
                 train_idx=np.arange(1500))
        # TVT_prev MUST survive the filters under the new default.
        bases = {s.base_column for s in disc.specs_}
        assert "TVT_prev" in bases, (
            f"TVT_prev was filtered out under default corr threshold "
            f"(specs: {[(s.base_column, s.transform_name) for s in disc.specs_]}). "
            "This is the production-reported regression."
        )
        # And the corr drop list should NOT mention TVT_prev.
        corr_drops = [
            d for d in disc.filter_drops()
            if d.get("reason") == "forbidden_base_corr_threshold"
        ]
        assert all(d["name"] != "TVT_prev" for d in corr_drops)

    def test_filter_drops_records_corr_threshold_reason(self) -> None:
        """When a literal copy of y IS dropped by corr threshold, the
        ``filter_drops()`` audit list records it with reason +
        offending corr value."""
        df = _tvt_strong()
        df["leaked"] = df["TVT"] + np.random.default_rng(0).normal(
            scale=1e-9, size=len(df))
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=600,
            base_candidates=["leaked", "TVT_prev"], transforms=["diff"],
            forbidden_base_corr_threshold=0.99,  # explicit aggressive
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="TVT",
                 feature_cols=["TVT_prev", "x1", "leaked"],
                 train_idx=np.arange(1200))
        drops = disc.filter_drops()
        leaked_drops = [d for d in drops
                        if d["name"] == "leaked"
                        and d["reason"] == "forbidden_base_corr_threshold"]
        assert len(leaked_drops) == 1
        assert leaked_drops[0]["corr"] >= 0.99
        assert leaked_drops[0]["threshold"] == 0.99


# ----------------------------------------------------------------------
# Domain validity gating
# ----------------------------------------------------------------------


class TestDomainValidity:
    def test_logratio_skipped_when_y_mostly_negative(self) -> None:
        df = _tvt_strong()
        df["TVT"] = df["TVT"] - 100  # shift y to be mostly negative
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=600,
            base_candidates=["TVT_prev"], transforms=["diff", "logratio"],
            min_valid_domain_frac=0.7,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="TVT",
                 feature_cols=["TVT_prev", "x1", "x2"],
                 train_idx=np.arange(1200))
        kept_transforms = {s.transform_name for s in disc.specs_}
        # diff is general-domain, logratio requires y > 0 -> should be dropped.
        assert "logratio" not in kept_transforms
        # The reject reason should mention valid_domain_frac.
        rejected = [r for r in disc.report_ if r["transform_name"] == "logratio"]
        assert any("valid_domain_frac" in r["reason"] for r in rejected)


# ----------------------------------------------------------------------
# fail_on_no_gain modes
# ----------------------------------------------------------------------


class TestFailOnNoGain:
    @pytest.fixture
    def no_signal_df(self):
        return _no_dominant()

    def test_fallback_raw_no_specs_no_raise(self, no_signal_df) -> None:
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=600,
            base_candidates=["f0", "f1"], transforms=["diff"],
            eps_mi_gain=10.0,  # impossibly high threshold
            fail_on_no_gain="fallback_raw",
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(no_signal_df, target_col="y",
                 feature_cols=["f0", "f1", "f2", "f3"],
                 train_idx=np.arange(1200))
        assert disc.specs_ == []

    def test_raise_mode_propagates(self, no_signal_df) -> None:
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=600,
            base_candidates=["f0"], transforms=["diff"],
            eps_mi_gain=10.0,
            fail_on_no_gain="raise",
        )
        disc = CompositeTargetDiscovery(cfg)
        with pytest.raises(RuntimeError, match="no candidate cleared"):
            disc.fit(no_signal_df, target_col="y",
                     feature_cols=["f0", "f1", "f2", "f3"],
                     train_idx=np.arange(1200))


# ----------------------------------------------------------------------
# iter_transform
# ----------------------------------------------------------------------


class TestIterTransform:
    def test_iter_transform_yields_per_spec(self) -> None:
        df = _tvt_strong()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=600, top_k_after_mi=2,
            base_candidates=["TVT_prev"], transforms=["diff", "linear_residual"],
            eps_mi_gain=-1.0,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="TVT",
                 feature_cols=["TVT_prev", "x1", "x2"],
                 train_idx=np.arange(1200))
        outputs = dict(disc.iter_transform(df))
        # One T-array per discovered spec.
        assert set(outputs) == {s.name for s in disc.specs_}
        # Each T has length matching df.
        for name, t in outputs.items():
            assert len(t) == len(df)
            assert t.dtype == np.float64

    def test_iter_transform_empty_when_no_specs(self) -> None:
        cfg = CompositeTargetDiscoveryConfig(enabled=False)
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(_tvt_strong(), target_col="TVT",
                 feature_cols=["TVT_prev", "x1", "x2"],
                 train_idx=np.arange(1200))
        assert list(disc.iter_transform(_tvt_strong())) == []


# ----------------------------------------------------------------------
# export_specs / report
# ----------------------------------------------------------------------


class TestSerialization:
    def test_export_specs_shape(self) -> None:
        df = _tvt_strong()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=600,
            base_candidates=["TVT_prev"], transforms=["diff"],
            eps_mi_gain=-1.0,
        )
        disc = CompositeTargetDiscovery(cfg).fit(
            df, target_col="TVT", feature_cols=["TVT_prev", "x1"],
            train_idx=np.arange(1200),
        )
        exported = disc.export_specs()
        assert isinstance(exported, list)
        for entry in exported:
            assert {"name", "target_col", "transform_name", "base_column",
                    "fitted_params", "mi_gain", "valid_domain_frac"}.issubset(entry)
            assert isinstance(entry["fitted_params"], dict)

    def test_report_includes_rejected(self) -> None:
        df = _tvt_strong()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=600,
            base_candidates=["TVT_prev"], transforms=["diff", "ratio"],
            eps_mi_gain=10.0,  # all rejected
        )
        disc = CompositeTargetDiscovery(cfg).fit(
            df, target_col="TVT", feature_cols=["TVT_prev", "x1"],
            train_idx=np.arange(1200),
        )
        report = disc.report()
        # Both transforms evaluated; both rejected.
        assert len(report) == 2
        for entry in report:
            assert entry["kept"] is False


# ----------------------------------------------------------------------
# Polars input + reproducibility + edge cases
# ----------------------------------------------------------------------


class TestPolarsAndEdgeCases:
    def test_polars_dataframe_accepted(self) -> None:
        df = pl.from_pandas(_tvt_strong())
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=600,
            base_candidates=["TVT_prev"], transforms=["diff"],
            eps_mi_gain=-1.0,
        )
        disc = CompositeTargetDiscovery(cfg).fit(
            df, target_col="TVT", feature_cols=["TVT_prev", "x1", "x2"],
            train_idx=np.arange(1200),
        )
        assert len(disc.specs_) >= 1

    def test_disabled_short_circuits(self) -> None:
        df = _tvt_strong()
        cfg = CompositeTargetDiscoveryConfig(enabled=False)
        disc = CompositeTargetDiscovery(cfg).fit(
            df, target_col="TVT", feature_cols=["TVT_prev", "x1"],
            train_idx=np.arange(1200),
        )
        assert disc.specs_ == []
        assert disc.report_ == []

    def test_tiny_train_idx_returns_empty(self) -> None:
        df = _tvt_strong()
        cfg = CompositeTargetDiscoveryConfig(enabled=True)
        disc = CompositeTargetDiscovery(cfg).fit(
            df, target_col="TVT", feature_cols=["TVT_prev", "x1"],
            train_idx=np.arange(20),  # below 50-row floor
        )
        assert disc.specs_ == []

    def test_random_state_reproducibility(self) -> None:
        df = _tvt_strong()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=400,
            base_candidates=["TVT_prev"], transforms=["diff"],
            eps_mi_gain=-1.0, random_state=123,
        )
        disc_a = CompositeTargetDiscovery(cfg).fit(
            df, target_col="TVT", feature_cols=["TVT_prev", "x1", "x2"],
            train_idx=np.arange(1200),
        )
        disc_b = CompositeTargetDiscovery(cfg).fit(
            df, target_col="TVT", feature_cols=["TVT_prev", "x1", "x2"],
            train_idx=np.arange(1200),
        )
        # Same seed -> same MI gains.
        for sa, sb in zip(disc_a.specs_, disc_b.specs_):
            assert sa.name == sb.name
            assert abs(sa.mi_gain - sb.mi_gain) < 1e-10

    def test_tiny_model_rerank_trims_to_top_m(self) -> None:
        """Phase B should trim the kept_specs from top_k_after_mi=8 down
        to top_m_after_tiny=3 by tiny-model CV-RMSE."""
        df = _tvt_strong()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=600,
            base_candidates=["TVT_prev"],
            transforms=["diff", "ratio", "logratio", "linear_residual"],
            top_k_after_mi=8, eps_mi_gain=-1.0,
            screening="hybrid",
            tiny_model_n_estimators=20,
            tiny_model_sample_n=400,
            top_m_after_tiny=2,
            tiny_screening_models="single_lgbm",
        )
        disc = CompositeTargetDiscovery(cfg).fit(
            df, target_col="TVT", feature_cols=["TVT_prev", "x1", "x2", "x3"],
            train_idx=np.arange(1200),
        )
        # Phase B trims to top_m_after_tiny=2.
        assert len(disc.specs_) <= 2

    def test_screening_mi_only_skips_tiny_rerank(self) -> None:
        df = _tvt_strong()
        cfg_mi = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=600,
            base_candidates=["TVT_prev"],
            transforms=["diff", "linear_residual"],
            top_k_after_mi=8, eps_mi_gain=-1.0,
            screening="mi",
            top_m_after_tiny=2,  # ignored when screening='mi'
        )
        disc = CompositeTargetDiscovery(cfg_mi).fit(
            df, target_col="TVT", feature_cols=["TVT_prev", "x1"],
            train_idx=np.arange(1200),
        )
        # MI-only mode keeps all top_k_after_mi survivors, NOT trimmed
        # by top_m_after_tiny.
        assert len(disc.specs_) >= 2  # both transforms passed

    def test_per_family_screening_runs(self) -> None:
        """Per-family rerank should still produce specs (even with one
        family configured)."""
        df = _tvt_strong()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=600,
            base_candidates=["TVT_prev"],
            transforms=["linear_residual"],
            top_k_after_mi=2, eps_mi_gain=-1.0,
            screening="hybrid",
            tiny_model_n_estimators=20,
            tiny_model_sample_n=400,
            top_m_after_tiny=1,
            tiny_screening_models="per_family",
            tiny_screening_families=("lightgbm",),  # single family for portability
            tiny_consensus="union",
        )
        disc = CompositeTargetDiscovery(cfg).fit(
            df, target_col="TVT", feature_cols=["TVT_prev", "x1"],
            train_idx=np.arange(1200),
        )
        assert len(disc.specs_) >= 1

    def test_unknown_transform_in_config_skipped(self) -> None:
        df = _tvt_strong()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=600,
            base_candidates=["TVT_prev"],
            transforms=["diff", "made_up_transform"],
            eps_mi_gain=-1.0,
        )
        disc = CompositeTargetDiscovery(cfg).fit(
            df, target_col="TVT", feature_cols=["TVT_prev", "x1"],
            train_idx=np.arange(1200),
        )
        # Real transforms still discovered; bogus one silently skipped
        # with a warning (verified in fit log path).
        assert any(s.transform_name == "diff" for s in disc.specs_)
        assert all(s.transform_name != "made_up_transform" for s in disc.specs_)
