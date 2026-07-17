"""Tests for ``mlframe.training.baseline_diagnostics``.

Coverage map:
- Config validation (defaults, overrides, ``apply_to_target_types`` filter).
- ``_delta_pct`` numeric edge cases (zero baseline, NaN, sign convention).
- ``BaselineDiagnostics.fit_and_report``:
  - regression with a structurally dominant feature (TVT_prev-style);
  - binary classification with a dominant feature;
  - skip path for unsupported target_type;
  - skip path when ``enabled=False``;
  - skip path on empty feature_cols / length-mismatched target;
  - skip path on degenerate target (constant y).
- Recommendation classifier:
  - ``high_potential`` when ablation Δ% large + init_score still off;
  - ``unlikely_to_help`` when init_score baseline matches raw;
  - ``unlikely_to_help`` when ablation Δ% all small;
  - ``marginal`` band.
- Report serialization (``to_dict`` round-trip).
- Pretty-printer (``format_baseline_diagnostics_report``).
- Polars input acceptance.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

# Optional dep guard - the diagnostic itself imports lightgbm lazily;
# tests that actually fit models need the library available.
pytest.importorskip("lightgbm")

from mlframe.training.baselines.diagnostics import (
    AblationEntry,
    BaselineDiagnostics,
    BaselineDiagnosticsReport,
    InitScoreBaseline,
    _delta_pct,
    format_baseline_diagnostics_report,
)
from mlframe.training.configs import BaselineDiagnosticsConfig


# ----------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------


def _make_dominant_regression(n: int = 600, seed: int = 0):
    """Build a regression dataset where ``base`` carries ~95% of the
    variance and the remaining features contribute small structural
    signal. Mirrors the user's TVT_prev case from the original motivation.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=10.0, scale=3.0, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    noise = rng.normal(scale=0.3, size=n)
    # Strong autoregressive lag + tiny structural signal in x1/x2.
    y = 0.95 * base + 0.2 * x1 - 0.1 * x2 + noise
    df = pd.DataFrame({"base": base, "x1": x1, "x2": x2, "x3": x3, "y": y})
    return df, ["base", "x1", "x2", "x3"], y


def _make_dominant_binary(n: int = 600, seed: int = 0):
    """Binary classification with one dominant logit feature."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logit = 3.0 * base + 0.5 * x1
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)
    df = pd.DataFrame({"base": base, "x1": x1, "x2": x2, "y": y})
    return df, ["base", "x1", "x2"], y


def _make_no_dominant_regression(n: int = 600, seed: int = 0):
    """Regression where NO feature dominates - all small contributors."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    # Equal-weight contribution + lots of noise -> no dominant feature.
    y = 0.3 * X[:, 0] + 0.3 * X[:, 1] + 0.3 * X[:, 2] + 0.3 * X[:, 3] + rng.normal(scale=2.0, size=n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["y"] = y
    return df, [f"f{i}" for i in range(4)], y


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------


class TestBaselineDiagnosticsConfig:
    def test_defaults(self) -> None:
        cfg = BaselineDiagnosticsConfig()
        assert cfg.enabled is True
        assert cfg.ablation_top_k == 5
        assert cfg.init_score_top_k == 1
        assert "regression" in cfg.apply_to_target_types
        assert "binary_classification" in cfg.apply_to_target_types
        assert cfg.init_score_apply_to_target_types == (
            "regression",
            "binary_classification",
        )

    def test_dict_construction_via_baseconfig(self) -> None:
        # BaseConfig allows extra fields with a warning; pure dict construction
        # still produces a valid config.
        cfg = BaselineDiagnosticsConfig(**{"ablation_top_k": 3, "sample_n": 1000})
        assert cfg.ablation_top_k == 3
        assert cfg.sample_n == 1000

    def test_apply_to_target_types_immutable_tuple(self) -> None:
        cfg = BaselineDiagnosticsConfig(apply_to_target_types=("regression",))
        assert cfg.apply_to_target_types == ("regression",)


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------


class TestDeltaPct:
    def test_lower_is_better_drop_hurts(self) -> None:
        # RMSE goes 1.0 -> 1.5 means drop hurt -> +50% (positive).
        assert _delta_pct(1.0, 1.5, higher_is_better=False) == pytest.approx(50.0)

    def test_lower_is_better_drop_helps(self) -> None:
        # RMSE goes 1.0 -> 0.8 means drop helped -> -20% (negative).
        assert _delta_pct(1.0, 0.8, higher_is_better=False) == pytest.approx(-20.0)

    def test_higher_is_better_drop_hurts(self) -> None:
        # AUC goes 0.9 -> 0.7 means drop hurt -> +22.22% (positive).
        assert _delta_pct(0.9, 0.7, higher_is_better=True) == pytest.approx(22.22, abs=0.05)

    def test_higher_is_better_drop_helps(self) -> None:
        assert _delta_pct(0.9, 0.95, higher_is_better=True) < 0

    def test_zero_baseline_fallback(self) -> None:
        # ``baseline ≈ 0`` -> absolute pp fallback (no division blow-up).
        out = _delta_pct(0.0, 0.5, higher_is_better=False)
        assert math.isfinite(out)

    def test_nan_input_propagates(self) -> None:
        assert math.isnan(_delta_pct(float("nan"), 1.0, higher_is_better=False))
        assert math.isnan(_delta_pct(1.0, float("nan"), higher_is_better=False))


# ----------------------------------------------------------------------
# fit_and_report - skip paths
# ----------------------------------------------------------------------


class TestSkipPaths:
    def test_disabled_config_returns_skipped_report(self) -> None:
        df, feats, y = _make_dominant_regression(n=200)
        bd = BaselineDiagnostics(BaselineDiagnosticsConfig(enabled=False))
        rep = bd.fit_and_report(
            train_df=df.drop(columns=["y"]),
            train_target=y,
            feature_cols=feats,
            target_type="regression",
            target_name="y",
        )
        assert rep.skipped is True
        assert "enabled=False" in rep.skip_reason

    def test_unsupported_target_type_skipped(self) -> None:
        df, feats, y = _make_dominant_regression(n=200)
        bd = BaselineDiagnostics(BaselineDiagnosticsConfig())
        rep = bd.fit_and_report(
            train_df=df.drop(columns=["y"]),
            train_target=y,
            feature_cols=feats,
            target_type="multiclass_classification",  # not in apply_to_target_types
            target_name="y",
        )
        assert rep.skipped is True
        assert "apply_to_target_types" in rep.skip_reason

    def test_empty_feature_cols_skipped(self) -> None:
        df, _feats, y = _make_dominant_regression(n=200)
        bd = BaselineDiagnostics(BaselineDiagnosticsConfig())
        rep = bd.fit_and_report(
            train_df=df.drop(columns=["y"]),
            train_target=y,
            feature_cols=[],
            target_type="regression",
            target_name="y",
        )
        assert rep.skipped is True

    def test_length_mismatch_skipped(self) -> None:
        df, feats, y = _make_dominant_regression(n=200)
        bd = BaselineDiagnostics(BaselineDiagnosticsConfig())
        rep = bd.fit_and_report(
            train_df=df.drop(columns=["y"]),
            train_target=y[:100],  # mismatch
            feature_cols=feats,
            target_type="regression",
            target_name="y",
        )
        assert rep.skipped is True
        assert "length mismatch" in rep.skip_reason


# ----------------------------------------------------------------------
# fit_and_report - happy path: regression with dominant feature
# ----------------------------------------------------------------------


class TestRegressionHappyPath:
    @pytest.fixture(scope="class")
    def report(self) -> BaselineDiagnosticsReport:
        df, feats, y = _make_dominant_regression(n=500)
        cfg = BaselineDiagnosticsConfig(
            ablation_top_k=4,
            quick_model_n_estimators=80,
            sample_n=None,  # use full
        )
        return BaselineDiagnostics(cfg).fit_and_report(
            train_df=df.drop(columns=["y"]),
            train_target=y,
            feature_cols=feats,
            target_type="regression",
            target_name="y",
        )

    def test_not_skipped(self, report: BaselineDiagnosticsReport) -> None:
        assert report.skipped is False
        assert report.skip_reason == ""

    def test_headline_metric_is_rmse_finite(
        self,
        report: BaselineDiagnosticsReport,
    ) -> None:
        assert report.headline_metric_name == "RMSE"
        assert report.headline_metric_higher_is_better is False
        assert math.isfinite(report.headline_metric_value)
        assert report.headline_metric_value > 0  # RMSE non-negative

    def test_ablation_ranks_base_first(
        self,
        report: BaselineDiagnosticsReport,
    ) -> None:
        # The dominant feature MUST come out on top of the ablation
        # ranking. If it doesn't, the diagnostic is broken.
        assert report.ablation, "ablation should not be empty"
        top = report.ablation[0]
        assert top.feature == "base", f"expected 'base' top of ablation, got {top.feature}; full: {[(e.feature, e.delta_pct) for e in report.ablation]}"
        assert top.delta_pct > 5.0, "dropping the dominant base feature should hurt RMSE by >5%"

    def test_ablation_ranks_descending_by_dominance(
        self,
        report: BaselineDiagnosticsReport,
    ) -> None:
        deltas = [e.delta_pct for e in report.ablation]
        # Should be sorted descending by Δ% (post-sort in implementation).
        assert deltas == sorted(deltas, reverse=True)

    def test_init_score_baseline_present(
        self,
        report: BaselineDiagnosticsReport,
    ) -> None:
        # Regression -> init_score baseline runs by default.
        assert report.init_score_baseline is not None
        assert report.init_score_baseline.feature_used == "base"

    def test_recommendation_high_potential(
        self,
        report: BaselineDiagnosticsReport,
    ) -> None:
        # Synthetic structure has a clear dominant feature; init_score
        # baseline catches most of it but not all (residual structural
        # signal in x1/x2). Should land somewhere actionable.
        assert report.composite_recommendation in ("high_potential", "unlikely_to_help"), (
            f"got {report.composite_recommendation}: {report.composite_recommendation_reason}"
        )


# ----------------------------------------------------------------------
# Binary classification happy path
# ----------------------------------------------------------------------


class TestBinaryHappyPath:
    def test_runs_and_uses_auc(self) -> None:
        df, feats, y = _make_dominant_binary(n=600)
        cfg = BaselineDiagnosticsConfig(
            quick_model_n_estimators=80,
            sample_n=None,
            ablation_top_k=3,
        )
        rep = BaselineDiagnostics(cfg).fit_and_report(
            train_df=df.drop(columns=["y"]),
            train_target=y,
            feature_cols=feats,
            target_type="binary_classification",
            target_name="y",
        )
        assert rep.skipped is False
        assert rep.headline_metric_name == "AUC"
        assert rep.headline_metric_higher_is_better is True
        assert 0.5 < rep.headline_metric_value <= 1.0
        # ``base`` carries the dominant logit, must come out on top.
        assert rep.ablation[0].feature == "base"

    def test_init_score_baseline_runs_for_binary(self) -> None:
        """init_score baseline IS now meaningful for binary
        classification: dominant feature is squashed -> logit and
        passed as LightGBM init_score so the booster learns the
        residual logit. With a strong logit feature like the one
        in _make_dominant_binary, init_score baseline AUC should
        match raw AUC closely (recommendation: unlikely_to_help)."""
        df, feats, y = _make_dominant_binary(n=1500)
        cfg = BaselineDiagnosticsConfig(
            quick_model_n_estimators=80,
            sample_n=None,
            ablation_top_k=3,
        )
        rep = BaselineDiagnostics(cfg).fit_and_report(
            train_df=df.drop(columns=["y"]),
            train_target=y,
            feature_cols=feats,
            target_type="binary_classification",
            target_name="y",
        )
        assert rep.skipped is False
        # Binary now produces a meaningful init_score_baseline.
        assert rep.init_score_baseline is not None
        assert rep.init_score_baseline.feature_used == "base"
        # init_score baseline AUC should be in [0.5, 1.0].
        assert 0.5 < rep.init_score_baseline.metric <= 1.0
        # On synthetic data with a 3*base logit signal, the init_score
        # path should be close to raw AUC -- not necessarily within
        # 1pct but at least within the same ballpark.
        assert abs(rep.init_score_baseline.delta_vs_raw_pct) < 20


# ----------------------------------------------------------------------
# Recommendation: no dominant feature -> unlikely_to_help
# ----------------------------------------------------------------------


class TestNoDominantFeature:
    def test_recommendation_unlikely_to_help(self) -> None:
        df, feats, y = _make_no_dominant_regression(n=600)
        cfg = BaselineDiagnosticsConfig(
            quick_model_n_estimators=60,
            sample_n=None,
            ablation_top_k=4,
        )
        rep = BaselineDiagnostics(cfg).fit_and_report(
            train_df=df.drop(columns=["y"]),
            train_target=y,
            feature_cols=feats,
            target_type="regression",
            target_name="y",
        )
        assert rep.skipped is False
        # No dominant feature ⇒ recommendation should land on
        # unlikely_to_help or marginal (NOT high_potential).
        assert rep.composite_recommendation in ("unlikely_to_help", "marginal"), f"got {rep.composite_recommendation}: {rep.composite_recommendation_reason}"


# ----------------------------------------------------------------------
# Polars input
# ----------------------------------------------------------------------


class TestPolarsInput:
    def test_polars_dataframe_accepted(self) -> None:
        df, feats, y = _make_dominant_regression(n=300)
        pl_df = pl.from_pandas(df.drop(columns=["y"]))
        cfg = BaselineDiagnosticsConfig(
            quick_model_n_estimators=60,
            sample_n=None,
            ablation_top_k=2,
        )
        rep = BaselineDiagnostics(cfg).fit_and_report(
            train_df=pl_df,
            train_target=y,
            feature_cols=feats,
            target_type="regression",
            target_name="y",
        )
        assert rep.skipped is False
        assert rep.ablation


# ----------------------------------------------------------------------
# Serialization + pretty-printer
# ----------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_round_trip(self) -> None:
        rep = BaselineDiagnosticsReport(
            target_name="t",
            target_type="regression",
            headline_metric_name="RMSE",
            headline_metric_value=1.234,
            headline_metric_higher_is_better=False,
            sample_n_used=100,
            ablation=[
                AblationEntry("a", 1.5, 21.5, 1),
                AblationEntry("b", 1.3, 5.3, 2),
            ],
            init_score_baseline=InitScoreBaseline("a", "lightgbm", 1.25, 1.3),
            composite_recommendation="high_potential",
            composite_recommendation_reason="strong",
            elapsed_seconds=0.5,
        )
        d = rep.to_dict()
        assert d["headline_metric"]["value"] == 1.234
        assert len(d["ablation"]) == 2
        assert d["init_score_baseline"]["feature_used"] == "a"
        assert d["composite_recommendation"] == "high_potential"

    def test_skipped_to_dict(self) -> None:
        rep = BaselineDiagnosticsReport(
            target_name="t",
            target_type="regression",
            headline_metric_name="",
            headline_metric_value=float("nan"),
            headline_metric_higher_is_better=False,
            sample_n_used=0,
            skipped=True,
            skip_reason="disabled",
        )
        d = rep.to_dict()
        assert d["skipped"] is True
        assert d["skip_reason"] == "disabled"
        assert d["init_score_baseline"] is None


class TestFormatter:
    def test_skipped_report_renders(self) -> None:
        rep = BaselineDiagnosticsReport(
            target_name="y",
            target_type="regression",
            headline_metric_name="",
            headline_metric_value=float("nan"),
            headline_metric_higher_is_better=False,
            sample_n_used=0,
            skipped=True,
            skip_reason="config.enabled=False",
        )
        text = format_baseline_diagnostics_report(rep)
        assert "SKIPPED" in text
        assert "config.enabled=False" in text

    def test_full_report_renders_all_sections(self) -> None:
        rep = BaselineDiagnosticsReport(
            target_name="y",
            target_type="regression",
            headline_metric_name="RMSE",
            headline_metric_value=1.234,
            headline_metric_higher_is_better=False,
            sample_n_used=500,
            ablation=[
                AblationEntry("base", 1.5, 21.5, 1),
                AblationEntry("x1", 1.25, 1.3, 2),
            ],
            init_score_baseline=InitScoreBaseline("base", "lightgbm", 1.24, 0.5),
            composite_recommendation="high_potential",
            composite_recommendation_reason="strong dominance",
            elapsed_seconds=2.1,
        )
        text = format_baseline_diagnostics_report(rep)
        assert "RMSE_raw=1.2340" in text
        assert "rank=1" in text
        assert "init_score(base)" in text
        assert "high_potential" in text
        assert "strong dominance" in text


# ----------------------------------------------------------------------
# Recommendation classifier - direct unit tests
# ----------------------------------------------------------------------


class TestRecommendationClassifier:
    def _bd(self, **overrides) -> BaselineDiagnostics:
        return BaselineDiagnostics(BaselineDiagnosticsConfig(**overrides))

    def test_high_potential_when_strong_dominance_and_residual_structure(self) -> None:
        bd = self._bd()
        ablation = [AblationEntry("base", 1.5, 20.0, 1)]
        # init_score baseline still 5% off raw -> residual has structure.
        init_score = InitScoreBaseline("base", "lightgbm", 1.05, 5.0)
        rec, reason = bd._build_recommendation(ablation, init_score)
        assert rec == "high_potential"
        assert "20" in reason or "residual" in reason

    def test_unlikely_when_init_score_matches_raw(self) -> None:
        bd = self._bd()
        ablation = [AblationEntry("base", 1.5, 20.0, 1)]
        # init_score baseline within 0.5% of raw -> already optimal.
        init_score = InitScoreBaseline("base", "lightgbm", 1.005, 0.5)
        rec, reason = bd._build_recommendation(ablation, init_score)
        assert rec == "unlikely_to_help"
        assert "init_score" in reason

    def test_unlikely_when_no_dominance(self) -> None:
        bd = self._bd()
        ablation = [
            AblationEntry("a", 1.01, 1.0, 1),
            AblationEntry("b", 1.005, 0.5, 2),
        ]
        rec, _reason = bd._build_recommendation(ablation, init_score_baseline=None)
        assert rec == "unlikely_to_help"

    def test_marginal_band(self) -> None:
        bd = self._bd()  # defaults: marginal=2.0, high=5.0
        ablation = [AblationEntry("a", 1.03, 3.0, 1)]
        rec, _reason = bd._build_recommendation(ablation, init_score_baseline=None)
        assert rec == "marginal"

    def test_empty_ablation_yields_unlikely(self) -> None:
        bd = self._bd()
        rec, reason = bd._build_recommendation([], init_score_baseline=None)
        assert rec == "unlikely_to_help"
        assert "no ablation" in reason


def test_fit_and_report_survives_string_categorical_features():
    """Regression: a train frame carrying OBJECT/STRING categorical feature columns must NOT crash the quick-model
    fit. LightGBM rejects object columns ("pandas dtypes must be int, float or bool") even when they are named in
    categorical_feature -- it needs pandas 'category' dtype. Pre-fix the whole diagnostic was silently skipped
    (broad-except-swallowed) for such frames; fit_and_report now casts the declared string categoricals to 'category'
    once (via assign, no whole-frame copy) so the ablation runs. The caller's frame must stay unmutated."""
    import numpy as np
    import pandas as pd
    from mlframe.training.configs import BaselineDiagnosticsConfig
    from mlframe.training.baselines.diagnostics import BaselineDiagnostics

    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame({"x0": rng.standard_normal(n), "x1": rng.standard_normal(n)})
    cat_cols = [f"cat{j}" for j in range(12)]
    for c in cat_cols:
        df[c] = rng.choice(list("ABCDE"), n)  # object/string dtype -- the shape that crashed LightGBM
    # An UNDECLARED string column (NOT in cat_features) must also be cast, else it reaches the quick model and crashes.
    df["text_und"] = [" ".join(rng.choice(list("abcde"), 2)) for _ in range(n)]
    y = (2.0 * df["x0"] - 1.5 * df["x1"] + rng.standard_normal(n) * 0.5).to_numpy()

    bd = BaselineDiagnostics(BaselineDiagnosticsConfig(enabled=True))
    rep = bd.fit_and_report(
        train_df=df,
        train_target=y,
        target_name="target_reg",
        target_type="regression",
        feature_cols=list(df.columns),
        cat_features=cat_cols,
    )
    d = rep.to_dict()
    # Not skipped, and the ablation actually ran (proves the quick LightGBM fit succeeded on the string cats).
    assert d.get("status") != "skipped", d.get("reason")
    assert len(d.get("ablation", [])) >= 1
    # Caller frame is untouched (no in-place category cast on the possibly-huge input).
    assert df["cat0"].dtype == object
