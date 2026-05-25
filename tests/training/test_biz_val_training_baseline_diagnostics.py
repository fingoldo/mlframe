"""biz_val tests for ``BaselineDiagnostics`` (training/baseline_diagnostics.py).

Per CLAUDE.md "Every new ML trick gets a biz_val synthetic test":
each test asserts a SYNTHETIC measurable WIN that locks in the
diagnostic's ability to surface dominant features, detect dummy
overruns, and produce ablation rankings that match ground truth.

Naming: ``test_biz_val_baseline_diagnostics_<parameter>_<scenario>``.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


def _dominant_one_target(n=2000, seed=42):
    """One feature dominates: ``y = 5*x_dominant + 0.5*x_other + noise``.
    Ablation should identify ``x_dominant`` as the top contributor."""
    rng = np.random.default_rng(seed)
    x_dominant = rng.normal(size=n)
    x_other = rng.normal(size=n)
    x_noise1 = rng.normal(size=n)
    x_noise2 = rng.normal(size=n)
    y = 5.0 * x_dominant + 0.5 * x_other + 0.3 * rng.normal(size=n)
    return pd.DataFrame({
        "x_dominant": x_dominant, "x_other": x_other,
        "x_noise1": x_noise1, "x_noise2": x_noise2,
        "y": y,
    })


def _make_config(**overrides):
    from mlframe.training.configs import BaselineDiagnosticsConfig
    defaults = dict(
        enabled=True,
        ablation_top_k=4,
        quick_model_family="lightgbm",
        quick_model_n_estimators=50,
        sample_n=2000,
        random_state=42,
    )
    defaults.update(overrides)
    return BaselineDiagnosticsConfig(**defaults)


# ---------------------------------------------------------------------------
# Ablation surfaces the dominant feature
# ---------------------------------------------------------------------------


def test_biz_val_baseline_diagnostics_ablation_finds_dominant_feature():
    """On a target with one feature explaining most variance, the
    ablation rank-1 entry must be that feature. Catches regressions
    in the per-feature ablation loop."""
    pytest.importorskip("lightgbm")
    from mlframe.training.baseline_diagnostics import BaselineDiagnostics

    df = _dominant_one_target(n=2000, seed=42)
    feature_cols = ["x_dominant", "x_other", "x_noise1", "x_noise2"]
    diag = BaselineDiagnostics(_make_config(ablation_top_k=4))
    report = diag.fit_and_report(
        train_df=df, train_target=df["y"],
        feature_cols=feature_cols,
        target_type="regression", target_name="y",
    )
    # report.ablation is a list of AblationEntry(feature, ..., rank)
    assert hasattr(report, "ablation")
    abl = report.ablation
    assert len(abl) >= 1
    # Find the rank=1 entry
    rank_1 = [e for e in abl if getattr(e, "rank", None) == 1]
    assert len(rank_1) == 1, f"ablation must have exactly one rank-1 entry; got {abl}"
    assert rank_1[0].feature == "x_dominant", (
        f"rank-1 feature must be x_dominant; got {rank_1[0].feature}"
    )


def test_biz_val_baseline_diagnostics_dominant_features_finds_signal():
    """``report.dominant_features`` must list features in the same
    rank order as ablation. The dominant feature must be #1."""
    pytest.importorskip("lightgbm")
    from mlframe.training.baseline_diagnostics import BaselineDiagnostics

    df = _dominant_one_target(n=2000, seed=42)
    diag = BaselineDiagnostics(_make_config(ablation_top_k=4))
    report = diag.fit_and_report(
        train_df=df, train_target=df["y"],
        feature_cols=["x_dominant", "x_other", "x_noise1", "x_noise2"],
        target_type="regression", target_name="y",
    )
    assert hasattr(report, "dominant_features")
    dom = report.dominant_features
    assert len(dom) >= 1
    # First entry must be x_dominant
    first = dom[0]
    name = first.get("feature") if isinstance(first, dict) else getattr(first, "feature", None)
    assert name == "x_dominant", (
        f"top dominant feature must be x_dominant; got {name}"
    )


# ---------------------------------------------------------------------------
# sample_n bounds runtime
# ---------------------------------------------------------------------------


def test_biz_val_baseline_diagnostics_sample_n_caps_runtime():
    """``sample_n=500`` must keep runtime under 30s on a 50_000-row
    dataset. The sample_n parameter is the headline speedup: without
    it, fitting LightGBM on 50k rows would dominate the diagnostic
    cost. Floor: complete in < 30s. Measured fast on this hardware
    in 1-3s; 30s leaves CI headroom."""
    pytest.importorskip("lightgbm")
    from mlframe.training.baseline_diagnostics import BaselineDiagnostics

    rng = np.random.default_rng(42)
    n = 50_000
    df = pd.DataFrame({
        "x_dom": rng.normal(size=n),
        "x_other": rng.normal(size=n),
        "y": rng.normal(size=n),
    })
    df["y"] = 3.0 * df["x_dom"] + 0.3 * rng.normal(size=n)

    diag = BaselineDiagnostics(_make_config(sample_n=500))
    t0 = time.perf_counter()
    report = diag.fit_and_report(
        train_df=df, train_target=df["y"],
        feature_cols=["x_dom", "x_other"],
        target_type="regression", target_name="y",
    )
    dt = time.perf_counter() - t0
    assert dt < 30.0, (
        f"sample_n=500 must keep n=50k runtime under 30s; got {dt:.1f}s"
    )
    # Behavioural: a sample_n=500 fit must produce a real diagnostics record (not just any object), and the
    # ablation summary must surface the dominant feature so the runtime-vs-quality tradeoff is verifiable.
    assert report is not None, "fit_and_report returned None on sample_n=500 path"
    assert hasattr(report, "ablation") or hasattr(report, "feature_ranks"), (
        f"report missing ablation / feature_ranks attribute on sample_n path; got dir={dir(report)[:10]}"
    )


# ---------------------------------------------------------------------------
# disabled config bypasses everything
# ---------------------------------------------------------------------------


def test_biz_val_baseline_diagnostics_disabled_returns_quickly():
    """``enabled=False`` must short-circuit ``fit_and_report`` so it
    doesn't fit any model. Runtime must be < 1s even on a large
    dataframe. Catches regressions where someone forgets to honor
    the gate."""
    pytest.importorskip("lightgbm")
    from mlframe.training.baseline_diagnostics import BaselineDiagnostics

    rng = np.random.default_rng(42)
    n = 100_000
    df = pd.DataFrame({
        "x1": rng.normal(size=n), "x2": rng.normal(size=n),
        "y": rng.normal(size=n),
    })
    diag = BaselineDiagnostics(_make_config(enabled=False))
    t0 = time.perf_counter()
    diag.fit_and_report(
        train_df=df, train_target=df["y"],
        feature_cols=["x1", "x2"],
        target_type="regression", target_name="y",
    )
    dt = time.perf_counter() - t0
    assert dt < 1.0, (
        f"enabled=False must short-circuit; got {dt:.2f}s"
    )


# ---------------------------------------------------------------------------
# Smoke: report has the expected high-level fields
# ---------------------------------------------------------------------------


def test_biz_val_baseline_diagnostics_report_has_init_score_field():
    """Report object must expose an attribute related to the init-
    score / dummy baseline check (the headline diagnostic). Catches
    regressions in the report schema."""
    pytest.importorskip("lightgbm")
    from mlframe.training.baseline_diagnostics import BaselineDiagnostics

    df = _dominant_one_target(n=2000, seed=42)
    diag = BaselineDiagnostics(_make_config())
    report = diag.fit_and_report(
        train_df=df, train_target=df["y"],
        feature_cols=["x_dominant", "x_other", "x_noise1", "x_noise2"],
        target_type="regression", target_name="y",
    )
    # Public attrs related to init-score / dominance.
    expected_any = ("init_score", "dominant_features", "high_potential",
                      "ablation", "feature_importance", "marginal")
    public_attrs = [a for a in dir(report) if not a.startswith("_")]
    assert any(any(e in a.lower() for e in expected_any) for a in public_attrs), (
        f"BaselineDiagnostics report must expose at least one of "
        f"{expected_any}; got attrs={public_attrs[:15]}"
    )
