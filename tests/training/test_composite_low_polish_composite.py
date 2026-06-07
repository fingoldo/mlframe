"""Regression tests for LOW + POLISH findings in composite-family.

Wave 4 + 5 of the 2026-05-17 audit. Each test pins a specific behavior-affecting
fix; tests pass on the post-fix code. Pure POLISH items (comment renames,
em-dash strips, dated-tag removal) are not tested individually since they have
no observable behavior change.

Categories covered:
- Public symbol re-exports remain importable after audit-tag comment scrubs.
- ``streaming_alpha_check_and_refit`` SE formula stays the residual-based form
  (would silently mis-detect drift if the regression of the audit-introduced
  fix were ever reverted).
- ``stacking_aware_gate`` narrowed NNLS except still falls back uniformly on
  degenerate inputs (preserved behavior with tighter exception type).
- ``composite_predictions_as_feature`` polars fallback narrowed to ImportError
  still works when polars is present (preserved happy path).
- ``CompositeSpec`` dataclass is still frozen + carries the multi-base field.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Public symbols survive the comment / docstring scrubs.
# ---------------------------------------------------------------------------


def test_polish_composite_public_symbols_importable() -> None:
    """Audit-tag stripping in composite.py must not break re-exports."""
    from mlframe.training.composite import (  # noqa: F401
        CompositeSpec,
        CompositeTargetEstimator,
        CompositeProvenance,
        CompositeCrossTargetEnsemble,
        CompositeTargetDiscovery,
        Transform,
        get_transform,
        list_transforms,
    )


def test_polish_composite_spec_frozen_with_multi_base() -> None:
    """``CompositeSpec`` is frozen + ``extra_base_columns`` defaults to empty tuple.

    Multi-base support is meant to be additive: legacy callers passing only
    ``base_column`` must keep working.
    """
    from mlframe.training.composite_spec import CompositeSpec

    spec = CompositeSpec(
        name="y-linres-x",
        target_col="y",
        transform_name="linear_residual",
        base_column="x",
        fitted_params={"alpha": 1.0, "beta": 0.0},
        mi_gain=0.1,
        mi_y=0.2,
        mi_t=0.3,
        valid_domain_frac=1.0,
        n_train_rows=100,
    )
    assert spec.extra_base_columns == ()
    # Frozen dataclass: mutation must raise.
    with pytest.raises((AttributeError, Exception)):
        spec.name = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Streaming alpha refit: residual-based SE survives.
# ---------------------------------------------------------------------------


def test_low_streaming_alpha_uses_residual_based_se() -> None:
    """For a clean linear y = 2*x + 0 with low noise, the buffer's fresh fit
    matches the deployed alpha closely, so z << z_threshold and no refit fires.

    Pre-audit-fix code used SE = y_std / (sqrt(n) * base_std), which dramatically
    inflated SE for high-R^2 regressors (z under-detected drift). Post-fix uses
    residual-based SE so z is correctly large when alpha really drifted.
    """
    from mlframe.training.composite_streaming import (
        streaming_alpha_check_and_refit,
    )

    rng = np.random.default_rng(0)
    n = 500
    base = rng.normal(0.0, 1.0, n)
    # True alpha=2.0; buffer matches deployed alpha exactly under low noise.
    y = 2.0 * base + rng.normal(0.0, 0.01, n)
    new_alpha, new_beta, info = streaming_alpha_check_and_refit(
        y_buffer=y,
        base_buffer=base,
        current_alpha=2.0,
        current_beta=0.0,
        z_threshold=3.0,
    )
    assert info["refit"] is False
    assert info["reason"] in ("no_drift", "degenerate_buffer")

    # Now genuinely drift the buffer (alpha shifted to 5.0).
    y_drift = 5.0 * base + rng.normal(0.0, 0.01, n)
    new_alpha2, new_beta2, info2 = streaming_alpha_check_and_refit(
        y_buffer=y_drift,
        base_buffer=base,
        current_alpha=2.0,
        current_beta=0.0,
        z_threshold=3.0,
    )
    assert info2["refit"] is True
    assert info2["reason"] == "drift_detected"
    assert new_alpha2 == pytest.approx(5.0, abs=0.1)


# ---------------------------------------------------------------------------
# Stacking-aware gate: narrowed exception type still triggers uniform fallback.
# ---------------------------------------------------------------------------


def test_low_stacking_gate_uniform_fallback_on_degenerate_input() -> None:
    """Too-few finite rows force the uniform-weights fallback branch. Tighter
    exception types around the NNLS call must not change observable behavior:
    survivors == input names; weights uniform."""
    from mlframe.training.composite_stacking import stacking_aware_gate

    # Only 2 rows of finite data, 3 transforms -> need >= max(3, 4) finite rows
    # for NNLS; uniform fallback fires.
    preds = {
        "t1": np.array([1.0, 2.0, np.nan, np.nan, np.nan]),
        "t2": np.array([1.0, 2.0, np.nan, np.nan, np.nan]),
        "t3": np.array([1.0, 2.0, np.nan, np.nan, np.nan]),
    }
    y = np.array([1.0, 2.0, np.nan, np.nan, np.nan])
    survivors, weights = stacking_aware_gate(preds, y)
    assert sorted(survivors) == ["t1", "t2", "t3"]
    for v in weights.values():
        assert v == pytest.approx(1.0 / 3.0, abs=1e-6)


# ---------------------------------------------------------------------------
# composite_predictions_as_feature happy path still works after narrowing
# the polars import-fallback to ImportError only.
# ---------------------------------------------------------------------------


def test_low_composite_predictions_as_feature_pandas_roundtrip() -> None:
    """Polars import-failure narrowed from bare-except to ImportError.

    The pandas fast path must continue to attach the prediction column
    without mutating the input frame.
    """
    pd = pytest.importorskip("pandas")
    from mlframe.training.composite_feature_stacking import (
        composite_predictions_as_feature,
    )

    class _FakeWrapper:
        transform_name = "linear_residual"
        base_column = "x"

        def predict(self, df):
            return np.arange(len(df), dtype=np.float64)

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    out = composite_predictions_as_feature(_FakeWrapper(), df)
    assert "composite_pred__linear_residual__x" in out.columns
    assert list(out["composite_pred__linear_residual__x"]) == [0.0, 1.0, 2.0]
    # Source df was NOT mutated (only the returned copy carries the new col).
    assert "composite_pred__linear_residual__x" not in df.columns


# ---------------------------------------------------------------------------
# Comments in composite_estimator.py for the docstring-em-dash strip:
# the docstring still describes the column-dropping behavior accurately.
# ---------------------------------------------------------------------------


def test_polish_drop_columns_docstring_preserved() -> None:
    """``_drop_columns`` docstring kept the WHY (LightGBM rejects object/string
    dtypes). The em-dash em-dash variant was replaced with a hyphen to honour
    the project-wide "no em-dash in prose" rule; the meaning must survive."""
    from mlframe.training.composite import CompositeTargetEstimator

    doc = CompositeTargetEstimator._drop_columns.__doc__ or ""
    assert "LightGBM" in doc
    assert "—" not in doc  # em-dash replaced
