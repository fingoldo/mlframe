"""T1#7 2026-05-18 lazy recovery of composite-target y-scale metrics.

When ``skip_wrap_pass_predict=True`` (default since 2026-05-18) the suite
populates ``models`` with wrapped CompositeTargetEstimator entries but
leaves ``metadata["composite_target_y_scale_metrics"]`` empty.

These tests pin the contract that:

1. Eager path: ``_run_composite_target_wrapping(skip_predict=False)`` fills
   the metric dict.
2. Skipped path: ``skip_predict=True`` leaves it empty (regression guard).
3. Lazy recovery: ``recover_composite_y_scale_metrics`` recovers the same
   numbers when invoked on the already-wrapped models (idempotent against
   the wrap step).
4. Biz value: the lazy-recovered metrics match the eager metrics exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.base import BaseEstimator, RegressorMixin


class _LinearInner(BaseEstimator, RegressorMixin):
    """T-scale inner regressor that fits ``T = T_true * scale + noise``."""

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def fit(self, X, y):
        # OLS on raw X to keep the test deterministic.
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        self.coef_, *_ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return (X_arr @ self.coef_) * self.scale


def _build_problem():
    """Tiny linear_residual composite: y = 1.5 * base + epsilon."""
    from mlframe.training.composite_estimator import CompositeTargetEstimator
    from mlframe.training.composite_transforms import get_transform

    rng = np.random.default_rng(42)
    n = 200
    base = rng.normal(50.0, 10.0, n)
    noise = rng.normal(0.0, 1.0, n)
    y = 1.5 * base + 3.0 + noise
    df = pd.DataFrame({"base": base, "x1": rng.normal(size=n)})

    transform = get_transform("linear_residual")
    params = transform.fit(y, base)
    T = transform.forward(y, base, params)

    inner = _LinearInner().fit(df[["base", "x1"]].values, T)
    wrapper = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner,
        transform_name="linear_residual",
        base_column="base",
        transform_fitted_params=params,
        y_train=y,
    )

    spec = {
        "name": "y-linres-base",
        "transform_name": "linear_residual",
        "base_column": "base",
        "fitted_params": params,
    }

    class _Entry:
        def __init__(self, m):
            self.model = m
            self.model_name = "LinearInner"

    # Build PRE-wrap models dict (raw inner) so the wrap step can exercise.
    pre_wrap_inner = _LinearInner().fit(df[["base", "x1"]].values, T)

    return {
        "df": df, "y": y, "base": base, "T": T, "spec": spec,
        "params": params, "wrapper": wrapper,
        "pre_wrap_inner": pre_wrap_inner,
        "Entry": _Entry,
    }


def _eager_run(problem):
    """Run the wrap phase with skip_predict=False on a fresh inner."""
    from mlframe.training.core._phase_composite_post import _run_composite_target_wrapping

    Entry = problem["Entry"]
    models = {
        "regression": {
            "y-linres-base": [Entry(problem["pre_wrap_inner"])],
        },
    }
    metadata: dict = {}
    n = len(problem["df"])
    train_idx = np.arange(int(0.7 * n))
    val_idx = np.arange(int(0.7 * n), n)
    _run_composite_target_wrapping(
        models=models,
        metadata=metadata,
        target_by_type={"regression": {"y": problem["y"]}},
        composite_specs_by_target_type={
            "regression": {"y": [problem["spec"]]},
        },
        filtered_train_idx=train_idx,
        filtered_train_df=problem["df"].iloc[train_idx].reset_index(drop=True),
        filtered_val_idx=val_idx,
        filtered_val_df=problem["df"].iloc[val_idx].reset_index(drop=True),
        test_idx=None,
        test_df_pd=None,
        skip_predict=False,
    )
    return models, metadata, train_idx, val_idx


def _skipped_then_recovered_run(problem):
    """Run with skip_predict=True, then call the lazy recovery helper."""
    from mlframe.training.core._phase_composite_post import (
        _run_composite_target_wrapping,
        recover_composite_y_scale_metrics,
    )

    Entry = problem["Entry"]
    # Fresh inner so both runs start from identical state.
    fresh_inner = _LinearInner().fit(
        problem["df"][["base", "x1"]].values, problem["T"],
    )
    models = {
        "regression": {
            "y-linres-base": [Entry(fresh_inner)],
        },
    }
    metadata: dict = {}
    n = len(problem["df"])
    train_idx = np.arange(int(0.7 * n))
    val_idx = np.arange(int(0.7 * n), n)
    train_df = problem["df"].iloc[train_idx].reset_index(drop=True)
    val_df = problem["df"].iloc[val_idx].reset_index(drop=True)

    # Phase 1: wrap with skip_predict=True; metadata stays empty.
    _run_composite_target_wrapping(
        models=models,
        metadata=metadata,
        target_by_type={"regression": {"y": problem["y"]}},
        composite_specs_by_target_type={
            "regression": {"y": [problem["spec"]]},
        },
        filtered_train_idx=train_idx,
        filtered_train_df=train_df,
        filtered_val_idx=val_idx,
        filtered_val_df=val_df,
        test_idx=None,
        test_df_pd=None,
        skip_predict=True,
    )
    skipped_metadata_snapshot = {
        k: v for k, v in metadata.items() if "y_scale_metrics" in k
    }

    # Phase 2: lazy recovery on the already-wrapped models.
    recover_composite_y_scale_metrics(
        models=models,
        metadata=metadata,
        target_by_type={"regression": {"y": problem["y"]}},
        composite_specs_by_target_type={
            "regression": {"y": [problem["spec"]]},
        },
        filtered_train_idx=train_idx,
        filtered_train_df=train_df,
        filtered_val_idx=val_idx,
        filtered_val_df=val_df,
        test_idx=None,
        test_df_pd=None,
    )

    return models, metadata, skipped_metadata_snapshot, train_idx, val_idx


class TestSkipPredictLeavesMetricsEmpty:
    """Regression guard: ``skip_predict=True`` leaves the metric dict empty
    (this is the documented contract that motivates the lazy recovery helper)."""

    def test_metrics_dict_empty_when_skipped(self) -> None:
        problem = _build_problem()
        _, metadata, snapshot, _, _ = _skipped_then_recovered_run(problem)
        # After the skip_predict=True call but BEFORE recovery, metadata
        # should not carry y-scale metrics for the composite.
        assert "composite_target_y_scale_metrics" not in snapshot, (
            "skip_predict=True should NOT populate "
            "composite_target_y_scale_metrics; got snapshot="
            f"{snapshot}"
        )


class TestRecoverComposeYScaleMetrics:
    """Lazy recovery populates the metric dict with the same shape as the
    eager path; idempotent against the already-wrapped models."""

    def test_recovery_populates_metric_dict(self) -> None:
        problem = _build_problem()
        _, metadata, _, _, _ = _skipped_then_recovered_run(problem)
        m = metadata.get("composite_target_y_scale_metrics", {})
        assert "regression" in m, f"recovery missing 'regression' key; got: {list(m.keys())}"
        per_target = m["regression"]
        assert "y-linres-base" in per_target, (
            f"recovery missing composite entry; got: {list(per_target.keys())}"
        )
        entries = per_target["y-linres-base"]
        assert isinstance(entries, list) and len(entries) >= 1, (
            f"composite entry must be a non-empty list; got: {entries!r}"
        )

    def test_recovery_is_idempotent(self) -> None:
        """Calling the helper twice on the same wrapped models produces the same numbers."""
        from mlframe.training.core._phase_composite_post import recover_composite_y_scale_metrics

        problem = _build_problem()
        models, metadata, _, train_idx, val_idx = _skipped_then_recovered_run(problem)
        first_metrics = metadata["composite_target_y_scale_metrics"]["regression"]["y-linres-base"][0]["metrics"]

        # Call again on the same wrapped models; must not double-wrap or change the numbers.
        recover_composite_y_scale_metrics(
            models=models,
            metadata=metadata,
            target_by_type={"regression": {"y": problem["y"]}},
            composite_specs_by_target_type={
                "regression": {"y": [problem["spec"]]},
            },
            filtered_train_idx=train_idx,
            filtered_train_df=problem["df"].iloc[train_idx].reset_index(drop=True),
            filtered_val_idx=val_idx,
            filtered_val_df=problem["df"].iloc[val_idx].reset_index(drop=True),
            test_idx=None,
            test_df_pd=None,
        )
        second_metrics = metadata["composite_target_y_scale_metrics"]["regression"]["y-linres-base"][0]["metrics"]
        # The metric numbers must match to within floating-point precision.
        for split in ("train", "val"):
            if split in first_metrics and split in second_metrics:
                for metric in ("RMSE", "MAE", "R2"):
                    if metric in first_metrics[split]:
                        assert np.isclose(
                            first_metrics[split][metric],
                            second_metrics[split][metric],
                            equal_nan=True,
                        ), (
                            f"second recovery diverged on {split}/{metric}: "
                            f"first={first_metrics[split][metric]}, "
                            f"second={second_metrics[split][metric]}"
                        )


class TestRecoveryBizValueMatchesEager:
    """biz_value: numbers from lazy recovery must MATCH numbers from the eager
    path. If they diverge the recovery is silently giving wrong answers."""

    def test_recovered_metrics_match_eager_path(self) -> None:
        problem = _build_problem()

        _, eager_metadata, _, _ = _eager_run(problem)
        eager_metrics = eager_metadata["composite_target_y_scale_metrics"]["regression"]["y-linres-base"][0]["metrics"]

        problem2 = _build_problem()
        _, lazy_metadata, _, _, _ = _skipped_then_recovered_run(problem2)
        lazy_metrics = lazy_metadata["composite_target_y_scale_metrics"]["regression"]["y-linres-base"][0]["metrics"]

        # Both paths used the same RNG-seeded data, same inner coefficients,
        # same wrap. The per-split metrics must be bit-for-bit identical
        # (no resampling, no parallelism that could reorder floats).
        for split in ("train", "val"):
            assert split in eager_metrics, f"eager missing split={split}"
            assert split in lazy_metrics, f"lazy missing split={split}"
            for metric in ("RMSE", "MAE", "R2"):
                if metric not in eager_metrics[split]:
                    continue
                ev = eager_metrics[split][metric]
                lv = lazy_metrics[split][metric]
                assert np.isclose(ev, lv, atol=1e-10, equal_nan=True), (
                    f"eager vs lazy diverged on {split}/{metric}: "
                    f"eager={ev}, lazy={lv}"
                )
