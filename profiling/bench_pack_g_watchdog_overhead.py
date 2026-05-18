"""HIGH#4 2026-05-18: measure Pack G runtime watchdog cost.

The watchdog adds two extra ``predict()`` calls per (entry, split) - one for
``wrapper.predict(X)`` and one implicit inside ``inner.predict(X)`` for the
additive-error invariant. On a wide model zoo this can dominate wall time.

This benchmark measures the OVERHEAD by running ``_run_composite_target_wrapping``
once with ``enable_watchdog=True`` and once with ``enable_watchdog=False`` on
the same set of wrapped models / splits.

Run: python profiling/bench_pack_g_watchdog_overhead.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


class _OLSInner(BaseEstimator, RegressorMixin):
    """Linear inner regressor for the bench - cheap predict so the watchdog
    cost stands out relative to inner cost."""

    def __init__(self) -> None:
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)


def _build_problem(n: int = 200_000, n_composites: int = 2, n_models: int = 5):
    """Synthetic linear residual problem with multiple composites and models
    in the dict so the watchdog has many (entry, split) pairs to check."""
    from mlframe.training.composite_estimator import CompositeTargetEstimator
    from mlframe.training.composite_transforms import get_transform

    rng = np.random.default_rng(2026)
    base = rng.normal(100.0, 20.0, n)
    y = 1.5 * base + 5.0 + rng.normal(0.0, 1.0, n)
    df = pd.DataFrame({
        "base": base,
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })

    transform = get_transform("linear_residual")
    params = transform.fit(y, base)
    T = transform.forward(y, base, params)

    class _Entry:
        def __init__(self, m):
            self.model = m
            self.model_name = type(m).__name__

    # Wrap n_models inner models per composite, n_composites composites.
    models = {"regression": {}}
    composite_specs_by_target_type = {"regression": {"y": []}}
    for c in range(n_composites):
        comp_name = f"y-linres-base-{c}"
        entries = []
        for _ in range(n_models):
            inner = _OLSInner().fit(df.values, T)
            wrapper = CompositeTargetEstimator.from_fitted_inner(
                fitted_inner=inner,
                transform_name="linear_residual",
                base_column="base",
                transform_fitted_params=params,
                y_train=y,
            )
            entries.append(_Entry(wrapper))
        models["regression"][comp_name] = entries
        composite_specs_by_target_type["regression"]["y"].append({
            "name": comp_name,
            "transform_name": "linear_residual",
            "base_column": "base",
            "fitted_params": params,
        })

    train_idx = np.arange(int(0.7 * n))
    val_idx = np.arange(int(0.7 * n), n)

    return {
        "models": models,
        "composite_specs_by_target_type": composite_specs_by_target_type,
        "target_by_type": {"regression": {"y": y}},
        "train_idx": train_idx, "val_idx": val_idx,
        "train_df": df.iloc[train_idx].reset_index(drop=True),
        "val_df": df.iloc[val_idx].reset_index(drop=True),
    }


def main() -> int:
    from mlframe.training.core._phase_composite_post import (
        _run_composite_target_wrapping,
    )

    print("=" * 70)
    print("HIGH#4 Pack G watchdog overhead measurement")
    print("=" * 70)

    # Build problem twice so we can run two independent dispatches without
    # mutation contamination (wrap step replaces inner with wrapper).
    ctx_on = _build_problem(n=200_000, n_composites=2, n_models=5)
    ctx_off = _build_problem(n=200_000, n_composites=2, n_models=5)
    print(f"Problem: n=200_000, n_composites=2, n_models=5 "
          f"-> 30 (entry, split) pairs per dispatch")

    def _run(ctx, *, enable_watchdog: bool) -> float:
        t0 = time.perf_counter()
        _run_composite_target_wrapping(
            models=ctx["models"],
            metadata={},
            target_by_type=ctx["target_by_type"],
            composite_specs_by_target_type=ctx["composite_specs_by_target_type"],
            filtered_train_idx=ctx["train_idx"],
            filtered_train_df=ctx["train_df"],
            filtered_val_idx=ctx["val_idx"],
            filtered_val_df=ctx["val_df"],
            test_idx=None,
            test_df_pd=None,
            skip_predict=False,
            enable_watchdog=enable_watchdog,
        )
        return time.perf_counter() - t0

    # Warm up.
    _run(_build_problem(n=10_000), enable_watchdog=True)

    # Median of 3 runs.
    on_times = []
    off_times = []
    for _ in range(3):
        ctx_on_fresh = _build_problem(n=200_000)
        ctx_off_fresh = _build_problem(n=200_000)
        on_times.append(_run(ctx_on_fresh, enable_watchdog=True))
        off_times.append(_run(ctx_off_fresh, enable_watchdog=False))

    med_on = float(np.median(on_times))
    med_off = float(np.median(off_times))
    overhead_pct = 100.0 * (med_on - med_off) / max(med_off, 1e-9)

    print()
    print(f"  watchdog ON:  median {med_on:.3f}s  (runs: {on_times})")
    print(f"  watchdog OFF: median {med_off:.3f}s  (runs: {off_times})")
    print(f"  overhead:     {overhead_pct:+.1f}%")
    print()
    if overhead_pct > 10.0:
        print("  ASSESSMENT: watchdog adds material overhead on this size; "
              "production callers with verified wrappers may want to disable.")
    else:
        print("  ASSESSMENT: watchdog overhead negligible on this size; "
              "keep enabled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
