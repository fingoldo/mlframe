"""Per-model immediate y-scale emit for composite targets.

A composite (e.g. ``TVT-diff-TVT_prev``) is a TRANSFORM of the original
target; the inner model's raw predictions are in the residual/composite
scale. Without this hook the operator sees NO per-model feedback for
composites until end-of-target (the T-scale chart is skipped + the T-scale
metric numbers are suppressed by upstream guards). The hook wraps the
freshly-fit inner in CompositeTargetEstimator and emits a TEST-split
y-scale chart + log line right after the fit, in the ORIGINAL target scale.

The wrap is idempotent (a second call on an already-wrapped entry reuses
it without rebuilding) so the end-of-target wrap-pass remains a safe
fallback for the full train/val/test metrics block + watchdog.
"""
from __future__ import annotations

import logging
import types

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.core._phase_composite_wrapping import (
    emit_per_model_composite_y_scale_test,
)


def _build_diff_setup():
    """y = base + small_noise -> diff target = y - base = small_noise.

    Inner learns the diff; wrapper.predict(test_df) must invert back to y.
    """
    rng = np.random.default_rng(0)
    n = 60
    # Two splits: first 30 train, last 30 test.
    base = rng.normal(11500, 600, n).astype(np.float64)
    diff_noise = rng.normal(0, 5, n).astype(np.float64)
    y_full = base + diff_noise
    feat = base / 1e3 + rng.normal(0, 0.1, n)  # weakly informative
    df = pd.DataFrame({"base": base, "feat": feat})
    test_idx = np.arange(30, n)
    train_idx = np.arange(0, 30)

    # Fit a tiny inner on the diff target on the train split.
    from sklearn.linear_model import Ridge
    inner = Ridge(alpha=1.0).fit(df.iloc[train_idx], diff_noise[train_idx])
    entry = types.SimpleNamespace(model=inner)
    spec = {
        "name": "TVT-diff-TVT_prev",
        "transform_name": "diff",
        "base_column": "base",
        "fitted_params": {},
    }
    return df, test_idx, y_full, spec, entry


def test_per_model_emit_wraps_inner_and_logs_y_scale(caplog) -> None:
    df, test_idx, y_full, spec, entry = _build_diff_setup()
    with caplog.at_level(logging.INFO,
                         logger="mlframe.training.core._phase_composite_wrapping"):
        emit_per_model_composite_y_scale_test(
            entry=entry, composite_spec=spec,
            orig_target_name="TVT", composite_name="TVT-diff-TVT_prev",
            target_name="TVT-diff-TVT_prev",
            y_full=y_full,
            test_idx=test_idx, test_df_pd=df.iloc[test_idx],
            plot_file=None, reporting_config=None,
        )
    # Inner is now wrapped in CompositeTargetEstimator (in-place).
    assert isinstance(entry.model, CompositeTargetEstimator), (
        f"entry.model not wrapped: {type(entry.model).__name__}"
    )
    # Per-model y-scale log line present.
    msgs = "\n".join(r.getMessage() for r in caplog.records)
    assert "y-scale, per-model immediate" in msgs, (
        f"y-scale per-model log line missing; got:\n{msgs}"
    )
    # Sanity: predict returns y-scale values (close to y_full, NOT close to diff).
    y_pred = np.asarray(entry.model.predict(df.iloc[test_idx]), dtype=np.float64)
    err_yscale = float(np.mean(np.abs(y_pred - y_full[test_idx])))
    err_diffscale = float(np.mean(np.abs(y_pred - 0.0)))  # mean diff target ~ 0
    assert err_yscale < err_diffscale, (
        f"wrapped predict still on composite scale: y-err={err_yscale}, "
        f"diff-err={err_diffscale}"
    )


def test_per_model_emit_is_idempotent_on_already_wrapped_entry() -> None:
    df, test_idx, y_full, spec, entry = _build_diff_setup()
    # First call: wraps.
    emit_per_model_composite_y_scale_test(
        entry=entry, composite_spec=spec,
        orig_target_name="TVT", composite_name="TVT-diff-TVT_prev",
        target_name="TVT-diff-TVT_prev",
        y_full=y_full, test_idx=test_idx, test_df_pd=df.iloc[test_idx],
    )
    assert isinstance(entry.model, CompositeTargetEstimator)
    wrapped_first = entry.model
    # Second call: must NOT rebuild the wrapper (the end-of-target pass
    # also re-runs and should idempotently no-op the wrap step).
    emit_per_model_composite_y_scale_test(
        entry=entry, composite_spec=spec,
        orig_target_name="TVT", composite_name="TVT-diff-TVT_prev",
        target_name="TVT-diff-TVT_prev",
        y_full=y_full, test_idx=test_idx, test_df_pd=df.iloc[test_idx],
    )
    assert entry.model is wrapped_first, (
        "second call rebuilt the wrapper instead of reusing it"
    )


def test_per_model_emit_swallows_missing_inputs() -> None:
    # Must never raise -- training continues regardless of reporting hook.
    df, test_idx, y_full, spec, _entry = _build_diff_setup()
    entry = types.SimpleNamespace(model=None)
    emit_per_model_composite_y_scale_test(
        entry=entry, composite_spec=spec,
        orig_target_name="TVT", composite_name="TVT-diff-TVT_prev",
        target_name="TVT-diff-TVT_prev",
        y_full=y_full, test_idx=test_idx, test_df_pd=df.iloc[test_idx],
    )
    # Inner was None -> no wrap, no crash.
    assert entry.model is None
