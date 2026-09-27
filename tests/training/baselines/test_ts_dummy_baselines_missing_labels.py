"""Time-series rule baselines are not built from a train series with many unlabelled rows taken out."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.baselines._dummy_baseline_regression import _compute_regression_baselines
from mlframe.training.configs import DummyBaselinesConfig
from mlframe.training.core._target_row_scope import target_row_scope
from mlframe.training.core._target_rows import build_target_rows, splits_of
from mlframe.training.core._training_context import TrainingContext


def _run():
    n_tr, n_va, n_te = 300, 60, 60
    rng = np.random.default_rng(0)
    frame = lambda n: pd.DataFrame({"x": rng.normal(size=n)})
    ts = np.arange(n_tr + n_va + n_te, dtype=np.float64)
    return _compute_regression_baselines(
        "y", frame(n_tr), frame(n_va), frame(n_te), rng.normal(size=n_tr), rng.normal(size=n_va), rng.normal(size=n_te),
        ts[:n_tr], ts[n_tr:n_tr + n_va], ts[n_tr + n_va:], None, DummyBaselinesConfig(),
    )[2]


def test_rule_baselines_are_skipped_inside_a_scope_with_many_gaps_and_built_outside():
    assert "skipped" not in (_run().get("ts_diagnostics") or {})
    ctx = TrainingContext()
    ctx.train_idx, ctx.val_idx, ctx.test_idx = np.arange(300), np.arange(300, 360), np.arange(360, 420)
    mask = np.random.default_rng(1).random(420) > 0.3
    with target_row_scope(ctx, build_target_rows(mask, splits_of(ctx))):
        assert "no label" in _run()["ts_diagnostics"]["skipped"]
