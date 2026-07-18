"""Sensor: baselines.diagnostics method-rebinding preserves identity + class invariants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlframe.training.baselines import diagnostics as parent
from mlframe.training.baselines import _baseline_diagnostics_ablation as ablation_mod
from mlframe.training.baselines import _baseline_diagnostics_init_score as init_score_mod
from mlframe.training.baselines import _baseline_diagnostics_quick_model as quick_model_mod
from mlframe.training.baselines import _baseline_diagnostics_recommend as recommend_mod
from mlframe.training.configs import BaselineDiagnosticsConfig


def test_w12b_baseline_diagnostics_methods_rebound():
    """W12b baseline diagnostics methods rebound."""
    cls = parent.BaselineDiagnostics
    assert cls._make_quick_model is quick_model_mod._make_quick_model
    assert cls._fit_quick_and_score is quick_model_mod._fit_quick_and_score
    assert cls._run_ablation is ablation_mod._run_ablation
    assert cls._fit_init_score_baseline is init_score_mod._fit_init_score_baseline
    assert cls._build_recommendation is recommend_mod._build_recommendation


def test_w12b_baseline_diagnostics_facade_under_budget():
    """W12b baseline diagnostics facade under budget."""
    facade_loc = sum(1 for _ in Path(parent.__file__).open(encoding="utf-8"))
    assert facade_loc < 750, f"baselines/diagnostics.py LOC={facade_loc} exceeds 750 budget"


def test_w12b_baseline_diagnostics_class_identity_preserved():
    """W12b baseline diagnostics class identity preserved."""
    cfg = BaselineDiagnosticsConfig(enabled=True)
    diag = parent.BaselineDiagnostics(cfg)
    assert isinstance(diag, parent.BaselineDiagnostics)


def test_w12b_baseline_diagnostics_smoke_fit_regression():
    """W12b baseline diagnostics smoke fit regression."""
    cfg = BaselineDiagnosticsConfig(enabled=True)
    diag = parent.BaselineDiagnostics(cfg)
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )
    y = (X["a"] + 0.3 * X["b"]).to_numpy()
    rep = diag.fit_and_report(X, y, ["a", "b", "c"], "regression", "smoke_target")
    assert isinstance(rep, parent.BaselineDiagnosticsReport)
    assert rep.target_name == "smoke_target"
    assert not rep.skipped
    assert np.isfinite(rep.headline_metric_value)
    # `a` dominates -- should be ranked 1 in ablation entries.
    assert len(rep.ablation) >= 1
