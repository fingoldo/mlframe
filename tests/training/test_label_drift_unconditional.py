"""Label-distribution drift report is auto-wired and runs unconditionally before training (A7-12).

SELECTION_BIAS.md claims ``compute_label_distribution_drift`` is wired into the suite automatically and fires before
training. The per-target diagnostics entrypoint takes no ``verbose`` flag and always computes + stamps the drift report
into metadata, so the diagnostic does not depend on verbose. This pins that contract so a future refactor cannot make
the drift computation/metadata conditional on verbose.
"""
from __future__ import annotations

import inspect

import numpy as np


def test_run_per_target_diagnostics_has_no_verbose_gate():
    """The diagnostics entrypoint must not accept a verbose flag -- proof the drift computation isn't verbose-gated."""
    from mlframe.training.core._phase_diagnostics import run_per_target_diagnostics

    params = set(inspect.signature(run_per_target_diagnostics).parameters)
    assert "verbose" not in params, "diagnostics gained a verbose param; ensure drift still runs unconditionally"


def test_label_drift_computed_and_stamped_into_metadata():
    """Calling the diagnostics entrypoint stamps a label_distribution_drift entry regardless of any verbose setting."""
    import pandas as pd
    from types import SimpleNamespace
    from mlframe.training.core._phase_diagnostics import run_per_target_diagnostics

    rng = np.random.default_rng(0)
    n = 300
    train_df = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    val_df = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    test_df = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    # Deliberate prior shift: train P(y=1)=0.3, val/test 0.7 -> drift report should populate.
    train_y = (rng.random(n) < 0.3).astype(int)
    val_y = (rng.random(n) < 0.7).astype(int)
    test_y = (rng.random(n) < 0.7).astype(int)

    bd_cfg = SimpleNamespace(enabled=False, apply_to_target_types=())
    metadata: dict = {}
    metadata = run_per_target_diagnostics(
        target_type="binary_classification",
        cur_target_name="tgt",
        current_train_target=train_y,
        current_val_target=val_y,
        current_test_target=test_y,
        filtered_train_df=train_df,
        filtered_val_df=val_df,
        filtered_test_df=test_df,
        baseline_diagnostics_config=bd_cfg,
        cat_features=[],
        metadata=metadata,
    )
    assert "label_distribution_drift" in metadata
    drift = metadata["label_distribution_drift"]["binary_classification"]["tgt"]
    assert drift is not None
