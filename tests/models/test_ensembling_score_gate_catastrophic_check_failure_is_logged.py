"""A failing catastrophic-drop check is logged at WARNING even when ``verbose`` is off.

The gate proceeds without the drop when its check raises, so the ensemble keeps a member that was meant to be removed. That outcome needs
to be visible in a normal run, not only in a verbose one.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np

from mlframe.models.ensembling import score_gate


def test_failed_k2_catastrophic_check_logs_a_warning_without_verbose(caplog):
    """A check that raises inside the K=2 gate proceeds without the drop and says so at WARNING level."""
    n = 40
    target = np.linspace(0.0, 1.0, n)
    members = [SimpleNamespace(), SimpleNamespace()]
    unusable = np.array(["x"] * n, dtype=object)
    with caplog.at_level(logging.WARNING, logger=score_gate.logger.name):
        out = score_gate.catastrophic_drop_k2(
            level_models_and_predictions=members,
            _gate_preds_for_check=[unusable, target],
            _gate_source_split="oof",
            _ensemble_member_tags=["a", "b"],
            _ensemble_short_tags=["a", "b"],
            ensemble_name="[a+b] ensemble",
            train_target_arr=target,
            val_target_arr=None,
            test_target_arr=None,
            k2_catastrophic_mae_ratio=20.0,
            verbose=False,
            res={},
        )
    assert out[-1] is False
    assert any(r.levelno == logging.WARNING and "catastrophic" in r.getMessage() for r in caplog.records)
