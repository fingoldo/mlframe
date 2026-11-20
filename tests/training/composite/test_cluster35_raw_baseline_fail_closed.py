"""Audit 2026-11-20 finding 35-F4: a non-finite raw baseline must not silently disable the requested raw-y gate.

With ``require_beats_raw_baseline`` set, the gate fails closed (no spec admitted) instead of shipping every candidate unchecked.
"""

from types import SimpleNamespace

import numpy as np

from mlframe.training.composite.discovery._tiny_rerank import _apply_raw_baseline_gate


def _stub(require: bool):
    """A discovery-like object carrying only the config the gate reads."""
    return SimpleNamespace(
        config=SimpleNamespace(require_beats_raw_baseline=require, raw_baseline_tolerance=1.02, honest_oof_selection_tolerance=1.05),
    )


def test_non_finite_raw_baseline_admits_no_spec_when_gate_is_required():
    """With the gate required and no finite raw baseline, the composite block yields nothing."""
    out = _apply_raw_baseline_gate(
        _stub(require=True), kept_specs=["spec_a", "spec_b"], agg_scores=[0.1, 0.2],
        raw_baseline=float("nan"), honest_oof_baseline=float("nan"), raw_per_seed_per_family={}, raw_per_bin_per_base={},
        spec_per_bin_rmse={}, per_bin_enabled=False, per_bin_tol=1.0, families=["ridge"], use_wilcoxon=False,
        y_screen=np.linspace(0.0, 1.0, 50),
    )
    assert out == ([], [0.1, 0.2])
