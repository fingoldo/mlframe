"""Audit 2026-11-20 finding 35-F36: a multi-base spec passed the structural gate without a word; the bypass is logged at WARNING now."""

import logging
from types import SimpleNamespace

import numpy as np

from mlframe.training.composite.discovery._yscale_holdout_gate import apply_structural_fragility_gate


def test_multi_base_spec_bypass_is_logged(caplog):
    """A spec with extra_base_columns survives the gate and a WARNING names it."""
    n = 80
    cfg = SimpleNamespace(structural_fragility_gate_enabled=True, random_state=0)
    self_ = SimpleNamespace(config=cfg, _group_ids_for_rerank=np.repeat(np.arange(8), 10))
    spec = SimpleNamespace(name="multi_spec", base_column="b", extra_base_columns=("c",), transform_name="linear_residual_multi")
    y = np.linspace(0.0, 1.0, n)
    with caplog.at_level(logging.WARNING):
        out = apply_structural_fragility_gate(self_, None, [spec], np.arange(n), y)
    assert out == [spec]
    assert any("multi-base spec multi_spec" in r.getMessage() for r in caplog.records)
