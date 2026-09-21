"""Guards that cannot evaluate a candidate drop it instead of waving it through.

The base-leakage guard kept a base whenever its check raised, the structural-fragility gate kept a spec whose base column
it could not read, and auto-chain kept a winning chain whose final fit failed, with empty params its inverse cannot use.
"""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pandas as pd

from mlframe.training.composite.discovery import _auto_chain
from mlframe.training.composite.discovery._fit_temporal import apply_base_leakage_guard
from mlframe.training.composite.discovery._yscale_holdout_gate import apply_structural_fragility_gate
from mlframe.training.composite.transforms import get_transform

from .test_biz_val_discovery_yscale_holdout_gate import _grouped_frame, _make_gate_ctx, _spec


def test_a_base_the_leakage_guard_cannot_check_is_dropped():
    """A text column raises inside the leakage check; it is dropped with the reason, not kept."""
    n = 200
    df = pd.DataFrame({"num": np.arange(n, dtype=float), "text": ["a"] * n})
    disc = types.SimpleNamespace()
    kept = apply_base_leakage_guard(disc, df, ["num", "text"], np.arange(n), np.arange(n, dtype=float) * 2.0 + 1.0, np.arange(n))
    assert "text" not in kept
    assert any(name == "text" and "leakage check failed" in reason for name, reason in disc._leaky_bases_dropped_)


def test_the_fragility_gate_rejects_a_spec_whose_base_it_cannot_read():
    """A base column missing from the frame is rejected with a ledger row, not kept as a survivor."""
    df, groups, y = _grouped_frame()
    disc = _make_gate_ctx(groups)
    ghost = dataclasses.replace(_spec("y-linres-ghost", alpha=1.0), base_column="no_such_column")
    out = apply_structural_fragility_gate(disc, df, [ghost], np.arange(len(df)), y)
    assert out == []
    assert any(r["stage"] == "structural_fragility" and "could not be read" in r["reason"] for r in disc.rejection_ledger)


def test_a_chain_whose_final_fit_fails_is_not_a_candidate():
    """A winning chain whose fit raises yields no candidate instead of one with empty params."""
    tf = get_transform("linear_residual")

    def _raise(*_a, **_k):
        raise ValueError("cannot fit")

    broken = dataclasses.replace(tf, fit=_raise)
    y = np.linspace(1.0, 10.0, 50)
    assert _auto_chain._fitted_chain_candidate(broken, "linear_residual", "cbrt", y=y, base=y.copy(), rmse=1.0) is None
