"""Regression: cheap FE candidate scans must be invariant to input column order.

The row-argmax / conditional-gate / pairwise-modular scans enumerate column
combinations under a budget and sort hits. Without a canonical column order +
secondary tie-break key, a reversed-column frame walks a different budgeted
combination prefix and breaks near-ties positionally, yielding a different set
of engineered candidates -- which then drives a column-order-dependent MRMR
selection (the [MRMR] case of test_reversed_columns_select_same_names).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._conditional_gate_fe import (
    cheap_conditional_gate_scan,
    cheap_row_argmax_scan,
)
from mlframe.feature_selection.filters._pairwise_modular_fe import cheap_modular_scan


def _hit_keys_argmax(hits):
    """Hit keys argmax."""
    return [tuple(sorted(str(c) for c in h.cols)) for h in hits]


def _hit_keys_gate(hits):
    """Hit keys gate."""
    return [(str(h.mode), tuple(sorted(str(c) for c in h.cols))) for h in hits]


def _hit_keys_mod(hits):
    """Hit keys mod."""
    return [(str(h.op), tuple(sorted(str(c) for c in h.cols))) for h in hits]


def test_row_argmax_scan_is_column_order_invariant():
    """Row argmax scan is column order invariant."""
    rng = np.random.default_rng(0)
    n = 2000
    cols = [f"f{i}" for i in range(8)]
    X = pd.DataFrame({c: rng.standard_normal(n) for c in cols})
    y = (np.argmax(X[["f0", "f3", "f5"]].to_numpy(), axis=1) > 0).astype(np.int64)
    fwd = cheap_row_argmax_scan(X, y, max_triples=20, seed=0)
    rev = cheap_row_argmax_scan(X[cols[::-1]], y, max_triples=20, seed=0)
    assert set(_hit_keys_argmax(fwd)) == set(_hit_keys_argmax(rev))


def test_conditional_gate_scan_is_column_order_invariant():
    """Conditional gate scan is column order invariant."""
    rng = np.random.default_rng(1)
    n = 2000
    cols = [f"f{i}" for i in range(8)]
    X = pd.DataFrame({c: rng.standard_normal(n) for c in cols})
    a, b, c = X["f0"].to_numpy(), X["f2"].to_numpy(), X["f4"].to_numpy()
    y = (np.where(c > 0.0, a, b) > 0.0).astype(np.int64)
    fwd = cheap_conditional_gate_scan(X, y, seed=0)
    rev = cheap_conditional_gate_scan(X[cols[::-1]], y, seed=0)
    assert set(_hit_keys_gate(fwd)) == set(_hit_keys_gate(rev))


def test_modular_scan_is_column_order_invariant():
    """Modular scan is column order invariant."""
    rng = np.random.default_rng(2)
    n = 2000
    cols = [f"f{i}" for i in range(6)]
    X = pd.DataFrame({c: rng.integers(0, 100, n) for c in cols})
    y = ((X["f0"].to_numpy() % 7) + (X["f3"].to_numpy() % 7) > 6).astype(np.int64)
    fwd = cheap_modular_scan(X, y, max_pairs=10, max_triples=10, seed=0)
    rev = cheap_modular_scan(X[cols[::-1]], y, max_pairs=10, max_triples=10, seed=0)
    assert set(_hit_keys_mod(fwd)) == set(_hit_keys_mod(rev))
