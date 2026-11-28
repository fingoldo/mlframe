"""Cluster 25 regression: the hard logical-constraint passes choose their backend once per call (audit 2026-11-20, finding 25-F53)."""

import numpy as np

import mlframe.inference._ktc_dispatch as ktc
from mlframe.inference.logical_constraints import apply_logical_constraints


def test_a_chain_needing_several_passes_looks_the_backend_up_once(monkeypatch):
    """A reversed 4-label chain takes more than one pass to settle; the backend lookup still happens exactly once and the result is consistent."""
    calls = []
    real = ktc.choose_logical_constraints_backend

    def counting(*a, **k):
        """Count the lookups, delegate to the real chooser."""
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(ktc, "choose_logical_constraints_backend", counting)
    preds = np.array([[0.9, 0.6, 0.3, 0.1]] * 5)
    rules = [(0, 1), (1, 2), (2, 3)]  # child <= parent for each pair
    out = apply_logical_constraints(preds, rules)
    assert len(calls) == 1
    assert (out[:, 0] <= out[:, 1]).all() and (out[:, 1] <= out[:, 2]).all() and (out[:, 2] <= out[:, 3]).all()
