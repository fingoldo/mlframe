"""Cluster 24 P3 regressions: the documented representative default matches the constructor, and the twin-prune module still works after its dead
local was removed (audit 2026-11-20, findings 24-F59 and 24-F66).

The _rep_member docstring named sum_fi as the default, but the constructor default is cluster_rep="first". The unused ``classes`` local in the twin
prune was removed; the pruning must still run.
"""

import numpy as np
import pandas as pd

from mlframe.feature_selection._hybrid_twin_prune import prune_conditional_twins
from mlframe.feature_selection.hybrid_selector import HybridSelector


def test_rep_member_docstring_names_the_constructor_default():
    """The constructor default is cluster_rep='first', and the docstring marks 'first' as the default, not sum_fi (24-F59)."""
    assert HybridSelector().cluster_rep == "first"
    doc = HybridSelector._rep_member.__doc__ or ""
    assert '"first" (the constructor default)' in doc


def test_twin_prune_runs_without_the_removed_local():
    """The pass runs after the dead local was removed: its output is an order-preserving subset of its input (24-F66).

    Whether a twin is actually dropped depends on the held-out gain test, so this checks the contract rather than one outcome.
    """
    rng = np.random.default_rng(0)
    n = 400
    base = rng.normal(size=n)
    X = pd.DataFrame({"a": base, "b": base + 1e-6 * rng.normal(size=n), "c": rng.normal(size=n)})
    y = (base + 0.3 * rng.normal(size=n) > 0).astype(int)
    fi = {"a": 0.5, "b": 0.4, "c": 0.01}
    selected = ["a", "b", "c"]
    kept = prune_conditional_twins(X, y, selected, fi, twin_corr=0.99, classification=True, seed=0)
    assert set(kept) <= set(selected)
    assert kept == [c for c in selected if c in kept]
