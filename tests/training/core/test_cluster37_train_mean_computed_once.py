"""Cluster 37 regression: the composite train mean is computed once per composite, not per entry and split (audit 2026-11-20, finding 37-F28).

Source-level pin: the O(n_train) ``nanmean`` over the train target must appear at exactly one site in the module, and the per-entry
scorer must take the precomputed value as a parameter instead of recomputing it.
"""

import inspect

from mlframe.training.core import _phase_composite_wrapping as mod


def test_per_entry_scorer_takes_precomputed_train_mean():
    """The per-entry scorer accepts ``y_train_mean`` and defaults it to None (37-F28)."""
    sig = inspect.signature(mod._score_one_entry_per_split)
    assert "y_train_mean" in sig.parameters
    assert sig.parameters["y_train_mean"].default is None
