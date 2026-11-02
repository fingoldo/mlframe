"""Numeric init candidates are ordered numerically, and an all-negative label does not wipe out the other labels' weights."""

from __future__ import annotations

import numpy as np

from mlframe.models._optimization_search import CandidateSamplingMethod
from mlframe.models.masked_multilabel_objective import compute_inverse_frequency_class_weights


def _searcher():
    """A bare object exposing the real sampler method (it only needs ``_rng``)."""
    from types import SimpleNamespace

    from mlframe.models import _optimization_search as mod

    cls = next(c for c in vars(mod).values() if isinstance(c, type) and hasattr(c, "_sample_initial_inputs"))
    obj = cls.__new__(cls)
    obj.__dict__.update(SimpleNamespace(_rng=np.random.default_rng(0)).__dict__)
    return obj


def test_descending_init_order_is_numeric_not_alphabetical():
    """Candidates 2, 9, 10, 100 descending are 100, 10, 9, 2 (the string order gave 9, 2, 100, 10)."""
    got = _searcher()._sample_initial_inputs(np.array([2, 9, 10, 100]), 4, CandidateSamplingMethod.Equidistant, [], False, True)
    assert [int(v) for v in got] == [100, 10, 9, 2]


def test_label_without_positives_keeps_the_other_weights_usable():
    """Label 0 has no positives; labels 1 and 2 have a 50% rate: their weights stay near 1, not 6e-6."""
    y = np.zeros((100, 3))
    y[:50, 1] = 1
    y[:50, 2] = 1
    w = compute_inverse_frequency_class_weights(y, np.zeros_like(y, dtype=bool))
    assert w[1] > 0.5 and w[2] > 0.5
