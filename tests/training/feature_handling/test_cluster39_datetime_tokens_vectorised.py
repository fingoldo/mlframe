"""Cluster 39 regression: vectorised datetime tokens are byte-identical to the per-element form (audit 2026-11-20, finding 39-F44)."""

import numpy as np

from mlframe.training.feature_handling._target_encoders_canon import _temporal_to_epoch_ns_tokens


def test_tokens_match_per_element_formula():
    """Each token is ``dt:<epoch-ns>``, exactly as the old Python loop produced (39-F44)."""
    ns = np.array([0, 1_577_882_400_000_000_000, -5], dtype=np.int64)
    expected = np.array(["dt:" + str(int(v)) for v in ns], dtype=object)
    got = _temporal_to_epoch_ns_tokens(ns, np.zeros(ns.shape, dtype=bool))
    assert got.tolist() == expected.tolist()


def test_null_rows_get_the_sentinel():
    """Null rows keep the reserved sentinel token (39-F44)."""
    ns = np.array([7, 8], dtype=np.int64)
    got = _temporal_to_epoch_ns_tokens(ns, np.array([False, True]))
    assert got[0] == "dt:7"
    assert got[1] != "dt:8"
