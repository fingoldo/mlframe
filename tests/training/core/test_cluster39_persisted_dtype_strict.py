"""Cluster 39 regression: persisted datetime dtype names resolve exactly, unknown names are refused (audit 2026-11-20, finding 39-F9)."""

import numpy as np
import pytest

from mlframe.training.core.predict import _resolve_persisted_dtype


def test_float16_is_not_collapsed_to_int8():
    """A float16 cyclical column keeps float16 rather than falling back to int8 (39-F9)."""
    assert _resolve_persisted_dtype("float16") is np.float16


@pytest.mark.parametrize("name", ["int8", "uint16", "float32", "float64"])
def test_known_names_resolve_to_their_dtype(name):
    """Existing integer and float names keep resolving to the matching numpy type (39-F9)."""
    assert _resolve_persisted_dtype(name) is np.dtype(name).type


@pytest.mark.parametrize("name", ["not_a_dtype", "datetime64[ns]", "object"])
def test_unknown_or_non_numeric_names_are_refused(name):
    """An unknown or non-numeric dtype name raises instead of silently casting to int8 (39-F9)."""
    with pytest.raises(ValueError):
        _resolve_persisted_dtype(name)
