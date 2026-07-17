"""Sensor tests for MRMR cross-target identity cache safety defaults.

Covers A-P0-003:
- `mrmr_identity_cache_include_y` defaults to True so target-y disambiguates the cache key
- `_mrmr_compute_x_fingerprint` mixes a content-cell sample so same-schema-different-content X
  frames produce DIFFERENT fingerprints
"""

import inspect

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import (
    MRMR,
    _mrmr_compute_x_fingerprint,
)


def test_mrmr_identity_cache_include_y_default_is_true():
    """Default must be True so target A's identity result cannot poison target B."""
    sig = inspect.signature(MRMR.__init__)
    p = sig.parameters["mrmr_identity_cache_include_y"]
    assert p.default is True, (
        f"mrmr_identity_cache_include_y default flipped from True to {p.default!r}; "
        "fix A-P0-003 requires include_y=True so same-X-different-y targets get separate cache slots."
    )


def test_x_fingerprint_differs_for_same_schema_different_content():
    """Two pandas frames with identical column names + dtypes + n_rows but DIFFERENT cell content
    must produce DIFFERENT fingerprints. Pre-fix path returned identical fingerprints."""
    rng = np.random.default_rng(0)
    X1 = pd.DataFrame(
        {
            "a": rng.normal(size=200).astype(np.float32),
            "b": rng.integers(0, 10, size=200).astype(np.int32),
        }
    )
    X2 = pd.DataFrame(
        {
            "a": rng.normal(size=200).astype(np.float32) + 100.0,  # shifted, very different content
            "b": rng.integers(0, 10, size=200).astype(np.int32),
        }
    )
    fp1 = _mrmr_compute_x_fingerprint(X1)
    fp2 = _mrmr_compute_x_fingerprint(X2)
    assert fp1 != fp2, "_mrmr_compute_x_fingerprint must include a cell-content sample; same-schema-different-content X currently collide on the same key."


def test_x_fingerprint_stable_for_same_content():
    """Repeated calls on the same frame must return the same fingerprint (sampling determinism)."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=300).astype(np.float64),
            "b": rng.integers(0, 5, size=300).astype(np.int32),
        }
    )
    fp_a = _mrmr_compute_x_fingerprint(X)
    fp_b = _mrmr_compute_x_fingerprint(X)
    assert fp_a == fp_b
