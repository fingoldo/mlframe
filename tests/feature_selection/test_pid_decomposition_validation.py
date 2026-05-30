"""Wave 9.1 loop-iter-24 regression: ``pid_decomposition`` MUST validate
input ranges symmetrically.

Pre-fix at ``_pid_decomposition.py:184-186``: ``joint[x1[i], x2[i],
y[i]] += 1.0`` accepted negative integer sentinels (e.g. ``-1`` from
NaN encoding) and silently wrapped them via numpy negative-indexing to
``joint[K_x1-1, ...]``. Upper-bound overflow (``x1[i] >= K_x1``)
correctly raised ``IndexError`` -- the asymmetric handling was the
smoking gun.

Effect: any caller that uses ``-1`` as a NaN sentinel (the convention
adopted by the discretizer post-iter-11) silently corrupted the PID
output. The four PID values (redundant, unique_x1, unique_x2,
synergistic) all looked numerically plausible but were tabulated against
the wrong joint distribution.

Severity: medium. The module is exported via ``__all__`` and called by
``training/feature_handling/locking.py`` and through the
``pid_synergy_bonus`` parameter; downstream consumers had no way to
detect the silent corruption.

Fix mirrors the iter-13 (factorize raise) + iter-22 (target_encoding
raise) explicit-raise pattern: validate both lower bound (>= 0) and
upper bound (< K) before tabulation; raise ``ValueError`` with the
offending values for diagnostics.
"""
from __future__ import annotations

import numpy as np
import pytest


def _setup_valid_xor():
    np.random.seed(0)
    n = 500
    x1 = np.random.randint(0, 2, n).astype(np.int64)
    x2 = np.random.randint(0, 2, n).astype(np.int64)
    y = x1 ^ x2
    return x1, x2, y


def test_negative_x1_raises():
    """``x1[i] = -1`` (NaN sentinel) must raise, not silently wrap."""
    from mlframe.feature_selection.filters._pid_decomposition import pid_decomposition
    x1, x2, y = _setup_valid_xor()
    x1[0] = -1
    with pytest.raises(ValueError, match="negative integer indices"):
        pid_decomposition(x1, x2, y, 2, 2, 2)


def test_negative_x2_raises():
    from mlframe.feature_selection.filters._pid_decomposition import pid_decomposition
    x1, x2, y = _setup_valid_xor()
    x2[3] = -1
    with pytest.raises(ValueError, match="negative integer indices"):
        pid_decomposition(x1, x2, y, 2, 2, 2)


def test_negative_y_raises():
    from mlframe.feature_selection.filters._pid_decomposition import pid_decomposition
    x1, x2, y = _setup_valid_xor()
    y[5] = -1
    with pytest.raises(ValueError, match="negative integer indices"):
        pid_decomposition(x1, x2, y, 2, 2, 2)


def test_upper_overflow_x1_raises_valueerror():
    """Upper bound (x1[i] >= K_x1) now raises ValueError too (was
    IndexError pre-fix). Either error type is acceptable; the test
    pins to ValueError per the iter-24 contract.
    """
    from mlframe.feature_selection.filters._pid_decomposition import pid_decomposition
    x1, x2, y = _setup_valid_xor()
    x1[0] = 5  # K_x1=2 means valid range 0..1
    with pytest.raises(ValueError, match="exceeds declared cardinality"):
        pid_decomposition(x1, x2, y, 2, 2, 2)


def test_valid_inputs_succeed():
    """Negative control: valid inputs (no negatives, no overflow) must
    succeed and produce all four PID components.
    """
    from mlframe.feature_selection.filters._pid_decomposition import pid_decomposition
    x1, x2, y = _setup_valid_xor()
    result = pid_decomposition(x1, x2, y, 2, 2, 2)
    assert set(result.keys()) == {"redundant", "unique_x1", "unique_x2",
                                    "synergistic", "total"}
    # XOR has high synergistic, low everything else.
    assert result["synergistic"] > 0.5
    assert result["total"] > 0.5


def test_empty_input_returns_zeros():
    """Edge case: empty arrays return all-zero PID."""
    from mlframe.feature_selection.filters._pid_decomposition import pid_decomposition
    empty = np.array([], dtype=np.int64)
    result = pid_decomposition(empty, empty, empty, 2, 2, 2)
    for v in result.values():
        assert v == 0.0
