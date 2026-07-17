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
    rng = np.random.default_rng(0)
    n = 500
    x1 = rng.integers(0, 2, n).astype(np.int64)
    x2 = rng.integers(0, 2, n).astype(np.int64)
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
    assert set(result.keys()) == {"redundant", "unique_x1", "unique_x2", "synergistic", "total"}
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


def test_x1_marginal_y_occupancy_equals_composite_y_occupancy():
    """The X1-marginal joint's y-occupancy is IDENTICAL to the composite joint's y-occupancy.

    This pins the invariant behind the A10 disposition (DOC, no behavior change): ``p_x1y = joint.sum(axis=1)`` has already
    summed over X2, so a y-bin is occupied in ``p_x1y`` iff some (x1, x2) row carries it. The ``total = I({X1,X2};Y)``
    Miller-Madow correction therefore reuses the X1-marginal y-count CORRECTLY -- a y-bin X2 occupies but X1 does not is
    still counted (it lands in ``p_x1y`` regardless of which x1 it pairs with). Guards a future "fix" that wrongly assumes
    the two counts can differ. Tested over many random joints including the X2-extends-support shape.
    """
    from mlframe.feature_selection.filters._pid_decomposition import _occupied_counts_2d

    rng = np.random.default_rng(7)
    for _ in range(500):
        K_x1, K_x2, K_y = (int(rng.integers(1, 4)) for _ in range(3))
        n = int(rng.integers(1, 40))
        x1 = rng.integers(0, K_x1, n)
        x2 = rng.integers(0, K_x2, n)
        y = rng.integers(0, K_y, n)
        flat = (x1 * K_x2 + x2) * K_y + y
        joint = np.bincount(flat, minlength=K_x1 * K_x2 * K_y).reshape(K_x1, K_x2, K_y).astype(np.float64)
        p_x1y = joint.sum(axis=1)
        _, k_y_x1_marginal = _occupied_counts_2d(p_x1y)
        k_y_composite = int((joint.sum(axis=(0, 1)) > 0.0).sum())
        assert k_y_x1_marginal == k_y_composite

    # Explicit X2-extends-support shape: X2 carries y-bins paired with x1=0; the X1-marginal still occupies every y-bin.
    x1 = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0], dtype=np.int64)
    x2 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    y = np.array([0, 0, 1, 1, 0, 1, 2, 2, 2, 2], dtype=np.int64)
    K_x1, K_x2, K_y = 2, 2, 3
    flat = (x1 * K_x2 + x2) * K_y + y
    joint = np.bincount(flat, minlength=K_x1 * K_x2 * K_y).reshape(K_x1, K_x2, K_y).astype(np.float64)
    _, k_y_x1_marginal = _occupied_counts_2d(joint.sum(axis=1))
    assert k_y_x1_marginal == int((joint.sum(axis=(0, 1)) > 0.0).sum()) == 3
