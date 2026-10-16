"""A raw operand is re-attached next to its engineered feature only when it adds information about y beyond that feature."""

import numpy as np

from mlframe.feature_selection.filters._mrmr_fit_impl._operand_information import operand_adds_information


def _codes(rng, n, k):
    """``n`` random integer codes with ``k`` levels."""
    return rng.integers(0, k, n).astype(np.int64)


def test_xor_operand_is_redundant_given_the_composite():
    """No operand of a 3-way XOR adds information next to the parity composite that carries the whole target."""
    rng = np.random.default_rng(0)
    n = 4000
    x1, x2, x3 = (rng.integers(0, 2, n) for _ in range(3))
    y = x1 ^ x2 ^ x3
    composite = x1 ^ x2 ^ x3  # the parity composite carries all of the target
    data = np.column_stack([x1, x2, x3, composite, y]).astype(np.int64)
    for operand in (0, 1, 2):
        assert not operand_adds_information(data, 4, operand, [3], 0.05, 0, f"x{operand}")


def test_operand_with_signal_beyond_the_composite_is_kept():
    """An operand the target also depends on directly, beyond the composite built from it, is kept."""
    rng = np.random.default_rng(1)
    n = 4000
    a, b = _codes(rng, n, 6), _codes(rng, n, 6)
    composite = (a * b) % 6  # a feature built from a and b
    y = (composite + 2 * a + rng.integers(0, 2, n)) % 6  # y also depends on a directly
    data = np.column_stack([a, b, composite, y]).astype(np.int64)
    assert operand_adds_information(data, 3, 0, [2], 0.05, 0, "a")


def test_pure_noise_operand_is_not_kept():
    """A pure-noise column offered as an operand adds nothing next to the engineered feature."""
    rng = np.random.default_rng(2)
    n = 4000
    a, noise = _codes(rng, n, 6), _codes(rng, n, 6)
    y = a % 3
    data = np.column_stack([a, noise, a, y]).astype(np.int64)  # the engineered feature is a itself
    assert not operand_adds_information(data, 3, 1, [2], 0.05, 0, "noise")
