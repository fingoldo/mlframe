"""RelaxMRMR-3D's conditional-MI terms must compute the quantity their formula names (mrmr_audit_2026-09-14 RO-1).

Both conditional terms had their argument slots transposed: the code computed ``I(X;Y|Z)`` where the
co-information decomposition in the surrounding comment -- and the filled variable's own comment -- require
``I(X;Z|Y)``. The pre-existing sign-direction tests pass on BOTH orientations, so they could never catch it.
These assert the terms against cases with an analytically known answer instead.
"""

import numpy as np
import pytest

from mlframe.feature_selection.filters._relaxmrmr_3d import (
    _cmi_xy_given_z_njit,
    _joint_mi_x_zw_given_y_njit,
)

LN2 = float(np.log(2.0))


@pytest.fixture
def xor_triple():
    """``x = z1 ^ z2`` with an independent ``y``: the PAIR determines x, neither member alone does."""
    rng = np.random.default_rng(0)
    n = 20000
    z1 = rng.integers(0, 2, n)
    z2 = rng.integers(0, 2, n)
    y = rng.integers(0, 2, n)
    return (z1 ^ z2), z1, z2, y


def test_joint_term_is_mi_between_x_and_the_pair_conditioned_on_y(xor_triple):
    """I(X; (Z1,Z2) | Y) == ln2 when the pair fully determines X and Y is independent."""
    x, z1, z2, y = xor_triple
    got = _joint_mi_x_zw_given_y_njit(x, z1, z2, y, 2, 2, 2, 2)
    assert got == pytest.approx(LN2, abs=1e-3)


def test_joint_term_is_zero_when_the_conditioning_variable_already_determines_x(xor_triple):
    """I(X; (Z1,Z2) | Y) == 0 when X == Y: conditioning on Y leaves nothing for the pair to explain.

    This is the assertion that separates the two orientations: the transposed form computed
    ``I(X; Y | Z1,Z2)``, which on X == Y is maximal (ln2), not zero.
    """
    _x, z1, z2, y = xor_triple
    got = _joint_mi_x_zw_given_y_njit(y.copy(), z1, z2, y, 2, 2, 2, 2)
    assert got == pytest.approx(0.0, abs=1e-3)


def test_single_term_is_zero_for_a_member_that_is_uninformative_alone(xor_triple):
    """I(X; Z1 | Y) == 0 for XOR: one member alone says nothing about X."""
    x, z1, _z2, y = xor_triple
    got = _cmi_xy_given_z_njit(x, z1, y, 2, 2, 2)
    assert got == pytest.approx(0.0, abs=1e-3)


def test_single_term_is_maximal_for_a_member_that_determines_x():
    """I(X; Z1 | Y) == ln2 when X == Z1 and Y is independent -- the complementary direction."""
    rng = np.random.default_rng(3)
    n = 20000
    z1 = rng.integers(0, 2, n)
    y = rng.integers(0, 2, n)
    got = _cmi_xy_given_z_njit(z1.copy(), z1, y, 2, 2, 2)
    assert got == pytest.approx(LN2, abs=1e-3)


def test_the_transposed_orientation_would_give_a_different_answer(xor_triple):
    """Teeth: the two orientations genuinely differ, so the tests above are not vacuous."""
    _x, z1, z2, y = xor_triple
    x_eq_y = y.copy()
    correct = _joint_mi_x_zw_given_y_njit(x_eq_y, z1, z2, y, 2, 2, 2, 2)
    # The old, transposed form: composite pair placed in the CONDITIONING slot.
    z_comp = z1.astype(np.int64) * 2 + z2.astype(np.int64)
    transposed = _cmi_xy_given_z_njit(x_eq_y, y, z_comp, 2, 2, 4)
    assert abs(correct - transposed) > 0.5, "the two orientations must not coincide on this fixture"
