"""Sensor tests for BorutaShap RNG isolation (A-P0-004).

BorutaShap.__init__ must NOT mutate the global np.random stream.
"""

import numpy as np

from mlframe.feature_selection.boruta_shap import BorutaShap


def test_borutashap_ctor_does_not_reseed_global_numpy_rng():
    """Constructing BorutaShap with a random_state must not alter np.random global state.

    Pre-fix code called np.random.seed(random_state) which leaked the seed into
    every downstream consumer of np.random in the same process.
    """
    # Snapshot the global state, draw a baseline value, then reset.
    state_before = np.random.get_state()
    expected = np.random.rand()
    np.random.set_state(state_before)

    # Constructing BorutaShap with a non-None seed must not disturb the global RNG.
    BorutaShap(random_state=42)

    got = np.random.rand()
    assert got == expected, "BorutaShap ctor mutated global np.random state; private rng required (A-P0-004)."


def test_borutashap_has_private_rng():
    bs = BorutaShap(random_state=123)
    assert hasattr(bs, "_rng"), "BorutaShap must hold a private np.random.Generator (A-P0-004)"
    assert isinstance(bs._rng, np.random.Generator)


def test_borutashap_private_rng_deterministic_across_instances():
    """Two BorutaShap instances with the same seed produce the same private-rng draws."""
    a = BorutaShap(random_state=7)
    b = BorutaShap(random_state=7)
    arr1 = a._rng.permutation(np.arange(50))
    arr2 = b._rng.permutation(np.arange(50))
    assert np.array_equal(arr1, arr2)
