"""Regression: the parallel-memset ``_combine_factorize_njit`` is bit-identical to the serial reference.

The first-seen factorize WALK is irreducibly sequential; only the large dense ``seen`` lookup buffer is
prange-initialised (gated on ``_FAC_PAR_MEMSET_MIN``). This pins that the parallelisation changed nothing
about the result -- the dense first-seen ids AND nclasses must equal the fully-serial form on every shape,
including heavy ties (low cardinality), the large-kmax cap-adjacent case that actually triggers the parallel
fill, the empty input, and the hash-fallback (kmax over the array cap).
"""

import numpy as np
import pytest

from mlframe.feature_selection.filters._mi_greedy_cmi_fe import (
    _combine_factorize_njit,
    _combine_factorize_serial_njit,
    _FAC_PAR_MEMSET_MIN,
    _FAC_ARRAY_CAP,
)


def _check(joint, c, mult):
    a_inv, a_nc = _combine_factorize_serial_njit(joint, c, mult)
    b_inv, b_nc = _combine_factorize_njit(joint, c, mult)
    assert a_nc == b_nc
    assert np.array_equal(a_inv, b_inv)
    return a_nc


@pytest.mark.parametrize(
    "n,jcard,ccard",
    [
        (5000, 50, 4),  # tiny, serial-fill branch
        (200_000, 2000, 20),  # below the parallel-memset gate
        (1_000_000, 2000, 20),  # below gate, large n
        (200_000, 8000, 20),  # spans the gate region
        (1_000_000, 20000, 20),  # above gate, parallel fill
        (1_000_000, 400000, 40),  # cap-adjacent kmax (~16M), heaviest parallel fill
    ],
)
def test_parmemset_matches_serial(n, jcard, ccard):
    rng = np.random.default_rng(n + jcard + ccard)
    joint = rng.integers(0, jcard, n).astype(np.int64)
    c = rng.integers(0, ccard, n).astype(np.int64)
    _check(joint, c, jcard)


def test_heavy_ties_low_cardinality():
    # extreme ties: the partition is dominated by repeated joint values
    rng = np.random.default_rng(7)
    n = 300_000
    joint = rng.integers(0, 5, n).astype(np.int64)
    c = rng.integers(0, 3, n).astype(np.int64)
    _check(joint, c, 5)


def test_empty_input():
    e = np.zeros(0, dtype=np.int64)
    _check(e, e, 1)


def test_hash_fallback_over_cap():
    # force kmax >= _FAC_ARRAY_CAP so both paths take the typed.Dict branch
    rng = np.random.default_rng(11)
    n = 50_000
    mult = _FAC_ARRAY_CAP  # joint + c*mult immediately exceeds the cap for c>=1
    joint = rng.integers(0, 1000, n).astype(np.int64)
    c = rng.integers(1, 50, n).astype(np.int64)
    _check(joint, c, mult)


def test_gate_constant_sane():
    # the parallel fill only kicks in for buffers worth the thread spin-up
    assert _FAC_PAR_MEMSET_MIN > 0
    assert _FAC_PAR_MEMSET_MIN < _FAC_ARRAY_CAP
