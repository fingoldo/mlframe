"""Sensor for the 2026-04-19 thinc/pytest-randomly seed-overflow fix
(tests/conftest.py::_patch_thinc_fix_random_seed_for_pytest_randomly_compat).

Root cause: thinc ships ``thinc.util.fix_random_seed`` as a
``pytest_randomly.random_seeder`` entry point. That function calls
``numpy.random.seed(seed)`` with no ``% 2**32`` clamp. pytest-randomly
invokes every registered random_seeder with
``seed = randomly_seed_option + crc32(nodeid_offset)``, a sum that
regularly exceeds 2**32. The thinc seeder then raised
``ValueError: Seed must be between 0 and 2**32 - 1`` during fixture
setup, which cascaded into ``previous item was not torn down properly``
for every test that followed in the session — all tests/training/
files went from green to 20-errors overnight.

Fix: session-scoped autouse fixture monkey-patches
``thinc.util.fix_random_seed`` to clamp seed modulo 2**32 before
calling the original. This file verifies the shim is in place and
the clamp actually happens.
"""
from __future__ import annotations

import pytest

try:
    import thinc.util as _thinc_util
    THINC_INSTALLED = True
except ImportError:  # pragma: no cover
    THINC_INSTALLED = False


pytestmark = pytest.mark.skipif(
    not THINC_INSTALLED,
    reason="thinc not installed — the shim is only needed when thinc is present",
)


def test_fix_random_seed_accepts_large_seed():
    """The bug: thinc.util.fix_random_seed(4414703545) previously
    raised ValueError inside numpy.random.seed because 4414703545 >
    2**32 - 1 = 4294967295. After the conftest shim clamps via
    % 2**32, calling with a large seed must succeed."""
    huge_seed = 4_414_703_545  # > 2**32, the exact value observed in prod
    # The shim is installed by the session autouse fixture. If it's
    # not in effect, this raises.
    _thinc_util.fix_random_seed(huge_seed)  # must not raise


def test_fix_random_seed_normal_seed_still_works():
    """False-positive sensor: the shim must not break small seeds."""
    _thinc_util.fix_random_seed(42)


def test_fix_random_seed_zero_still_works():
    """Edge: seed=0 is valid. Pre-clamp or post-clamp, must succeed."""
    _thinc_util.fix_random_seed(0)


def test_shim_is_wrapper_not_original():
    """The session fixture replaces the attribute with a wrapper.
    If someone refactors and accidentally removes the wrap, this
    sensor trips."""
    # The wrapper closes over _original_fix and calls it with clamped
    # seed. The callable name won't be 'fix_random_seed' if wrapped.
    fn = _thinc_util.fix_random_seed
    # Can either check name differs OR check __closure__ is set
    # (our wrapper has a closure; the original function doesn't).
    is_wrapped = (
        fn.__name__ != "fix_random_seed"
        or (fn.__closure__ is not None and len(fn.__closure__) > 0)
    )
    assert is_wrapped, (
        "thinc.util.fix_random_seed is not the session shim — "
        "pytest-randomly seed overflow protection regressed"
    )
