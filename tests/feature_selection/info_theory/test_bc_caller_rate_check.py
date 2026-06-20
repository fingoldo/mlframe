"""Wave 9.1 loop-iter-10 regression: BC (Besag-Clifford) early-stop
caller-side check MUST be rate-based, not budget-absolute.

Pre-fix: ``mi_direct`` with ``parallelism='bc'`` checked
``nfailed >= max_failed`` where ``max_failed = int(npermutations *
(1 - min_nonzero_confidence))``. That budget-absolute threshold
assumes the full budget ran. But BC may exit at ``n_checked`` as low
as ``min_perms=30`` (when its Wilson CI on the failure rate proves the
null hypothesis is decisively true or false). At that small
``n_checked`` the ``nfailed`` count is also small, so candidates whose
actual p-value sits above the caller's ``1 - min_nonzero_confidence``
threshold get silently ACCEPTED by BC while the equivalent
``parallelism='outer'`` (which always runs the full budget) correctly
rejects.

Effect: with ``parallelism='bc'`` and tight ``min_nonzero_confidence``
(0.99+), weakly-informative noise features at the [0.01, 0.05] corridor
slip through and get nonzero ``original_mi``. Downstream
``_confirm_predictor.py:594`` multiplies ``next_best_gain *= confidence``
so these false-positives crowd out legitimate weaker-MI candidates.

Fix at permutation.py:364 - add a rate-based clause to the BC caller
check: ``(nfailed / n_checked) >= (1 - min_nonzero_confidence)``. The
budget-absolute clause stays (handles BC's other early-exit path where
the CI is decisively below p_low). For non-BC paths the rate form is
arithmetically equivalent so no behavioural change there.
"""
from __future__ import annotations

import numpy as np
import pytest


def _noise_corridor_data(seed: int = 8, n: int = 600):
    """Constructed to land BC in the [0.01, 0.05] corridor:
    a weakly-informative 4-class x with ~2.5% y-correlation. Pre-fix
    BC accepted; outer rejected.
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 4, n).astype(np.int32)
    y = np.where(rng.random(n) < 0.025, x % 2, rng.integers(0, 2, n)).astype(np.int32)
    factors = np.column_stack([x, y]).astype(np.int32)
    nbins = np.array([4, 2], dtype=np.int32)
    return factors, nbins


def test_bc_and_outer_agree_on_noise_rejection_with_tight_mnc():
    """Pre-fix: BC accepted, outer rejected. Post-fix: both reject."""
    from mlframe.feature_selection.filters.permutation import mi_direct
    factors, nbins = _noise_corridor_data()
    mi_outer, _ = mi_direct(
        factors, x=(0,), y=(1,), factors_nbins=nbins,
        npermutations=3000, min_nonzero_confidence=0.99,
        parallelism="outer", prefer_gpu=False,
    )
    mi_bc, _ = mi_direct(
        factors, x=(0,), y=(1,), factors_nbins=nbins,
        npermutations=3000, min_nonzero_confidence=0.99,
        parallelism="bc", prefer_gpu=False,
    )
    # Outer (always full budget) is the ground truth - if it rejects
    # at this mnc, BC must too.
    if mi_outer == 0.0:
        assert mi_bc == 0.0, (
            f"BC accepted noise candidate at mi={mi_bc} while outer "
            f"correctly rejected with the same data. "
            f"BC's small-n_checked early-stop didn't enforce the "
            f"caller's mnc=0.99 rate."
        )


def test_bc_signal_still_accepts():
    """Negative control: a clearly-significant signal must still be
    ACCEPTED by BC (the fix tightened the rejection check, not the
    acceptance path).
    """
    from mlframe.feature_selection.filters.permutation import mi_direct
    rng = np.random.default_rng(7)
    n = 800
    x = rng.integers(0, 8, n).astype(np.int32)
    y = np.where(rng.random(n) < 0.55, x % 2,
                  rng.integers(0, 2, n)).astype(np.int32)
    factors = np.column_stack([x, y]).astype(np.int32)
    nbins = np.array([8, 2], dtype=np.int32)
    mi_bc, conf_bc = mi_direct(
        factors, x=(0,), y=(1,), factors_nbins=nbins,
        npermutations=3000, min_nonzero_confidence=0.99,
        parallelism="bc", prefer_gpu=False,
    )
    assert mi_bc > 0.0, "BC must accept clearly-significant signal"
    assert conf_bc >= 0.95


@pytest.mark.parametrize("mnc", [0.95, 0.99, 0.999])
def test_bc_caller_check_parametric_mnc(mnc):
    """The fix must scale across the practical ``min_nonzero_confidence``
    range. For each mnc, BC's accept/reject must agree with outer.
    """
    from mlframe.feature_selection.filters.permutation import mi_direct
    factors, nbins = _noise_corridor_data(seed=8)
    mi_outer, _ = mi_direct(
        factors, x=(0,), y=(1,), factors_nbins=nbins,
        npermutations=3000, min_nonzero_confidence=mnc,
        parallelism="outer", prefer_gpu=False,
    )
    mi_bc, _ = mi_direct(
        factors, x=(0,), y=(1,), factors_nbins=nbins,
        npermutations=3000, min_nonzero_confidence=mnc,
        parallelism="bc", prefer_gpu=False,
    )
    # Either both accept (with possibly slightly different MI values
    # from BC's smaller n_checked sample variance) or both reject.
    outer_rejected = (mi_outer == 0.0)
    bc_rejected = (mi_bc == 0.0)
    assert outer_rejected == bc_rejected, (
        f"mnc={mnc}: BC and outer must agree on accept/reject. "
        f"outer_mi={mi_outer}, bc_mi={mi_bc}."
    )
