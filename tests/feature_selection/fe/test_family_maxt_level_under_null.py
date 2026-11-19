"""The family-wise permutation gate keeps its level: on candidates that are independent of the target, some candidate passes in at most about ``alpha`` of the repetitions.

The family gate passes a candidate when its MI exceeds the permutation null of the family maximum, so the probability that ANY candidate of a family of pure noise passes is the gate's
family-wise error rate. The audit of 2026-10-15 (statistics critique F08) asked for this check; measured when it was added: 0.047 at alpha 0.05 and 0.010 at alpha 0.01, 300 repetitions.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._family_maxt_gate import family_maxt_pass_codes

_REPS = 300
_K, _N, _BINS, _PERMS = 8, 600, 5, 199


def _any_pass_rate(alpha: float, seed: int) -> float:
    """Share of repetitions in which at least one of ``_K`` noise columns passes the family gate."""
    rng = np.random.default_rng(seed)
    hits = 0
    for rep in range(_REPS):
        cols = rng.integers(0, _BINS, (_K, _N)).astype(np.int16)
        nx = np.full(_K, _BINS, dtype=np.int64)
        y = rng.integers(0, 2, _N)
        hits += bool(np.any(family_maxt_pass_codes(cols, nx, y, alpha=alpha, n_perm=_PERMS, seed=rep)))
    return hits / _REPS


@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_pure_noise_family_passes_at_most_alpha_of_the_time(alpha):
    """The pass rate under H0 does not exceed alpha by more than three Monte Carlo standard errors."""
    rate = _any_pass_rate(alpha, seed=11)
    bound = alpha + 3.0 * np.sqrt(alpha * (1.0 - alpha) / _REPS)
    assert rate <= bound, f"family-wise pass rate {rate:.4f} at alpha {alpha} exceeds {bound:.4f} on pure noise"


def test_a_planted_signal_passes_the_same_gate():
    """The gate is not simply closed: a column that determines the target passes at alpha 0.01."""
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, _N)
    cols = rng.integers(0, _BINS, (_K, _N)).astype(np.int16)
    cols[3] = (y * 3 + rng.integers(0, 2, _N)).astype(np.int16)  # a column that follows the target
    flags = family_maxt_pass_codes(cols, np.full(_K, _BINS, dtype=np.int64), y, alpha=0.01, n_perm=_PERMS, seed=0)
    assert bool(flags[3]) and int(np.count_nonzero(flags)) == 1


def test_the_analytic_bound_never_certifies_pure_noise_and_certifies_a_planted_signal():
    """The chi-square triage (pair-family critique F09) is a conservative bound: over 300 noise families it certifies no pass, and it certifies the column that follows the target."""
    from mlframe.feature_selection.filters._family_maxt_gate import _analytic_triage, mi_all_columns_njit

    rng = np.random.default_rng(3)
    nx = np.full(_K, _BINS, dtype=np.int64)
    starts = np.arange(_K + 1)
    certified = 0
    for _ in range(_REPS):
        cols = rng.integers(0, _BINS, (_K, _N)).astype(np.int16)
        y = rng.integers(0, 2, _N).astype(np.int64)
        status = _analytic_triage(cols, nx, y, 2, mi_all_columns_njit(cols, nx, y, 2), starts, 0.01)
        certified += bool(status is not None and (status == 1).any())
    assert certified == 0, f"the analytic bound certified a pass on pure noise in {certified} of {_REPS} families"
    y = rng.integers(0, 2, _N).astype(np.int64)
    cols = rng.integers(0, _BINS, (_K, _N)).astype(np.int16)
    cols[3] = (y * 3 + rng.integers(0, 2, _N)).astype(np.int16)
    status = _analytic_triage(cols, nx, y, 2, mi_all_columns_njit(cols, nx, y, 2), starts, 0.01)
    assert status is not None and int(status[3]) == 1
