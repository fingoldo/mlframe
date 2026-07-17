"""Regression sensor for the iter48 no-alloc rewrite of ``_corr_sq_centered``.

The hot periodogram helper computes the centered sum-of-squares + numerator from
RAW ``v`` dot products (``v_ss = v@v - sum(v)^2/n``, ``num = v @ yc``) instead of
allocating a ``vc = v - v.mean()`` temporary. This is mathematically equal to the
centered form to ~1e-13 (single ULP); the test pins that equivalence so a future
"just center it again" or an algebra slip is caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe import (
    _corr_sq_centered,
    _periodogram_power,
)


def _ref_corr_sq(v, y_centered, y_ss):
    """The pre-iter48 centered reference: allocate vc explicitly."""
    vc = v - v.mean()
    v_ss = float(vc @ vc)
    if v_ss < 1e-24 or y_ss < 1e-24:
        return 0.0
    num = float(vc @ y_centered)
    return (num * num) / (v_ss * y_ss)


@pytest.mark.parametrize("n", [16, 533, 1100, 1667, 3333, 20000])
def test_corr_sq_centered_matches_centered_reference(n):
    rng = np.random.default_rng(n)
    z = np.sort(rng.random(n))
    max_diff = 0.0
    for _ in range(50):
        fv = 0.25 + 6.0 * rng.random()
        v = np.sin(2.0 * np.pi * fv * z)
        y = np.sin(2.0 * np.pi * (0.25 + 6.0 * rng.random()) * z) + 0.3 * rng.standard_normal(n)
        yc = y - y.mean()
        y_ss = float(yc @ yc)
        got = _corr_sq_centered(v, yc, y_ss)
        ref = _ref_corr_sq(v, yc, y_ss)
        max_diff = max(max_diff, abs(got - ref))
    assert max_diff < 1e-12, f"no-alloc corr_sq diverged {max_diff:.2e} from centered reference at n={n}"


def test_corr_sq_centered_degenerate_constant_v_returns_zero():
    yc = np.array([1.0, -1.0, 0.5, -0.5])
    assert _corr_sq_centered(np.ones(4), yc, float(yc @ yc)) == 0.0


@pytest.mark.parametrize("n", [1667, 4000, 20000])  # below + above the parallel gate
def test_power_centered_parallel_path_matches_serial_reference(n):
    """_power_centered routes to the parallel fused njit kernel at n>=_POWER_CENTERED_PAR_MIN_N; it must match
    the independent centered periodogram reference (_periodogram_power) to ~1e-12 on both sides of the gate."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe import (
        _power_centered,
        _periodogram_power,
    )

    rng = np.random.default_rng(n)
    z = np.sort(rng.random(n))
    max_rel = 0.0
    for _ in range(15):
        f = 0.3 + 6.0 * rng.random()
        y = np.sin(2.0 * np.pi * (0.3 + 6.0 * rng.random()) * z) + 0.3 * rng.standard_normal(n)
        yc = y - y.mean()
        y_ss = float(yc @ yc)
        ref = _periodogram_power(z, y, f)
        got = _power_centered(z, yc, y_ss, f)
        max_rel = max(max_rel, abs(ref - got) / max(abs(ref), 1e-30))
    assert max_rel < 1e-10, f"_power_centered diverged {max_rel:.2e} from the centered reference at n={n}"


@pytest.mark.parametrize("n", [4000, 4096, 20000])  # at + above the parallel gate
def test_power_centered_parallel_path_bitidentical_across_thread_counts(n):
    """The parallel periodogram kernel ``_power_centered_fused_par_njit`` must return a BIT-IDENTICAL float
    regardless of the live numba thread count.

    It is the only ``@njit(parallel=True)`` kernel on the CPU FE path with a genuine cross-thread float
    reduction. A naive numba auto-reduction over ``prange(n)`` lets the per-thread partial-COMBINE order drift
    with the thread schedule (float ``+`` is non-associative), so the result wobbles ~1e-15 across process
    starts -- enough to FLIP a razor-tie frequency argmax in ``_refine_peak_freq`` and silently diverge the CPU
    MRMR selection run-to-run. The kernel uses a FIXED contiguous-block reduction (constant block count, fixed
    combine order) so the output is identical across thread counts. This pins that determinism: pre-block-fix
    (auto-reduction) the bytes differed across 1/4/8 threads; post-fix they are identical."""
    import numba

    from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe import (
        _power_centered_fused_par_njit,
    )

    rng = np.random.default_rng(n + 7)
    z = np.ascontiguousarray(np.sort(rng.random(n)))
    y = np.sin(2.0 * np.pi * 3.0 * z) + 0.3 * rng.standard_normal(n)
    yc = np.ascontiguousarray(y - y.mean())
    y_ss = float(yc @ yc)
    freqs = [0.05 + 0.0125 * k for k in range(120)]

    prev = numba.get_num_threads()
    try:
        results = {}
        for nthreads in (1, 2, 4):
            numba.set_num_threads(nthreads)
            vals = np.array(
                [_power_centered_fused_par_njit(z, yc, y_ss, float(f)) for f in freqs],
                dtype=np.float64,
            )
            results[nthreads] = vals.tobytes()
    finally:
        numba.set_num_threads(prev)

    ref = results[1]
    for nthreads, b in results.items():
        assert b == ref, (
            f"_power_centered_fused_par_njit not bit-identical at n={n}: thread count {nthreads} differs "
            f"from 1 thread -- the cross-thread float reduction order is not deterministic."
        )


def test_periodogram_power_nonnegative_and_phase_invariant():
    rng = np.random.default_rng(7)
    n = 1000
    z = np.sort(rng.random(n))
    f = 2.3
    for phi in (0.0, 0.7, 1.9, 3.0):
        y = np.sin(2.0 * np.pi * f * z + phi)
        p = _periodogram_power(z, y, f)
        assert p >= 0.0
        assert p > 0.9, f"genuine tone power should be ~1, got {p:.3f} at phi={phi}"
