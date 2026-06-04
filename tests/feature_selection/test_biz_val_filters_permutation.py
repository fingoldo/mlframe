"""biz_val tests for permutation MI variants
(feature_selection/filters/permutation.py).

Per CLAUDE.md "Every new ML trick gets a biz_val synthetic test":
each test asserts a SYNTHETIC measurable WIN of an alternate
permutation-MI implementation over the baseline ``parallel_mi``
fixed-budget path.

Naming: ``test_biz_val_permutation_<variant>_<scenario>``.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")


def _make_strong_signal(n=5000, seed=42):
    """Strong signal: ``y = sign(x + small_noise)``. KSG MI is
    high (~0.5 nats); every permutation should FAIL to recover it,
    so the Besag-Clifford early-stop test should converge fast on a
    "no-failure" verdict and exit before the full budget."""
    from mlframe.feature_selection.filters.discretization import discretize_array
    rng = np.random.default_rng(seed)
    x_cont = rng.normal(size=n)
    y = (x_cont + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    x_bin = discretize_array(arr=x_cont, n_bins=10, method="quantile",
                              dtype=np.int32)
    return x_bin, y


def _make_no_signal(n=5000, seed=42):
    """No-signal: independent x and y. Original MI is small / noise;
    permutations FAIL fast (every shuffle reaches roughly the same
    MI), and the early-stop should also exit early on a "definitely
    no signal" verdict."""
    from mlframe.feature_selection.filters.discretization import discretize_array
    rng = np.random.default_rng(seed)
    x_cont = rng.normal(size=n)
    y = rng.integers(0, 2, n).astype(np.int64)
    x_bin = discretize_array(arr=x_cont, n_bins=10, method="quantile",
                              dtype=np.int32)
    return x_bin, y


def _classes_and_mi(x_bin, y):
    from mlframe.feature_selection.filters.info_theory import (
        compute_mi_from_classes, merge_vars,
    )
    factors = np.column_stack([x_bin, y]).astype(np.int32)
    factors_nbins = np.array([10, 2], dtype=np.int64)
    cx, fx, _ = merge_vars(factors, (0,), None, factors_nbins, dtype=np.int32)
    cy, fy, _ = merge_vars(factors, (1,), None, factors_nbins, dtype=np.int32)
    mi = float(compute_mi_from_classes(
        classes_x=cx, freqs_x=fx, classes_y=cy, freqs_y=fy, dtype=np.int32))
    return cx, fx, cy, fy, mi


def _warmup_njit():
    """Warm both njit kernels so the first-call JIT compile cost
    doesn't pollute the test timing measurement."""
    from mlframe.feature_selection.filters.permutation import (
        parallel_mi, parallel_mi_besag_clifford,
    )
    cx, fx, cy, fy, mi = _classes_and_mi(*_make_strong_signal(n=1000, seed=0))
    parallel_mi(cx, fx, cy, fy, 10, mi, max_failed=10, dtype=np.int32)
    parallel_mi_besag_clifford(cx, fx, cy, fy, 100, mi, np.uint64(0),
                                 dtype=np.int32)


# ---------------------------------------------------------------------------
# Besag-Clifford early stop
# ---------------------------------------------------------------------------


def test_biz_val_permutation_besag_clifford_2x_faster_strong_signal():
    """Besag-Clifford early-stop must be >=2x faster than the full-budget ``parallel_mi`` on a strong-signal target. Floor lowered from 3x to 2x after parallel_mi switched from
    np.random.shuffle (process-global RNG) to inline LCG Fisher-Yates (~6x faster shuffle); the legitimate-speedup margin shrinks because the baseline is now also fast."""
    from mlframe.feature_selection.filters.permutation import (
        parallel_mi, parallel_mi_besag_clifford,
    )
    _warmup_njit()
    cx, fx, cy, fy, mi = _classes_and_mi(*_make_strong_signal(n=5000, seed=42))
    N_PERMS = 1000

    t0 = time.perf_counter()
    nf_full, nt_full = parallel_mi(cx, fx, cy, fy, N_PERMS, mi,
                                       max_failed=N_PERMS, dtype=np.int32)
    t_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    nf_bc, nt_bc = parallel_mi_besag_clifford(
        cx, fx, cy, fy, N_PERMS, mi, np.uint64(0), dtype=np.int32)
    t_bc = time.perf_counter() - t0

    speedup = t_full / max(t_bc, 1e-6)
    assert speedup >= 2.0, (
        f"Besag-Clifford must be >=2x faster than full on strong-signal "
        f"target; got {speedup:.1f}x ({t_full*1000:.1f}ms vs "
        f"{t_bc*1000:.1f}ms)"
    )


def test_biz_val_permutation_besag_clifford_stops_before_full_budget():
    """Besag-Clifford must STOP before the full ``npermutations`` are
    consumed on a strong-signal target. Measured: ~381 of 1000.
    Floor: ``nchecked < 0.7 * npermutations``."""
    from mlframe.feature_selection.filters.permutation import (
        parallel_mi_besag_clifford,
    )
    _warmup_njit()
    cx, fx, cy, fy, mi = _classes_and_mi(*_make_strong_signal(n=5000, seed=42))
    N_PERMS = 1000
    nf, nt = parallel_mi_besag_clifford(
        cx, fx, cy, fy, N_PERMS, mi, np.uint64(0), dtype=np.int32)
    assert nt < int(0.7 * N_PERMS), (
        f"BC must early-stop with nchecked < {int(0.7 * N_PERMS)}; "
        f"got nchecked={nt}"
    )
    # AND it must agree with the full-budget verdict (no failures
    # for strong signal -- both should report nfailed=0).
    assert nf == 0, (
        f"On strong-signal target every permutation should fail to "
        f"recover the MI; BC reports nfailed={nf}"
    )


def test_biz_val_permutation_besag_clifford_agrees_with_full_no_signal():
    """On a no-signal (random y) target, BC and full must agree on
    the verdict ('signal not significant'). Both should report
    a non-zero ``nfailed`` (most permutations succeed at matching the
    original MI).

    Pre-fix-B22 (older code) the BC could return without ever
    checking on degenerate targets; this test catches that regression."""
    from mlframe.feature_selection.filters.permutation import (
        parallel_mi, parallel_mi_besag_clifford,
    )
    _warmup_njit()
    cx, fx, cy, fy, mi = _classes_and_mi(*_make_no_signal(n=5000, seed=42))
    N_PERMS = 500

    nf_full, nt_full = parallel_mi(cx, fx, cy, fy, N_PERMS, mi,
                                       max_failed=N_PERMS, dtype=np.int32)
    nf_bc, nt_bc = parallel_mi_besag_clifford(
        cx, fx, cy, fy, N_PERMS, mi, np.uint64(0), dtype=np.int32)

    # On a no-signal target, original_mi may be ~0; both implementations
    # should return SOMETHING (not crash, not return -1). The structural
    # invariant: both must complete and return non-negative counts.
    assert nf_full >= 0 and nt_full >= 0
    assert nf_bc >= 0 and nt_bc >= 0
    # The "no signal" verdict aligns: full's failure rate (nfailed /
    # nt_full) and BC's failure rate must be close (both ~50-100% on
    # genuine no-signal data).
    rate_full = nf_full / max(nt_full, 1)
    rate_bc = nf_bc / max(nt_bc, 1)
    # Broad tolerance: BC's adaptive stopping can land at different
    # failure rates than fixed-budget. Just require the two rates
    # don't disagree wildly (both indicating same direction).
    assert abs(rate_full - rate_bc) <= 0.5, (
        f"BC and full failure rates must roughly agree on no-signal; "
        f"full={rate_full:.2f} (nfailed={nf_full}/{nt_full}), "
        f"BC={rate_bc:.2f} (nfailed={nf_bc}/{nt_bc})"
    )


# ---------------------------------------------------------------------------
# Reproducibility invariant for ``parallelism="inner"`` (Phase 1 prange)
# ---------------------------------------------------------------------------


def test_biz_val_permutation_inner_prange_reproducible_across_n_workers():
    """Phase 1's ``parallelism='inner'`` (numba prange + per-iteration
    LCG seed) must produce IDENTICAL ``(nfailed, n_checked)`` across
    different effective worker counts, for a fixed ``base_seed``.
    Catches regressions where someone naively swaps the LCG strategy
    for one that depends on thread schedule."""
    from mlframe.feature_selection.filters.permutation import (
        parallel_mi_prange,
    )
    _warmup_njit()
    cx, fx, cy, fy, mi = _classes_and_mi(*_make_strong_signal(n=3000, seed=42))
    N_PERMS = 200

    # Run twice with the same base_seed -- must produce identical
    # results. The reproducibility invariant is: per-iteration
    # deterministic seed, no thread-schedule influence.
    a_failed, a_checked = parallel_mi_prange(
        cx, fx, cy, fy, N_PERMS, mi, np.uint64(42), dtype=np.int32)
    b_failed, b_checked = parallel_mi_prange(
        cx, fx, cy, fy, N_PERMS, mi, np.uint64(42), dtype=np.int32)
    assert (a_failed, a_checked) == (b_failed, b_checked), (
        f"prange must be reproducible across runs with same seed; "
        f"got run1=({a_failed}, {a_checked}) vs run2=({b_failed}, {b_checked})"
    )


# ---------------------------------------------------------------------------
# parallel_mi(npermutations=0) guard (B22 fix from earlier session)
# ---------------------------------------------------------------------------


def test_biz_val_permutation_zero_permutations_returns_zero_no_crash():
    """``parallel_mi`` with ``npermutations=0`` must return ``(0, 0)``
    (no permutations checked, none failed) without raising
    ``UnboundLocalError``. Pre-fix-B22 behaviour was a crash.
    Caller-side mi_direct also must guard against division-by-zero."""
    from mlframe.feature_selection.filters.permutation import parallel_mi
    _warmup_njit()
    cx, fx, cy, fy, mi = _classes_and_mi(*_make_strong_signal(n=1000, seed=42))
    nf, nt = parallel_mi(cx, fx, cy, fy, 0, mi, max_failed=10, dtype=np.int32)
    assert nf == 0 and nt == 0, (
        f"npermutations=0 must return (0, 0); got ({nf}, {nt})"
    )


# ---------------------------------------------------------------------------
# Empirical permutation-null relevance debiasing (the *_with_null kernels +
# mi_direct(return_null_mean=True)). The screen subtracts the per-feature null
# mean from the observed MI to demote spuriously-inflated columns; these tests
# pin (a) the null kernels are bit-identical to the legacy kernels on
# (nfailed, nchecked) and additionally return the per-permutation MI sum, and
# (b) the null mean separates genuine signal from noise.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_pair", ["prange", "bc", "worker"])
def test_biz_val_null_kernels_bit_identical_to_legacy(kernel_pair):
    """The ``*_with_null`` siblings MUST return ``(nfailed, nchecked)`` bit-identical to their legacy counterparts (same LCG stream, same exceedance/early-stop logic)
    and additionally a finite non-negative per-permutation MI sum. Guards against a future edit that lets the null accumulation perturb the confidence contract."""
    from mlframe.feature_selection.filters.permutation import (
        parallel_mi, parallel_mi_with_null,
        parallel_mi_prange, parallel_mi_prange_with_null,
        parallel_mi_besag_clifford, parallel_mi_besag_clifford_with_null,
    )
    cx, fx, cy, fy, mi = _classes_and_mi(*_make_strong_signal(n=1500, seed=11))
    if kernel_pair == "prange":
        a = parallel_mi_prange(cx, fx, cy, fy, 40, mi, np.uint64(123), dtype=np.int32)
        b = parallel_mi_prange_with_null(cx, fx, cy, fy, 40, mi, np.uint64(123), dtype=np.int32)
    elif kernel_pair == "bc":
        a = parallel_mi_besag_clifford(cx, fx, cy, fy, 1000, mi, np.uint64(0), dtype=np.int32)
        b = parallel_mi_besag_clifford_with_null(cx, fx, cy, fy, 1000, mi, np.uint64(0), dtype=np.int32)
    else:
        a = parallel_mi(cx, fx, cy, fy, 40, mi, max_failed=100, dtype=np.int32, base_seed=np.uint64(123))
        b = parallel_mi_with_null(cx, fx, cy, fy, 40, mi, max_failed=100, dtype=np.int32, base_seed=np.uint64(123))
    assert a == b[:2], f"{kernel_pair}: (nfailed, nchecked) diverged: legacy {a} vs with_null {b[:2]}"
    assert len(b) == 3 and np.isfinite(b[2]) and b[2] >= 0.0, f"{kernel_pair}: null MI sum invalid: {b}"


def test_biz_val_mi_direct_return_null_mean_backward_compatible():
    """``mi_direct`` default (``return_null_mean=False``) returns the legacy 2-tuple; ``return_null_mean=True`` returns a 4-tuple ``(observed, confidence, null_mean, p_value)``
    whose first two elements match the legacy call. Pins the contract so the screen caller and any future consumer reading the p-value are unaffected by an accidental reshape."""
    from mlframe.feature_selection.filters.permutation import mi_direct
    x_bin, y = _make_strong_signal(n=1500, seed=7)
    factors = np.column_stack([x_bin, y]).astype(np.int32)
    factors_nbins = np.array([10, 2], dtype=np.int64)
    legacy = mi_direct(factors, x=(0,), y=(1,), factors_nbins=factors_nbins,
                       npermutations=16, base_seed=3, prefer_gpu=False)
    assert isinstance(legacy, tuple) and len(legacy) == 2
    quad = mi_direct(factors, x=(0,), y=(1,), factors_nbins=factors_nbins,
                     npermutations=16, base_seed=3, prefer_gpu=False, return_null_mean=True)
    assert len(quad) == 4
    obs, conf, null_mean, p_value = quad
    assert obs == pytest.approx(legacy[0]), "observed MI must match legacy call"
    assert np.isfinite(null_mean) and null_mean >= 0.0, f"null mean must be finite >=0; got {null_mean}"
    assert 0.0 <= p_value <= 1.0, f"p_value must be a probability; got {p_value}"
    assert conf == pytest.approx(1.0 - p_value), "confidence and p_value must be complementary"
    # On strong signal the observed MI must dominate the null mean (so debiasing keeps the relevance) AND the feature is permutation-significant (p_value ~ 0).
    assert obs > 5.0 * null_mean, f"strong signal: observed {obs:.4f} should be >> null mean {null_mean:.4f}"
    assert p_value < 0.05, f"strong signal must be permutation-significant; got p_value={p_value:.4f}"


def test_biz_val_null_mean_demotes_noise_keeps_signal():
    """The empirical null mean must separate genuine signal from noise: a strong-signal column has ``observed >> null_mean`` (relevance survives debiasing) while a
    no-signal column has ``observed ~ null_mean`` (debiased relevance collapses toward 0). This is the core ranking-correction the screen relies on."""
    from mlframe.feature_selection.filters.permutation import mi_direct
    fnb = np.array([10, 2], dtype=np.int64)

    xs, ys = _make_strong_signal(n=2000, seed=5)
    sig = np.column_stack([xs, ys]).astype(np.int32)
    obs_s, _, null_s, _ = mi_direct(sig, x=(0,), y=(1,), factors_nbins=fnb,
                                    npermutations=16, base_seed=1, prefer_gpu=False, return_null_mean=True)

    xn, yn = _make_no_signal(n=2000, seed=5)
    noise = np.column_stack([xn, yn]).astype(np.int32)
    obs_n, _, null_n, _ = mi_direct(noise, x=(0,), y=(1,), factors_nbins=fnb,
                                    npermutations=16, base_seed=1, prefer_gpu=False, return_null_mean=True)

    debiased_sig = max(0.0, obs_s - null_s)
    debiased_noise = max(0.0, obs_n - null_n)
    assert debiased_sig > 10.0 * max(debiased_noise, 1e-9), (
        f"debiased signal relevance ({debiased_sig:.5f}) must dominate debiased noise ({debiased_noise:.5f})"
    )


def test_biz_val_significance_gate_keeps_weak_signal_demotes_noise():
    """SIGNIFICANCE-GATED debiasing: the permutation p-value separates weak-but-real signal from spurious noise EVEN when both carry a high null mean. A weak genuine signal
    sits ABOVE its null distribution (few/no permutations beat observed => small p => SIGNIFICANT => keep full observed MI), while pure noise sits WITHIN its null (many
    permutations beat observed => large p => NOT significant => subtract null mean). This is the discriminator the null mean alone cannot provide; the screen relies on it to
    protect weak signal the flat null-mean subtraction would over-correct away."""
    from mlframe.feature_selection.filters.permutation import mi_direct
    fnb = np.array([10, 2], dtype=np.int64)
    alpha = 0.05

    # A weak (low-coefficient) genuine signal: observed MI is modest and the coarse-binning null mean is a sizable fraction of it, yet it is permutation-significant.
    rng = np.random.default_rng(17)
    n = 3000
    z = rng.standard_normal(n)
    logit = 0.45 * z  # weak but real
    yw = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.int32)
    xw = np.clip((z - z.min()) / (z.max() - z.min()) * 10, 0, 9).astype(np.int32)
    weak = np.column_stack([xw, yw]).astype(np.int32)
    obs_w, _, null_w, p_w = mi_direct(weak, x=(0,), y=(1,), factors_nbins=fnb,
                                      npermutations=64, base_seed=2, prefer_gpu=False, return_null_mean=True)

    xn, yn = _make_no_signal(n=n, seed=5)
    noise = np.column_stack([xn, yn]).astype(np.int32)
    obs_n, _, null_n, p_n = mi_direct(noise, x=(0,), y=(1,), factors_nbins=fnb,
                                      npermutations=64, base_seed=2, prefer_gpu=False, return_null_mean=True)

    assert p_w < alpha, f"weak genuine signal must be significant (p<{alpha}); got p={p_w:.4f}, obs={obs_w:.5f}, null={null_w:.5f}"
    assert p_n >= alpha, f"pure noise must be non-significant (p>={alpha}); got p={p_n:.4f}"

    # Significance-gated relevance: keep full observed when significant, subtract null when not.
    rel_w = obs_w if p_w < alpha else max(0.0, obs_w - null_w)
    rel_n = obs_n if p_n < alpha else max(0.0, obs_n - null_n)
    assert rel_w == pytest.approx(obs_w), "weak-but-real signal must keep its FULL observed MI under the significance gate"
    assert rel_w > 5.0 * max(rel_n, 1e-9), (
        f"significance-gated weak signal relevance ({rel_w:.5f}) must dominate gated noise ({rel_n:.5f})"
    )
