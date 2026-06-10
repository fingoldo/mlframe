"""Cross-fit recipe warm-start prior (backlog idea #20) -- invariant tests.

The prior is bench-rejected (no iters/wall win; see
``profiling/bench_warmstart_probe.py`` + the in-code note in
``_hermite_fe_optimise_pair.py``) and ships OPT-IN / default-OFF. These tests
pin the two load-bearing properties that keep the opt-in hook safe:

  1. default-OFF (``cross_fit_prior_seeds=None`` / empty) is BYTE-IDENTICAL to
     the legacy warm-start population -- enabling the parameter never silently
     perturbs a fit that did not ask for it.
  2. a seeded fit still produces a result re-derived on THIS data, and a
     size-mismatched prior seed is silently skipped (never crashes / never
     leaks a stale-degree coefficient vector into the live population).
"""
import numpy as np
import pytest

from mlframe.feature_selection.filters._hermite_fe_optimise_pair import optimise_hermite_pair


def _pair_target(n=3000, seed=0):
    """Non-monotone-inner product target: the regime where the pair CMA must
    actually search (a plain identity-operand mul cannot represent it)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    inner = (a ** 3 - 2.0 * a) * (b ** 2 - 1.0)
    y = (inner + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    return a, b, y


_COMMON = dict(
    discrete_target=True, max_degree=4, min_degree=2, n_trials=250,
    seed=42, sweep_degrees=True, basis="chebyshev", optimizer="cma_batch",
    multi_fidelity=False,
)


def test_default_off_byte_identical():
    """None / [] cross-fit prior == legacy: coef_a, coef_b, mi, bin_func all equal."""
    a, b, y = _pair_target()
    base = optimise_hermite_pair(x_a=a, x_b=b, y=y, **_COMMON)
    assert base is not None
    none_seed = optimise_hermite_pair(x_a=a, x_b=b, y=y, cross_fit_prior_seeds=None, **_COMMON)
    empty_seed = optimise_hermite_pair(x_a=a, x_b=b, y=y, cross_fit_prior_seeds=[], **_COMMON)
    for other in (none_seed, empty_seed):
        assert other is not None
        assert np.array_equal(base.coef_a, other.coef_a)
        assert np.array_equal(base.coef_b, other.coef_b)
        assert base.mi == other.mi
        assert base.bin_func_name == other.bin_func_name


def test_size_mismatched_prior_seed_silently_skipped():
    """A prior seed whose length matches NO (ca_size, cb_size) for any swept
    degree must not crash and must not alter the byte-identical result."""
    a, b, y = _pair_target()
    base = optimise_hermite_pair(x_a=a, x_b=b, y=y, **_COMMON)
    # A 3-element vector cannot equal ca_size+cb_size for any degree>=2 pair
    # (smallest is 2*(min_degree+1) = 6); it is silently skipped.
    bad = optimise_hermite_pair(
        x_a=a, x_b=b, y=y, cross_fit_prior_seeds=[np.zeros(3)], **_COMMON,
    )
    assert bad is not None
    assert np.array_equal(base.coef_a, bad.coef_a)
    assert np.array_equal(base.coef_b, bad.coef_b)
    assert base.mi == bad.mi


def test_seeded_path_rescored_on_this_data():
    """The INVARIANT: a prior seed only changes search INIT, never admission.

    Feed the global optimum coefficients as a prior into an INDEPENDENT-noise
    target (y has no relation to a, b). The seed cannot manufacture a survivor:
    the winner is re-scored on THIS data and the permutation noise-floor /
    baseline-uplift gates reject it, so the result is None exactly as in the
    cold (no-prior) run. Warm-start changes order, never which features admit.
    """
    rng = np.random.default_rng(1)
    a = rng.normal(size=3000)
    b = rng.normal(size=3000)
    y = rng.integers(0, 2, size=3000)  # pure noise target, independent of a,b
    # Coefficients that WOULD reconstruct a strong product on a real target.
    strong = _pair_target()
    ref = optimise_hermite_pair(x_a=strong[0], x_b=strong[1], y=strong[2], **_COMMON)
    assert ref is not None
    prior = [np.concatenate([ref.coef_a, ref.coef_b])]

    cold = optimise_hermite_pair(x_a=a, x_b=b, y=y, **_COMMON)
    warm = optimise_hermite_pair(x_a=a, x_b=b, y=y, cross_fit_prior_seeds=prior, **_COMMON)
    # On a noise target both must agree on admission (None == not admitted).
    assert (cold is None) == (warm is None), (
        "cross-fit prior changed ADMISSION on a noise target -- LEAK BUG: "
        f"cold={'None' if cold is None else 'kept'} warm={'None' if warm is None else 'kept'}"
    )
