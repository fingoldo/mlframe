"""`_y_train_clip_bounds` must be bit-identical to the two-call form it replaced -- measured on the function.

The single `np.quantile(finite, (0.001, 0.999))` call does one sort instead of two, and the claim attached
to it is bit-identity, not approximation. The test that carried that claim did not check it: it declared a
`_reference_bounds` helper nothing ever called, and then compared `np.quantile(y, 0.001)` against
`np.quantile(y, (0.001, 0.999))[0]` -- a numpy-versus-numpy identity that holds whatever production does.
It imported `_y_train_clip_bounds` and never called it, so changing the probability pair, reverting to two
separate calls, or dropping the `isfinite` filter all left it green.

The NaN filter and the degenerate-span branch had no coverage at all: the one test that did reach
production used all-finite, non-constant input.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mlframe.training.composite.estimator import _Y_CLIP_HIGH_FRAC, _Y_CLIP_LOW_FRAC, _y_train_clip_bounds


def _reference_bounds(y_train: np.ndarray) -> tuple[float, float]:
    """The pre-optimisation form: filter, then TWO separate quantile calls, then the same envelope."""
    finite = y_train[np.isfinite(y_train)]
    if finite.size == 0:
        return float("-inf"), float("inf")
    q_low = float(np.quantile(finite, 0.001))
    q_high = float(np.quantile(finite, 0.999))
    span = q_high - q_low
    if span <= 0:
        med = float(np.median(finite))
        return med - 0.1 * abs(med) - 1e-6, med + 0.1 * abs(med) + 1e-6
    return q_low - (1.0 - _Y_CLIP_LOW_FRAC) * span, q_high + (_Y_CLIP_HIGH_FRAC - 1.0) * span


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("n", [1000, 50_000])
def test_the_function_is_bit_identical_to_the_two_call_reference(seed: int, n: int):
    """Equality, not approximation: the one-sort rewrite claimed bit-identity."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n).astype(np.float64)
    assert _y_train_clip_bounds(y) == _reference_bounds(y)


@pytest.mark.parametrize("seed", [0, 3])
def test_non_finite_values_are_excluded_before_the_quantiles(seed: int):
    """The `isfinite` filter had no coverage; without it a single inf drags both bounds to infinity."""
    rng = np.random.default_rng(seed)
    clean = rng.standard_normal(5_000).astype(np.float64)
    dirty = np.concatenate([clean, [np.nan, np.inf, -np.inf, np.nan]])
    rng.shuffle(dirty)

    assert _y_train_clip_bounds(dirty) == _reference_bounds(dirty)
    lo, hi = _y_train_clip_bounds(dirty)
    assert np.isfinite(lo) and np.isfinite(hi), "a non-finite value reached the quantiles"


def test_an_all_non_finite_target_gives_an_unbounded_envelope():
    """The empty-after-filter branch: nothing to clip against, so nothing is clipped."""
    y = np.array([np.nan, np.inf, -np.inf], dtype=np.float64)
    assert _y_train_clip_bounds(y) == (float("-inf"), float("inf"))


def test_a_constant_target_falls_back_to_a_wiggle_around_the_median():
    """The degenerate-span branch, also uncovered: a zero envelope must not collapse to a single point."""
    y = np.full(1_000, 4.0, dtype=np.float64)
    lo, hi = _y_train_clip_bounds(y)
    assert (lo, hi) == _reference_bounds(y)
    assert lo < 4.0 < hi, f"the constant-target fallback produced a degenerate envelope ({lo}, {hi})"


def test_the_envelope_is_asymmetric_in_the_documented_direction():
    """0.9 spans below, 9 above: a symmetric clip would bite legitimate upper-tail predictions."""
    rng = np.random.default_rng(5)
    y = rng.standard_normal(20_000).astype(np.float64)
    lo, hi = _y_train_clip_bounds(y)
    q_low, q_high = (float(v) for v in np.quantile(y, (0.001, 0.999)))
    span = q_high - q_low
    assert lo == pytest.approx(q_low - (1.0 - _Y_CLIP_LOW_FRAC) * span, abs=0.0)
    assert hi == pytest.approx(q_high + (_Y_CLIP_HIGH_FRAC - 1.0) * span, abs=0.0)
    assert (hi - q_high) > (q_low - lo), "the extension is no longer wider above than below"


def test_one_sort_is_not_slower_than_two():
    """The perf sentinel now times the function rather than two local lambdas.

    A generous bound: this is guarding against an accidental revert to two sorts, not measuring a speedup,
    and the machine may be loaded.
    """
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200_000).astype(np.float64)
    _y_train_clip_bounds(y)  # warm

    def _timed(fn) -> float:
        """Best of three wall-clock runs, in seconds."""
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            fn(y)
            best = min(best, time.perf_counter() - t0)
        return best

    assert _timed(_y_train_clip_bounds) <= _timed(_reference_bounds) * 1.5
