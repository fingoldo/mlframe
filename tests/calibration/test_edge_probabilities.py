"""Edge-case coverage for ``mlframe.calibration.probabilities`` generators.

Covers empty / single-sample inputs, all-0 / all-1 outcomes, flip_percent 0 vs 1,
chunk-size-vs-residual splitting, NaN propagation, single-class (undefined ROC AUC)
and non-convergence (best-so-far) for the similar-prob generators.

Behaviour verified against the real API before assertions were pinned:
  - ``fast_roc_auc`` returns NaN (not a raise) on single-class outcomes, so
    ``generate_similar_probs`` cannot score any candidate and returns ``None``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

from mlframe.calibration.probabilities import (
    generate_probs_from_outcomes,
    generate_similar_probs,
    generate_similar_probs_by_ranking,
)


def test_generate_probs_empty_returns_empty_float32():
    """Generate probs empty returns empty float32."""
    out = generate_probs_from_outcomes(np.array([], dtype=np.int64), chunk_size=5)
    assert out.shape == (0,)
    assert out.dtype == np.float32


@pytest.mark.parametrize("outcome", [0, 1])
def test_generate_probs_single_sample(outcome):
    # n=1 falls entirely into the residual tail (n // chunk_size == 0 chunks).
    """Generate probs single sample."""
    out = generate_probs_from_outcomes(np.array([outcome], dtype=np.int64), chunk_size=5, random_state=3)
    assert out.shape == (1,)
    assert np.isfinite(out).all()
    assert 0.0 <= out[0] <= 1.0


def test_generate_probs_flip_percent_0_vs_1_on_all_zero():
    # flip=0: no outcome flips -> every chunk freq==0 -> probs cluster near 0.
    # flip=1: every outcome flips 0->1 -> every chunk freq==1 -> probs cluster near 1.
    """Generate probs flip percent 0 vs 1 on all zero."""
    outcomes = np.zeros(80, dtype=np.int64)
    lo = generate_probs_from_outcomes(outcomes, chunk_size=8, flip_percent=0.0, random_state=1)
    hi = generate_probs_from_outcomes(outcomes, chunk_size=8, flip_percent=1.0, random_state=1)
    assert lo.mean() < 0.2, f"flip=0 on all-zero should stay near 0, got mean {lo.mean()}"
    assert hi.mean() > 0.8, f"flip=1 on all-zero should invert to near 1, got mean {hi.mean()}"
    assert hi.mean() > lo.mean() + 0.6


def test_generate_probs_chunk_vs_residual_matrix():
    # n=25, chunk_size=10 -> two full 10-row chunks + a 5-row residual tail.
    """Generate probs chunk vs residual matrix."""
    outcomes = np.array(([0, 1] * 12) + [1], dtype=np.int64)
    assert outcomes.shape == (25,)
    out = generate_probs_from_outcomes(outcomes, chunk_size=10, random_state=2)
    assert out.shape == (25,)
    assert np.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_generate_probs_nan_outcome_propagates_not_masked():
    # The kernel uses plain .mean() (NOT nanmean) on the outcome chunk, so a NaN outcome
    # deliberately poisons its chunk's probs rather than being silently dropped.
    """Generate probs nan outcome propagates not masked."""
    outcomes = np.array([0.0, np.nan, 1.0, 1.0])
    out = generate_probs_from_outcomes(outcomes, chunk_size=2, random_state=0)
    assert out.shape == (4,)
    assert np.isnan(out[:2]).all(), "NaN outcome must propagate into its chunk's probs"
    assert np.isfinite(out[2:]).all(), "clean chunk must remain finite"


def test_generate_similar_probs_single_class_returns_none():
    # Single-class outcomes make ROC AUC undefined (NaN), so no candidate can be scored
    # and the best-so-far tracker is never updated -> None. Documents the degenerate path;
    # the np.ndarray return annotation does not hold for single-class input.
    """Generate similar probs single class returns none."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, size=40)
    out = generate_similar_probs(p, np.zeros(40, dtype=int), n_iterations=5, random_state=0)
    assert out is None


def test_generate_similar_probs_nonconvergence_returns_best_so_far():
    # Large noise_scale + few iterations -> the rtol=0.01 convergence check never trips,
    # so the function returns the best-so-far candidate (finite ndarray), never None, for
    # a valid two-class target.
    """Generate similar probs nonconvergence returns best so far."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, size=60)
    y = (rng.random(60) > 0.5).astype(int)
    out = generate_similar_probs(p, y, noise_scale=0.7, n_iterations=3, random_state=0)
    assert out is not None
    assert out.shape == p.shape
    assert np.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 1.0).all()


@pytest.mark.parametrize("n_bins", [1, 5])
def test_generate_similar_probs_by_ranking_noise0_preserves_multiset(n_bins):
    # noise_scale=0 only shuffles within rank-bins, so the output is a permutation of the
    # input (identical multiset), never new values.
    """Generate similar probs by ranking noise0 preserves multiset."""
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.95, size=40)
    y = (rng.random(40) > 0.5).astype(int)
    out = generate_similar_probs_by_ranking(p, y, n_bins=n_bins, noise_scale=0.0, random_state=0)
    assert out.shape == p.shape
    assert sorted(out.tolist()) == pytest.approx(sorted(p.tolist()))
