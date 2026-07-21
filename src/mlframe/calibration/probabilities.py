"""Synthetic-probability generation for calibration testing.

- ``generate_probs_from_outcomes``: derive plausible predicted-probability vectors from known binary
  outcomes (chunked, roughly-calibrated).
- ``generate_similar_probs_logit_space``: perturb an existing probability vector via Gaussian noise in
  logit space.
- ``generate_similar_probs_random_walk``: perturb an existing probability vector via a bounded random walk.
- ``generate_similar_probs``: perturb an existing probability vector, iterating to keep it close to the
  original Brier Score and ROC AUC.
- ``generate_similar_probs_by_ranking``: rank-preserving bin-shuffle + noise variant of the above.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numba import njit
from scipy.stats import rankdata
from sklearn.utils import check_random_state
# Project numba metrics, proven sklearn-equivalent by mlframe.metrics tests; used here over
# sklearn.metrics.{roc_auc_score,brier_score_loss} to avoid sklearn call overhead in the hot loop.
from mlframe.metrics.core import fast_roc_auc, fast_brier_score_loss


@njit(cache=True, nogil=True)
def _generate_probs_from_outcomes_kernel(
    outcomes: np.ndarray,
    indices: np.ndarray,
    bin_offsets: np.ndarray,
    noise: np.ndarray,
    chunk_size: int,
    scale: float,
    nbins: int,
    flip_percent: float,
) -> np.ndarray:
    """Pure-compute njit core: all randomness (``indices`` permutation, per-bin ``bin_offsets``,
    per-row ``noise``) is drawn by the Python wrapper from a per-call seeded Generator and passed
    in, so this kernel is deterministic AND thread-safe (no process-global / njit-global RNG state).
    ``noise`` is a length-n uniform-[0,1) draw sliced per chunk in the same order the old per-chunk
    ``np.random.random(size=chunk_size)`` consumed it.
    """
    n = len(outcomes)
    # ``np.zeros`` (not empty): the chunk loop stops at ``r = (n // chunk_size) * chunk_size < n``
    # when ``n % chunk_size != 0``; the residual block below fills the tail, but zero-init keeps
    # the worst case a recognisable tail of zeros rather than uninitialised garbage.
    probs = np.zeros(n, dtype=np.float32)

    if flip_percent:
        # Without-replacement flip: ``indices`` is already a uniform permutation, so its first
        # ``floor(n*flip_percent)`` entries are a without-replacement sample (WITH-replacement
        # np.random.choice would let duplicates cancel and flip fewer than requested).
        flip_size = int(n * flip_percent)
        if flip_size:
            outcomes = outcomes.copy()
            flip_indices = indices[:flip_size]
            outcomes[flip_indices] = 1 - outcomes[flip_indices]

    lo = 0  # left border
    for idx in range(n // chunk_size):  # traverse randomly selected chunks/subsets of original data
        r = (idx + 1) * chunk_size  # right border
        freq = outcomes[lo:r].mean()  # find real event occurring frequency in current chunk of observation

        # add pregenerated offset for particular bin. Clamp bin_idx to nbins-1 so that
        # freq==1.0 (int(1.0*nbins) == nbins) does not index out of bounds.
        bin_idx = int(freq * nbins)
        if bin_idx >= nbins:
            bin_idx = nbins - 1
        freq = freq + bin_offsets[bin_idx]

        # add small symmetric random noise. it must be higher when freq approaches [0;1] borders.
        probs[lo:r] = freq + (noise[lo:r] - 0.5) * scale * np.abs(freq - 0.5)

        lo = r

    # Residual tail rows ``[lo:n]`` the chunked loop skipped when ``n % chunk_size != 0``.
    if lo < n:
        freq = outcomes[lo:n].mean()
        bin_idx = int(freq * nbins)
        if bin_idx >= nbins:
            bin_idx = nbins - 1
        freq = freq + bin_offsets[bin_idx]
        probs[lo:n] = freq + (noise[lo:n] - 0.5) * scale * np.abs(freq - 0.5)

    return np.clip(probs, 0.0, 1.0)


def generate_probs_from_outcomes(
    outcomes: np.ndarray, chunk_size: int = 20, scale: float = 0.1, nbins: int = 10, bins_std: float = 0.1, flip_percent: float = 0.6, random_state: int = 0
) -> np.ndarray:
    """Generate hypothetical, roughly-calibrated ground-truth probs from known outcomes.

    Our model probs will (hopefully) be calibrated, so the synthetic probs must be calibrated too, with
    some fitness, and cover a broad prob range:

    0)  if flip_percent is specified, a random WITHOUT-replacement portion of data has zeroes/ones flipped
        (lowers ROC AUC): exactly ``floor(n*flip_percent)`` distinct observations flip.
    1) work with small random chunks/subsets of data;
    2) for every chunk compute its real freq;
    3) draw each observation's 'exact' prob around the chunk freq, plus a constant per-bin ``bins_std`` noise.

    Final result is clipped to [0,1].

    Reproducibility / thread-safety: ALL randomness (the row permutation, per-bin offsets, per-row noise) is
    drawn HERE from a per-call ``np.random.default_rng(random_state)`` and passed into the pure-compute njit
    kernel. The previous version called ``np.random.seed(random_state)`` INSIDE the njit body, which mutates
    numba's njit-GLOBAL RNG stream -- racy under concurrent threaded callers (one thread's seed clobbers
    another's mid-flight, so results depended on inter-thread ordering). Threading a per-call Generator makes
    the output a deterministic function of ``random_state`` alone, independent of concurrency, and leaves both
    numpy's and numba's global RNG state untouched.
    """
    outcomes = np.asarray(outcomes)
    n = len(outcomes)
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n).astype(np.int64)
    bin_offsets = (rng.random(size=nbins) - 0.5) * bins_std
    # One length-n uniform draw sliced per chunk (same consumption order as the old per-chunk draws).
    noise = rng.random(size=n) if n else np.empty(0, dtype=np.float64)
    return np.asarray(_generate_probs_from_outcomes_kernel(outcomes, indices, bin_offsets, noise, int(chunk_size), float(scale), int(nbins), float(flip_percent)))


from scipy.special import logit, expit


def generate_similar_probs_logit_space(
    predicted_probs: np.ndarray, true_outcomes: np.ndarray, noise_scale: float = 0.05, random_state: Optional[int] = None
) -> np.ndarray:
    """
    Perturb probabilities by applying noise in logit space (log-odds),
    then converting back to probability space.

    Args:
        predicted_probs (np.ndarray): Original predicted probabilities (0-1).
        true_outcomes (np.ndarray): True binary outcomes (0 or 1).
        noise_scale (float): Scale of Gaussian noise applied in logit space.

    Returns:
        np.ndarray: A new set of perturbed probabilities.
    """
    # Convert probabilities to logit (log-odds) space
    rng = check_random_state(random_state)
    logit_probs = logit(np.clip(predicted_probs, 1e-6, 1 - 1e-6))  # Avoid log(0) issues

    # Add Gaussian noise in logit space
    perturbed_logit = logit_probs + rng.normal(0, noise_scale, size=logit_probs.shape)

    # Convert back to probability space using the sigmoid function
    similar_probs = expit(perturbed_logit)

    return np.asarray(similar_probs)


def generate_similar_probs_random_walk(
    predicted_probs: np.ndarray, true_outcomes: np.ndarray, step_size: float = 0.05, n_steps: int = 1, random_state: Optional[int] = None
) -> np.ndarray:
    """
    Generates perturbed probabilities using a small random walk.

    Args:
        predicted_probs (np.ndarray): Original predicted probabilities (0-1).
        true_outcomes (np.ndarray): True binary outcomes (0 or 1).
        step_size (float): Size of each random walk step.
        n_steps (int): Number of random walk steps.

    Returns:
        np.ndarray: A new set of perturbed probabilities.
    """
    rng = check_random_state(random_state)
    similar_probs = predicted_probs.copy()

    for _ in range(n_steps):
        # Add or subtract random small values for the random walk
        random_step = rng.uniform(-step_size, step_size, size=predicted_probs.shape)
        similar_probs += random_step

        # Ensure probabilities stay within [0, 1]
        similar_probs = np.clip(similar_probs, 0, 1)

    return np.asarray(similar_probs)


def generate_similar_probs(
    predicted_probs: np.ndarray, true_outcomes: np.ndarray, noise_scale: float = 0.05, n_iterations: int = 100, random_state: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    Generates a random ndarray of similar probabilities that has approximately the same
    Brier Score and ROC AUC as the input predicted_probs.

    Args:
        predicted_probs (np.ndarray): Original predicted probabilities (0-1).
        true_outcomes (np.ndarray): True binary outcomes (0 or 1).
        noise_scale (float): Standard deviation of the noise to add.
        n_iterations (int): Number of iterations for fine-tuning the noise scale.

    Returns:
        np.ndarray: A similar_probs array with approximately the same Brier Score and ROC AUC, or
            ``None`` for degenerate single-class ``true_outcomes`` (ROC AUC is undefined, so the joint
            metric distance is NaN and no candidate is ever accepted).
    """
    rng = check_random_state(random_state)
    original_brier_score = fast_brier_score_loss(true_outcomes, predicted_probs)
    original_auc = fast_roc_auc(true_outcomes, predicted_probs)

    similar_probs = None
    # Track the best candidate seen so far (minimum joint distance to the target metrics)
    # so that non-convergence returns the closest approximation rather than whatever the
    # final iteration produced. The previous impl returned `noisy_probs` (the last draw),
    # which could be arbitrarily far from the originals.
    best_probs = None
    best_score = np.inf

    for _ in range(n_iterations):
        # Add Gaussian noise and clip the values to keep them in [0, 1]
        noisy_probs = predicted_probs + rng.normal(loc=0.0, scale=noise_scale, size=predicted_probs.shape)
        noisy_probs = np.clip(noisy_probs, 0, 1)

        # Calculate new Brier Score and AUC
        new_brier_score = fast_brier_score_loss(true_outcomes, noisy_probs)
        new_auc = fast_roc_auc(true_outcomes, noisy_probs)

        score = abs(new_brier_score - original_brier_score) / max(abs(original_brier_score), 1e-12) + abs(new_auc - original_auc) / max(
            abs(original_auc), 1e-12
        )
        if score < best_score:
            best_score = score
            best_probs = noisy_probs

        # Check if the new metrics are close to the original
        if np.isclose(new_brier_score, original_brier_score, rtol=0.01) and np.isclose(new_auc, original_auc, rtol=0.01):
            similar_probs = noisy_probs
            break

    # If the loop doesn't converge, return the best-so-far (closest joint Brier+AUC match),
    # not the final iteration's draw.
    return similar_probs if similar_probs is not None else best_probs


def generate_similar_probs_by_ranking(
    predicted_probs: np.ndarray,
    true_outcomes: np.ndarray,
    n_bins: int = 10,
    noise_scale: float = 0.001,
    n_iterations: int = 1,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generates a new set of probabilities by shuffling within ranked bins,
    preserving the ranking and (over ``n_iterations`` candidate draws) verifying it stays close to the
    ORIGINAL Brier Score and ROC AUC against ``true_outcomes`` -- the same closeness-tracking pattern
    :func:`generate_similar_probs` uses, since a single rank-preserving shuffle+noise draw is only
    APPROXIMATELY metric-preserving, not verified.

    Args:
        predicted_probs (np.ndarray): Original predicted probabilities (0-1).
        true_outcomes (np.ndarray): True binary outcomes (0 or 1).
        n_bins (int): Number of bins to group the probabilities into (e.g., quantiles).
        noise_scale (float): Amount of random noise to add after shuffling to introduce variation.
        n_iterations (int): Number of candidate draws to try, keeping the one closest to the original
            Brier Score/ROC AUC (early-stops once a draw is within 1% of both). ``1`` (default) reproduces
            the original single-draw behavior bit-for-bit, with no metric verification.

    Returns:
        np.ndarray: A new set of similar probabilities.
    """
    rng = check_random_state(random_state)
    n = len(predicted_probs)

    def _one_draw() -> np.ndarray:
        """One rank-preserving shuffle-within-bin + noise draw over ``predicted_probs``."""
        # Get ranks of predicted probabilities
        ranks = rankdata(predicted_probs, method="ordinal")
        # Create bins based on the ranks (grouping into approximately equal-sized bins)
        bins = np.floor_divide(ranks * n_bins, n)

        similar_probs = predicted_probs.copy()

        # Shuffle within each bin.
        # bench-attempt-rejected (2026-07): replacing this np.unique + per-bin np.where scan with a
        # single stable-argsort-by-bin + contiguous-slice walk was bit-identical (stable sort preserves
        # ascending intra-bin index order -> same rng.shuffle draw order) but SLOWER at every tested cell
        # (n in {1e5,1e6} x n_bins in {10,100}): e.g. n=1e6,n_bins=10 old 96ms vs argsort 220ms/contig
        # 223ms (0.43x). The O(n_bins*N) boolean scans are vectorized cache-friendly passes and n_bins is
        # small, so they beat the O(N log N) argsort + fancy-index gather/scatter.
        for bin_value in np.unique(bins):
            bin_indices = np.where(bins == bin_value)[0]
            # Extract the probabilities in this bin
            bin_probs = similar_probs[bin_indices]
            # Shuffle them
            rng.shuffle(bin_probs)
            # Assign the shuffled values back to their positions
            similar_probs[bin_indices] = bin_probs

        if noise_scale:
            # Add small noise to ensure variation
            similar_probs += rng.normal(0, noise_scale, size=predicted_probs.shape)

            # Ensure probabilities are still between 0 and 1
            similar_probs = np.clip(similar_probs, 0, 1)

        return np.asarray(similar_probs)

    if n_iterations <= 1:
        return _one_draw()

    original_brier_score = fast_brier_score_loss(true_outcomes, predicted_probs)
    original_auc = fast_roc_auc(true_outcomes, predicted_probs)

    best_probs: Optional[np.ndarray] = None
    best_score = np.inf
    for _ in range(n_iterations):
        candidate = _one_draw()
        new_brier_score = fast_brier_score_loss(true_outcomes, candidate)
        new_auc = fast_roc_auc(true_outcomes, candidate)
        score = abs(new_brier_score - original_brier_score) / max(abs(original_brier_score), 1e-12) + abs(new_auc - original_auc) / max(
            abs(original_auc), 1e-12
        )
        # `best_probs is None` keeps the first draw as a fallback even when `score` comes out NaN (degenerate
        # true_outcomes/predicted_probs, e.g. a single-class fold) -- `nan < best_score` is always False, so
        # relying on the score comparison alone would leave best_probs unset and return None from a function
        # typed to always return an ndarray.
        if best_probs is None or score < best_score:
            best_score = score
            best_probs = candidate
        if np.isclose(new_brier_score, original_brier_score, rtol=0.01) and np.isclose(new_auc, original_auc, rtol=0.01):
            return candidate

    assert best_probs is not None  # guaranteed by the "first draw" fallback above (n_iterations > 1 here)
    return best_probs
