"""Denominator/scale-jump detection to unscale anonymized integer features.

COMPETITION/EXPLORATORY USE ONLY -- NOT FOR PRODUCTION.

Kaggle competitions frequently anonymize originally-integer features by rescaling
and adding small noise (e.g. ``x = true_int * denom + noise``). This module recovers
the likely quantization step (``denom``'s reciprocal) of such a feature by sorting its
unique values, taking successive differences, and finding the dominant small gap that
all (or nearly all) other gaps are near-integer multiples of -- then de-rounds the
feature back toward its true integer values.

Real production systems do not deliberately obfuscate their own integer fields this
way, so this is a de-anonymization/reverse-engineering diagnostic with essentially no
production applicability. It must never be imported by production mlframe code or
wired into default pipelines -- see ``mlframe.competition`` package docstring and
``MLFRAME_IDEAS_competitions.md`` ("Denominator/scale-jump detection...").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["detect_quantization_step", "derounded_feature", "rank_features_by_quantization_confidence", "QuantizationRankResult"]


def detect_quantization_step(
    x: np.ndarray,
    *,
    min_gap: float = 1e-9,
    max_candidates: int = 500,
) -> float:
    """Recover the likely quantization step of a noised/scaled integer feature.

    COMPETITION/EXPLORATORY USE ONLY -- see module docstring.

    Sorts the finite values of ``x``, collapses repeated/near-repeated draws of the same
    underlying pre-noise value into per-cluster centroids (many samples can share a true
    integer with independent noise, which would otherwise pollute the raw successive-diff
    gaps), then computes successive differences between cluster centroids and picks the
    smallest gap ``g`` that best explains the rest of that gap distribution: for each
    candidate the "explained" score is the fraction of all centroid gaps that land close to
    an integer multiple of the candidate. The candidate with the highest explained score
    wins; ties broken by preferring the larger step (coarser, more likely the true
    denominator rather than a noise-induced sub-multiple).

    Args:
        x: 1-D array of (possibly noised/scaled) numeric values.
        min_gap: gaps smaller than this are treated as duplicates/noise and dropped.
        max_candidates: cap on how many smallest centroid gaps are tried as step candidates
            (keeps the search cheap on large/high-cardinality inputs).

    Returns:
        The estimated quantization step (float). Returns ``float("nan")`` if fewer than 2
        finite, sufficiently-distinct clusters are present (nothing to quantify).
    """
    real_gaps = _centroid_gaps(x, min_gap=min_gap)
    if real_gaps is None or real_gaps.size == 0:
        return float("nan")

    candidates = np.unique(real_gaps)
    if candidates.size > max_candidates:
        candidates = candidates[:max_candidates]  # smallest gaps first (uniqued+sorted)

    scores = np.empty(candidates.size)
    for i, cand in enumerate(candidates):
        ratios = real_gaps / cand
        nearest_int = np.round(ratios)
        nearest_int[nearest_int == 0] = 1  # a gap smaller than a full step still counts as 1 unit of noise
        # Residual in units of the candidate step itself (NOT divided by nearest_int): dividing by
        # nearest_int would make the relative tolerance trivially satisfied for absurdly tiny candidates
        # (a huge integer multiple is always "relatively close" to its rounded value), spuriously favoring
        # near-zero steps over the true one.
        residual = np.abs(ratios - nearest_int)
        scores[i] = float(np.mean(residual < 0.15))

    best_score = float(scores.max())
    # Many single-step candidates near the true scale all explain the gap population near-equally
    # well (they're all "close enough" to be within tolerance of most gaps); picking any ONE of them
    # (e.g. the largest, or the first) is an arbitrary, noisy choice among near-ties. The median of
    # ALL top-scoring candidates is a far less noisy point estimate of the true step.
    top_candidates = candidates[scores >= best_score - 1e-9]
    best_step = float(np.median(top_candidates))

    return _refine_step_by_regression(real_gaps, best_step)


def _refine_step_by_regression(gaps: np.ndarray, approx_step: float) -> float:
    """Sharpen a coarse step estimate via a through-origin least-squares fit over cumulative gaps.

    The candidate search above locks onto a step that best explains gap MULTIPLES, but each
    individual gap still carries its own noise; averaging that noise away needs many gaps at once,
    not just one. Reconstructing integer indices ``k`` for the cumulative positions and regressing
    position on ``k`` (``position ~= step * k``, through the origin) uses every gap simultaneously,
    which cancels per-gap noise far more than the single winning candidate gap does -- this matters
    when de-rounding needs sub-percent precision to avoid drifting past the half-step boundary at
    large integer values.
    """
    if not np.isfinite(approx_step) or approx_step <= 0 or gaps.size == 0:
        return approx_step
    positions = np.concatenate(([0.0], np.cumsum(gaps)))
    k = np.round(positions / approx_step)
    denom = float(np.sum(k * k))
    if denom <= 0:
        return approx_step
    refined = float(np.sum(positions * k) / denom)
    return refined if np.isfinite(refined) and refined > 0 else approx_step


def _noise_step_split_threshold(sorted_unique_gaps: np.ndarray) -> "float | None":
    """Find the log-scale valley separating intra-cluster noise gaps from real inter-value gaps.

    When many samples share a true (pre-noise) value, sorting the noised values produces two
    log-scale-separated gap populations: many tiny within-cluster spacings (noise-scale) and a
    minority of much larger between-cluster spacings (quantization-step-scale) -- exactly the
    "spike in the gap distribution" the technique is named for. Returns the geometric-mean
    threshold at the valley, or ``None`` if no clear valley is found (e.g. too few points, or no
    real clustering present at all).
    """
    if sorted_unique_gaps.size < 3:
        return None
    # Skip the very smallest gaps before hunting for the split: with few points at the extreme low
    # end, a single lone tiny gap next to its nearest neighbor produces a huge log-jump purely from
    # sparsity (few samples -> large relative spacing), which would otherwise dominate a plain
    # global argmax and split off almost nothing. The real noise/step valley reliably shows up once
    # that sparse tail is skipped, since the noise-gap population itself has many members clustered
    # together.
    skip = max(5, int(0.01 * sorted_unique_gaps.size))
    if skip >= sorted_unique_gaps.size - 1:
        return None
    log_gaps = np.log(sorted_unique_gaps[skip:])
    jumps = np.diff(log_gaps)
    split_local = int(np.argmax(jumps))
    split = skip + split_local + 1
    return float(np.sqrt(sorted_unique_gaps[split - 1] * sorted_unique_gaps[split]))


def _centroid_gaps(x: np.ndarray, *, min_gap: float) -> "np.ndarray | None":
    """Cluster raw values by a noise/step valley threshold and return gaps between cluster centroids.

    Using RAW successive-unique-value gaps directly (rather than cluster centroids) systematically
    biases the recovered step downward whenever many samples share a true value: the boundary gap
    between two clusters' extreme (max of one, min of next) noised values is shorter than the true
    center-to-center spacing by roughly the sum of each cluster's noise spread. Collapsing each
    cluster to its mean before differencing removes that boundary-shrinkage bias.
    """
    finite = np.sort(x[np.isfinite(x)])
    if finite.size < 2:
        return None

    uniq = np.unique(finite)
    if uniq.size < 2:
        return None
    unique_gaps = np.diff(uniq)
    unique_gaps = unique_gaps[unique_gaps > min_gap]
    if unique_gaps.size == 0:
        return None

    threshold = _noise_step_split_threshold(np.sort(unique_gaps))
    if threshold is None:
        # No detectable intra-cluster noise population -- values are already effectively distinct
        # points, so the raw successive-value gaps ARE the between-value gaps.
        return unique_gaps

    all_diffs = np.diff(finite)
    breaks = np.where(all_diffs > threshold)[0] + 1
    boundaries = np.concatenate(([0], breaks, [finite.size]))
    if boundaries.size < 3:
        return unique_gaps
    centers = np.array([finite[boundaries[i] : boundaries[i + 1]].mean() for i in range(boundaries.size - 1)])
    centroid_gaps = np.diff(centers)
    centroid_gaps = centroid_gaps[centroid_gaps > min_gap]
    return centroid_gaps if centroid_gaps.size > 0 else unique_gaps


def derounded_feature(x: np.ndarray, step: float) -> np.ndarray:
    """De-round a scaled/noised feature back toward its true integer-multiple grid.

    COMPETITION/EXPLORATORY USE ONLY -- see module docstring.

    Args:
        x: original (noised/scaled) values.
        step: quantization step, e.g. from ``detect_quantization_step``.

    Returns:
        ``round(x / step) * step``, elementwise; ``x`` unchanged where ``step`` is
        non-finite or non-positive (nothing to de-round).
    """
    if not np.isfinite(step) or step <= 0:
        return np.array(x, dtype=float)
    return np.asarray(np.round(np.asarray(x, dtype=float) / step) * step, dtype=float)


@dataclass
class QuantizationRankResult:
    """Per-feature quantization-step diagnostics, ranked by recovery confidence.

    COMPETITION/EXPLORATORY USE ONLY -- see module docstring.
    """

    feature_names: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    confidences: list = field(default_factory=list)


def rank_features_by_quantization_confidence(
    features: dict,
    *,
    min_gap: float = 1e-9,
    max_candidates: int = 500,
) -> QuantizationRankResult:
    """Rank a set of numeric features by how confidently a quantization step was recovered.

    COMPETITION/EXPLORATORY USE ONLY -- see module docstring.

    Confidence is the fraction of successive-value gaps explained by the recovered step
    (same statistic ``detect_quantization_step`` optimizes internally) -- higher means the
    feature more plausibly began life as a scaled/noised integer field.

    Args:
        features: mapping of feature name -> 1-D numeric array.
        min_gap, max_candidates: forwarded to ``detect_quantization_step``.

    Returns:
        ``QuantizationRankResult`` with features sorted by descending confidence.
    """
    rows = []
    for name, arr in features.items():
        step = detect_quantization_step(np.asarray(arr, dtype=float), min_gap=min_gap, max_candidates=max_candidates)
        uniq = np.unique(np.asarray(arr, dtype=float)[np.isfinite(arr)])
        if uniq.size < 2 or not np.isfinite(step) or step <= 0:
            confidence = 0.0
        else:
            gaps = np.diff(uniq)
            gaps = gaps[gaps > min_gap]
            if gaps.size == 0:
                confidence = 0.0
            else:
                # Scored against the FULL, unfiltered gap set (not the noise-filtered candidate pool used
                # for step search) so the score is an honest, non-circular discriminator: genuinely
                # quantized data has essentially no gaps left unexplained, while non-quantized data still
                # has plenty of gaps the winning candidate can't explain, even though that same candidate
                # was chosen to fit the filtered pool well.
                ratios = gaps / step
                nearest_int = np.round(ratios)
                nearest_int[nearest_int == 0] = 1
                residual = np.abs(ratios - nearest_int)
                confidence = float(np.mean(residual < 0.15))
        rows.append((name, step, confidence))

    rows.sort(key=lambda r: r[2], reverse=True)
    result = QuantizationRankResult()
    for name, step, confidence in rows:
        result.feature_names.append(name)
        result.steps.append(step)
        result.confidences.append(confidence)
    return result
