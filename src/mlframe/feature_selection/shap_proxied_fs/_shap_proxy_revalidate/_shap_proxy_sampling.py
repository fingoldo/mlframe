"""Anchor-subset sampling primitives for the SHAP-proxy trust guard.

Pure numpy helpers: softmax weighting of unit F-scores, WRSwoR column draws, a Zipf cardinality
prior, and the stratified anchor-subset sampler. Leaf module -- no back-import to the revalidate
parent, so the loss / refine siblings can depend on it freely.
"""

from __future__ import annotations

import numpy as np


def _softmax_weights(scores, temperature="auto"):
    """Normalise ``scores`` to a length-N probability vector via softmax with a temperature knob.

    ``scores`` is the proxy/unit-space F-score vector (length n_anchor_columns). -inf sentinels for
    constant/degenerate columns sink to zero probability. NaN/non-finite entries get the minimum
    finite score so they remain pickable but at the noise floor (never the high-prior tier).

    ``temperature='auto'`` (default since iter97): set temperature to the std of finite scores so the
    softmax is scale-invariant. Raw F-statistics span O(1)-O(100+) on real cohorts; a fixed
    ``temperature=1.0`` then collapses softmax to near-one-hot on the top entry (effective sample size
    ~1) and the stratified anchor sampler draws essentially identical anchors, killing the trust-guard
    Spearman signal. Standardising by the score std keeps the softmax entropy bounded regardless of
    whether the caller passes raw F-scores (std ~40) or log/z-scored values (std ~1). Passing a
    numeric ``temperature`` restores the legacy fixed-divisor behaviour (used by the unit-test fixture
    where weights have a calibrated [0,10] range).

    Returns a length-N float64 vector that sums to 1; falls back to a uniform vector when every entry
    is non-finite (degenerate input, e.g. all-constant working frame) so callers never crash."""
    s = np.asarray(scores, dtype=np.float64).copy()
    finite = np.isfinite(s)
    if not finite.any():
        return np.full(s.shape, 1.0 / max(1, s.size), dtype=np.float64)
    # Replace non-finite entries with the min finite (so they have negligible probability after softmax
    # but never NaN out the normalisation); subtract the max for numerical stability.
    s[~finite] = s[finite].min()
    if isinstance(temperature, str) and temperature == "auto":
        # Scale-invariant: divide by the std of finite scores (clamped from below so a near-constant
        # vector doesn't blow up to a one-hot via tiny-number division). With std-normalised input the
        # softmax entropy is bounded ~ log(n) / e regardless of the raw score magnitude.
        std = float(np.std(s[finite]))
        temp = max(1e-3, std)
    else:
        temp = max(1e-12, float(temperature))
    s = s / temp
    s -= s.max()
    w = np.exp(s)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(s.shape, 1.0 / max(1, s.size), dtype=np.float64)
    return w / total


def _weighted_choice_no_replace(rng, n, k, probs):
    """Sample ``k`` distinct indices from ``range(n)`` without replacement, weighted by ``probs``.

    Uses the Efraimidis-Spirakis exponential-key reservoir trick: key_i = -log(U_i)/p_i, take the k
    smallest. O(n) per call, correct under WRSwoR weights (numpy's ``rng.choice(..., replace=False)``
    with ``p=`` already implements this, but we wrap to handle zero-weight rows + degenerate cases
    without raising)."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.sum() <= 0 or not np.all(np.isfinite(probs)):
        return rng.choice(n, size=k, replace=False)
    # Mass might be concentrated on < k entries; expand zero-weight rows to a tiny epsilon so the
    # sampler can still draw k distinct picks (the leakage is bounded by ``eps * (n-nnz)``).
    nnz = int((probs > 0).sum())
    if nnz < k:
        eps = max(1e-12, probs[probs > 0].min() * 1e-6) if nnz > 0 else 1.0
        probs = np.where(probs > 0, probs, eps)
    probs = probs / probs.sum()
    return rng.choice(n, size=k, replace=False, p=probs)


def _zipf_card_probs(min_card, max_card, alpha):
    """Build a length-(max_card-min_card+1) probability vector ``p(k) ∝ k^(-alpha)`` over the closed
    range ``[min_card, max_card]``. Used by the Zipf cardinality prior in ``_sample_anchor_subsets``.

    Pulled out as a small helper so unit tests can inspect the prior directly (mean k under Zipf is a
    cheap structural assertion of "small-k-heavy"). ``alpha`` is clamped to ``>=0``; ``alpha=0`` is
    mathematically uniform in ``k`` (every entry equals ``1/range``). Returns a float64 vector that
    sums to 1; never NaN/zero for ``min_card>=1`` (we never raise ``0**alpha``)."""
    alpha = max(0.0, float(alpha))
    ks = np.arange(int(min_card), int(max_card) + 1, dtype=np.float64)
    # ``ks`` starts at ``min_card`` which the calling sampler clamps to ``>=1`` (semantics preserved),
    # so ``ks ** -alpha`` is finite for every entry.
    w = ks ** (-alpha) if alpha > 0 else np.ones_like(ks)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        # Degenerate alpha or empty range -> uniform fallback so callers never crash.
        return np.full(ks.shape, 1.0 / max(1, ks.size), dtype=np.float64)
    return w / total


def _sample_anchor_subsets(n_features, n_anchors, rng, min_card=1, max_card=None, *,
                           weights=None, uniform_tail_frac=0.2, cardinality_dist="uniform",
                           zipf_alpha=1.0):
    """Sample distinct anchor subsets of varying cardinality.

    ``cardinality_dist`` controls how each anchor's column count ``k`` is drawn over
    ``[min_card, max_card]``:

      - ``'uniform'`` (default; pre-iter15 behaviour): each ``k`` is drawn uniformly in
        ``[min_card, max_card]``. Matches the legacy sampler bit-for-bit with identical ``rng`` state.
        Default after the iter15 honest-negative bench (see below): on the iter14 width=6000 regime
        (two_stage prefilter -> 400-col cohort) the Zipf prior consistently REGRESSED Spearman across
        ``alpha`` in {0.25, 0.5, 1.0}, monotonically with alpha (alpha=1.0: -0.183; alpha=0.5: -0.023;
        alpha=0.25: -0.013). Hypothesis was that small-k anchors give honest models a wider loss range
        to rank, but at the iter14 regime the post-prefilter cohort already concentrates informatives;
        small-k samples land in the "all-noise or all-signal" extremes where the proxy and honest
        agree TRIVIALLY (no nuance for Spearman to rank), while large-k samples land in the
        interesting informative-mix-vs-noise-mix middle where the proxy is actually being asked to
        rank. Recall@k DID improve under Zipf (1.0 vs 0.833) and recovery was preserved (10/12 across
        all alphas) -- so the prior may pay in other regimes (e.g. callers with low-redundancy data or
        no prefilter). Kept as an opt-in knob for that use case.
      - ``'zipf'`` (opt-in; iter15): ``P(k) ∝ k^(-zipf_alpha)``. Small-k anchors are FAR more common
        (k=1..~10 dominate). ``zipf_alpha=1.0`` is the canonical 1/k Zipf; ``alpha=0`` degenerates to
        uniform-k. Lever is exposed but defaulted OFF after the iter15 honest-negative finding above.

    The column-content sampler is independent of ``cardinality_dist``:

      Default column draw (``weights is None``): ``k`` columns chosen uniformly at random without
      replacement.

      Weighted mode (``weights`` supplied, length ``n_features``): ``k`` columns split between a
      quality-weighted core (``1 - uniform_tail_frac`` of ``k``) drawn by softmax(weights) without
      replacement, and a uniform tail (``uniform_tail_frac`` of ``k``) drawn uniformly from the
      remaining columns. The uniform tail keeps coverage of tail-of-distribution cases the F-score
      under-represents (e.g. pure-interaction informatives with weak marginals). ``uniform_tail_frac``
      = 0 -> pure-weighted, 1.0 -> uniform column draw (cardinality prior still applies).

    Weighted columns whose F-score is -inf (constants) or non-finite get the noise-floor probability
    via ``_softmax_weights`` so they're never the high-prior tier but stay technically reachable
    through the uniform tail."""
    max_card = n_features if max_card is None else min(max_card, n_features)
    use_weights = weights is not None
    if use_weights:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != n_features:
            # Defensive: misaligned weights silently degrade to uniform rather than crash the guard.
            use_weights = False
    if use_weights:
        probs_all = _softmax_weights(weights)
    # Cardinality prior: pre-build the Zipf probs once (cheap, length max_card-min_card+1) so the per-
    # anchor draw is a single ``rng.choice`` rather than rebuilding weights inside the loop. Uniform
    # mode keeps the legacy ``rng.integers`` call so the bit-for-bit guarantee versus pre-iter15
    # behaviour is preserved (same RNG state -> same anchor list).
    card_mode = str(cardinality_dist).lower()
    if card_mode not in ("zipf", "uniform"):
        raise ValueError(
            f"_sample_anchor_subsets: cardinality_dist must be 'zipf' or 'uniform', got {cardinality_dist!r}")
    if card_mode == "zipf":
        card_values = np.arange(int(min_card), int(max_card) + 1, dtype=np.int64)
        card_probs = _zipf_card_probs(min_card, max_card, zipf_alpha)
    anchors = set()
    guard = 0
    max_guard = n_anchors * 50
    while len(anchors) < n_anchors and guard < max_guard:
        guard += 1
        if card_mode == "zipf":
            k = int(rng.choice(card_values, p=card_probs))
        else:
            k = int(rng.integers(min_card, max_card + 1))
        if use_weights and k >= 2 and 0.0 < uniform_tail_frac < 1.0:
            n_uniform = max(1, int(round(uniform_tail_frac * k)))
            n_uniform = min(n_uniform, k - 1)  # ensure at least one weighted pick
            n_weighted = k - n_uniform
            weighted_pick = _weighted_choice_no_replace(rng, n_features, n_weighted, probs_all)
            # Uniform-tail draws from the COMPLEMENT of the weighted picks so the two passes don't
            # collide (no replacement across the full anchor) and the cardinality is exactly k.
            mask = np.ones(n_features, dtype=bool)
            mask[weighted_pick] = False
            remaining = np.nonzero(mask)[0]
            if remaining.size < n_uniform:
                # Pathological: weighted picks already covered all columns. Just take whatever remains.
                tail_pick = remaining
                combined = np.concatenate([weighted_pick, tail_pick])
            else:
                tail_pick = rng.choice(remaining, size=n_uniform, replace=False)
                combined = np.concatenate([weighted_pick, tail_pick])
            cols = tuple(sorted(int(c) for c in combined))
        elif use_weights:
            # k == 1 (or uniform_tail_frac at the boundary): single pick by weight (or uniform tail).
            if k == 1 and uniform_tail_frac < 1.0:
                pick = _weighted_choice_no_replace(rng, n_features, 1, probs_all)
            else:
                pick = rng.choice(n_features, size=k, replace=False)
            cols = tuple(sorted(int(c) for c in pick))
        else:
            cols = tuple(sorted(rng.choice(n_features, size=k, replace=False).tolist()))
        anchors.add(cols)
    return [list(a) for a in anchors]
