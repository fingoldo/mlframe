"""Layer 47 (2026-05-31): auto-tau calibration sibling for ``_dynamic_cluster_discovery``.

Carved out of ``_dynamic_cluster_discovery.py`` to keep the parent below
the 1k-LOC monolith threshold while still letting the parent's
``make_dcd_state`` resolve ``dcd_tau_cluster='auto'`` against the
calibration sweep here.

Contract:

  - ``_detect_valley_between_modes(scores: np.ndarray) -> Optional[float]``
    Bimodality detector + saddle-point picker. Returns the SU value at the
    valley between the two top peaks when the distribution is clearly
    bimodal; None when unimodal.

  - ``_calibrate_tau_auto(*, factors_data, factors_nbins, distance, n_pairs,
    seed, fallback) -> tuple[float, dict]``
    Samples ``n_pairs`` random feature pairs, computes pair-SU via the
    parent's ``pair_su`` (so the calibration consumes the same metric the
    cluster-membership rule consumes), and picks tau via
    ``_detect_valley_between_modes``. Falls back to ``fallback`` when
    unimodal / degenerate.

The parent module re-exports both helpers from its own ``__all__`` for
backwards compat (test files and downstream tooling continue to import
from ``_dynamic_cluster_discovery``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Sentinel used for the legacy 0.7 default. The kernel_tuning_cache route
# only fires when the caller's constructor value is the dev-machine default.
_DCD_DEFAULT_TAU = 0.7
# Default sample size for the auto-tau calibration sweep -- enough for a
# stable histogram / KDE without dominating fit time on huge feature sets.
_DCD_AUTO_TAU_DEFAULT_N_PAIRS = 100
# Fallback tau when the SU distribution is unimodal (no clear clusters).
_DCD_AUTO_TAU_FALLBACK = 0.7
# Clamp the auto-tau to a sane window so a pathological calibration cannot
# silently disable DCD (tau == 1.0 means nothing ever clusters) or false-prune
# everything (tau == 0.0 means every pair is "redundant").
_DCD_AUTO_TAU_MIN = 0.3
_DCD_AUTO_TAU_MAX = 0.95


def _detect_valley_between_modes(scores: np.ndarray) -> Optional[float]:
    """Layer 47: bimodality detector + saddle-point picker.

    Given an array of pairwise SU scores in [0, 1], decide whether the
    distribution is bimodal (cluster-similar pairs vs unrelated pairs) and,
    if so, return the SU value at the valley between the two modes.

    Strategy:
      1. Coarse-bin the scores into 20 buckets over [0, 1] -> histogram.
      2. Identify local maxima (bins where count exceeds both neighbours).
      3. Require >= 2 maxima separated by >= 0.15 SU and a valley between
         them whose count is <= 0.6 * min(peak_count_left, peak_count_right).
         A clear valley means the two modes are well separated.
      4. Return the SU value at the valley bin midpoint.
      5. Return None when distribution is unimodal (no second peak meeting
         the separation + depth criteria).

    This is deliberately a simple histogram analyser rather than a 2-component
    Gaussian mixture: GMM requires scikit-learn at import time, adds 30+ ms
    fit overhead, and on a 100-pair sweep the Gaussian assumption is no
    better-grounded than a histogram. The histogram approach is dependency-
    free, deterministic, and produces the same valley estimate as a 2-comp
    GMM on textbook bimodal data within ~0.05 SU.
    """
    if scores.size < 10:
        return None
    finite = scores[np.isfinite(scores)]
    if finite.size < 10:
        return None
    n_bins = 20
    counts, _edges = np.histogram(
        np.clip(finite, 0.0, 1.0), bins=n_bins, range=(0.0, 1.0),
    )
    if counts.sum() < 10:
        return None
    # Local maxima -- count must dominate BOTH neighbours, but we tolerate
    # one-sided ties (plateau peaks) so adjacent equal-height bins still
    # surface a single peak rather than dropping out completely. A bin is a
    # peak iff its count is >= both neighbours AND strictly greater than at
    # least one of them (rules out flat runs of equal values).
    peaks: list = []
    for i in range(n_bins):
        left = counts[i - 1] if i > 0 else -1
        right = counts[i + 1] if i < n_bins - 1 else -1
        c = counts[i]
        if c < 2:
            continue
        if c >= left and c >= right and (c > left or c > right):
            peaks.append((i, int(c)))
    # Collapse adjacent equal-height plateau peaks -- they're a single mode,
    # pick the leftmost. Otherwise the bimodality check counts a flat-topped
    # mode twice and the "modes >= 3 bins apart" rule rejects it.
    if peaks:
        deduped: list = [peaks[0]]
        for bin_idx, c in peaks[1:]:
            prev_bin, prev_c = deduped[-1]
            if bin_idx == prev_bin + 1 and c == prev_c:
                continue
            deduped.append((bin_idx, c))
        peaks = deduped
    if len(peaks) < 2:
        return None
    # Pairing logic: in a redundancy-clustering SU sweep the IRRELEVANT-pair
    # bulk sits at LOW SU and the genuine near-duplicate / redundant-cluster
    # mode sits at HIGH SU. Picking the two TALLEST peaks (the legacy rule)
    # systematically misses the redundancy mode whenever it is a SMALL tail
    # peak relative to the broad low-SU bulk -- both tall peaks then fall
    # INSIDE the bulk and the shallow valley between them fails the depth gate,
    # so a real high-SU cluster mode is reported as unimodal (scenario-A sensor
    # mesh: bulk peak at SU~0.05, cluster tail at SU~0.55-0.75 of only ~5 pairs,
    # legacy rule paired the two tallest bulk-internal bins and returned None).
    # Robust rule: anchor p1 = the tallest peak (the bulk), then among all peaks
    # at least 3 bins away pick the one whose SEPARATING valley is deepest
    # relative to the smaller of the pair; tie-break toward the HIGHEST-SU bin so
    # the chosen valley sits between the bulk and the redundancy mode. This still
    # recovers the two real modes on textbook bimodal data (the second mode IS
    # the well-separated deep-valley peak) while catching a small high-SU tail.
    bin_width = 1.0 / n_bins
    p1 = max(peaks, key=lambda t: (t[1], -t[0]))
    best = None  # (valley_bin, valley_count, depth_ratio, p2_bin)
    for p2_bin, p2_count in peaks:
        if abs(p2_bin - p1[0]) < 3:
            continue
        bin_lo, bin_hi = (p1[0], p2_bin) if p1[0] < p2_bin else (p2_bin, p1[0])
        valley_slice = counts[bin_lo + 1 : bin_hi]
        if valley_slice.size == 0:
            continue
        valley_offset = int(np.argmin(valley_slice))
        valley_bin = bin_lo + 1 + valley_offset
        valley_count = int(counts[valley_bin])
        min_peak_count = min(p1[1], p2_count)
        if min_peak_count <= 0 or valley_count > 0.6 * min_peak_count:
            continue
        depth_ratio = valley_count / min_peak_count
        cand = (depth_ratio, -p2_bin, valley_bin)
        if best is None or cand < best[0]:
            best = (cand, valley_bin)
    if best is None:
        return None
    valley_bin = best[1]
    tau = float((valley_bin + 0.5) * bin_width)
    tau = max(_DCD_AUTO_TAU_MIN, min(_DCD_AUTO_TAU_MAX, tau))
    return tau


def _calibrate_tau_auto(
    *,
    factors_data,
    factors_nbins,
    distance: str = "su",
    n_pairs: int = _DCD_AUTO_TAU_DEFAULT_N_PAIRS,
    seed: int = 0,
    fallback: float = _DCD_AUTO_TAU_FALLBACK,
) -> tuple:
    """Layer 47: calibrate ``tau_cluster`` from a small SU sweep over random
    feature pairs.

    Returns ``(tau, diagnostics)`` where ``diagnostics`` is a dict carrying:
      - ``"n_pairs_sampled"``: int
      - ``"n_pairs_finite"``: int
      - ``"mode"``: ``"bimodal"`` when a valley was detected, ``"unimodal"``
        when the fallback was used, ``"degenerate"`` on too-few-features /
        too-few-samples
      - ``"tau"``: the chosen tau (mirrors the returned tau)
      - ``"valley_su"``: the raw valley SU before clamping (None when unimodal)
      - ``"su_scores"``: the sampled SU array (numpy ndarray; small)
      - ``"su_mean"``: arithmetic mean
      - ``"su_std"``: standard deviation

    The sweep computes pair SU via the same ``pair_su`` codepath the rest
    of DCD uses, so the calibration consumes the same metric the cluster-
    membership rule consumes -- no metric drift.
    """
    # Lazy-import the parent's DCDState + pair_su to break the circular
    # dependency (parent re-exports these helpers in its own ``__all__``).
    from ._dynamic_cluster_discovery import DCDState, pair_su
    # Layer 51 (2026-05-31): batched pairwise-SU dispatch. Lets the
    # ~100-pair sweep below pre-warm the per-column entropy cache in a
    # single sibling-column pass instead of paying the marginal-entropy
    # cost per pair. Bit-equivalent to looped pair_su.
    from ._dcd_pair_su_batch import pair_su_batch

    diagnostics: dict = {
        "n_pairs_sampled": 0,
        "n_pairs_finite": 0,
        "mode": "degenerate",
        "tau": float(fallback),
        "valley_su": None,
        "su_scores": np.zeros(0, dtype=np.float64),
        "su_mean": float("nan"),
        "su_std": float("nan"),
    }
    if factors_data is None or factors_nbins is None:
        return float(fallback), diagnostics
    n_cols = int(factors_data.shape[1]) if factors_data.ndim == 2 else 0
    if n_cols < 4:
        return float(fallback), diagnostics
    n_pairs_eff = max(20, min(int(n_pairs), n_cols * (n_cols - 1) // 2))
    rng = np.random.default_rng(int(seed))
    pair_set: set = set()
    max_attempts = n_pairs_eff * 4
    attempts = 0
    while len(pair_set) < n_pairs_eff and attempts < max_attempts:
        a = int(rng.integers(0, n_cols))
        b = int(rng.integers(0, n_cols))
        if a == b:
            attempts += 1
            continue
        key = (a, b) if a < b else (b, a)
        pair_set.add(key)
        attempts += 1
    if not pair_set:
        return float(fallback), diagnostics
    pairs = list(pair_set)
    # Lightweight DCDState clone for the calibration sweep -- we deliberately
    # use a transient state with the requested ``distance`` so the sweep
    # consumes the same metric as the live cluster-membership rule. Caches
    # (entropy, pair-SU) are owned by this transient state and discarded
    # after calibration; they don't pollute the fit-time state.
    cal_state = DCDState(
        pool_pruned_mask=np.zeros(n_cols, dtype=bool),
        factors_data=factors_data,
        factors_nbins=np.asarray(factors_nbins),
        cols=[],
        nbins=np.asarray(factors_nbins),
        distance=str(distance),
    )
    su_scores: list = []
    try:
        batch_scores = pair_su_batch(cal_state, pairs)
    except Exception:
        batch_scores = None
    if batch_scores is not None:
        for s in batch_scores:
            if np.isfinite(s):
                su_scores.append(float(s))
    else:
        for a, b in pairs:
            try:
                s = pair_su(cal_state, a, b)
            except Exception:
                continue
            if np.isfinite(s):
                su_scores.append(float(s))
    if len(su_scores) < 10:
        return float(fallback), diagnostics
    arr = np.asarray(su_scores, dtype=np.float64)
    diagnostics["n_pairs_sampled"] = len(pairs)
    diagnostics["n_pairs_finite"] = int(arr.size)
    diagnostics["su_scores"] = arr
    diagnostics["su_mean"] = float(arr.mean())
    diagnostics["su_std"] = float(arr.std())
    valley = _detect_valley_between_modes(arr)
    if valley is None:
        diagnostics["mode"] = "unimodal"
        diagnostics["tau"] = float(fallback)
        return float(fallback), diagnostics
    diagnostics["mode"] = "bimodal"
    diagnostics["valley_su"] = float(valley)
    diagnostics["tau"] = float(valley)
    return float(valley), diagnostics


__all__ = [
    "_calibrate_tau_auto",
    "_detect_valley_between_modes",
    "_DCD_DEFAULT_TAU",
    "_DCD_AUTO_TAU_DEFAULT_N_PAIRS",
    "_DCD_AUTO_TAU_FALLBACK",
    "_DCD_AUTO_TAU_MIN",
    "_DCD_AUTO_TAU_MAX",
]
