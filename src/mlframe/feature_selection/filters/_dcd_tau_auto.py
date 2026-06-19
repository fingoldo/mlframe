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
# A separated high-SU population only counts as a genuine redundant-pair mode
# when its median SU clears this floor: SU >= 0.5 means a pair shares the
# majority of its information (true near-duplicate), whereas small-sample /
# quantization noise among independent features tops out well below it.
_DCD_REDUNDANCY_FLOOR = 0.5


def _detect_valley_between_modes(scores: np.ndarray) -> Optional[float]:
    """Decide whether the pairwise-SU distribution holds a separated high-SU
    (redundant-pair) population on top of a dominant low-SU (independent-pair)
    bulk, and if so return the SU value separating them.

    A robust separation test rather than a histogram peak/valley analyser. The
    prior peak-pair-with-deep-valley approach required the high-SU mode to form
    a dense histogram peak (count >= 2 in a single 0.05-wide bin) with a near-
    empty valley between two well-formed peaks. That assumption holds for a
    textbook two-Gaussian mixture, but breaks on realistic data: when genuine
    near-duplicate pairs are a small minority of all pairs (e.g. 5 sensor packs
    of 3 -> 15 within-pack pairs out of 190), a uniform-random ~100-pair
    calibration sweep draws only a handful of high-SU pairs. Those scatter
    across several tail bins as singletons, never forming a peak, so the genuine
    bimodal structure was misread as unimodal and tau fell back to 0.7 -- which
    sits ABOVE the within-pack SU (~0.65), collapsing all clustering.

    Strategy (sample-size robust, no dense high peak required):
      1. Locate the bulk with a robust median + MAD (independent pairs cluster
         near SU ~ 0, so the median tracks the bulk even with a high tail).
      2. Build an upper fence ``max(med + 3*1.4826*MAD, med + 0.15)`` above the
         bulk; pairs above it are the candidate high-SU (redundant) population.
      3. Require that population to be a real, separated MINORITY of GENUINELY
         REDUNDANT pairs: at least ``max(3, 3%)`` of pairs above the fence, the
         bulk still >= 50% of pairs, a clear gap (>= 0.10 SU) between the bulk's
         95th percentile and the high population's 5th percentile, AND the high
         population's median SU >= ``_DCD_REDUNDANCY_FLOOR`` (0.5 -- a pair must
         share the majority of its information to count as redundant). The last
         gate rejects small-sample / quantization artefacts: among independent
         noise features a handful of spurious pairs can land at SU ~0.3-0.45 and
         clear the MAD fence, but they sit BELOW the redundancy floor, so they
         no longer manufacture a false bimodal split (the pure-noise fallback).
      4. tau = midpoint of that gap, clamped to ``[_DCD_AUTO_TAU_MIN,
         _DCD_AUTO_TAU_MAX]``.
      5. Return None when no separated high population meets the gate (truly
         unimodal noise) -- the caller then falls back to the default tau.

    Dependency-free + deterministic (no GMM / scikit-learn import), and unlike
    a variance-split (Otsu) it returns None on genuinely unimodal data instead
    of always manufacturing a threshold.
    """
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size < 10:
        return None
    s = np.clip(s, 0.0, 1.0)
    total = int(s.size)
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med))) + 1e-9
    fence = max(med + 3.0 * 1.4826 * mad, med + 0.15)
    high = s[s > fence]
    if high.size < max(3, 0.03 * total):
        return None
    low = s[s <= fence]
    if low.size < 0.5 * total:
        return None
    if float(np.median(high)) < _DCD_REDUNDANCY_FLOOR:
        return None
    low_hi = float(np.quantile(low, 0.95))
    high_lo = float(np.quantile(high, 0.05))
    if high_lo - low_hi < 0.10:
        return None
    tau = 0.5 * (low_hi + high_lo)
    return float(max(_DCD_AUTO_TAU_MIN, min(_DCD_AUTO_TAU_MAX, tau)))


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
