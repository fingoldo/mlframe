"""Noise-aware absolute MI floor for ``hybrid_orth_mi_fe``'s two-gate selection (carved out of the package ``__init__`` to keep it within the module-size budget)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__package__)


def hybrid_abs_mi_floor(scores: pd.DataFrame, min_abs_mi_frac: float, _h_y: float) -> float:
    """Absolute engineered-MI floor: the larger of the legacy fraction-of-max floor and the MAD-based noise floors that sit below ``H(y)``."""
    raw_baselines = scores["baseline_mi"].to_numpy()
    max_raw_baseline = float(raw_baselines.max()) if raw_baselines.size else 0.0
    legacy_floor = float(min_abs_mi_frac) * max_raw_baseline
    # Layer 27 noise-aware floor: use MEDIAN-based stats (robust to a few
    # real signals dragging the mean up). On all-noise frames every
    # baseline_mi sits in a tight band, median + 3*MAD bounds the band
    # tightly. On signal frames the true signal is an outlier above the
    # noise band and median+3*MAD remains in the noise band, so the
    # legacy ``frac * max(raw_baseline)`` dominates and legitimate signals
    # qualify as before.
    # Bonferroni-aware sigma scale: pure sqrt(2 ln 2p) under-counts the chi-
    # square-like right tail of plug-in MI's noise distribution. Empirically
    # n=1500 binary y with 10 bins produces noise MIs that exceed Gaussian
    # tails by ~1-1.5 sigma worth of probability mass. Anchor at the larger
    # of: (a) Bonferroni for the candidate count, (b) a 5-sigma floor that
    # bounds the chi-square right tail at ~1e-7 per slot. For 40 candidates
    # this gives 5.0; for 1000 it gives ~5.8 (asymptotically Bonferroni-driven).
    n_cands = int(raw_baselines.size)
    sigma_thresh = max(
        5.0,
        float(np.sqrt(2.0 * np.log(max(2.0, 2.0 * n_cands))) + 1.5),
    )
    # This used to hard-gate on
    # raw_baselines.size >= 4 (below that, noise_floor silently defaulted to 0.0 - a full no-op).
    # raw_baselines has one entry per SCORED ENGINEERED COLUMN (~len(degrees) * len(surviving raw
    # cols)), not per raw source column, so with the default 2 degrees the guard was bypassed whenever
    # at most 1 raw numeric column reached the scan (mostly-categorical frames, _dedup_collinear_source_cols
    # collapsing redundant sensors down to 1 survivor, or an explicit short `cols=`) - exactly the
    # degenerate-pool-size case the Layer 27 floor exists to guard, defeating BOTH documented safety
    # gates simultaneously. median/MAD are well-defined at any n >= 1 (n=1: MAD=0, so noise_floor
    # collapses to that single baseline_mi value itself - a strict, sound floor given zero evidence of
    # the noise distribution's spread, not a no-op); always compute it instead of gating on count.
    med = float(np.median(raw_baselines)) if raw_baselines.size else 0.0
    mad = float(np.median(np.abs(raw_baselines - med))) if raw_baselines.size else 0.0
    # 1.4826 * MAD ~= std for a normal distribution.
    noise_floor = med + sigma_thresh * 1.4826 * mad
    # Layer 27 follow-up: also compute a noise floor on the ENGINEERED MI
    # distribution. On all-noise frames the engineered cols inherit the same
    # noise structure; the top engineered_mi can be artifactually 2-4x the
    # median by pure tail sampling. Bound engineered_mi above the engineered
    # median+sigma*MAD too - legitimate signals are statistical outliers in
    # the engineered distribution AS WELL.
    # Matching fix to the raw-baseline floor
    # above - see that comment for the rationale (no size>=4 hard gate; median/MAD are well-defined at
    # any n>=1).
    eng_mis = scores["engineered_mi"].to_numpy()
    med_e = float(np.median(eng_mis)) if eng_mis.size else 0.0
    mad_e = float(np.median(np.abs(eng_mis - med_e))) if eng_mis.size else 0.0
    eng_noise_floor = med_e + sigma_thresh * 1.4826 * mad_e
    # SANITY BOUND on the two MAD-derived floors: ``MI(X; y) <= H(y)`` is a hard information-theoretic
    # ceiling, so a floor at or above H(y) is unsatisfiable BY CONSTRUCTION -- it does not "reject noise", it
    # rejects everything, silently disabling the whole family with no error and no log.
    # That is exactly what happens on a SMALL candidate set: MAD estimates the spread of a NOISE band, but
    # with only a couple of candidates (e.g. cols=["x1"], degrees=(2,3) -> 2 candidates, one signal + one
    # dud) the MAD measures the signal-to-noise SEPARATION instead, and multiplying it by ~5*1.4826 pushes
    # the floor far past H(y) (measured: floor 2.21 nats vs H(y)=0.62 on the He_2 quadratic fixture).
    # A floor that is genuinely discriminating always sits below H(y); one that does not is degenerate, so
    # drop it rather than let it veto every column. All-noise protection is untouched: noise MIs sit far
    # below H(y), so a meaningful floor still rejects them.
    _mad_floors = [f for f in (noise_floor, eng_noise_floor) if not (_h_y > 0.0 and f >= _h_y)]
    if len(_mad_floors) < 2:
        logger.debug(
            "hybrid_orth_mi_fe: dropped a MAD noise floor at/above H(y)=%.4f (raw=%.4f, eng=%.4f, n_cands=%d) "
            "-- unsatisfiable by construction, would have vetoed every candidate.",
            _h_y, noise_floor, eng_noise_floor, n_cands,
        )
    abs_floor = max([legacy_floor, *_mad_floors])
    return abs_floor
