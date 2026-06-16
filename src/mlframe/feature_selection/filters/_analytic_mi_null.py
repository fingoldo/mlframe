"""Analytic large-n null for the MI permutation test (2026-06-16).

The mRMR confidence / debiasing step gates features by a PERMUTATION null: shuffle ``y`` many
times, measure how often the shuffled MI ties/beats the observed MI (the p-value), and average the
shuffled MIs (the null-mean bias floor). Profiling at scale (D:/Temp/bench_scaling: 400k fit) showed
this permutation null is the dominant large-n cost -- thousands of O(n) shuffles per FE scan, routed
to a cupy ``argsort`` permutation generator that was 72% of the 400k wall.

At large n the permutation null has an exact asymptotic form, so the shuffles are unnecessary:
  * plug-in MI of two INDEPENDENT discrete variables has the Miller-Madow bias
        E[MI_hat] approx (Bx - 1) * (By - 1) / (2 * N)        [nats]
    where Bx, By are the numbers of OCCUPIED bins -- this IS the permutation null mean.
  * the G-test / likelihood-ratio statistic ``2 * N * MI`` (MI in nats) is asymptotically
        chi-square with df = (Bx - 1) * (By - 1)
    under independence, so the permutation p-value approx ``chi2.sf(2 * N * MI, df)``.

Empirically validated against the actual permutation kernel (mi_direct, npermutations=64) across
n in {5k, 20k, 50k, 200k}: the analytic null mean matches the permutation null mean to 3+ digits
even at n=5000, and the analytic p reproduces the significance decision (signal -> ~0, noise -> high).
See D:/Temp/validate_analytic_null.py for the comparison table.

IMPORTANT validity conditions (the caller MUST gate on these):
  * MI must be RAW (nats), NOT symmetric-uncertainty-normalised -- the 2*N*MI ~ chi2 identity only
    holds for raw MI. Gate on ``not use_su_normalization()``.
  * Large n -- the asymptotic tightens with N. Default floor ``_ANALYTIC_NULL_MIN_N`` (env-tunable),
    above which the analytic path replaces permutations; below it the cheap permutation path runs
    unchanged (small-n behaviour byte-for-byte preserved).
"""
from __future__ import annotations

import os

import numpy as np

try:  # scipy is a hard mlframe dep, but keep the import defensive so an env without it degrades.
    from scipy.stats import chi2 as _chi2
    _HAVE_CHI2 = True
except Exception:  # pragma: no cover
    _HAVE_CHI2 = False


# Minimum n at which the analytic null replaces the permutation null. Validated tight from ~20k;
# defaulted higher (50k) so the analytic path engages only where the permutation cost is material
# AND the asymptotic is very accurate, leaving the well-trodden small/mid-n permutation path
# byte-for-byte unchanged. Env-tunable; a future kernel_tuning_cache sweep can refine per host.
_ANALYTIC_NULL_MIN_N_DEFAULT = 50_000


def analytic_null_enabled() -> bool:
    """Off-switch: ``MLFRAME_MI_ANALYTIC_NULL=0`` forces the legacy permutation path everywhere."""
    return _HAVE_CHI2 and os.environ.get("MLFRAME_MI_ANALYTIC_NULL", "1").strip() not in ("0", "false", "False")


def analytic_null_min_n() -> int:
    raw = os.environ.get("MLFRAME_MI_ANALYTIC_NULL_MIN_N", "").strip()
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return _ANALYTIC_NULL_MIN_N_DEFAULT


def analytic_mi_null(original_mi: float, n_rows: int, n_bins_x: int, n_bins_y: int) -> tuple[float, float]:
    """Return ``(null_mean, p_value)`` for the MI permutation test, computed analytically.

    ``original_mi`` MUST be raw MI in NATS (not SU-normalised). ``n_bins_x`` / ``n_bins_y`` are the
    numbers of OCCUPIED bins of x / y (i.e. ``len(freqs_x)`` / ``len(freqs_y)`` from ``merge_vars``).

    ``null_mean`` is the Miller-Madow plug-in bias ``(Bx-1)(By-1)/(2N)``; ``p_value`` is the G-test
    tail ``chi2.sf(2N*MI, df)``. Degenerate cases (df <= 0, N <= 0, no chi2) return ``(0.0, 1.0)`` --
    an uninformative feature is maximally non-significant, matching the permutation path's no-perm
    default.
    """
    df = (int(n_bins_x) - 1) * (int(n_bins_y) - 1)
    if df <= 0 or n_rows <= 0:
        return 0.0, 1.0
    null_mean = df / (2.0 * float(n_rows))
    if not _HAVE_CHI2 or original_mi <= 0.0:
        # no observed signal -> sits at/below its null -> non-significant.
        return null_mean, 1.0
    g_stat = 2.0 * float(n_rows) * float(original_mi)
    p_value = float(_chi2.sf(g_stat, df))
    # clamp into [0, 1] against any FP underflow/overflow at the extreme tail.
    if p_value < 0.0:
        p_value = 0.0
    elif p_value > 1.0:
        p_value = 1.0
    return null_mean, p_value


__all__ = ["analytic_mi_null", "analytic_null_enabled", "analytic_null_min_n"]
