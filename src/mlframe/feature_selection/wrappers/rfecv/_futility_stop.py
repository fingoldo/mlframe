"""Futility stop of the RFECV outer loop: quit once further shrinking is very unlikely to move the ``one_se_max`` pick off the full feature set.

Why stopping is selection-neutral.  ``one_se_max`` keeps the LARGEST evaluated N whose CV mean is >= ``m* - h*`` (best mean minus the band half-width of the
best-scoring N).  Iteration 0 evaluates the full set ``P``, the largest N there is, so while ``m_0 >= m* - h*`` holds the pick IS the full set, and a later
evaluation can change it only by becoming the new best with ``m_j - h_j > m_0`` -- i.e. only a subset that beats the full set by more than its own band.
Everything else an extra iteration can do (add in-band sizes, lower means) leaves the pick at ``P``.  The rule therefore stops only while the pick is ``P``
(a pick already below ``P`` has unevaluated in-band sizes above it that a stop could miss, so the rule never fires then) and only when evidence says no
remaining subset is likely to clear that bar.

Evidence.  Folds are shared across iterations, so the per-fold differences ``d_j = a_j - a_0`` remove the fold-difficulty component and their standard
error ``s_j = std(d_j) / sqrt(k)`` is tighter than the band the pick is judged against whenever the data are large enough for pairing to matter; on small
noisy frames it is not, the upper bound never clears the bar and the rule simply stays silent
(measured: ~1% stop rate on 13 small datasets, ~40% on many-row ones).  A subset ``j`` is "futile" when its one-sided upper confidence
bound on the true gain, ``mean(d_j) + t_{1-alpha, k-1} * s_j``, stays below the band half-width ``h_j`` it would have to clear (intersection-union logic: the
claim "no evaluated subset is a winner" needs every bound below the bar, so no multiplicity correction is required).  Because unevaluated sizes are
extrapolated from evaluated ones, two more guards apply: at least ``min_iters`` subset sizes must be evaluated, and the paired gain must not be trending up
(no one-sided significant positive OLS slope of the gains over the last ``patience`` evaluations, and none of a gain rising as N falls across
the smallest evaluated sizes).  ``patience`` scales with the
iterations still available, so a long remaining horizon demands a longer flat stretch before quitting (a dip that precedes a rise is the main failure mode).

``anchor='pick'`` (opt-in) generalises the baseline from the full set to whichever size is the current ``one_se_max`` pick; unevaluated in-band sizes above that pick
then remain possible, so it is not selection-neutral by construction
(measured same-N 94% small / 87.5% many-row at alpha 0.05, OOS unchanged on aggregate).

Scope: only ``one_se_max`` (``auto`` resolves to it) with ``feature_cost == 0`` and no ``max_nfeatures`` cap; for ``argmax`` any improvement matters and for
``one_se_min`` the point of shrinking is finding a smaller in-band subset, so the rule is never armed there.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Sequence

import numpy as np

from ._one_se_band import band_half_width, split_rule_band

FUTILITY_RULES = ("auto", "one_se_max", "one_se_max_foldstd")
_MIN_TREND_WINDOW = 4
_TREND_P = 0.10


@dataclass(frozen=True)
class FutilityVerdict:
    """Outcome of one futility evaluation; ``evidence`` fields are filled whenever the baseline and at least one subset could be compared."""

    stop: bool
    why: str
    n_evaluated: int = 0
    baseline_mean: float = float("nan")
    best_mean: float = float("nan")
    best_n: int = 0
    paired_gain: float = float("nan")
    paired_se: float = float("nan")
    upper_bound: float = float("nan")
    bar: float = float("nan")

    def describe(self) -> str:
        """One-line evidence string for ``stop_reason`` and the end-of-fit summary."""
        return (
            f"futility (no subset beat the anchor's CV mean {self.baseline_mean:.5g} by its band: {self.n_evaluated} smaller sizes tried, "
            f"best {self.best_mean:.5g} at {self.best_n} features, paired gain {self.paired_gain:+.3g} +/- {self.paired_se:.3g} SE, "
            f"upper bound {self.upper_bound:+.3g} < bar {self.bar:.3g}; "
            f"pass futility_stop=False to keep searching)"
        )


@lru_cache(maxsize=256)
def _t_crit(df: int, alpha: float) -> float:
    """One-sided Student-t quantile ``t_{1-alpha, df}`` (normal quantile once df is large)."""
    from scipy.stats import t as _t

    return float(_t.ppf(1.0 - alpha, max(df, 1)))


def futility_armed(self: Any) -> bool:
    """Whether the stop may run for this configuration at all (rule/feature_cost/max_nfeatures/special indices gates)."""
    if not getattr(self, "futility_stop", False):
        return False
    rule = getattr(self, "n_features_selection_rule", "auto")
    if rule not in FUTILITY_RULES:
        return False
    if getattr(self, "feature_cost", 0.0):
        return False
    if getattr(self, "max_nfeatures", None) is not None:
        return False
    return getattr(self, "special_feature_indices", None) is None


def winners_from_trace(trace: Sequence[tuple], mean_w: float = 1.0, std_w: float = 0.0) -> tuple:
    """Collapse per-iteration ``(N, fold_scores)`` into the curve RFECV stores: per N the evaluation with the best ``mean*w - std*w'``, plus the order.

    Returns ``(curve, order)``: ``curve[N] -> float array`` and ``order`` the N of each first-seen evaluation in iteration order.
    """
    curve: dict = {}
    best_final: dict = {}
    order: list = []
    for n, scores in trace:
        arr = np.asarray(scores, dtype=float)
        fin = arr[np.isfinite(arr)]
        if fin.size == 0:
            continue
        mean, std = float(fin.mean()), float(fin.std())
        final = mean * mean_w - std * std_w
        if n not in curve:
            order.append(n)
        if n not in best_final or final > best_final[n]:
            best_final[n], curve[n] = final, arr
    return curve, order


def _band(arr: np.ndarray, band: str) -> float:
    """Band half-width of one N's fold scores (``std / sqrt(k)`` or raw fold std for the legacy band)."""
    fin = arr[np.isfinite(arr)]
    if fin.size == 0:
        return float("nan")
    return float(band_half_width(np.array([np.std(fin)]), np.array([fin.size]), band)[0])


def _trending_up(recent: Sequence[float], x: Optional[Sequence[float]] = None, sign: float = 1.0) -> bool:
    """One-sided significant OLS slope of ``recent`` against ``x`` (default: evaluation order) in the direction ``sign`` (+1 rising, -1 falling)."""
    from scipy.stats import linregress

    xs = np.arange(len(recent), dtype=float) if x is None else np.asarray(x, dtype=float)
    if len(set(recent)) < 2 or len(set(xs.tolist())) < 2:
        return False
    fit = linregress(xs, np.asarray(recent, dtype=float))
    return bool(sign * fit.slope > 0 and fit.pvalue / 2.0 < _TREND_P)


def patience_for(remaining: int, patience_frac: float, floor: int = 2) -> int:
    """Flat-stretch length demanded before quitting: a fraction of the iterations still available, never below ``floor``."""
    return max(floor, int(math.ceil(patience_frac * max(remaining, 0))))


def futility_verdict(
    trace: Sequence[tuple],
    *,
    min_iters: int = 5,
    alpha: float = 0.05,
    patience_frac: float = 0.1,
    remaining: int = 0,
    full_n: Optional[int] = None,
    anchor: str = "full",
    rule: str = "one_se_max",
    mean_w: float = 1.0,
    std_w: float = 0.0,
) -> FutilityVerdict:
    """Decide whether the search may stop now given the evaluation ``trace`` (iteration 0 must be the full set).

    ``full_n`` (the search universe size) guards against a resumed run whose trace lacks iteration 0.
    ``remaining`` is the number of further iterations the run could still spend (sizes left, ``max_refits`` left, time budget left); it only scales
    the patience.  Every unverifiable situation (no baseline, a single fold, NaN folds, mismatched fold counts) returns ``stop=False``.
    """
    curve, order = winners_from_trace(trace, mean_w, std_w)
    if not trace or trace[0][0] not in curve:
        return FutilityVerdict(False, "no full-set baseline")
    if full_n is not None and int(trace[0][0]) != int(full_n):
        return FutilityVerdict(False, "trace does not start at the full set (resumed run)")
    full_n = int(trace[0][0])
    if any(n > full_n for n in curve):
        return FutilityVerdict(False, "baseline is not the largest evaluated size")
    n_eval = len([n for n in order if n != full_n and n > 0])
    if n_eval < max(1, min_iters):
        return FutilityVerdict(False, "too few sizes evaluated", n_evaluated=n_eval)
    _, band = split_rule_band("one_se_max_foldstd" if rule == "one_se_max_foldstd" else "one_se_max")
    means = {n: float(curve[n][np.isfinite(curve[n])].mean()) for n in curve}
    best_n = max(means, key=lambda n: (means[n], n))
    in_band_max = max(n for n in means if means[n] >= means[best_n] - _band(curve[best_n], band))
    if anchor == "pick":
        anchor_n = in_band_max
    elif anchor == "full":
        anchor_n = full_n
        if in_band_max != full_n:
            return FutilityVerdict(False, "a smaller subset already moved the pick off the full set", n_evaluated=n_eval)
    else:
        raise ValueError(f"anchor must be 'full' or 'pick'; got {anchor!r}")
    base = curve[anchor_n]
    k = int(base.size)
    if k < 2 or not np.isfinite(base).all():
        return FutilityVerdict(False, "baseline has <2 finite folds")
    subs = [n for n in order if n != anchor_n and n > 0]
    m0 = means[anchor_n]

    gains, ses, ubs, bars = [], [], [], []
    tcrit = _t_crit(k - 1, alpha)
    for n in subs:
        a = curve[n]
        if a.size != k or not np.isfinite(a).all():
            return FutilityVerdict(False, "fold counts differ or NaN folds; paired test unavailable", n_evaluated=n_eval)
        d = a - base
        s = float(np.std(d, ddof=1) / math.sqrt(k))
        g = float(np.mean(d))
        gains.append(g)
        ses.append(s)
        ubs.append(g + tcrit * s)
        bars.append(_band(a, band))
    gi = int(np.argmax(gains))
    ev: dict[str, Any] = dict(n_evaluated=n_eval, baseline_mean=m0, best_mean=means[subs[gi]], best_n=int(subs[gi]), paired_gain=gains[gi], paired_se=ses[gi])
    worst = int(np.argmax(np.asarray(ubs) - np.asarray(bars)))
    ev.update(upper_bound=ubs[worst], bar=bars[worst])
    if any(ub >= bar for ub, bar in zip(ubs, bars)):
        return FutilityVerdict(False, "an evaluated subset could still beat the full set", **ev)

    window = max(patience_for(remaining, patience_frac), _MIN_TREND_WINDOW)
    if n_eval <= window:
        return FutilityVerdict(False, "patience not yet satisfied", **ev)
    if _trending_up(gains[-window:]):
        return FutilityVerdict(False, "paired gain still trending up", **ev)
    # Dip-then-rise guard: among the smallest evaluated sizes the gain must not be climbing towards the unevaluated territory below them.
    low = sorted(range(len(subs)), key=lambda i: subs[i])[:window]
    if len(low) >= _MIN_TREND_WINDOW and _trending_up([gains[i] for i in low], [math.log(subs[i]) for i in low], sign=-1.0):
        return FutilityVerdict(False, "paired gain climbs towards smaller unevaluated sizes", **ev)
    return FutilityVerdict(True, "futile", **ev)


def remaining_iterations(self: Any, state: Any, n_total: int, max_refits: Optional[int], max_runtime_mins: Optional[float], elapsed_s: float) -> int:
    """Iterations the run could still spend before another stop fires: sizes left, ``max_refits`` left, and the runtime budget over the mean iteration time."""
    left = max(n_total - state.nsteps, 0)
    if max_refits:
        left = min(left, max(max_refits - state.nsteps, 0))
    if max_runtime_mins and state.iter_durations:
        mean_s = float(np.mean(state.iter_durations))
        if mean_s > 0:
            left = min(left, max(int((max_runtime_mins * 60 - elapsed_s) / mean_s), 0))
    return int(left)
