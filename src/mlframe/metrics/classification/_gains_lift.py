"""Cumulative-gains / lift curves, gains table, and the exponential scoring loss (PZAD err_scoreandcurves).

The scoring-and-curves lecture (Дьяконов 2020) covers, beyond ROC/PR/AUC/KS (already in mlframe),
the single-population ranking-evaluation curves used for marketing / targeting decisions:

- **Cumulative gains curve** (aka CAP / Lorenz curve, slides 36-51): sort objects by score descending;
  the x-axis is the fraction of the population targeted (``PR = (TP+FP)/m``) and the y-axis is the
  fraction of positives captured (``gain = TPR``). "If we call the top 20%, what share of responders
  do we reach?"
- **Lift curve** (slides 53-54): ``lift = gain / fraction_targeted`` — how many times better than random
  targeting the top-k is. Tends to 1 as the whole population is targeted.
- **Gains table** (slide 58, the decile "Gains Table"): per-decile cumulative captured-% and lift, the
  table a business reads to pick how deep to call.
- **exploss** (slides 19-22): a proper scoring rule ``y·sqrt((1-a)/a) + (1-y)·sqrt(a/(1-a))`` whose
  per-object minimizer is the true probability (like log-loss / Brier); the boosting exponential loss
  written on the probability scale.

These differ from mlframe's per-group LTR lift (`metrics.ranking`) — here it is one population, binary y.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cumulative_gains_curve", "lift_curve", "gains_table", "exploss"]


def _sorted_cumulative(y_true: np.ndarray, y_score: np.ndarray):
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    ys = np.ascontiguousarray(y_score, dtype=np.float64)
    if yt.shape[0] != ys.shape[0]:
        raise ValueError("y_true and y_score length mismatch.")
    n = yt.shape[0]
    order = np.argsort(-ys, kind="stable")
    pos_sorted = (yt[order] > 0.5).astype(np.float64)
    cum_pos = np.cumsum(pos_sorted)
    total_pos = cum_pos[-1] if n else 0.0
    return n, cum_pos, total_pos


def cumulative_gains_curve(y_true: np.ndarray, y_score: np.ndarray):
    """Cumulative gains (CAP) curve. Returns ``(fraction_targeted, gain)`` arrays, both starting at 0.

    ``fraction_targeted[i]`` = share of the population in the top-i by score; ``gain[i]`` = share of all
    positives captured there. The diagonal is random targeting; a perfect model rises to 1 by the
    positive-prevalence fraction.
    """
    n, cum_pos, total_pos = _sorted_cumulative(y_true, y_score)
    if n == 0 or total_pos == 0:
        return np.array([0.0]), np.array([0.0])
    frac = np.arange(1, n + 1, dtype=np.float64) / n
    gain = cum_pos / total_pos
    return np.concatenate(([0.0], frac)), np.concatenate(([0.0], gain))


def lift_curve(y_true: np.ndarray, y_score: np.ndarray):
    """Lift curve. Returns ``(fraction_targeted, lift)`` where ``lift = gain / fraction_targeted``.

    ``lift[i]`` is how many times more positives the top-i captures than random targeting of the same size.
    The first point (smallest top set) has the largest lift; lift tends to 1 as the whole population is included.
    """
    n, cum_pos, total_pos = _sorted_cumulative(y_true, y_score)
    if n == 0 or total_pos == 0:
        return np.array([]), np.array([])
    frac = np.arange(1, n + 1, dtype=np.float64) / n
    gain = cum_pos / total_pos
    return frac, gain / frac


def gains_table(y_true: np.ndarray, y_score: np.ndarray, *, n_bins: int = 10):
    """Decile (n_bins) gains table. Returns a dict of per-bin arrays (length ``n_bins``).

    Keys: ``bin`` (1..n_bins), ``cum_fraction`` (cumulative population share), ``captured`` (positives in
    this bin), ``cum_captured_pct`` (cumulative share of all positives), ``lift`` (cumulative lift). The
    classic marketing "Gains Table" for choosing how deep to target.
    """
    if n_bins < 1:
        raise ValueError(f"gains_table: n_bins must be >= 1, got {n_bins}.")
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    ys = np.ascontiguousarray(y_score, dtype=np.float64)
    if yt.shape[0] != ys.shape[0]:
        raise ValueError("gains_table: y_true and y_score length mismatch.")
    n = yt.shape[0]
    if n == 0:
        raise ValueError("gains_table: empty input.")
    order = np.argsort(-ys, kind="stable")
    pos_sorted = (yt[order] > 0.5).astype(np.float64)
    total_pos = pos_sorted.sum()
    # near-equal contiguous bins over the score-sorted population
    edges = np.linspace(0, n, n_bins + 1).astype(np.int64)
    captured = np.empty(n_bins, dtype=np.float64)
    cum_count = np.empty(n_bins, dtype=np.float64)
    for b in range(n_bins):
        captured[b] = pos_sorted[edges[b] : edges[b + 1]].sum()
        cum_count[b] = edges[b + 1]
    cum_captured = np.cumsum(captured)
    cum_fraction = cum_count / n
    cum_captured_pct = cum_captured / total_pos if total_pos > 0 else np.zeros(n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(cum_fraction > 0, cum_captured_pct / cum_fraction, 0.0)
    return {
        "bin": np.arange(1, n_bins + 1),
        "cum_fraction": cum_fraction,
        "captured": captured,
        "cum_captured_pct": cum_captured_pct,
        "lift": lift,
    }


def exploss(y_true: np.ndarray, y_prob: np.ndarray, *, eps: float = 1e-12) -> float:
    """Exponential scoring loss on the probability scale: mean of ``y·sqrt((1-a)/a) + (1-y)·sqrt(a/(1-a))``.

    A proper scoring rule (per-object minimizer is the true probability, like log-loss / Brier); the boosting
    exponential loss re-expressed for probabilistic outputs. ``y_prob`` is clipped to ``[eps, 1-eps]``.
    """
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    a = np.clip(np.ascontiguousarray(y_prob, dtype=np.float64), eps, 1.0 - eps)
    if yt.shape[0] != a.shape[0]:
        raise ValueError("exploss: y_true and y_prob length mismatch.")
    if yt.shape[0] == 0:
        return np.nan
    per = yt * np.sqrt((1.0 - a) / a) + (1.0 - yt) * np.sqrt(a / (1.0 - a))
    return float(per.mean())
