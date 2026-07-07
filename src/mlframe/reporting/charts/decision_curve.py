"""Decision-curve analysis (net-benefit) chart builder.

Decision-curve analysis (Vickers & Elkin 2006) answers a question ROC / PR
cannot: across the clinically/operationally plausible range of threshold
probabilities ``pt``, does acting on the model's predictions yield more NET
BENEFIT than the two trivial policies -- treat everyone (act on all) and treat
no one (act on none)? It is the standard "is this model worth deploying"
diagnostic in medical ML and any cost-sensitive screening setting.

At a threshold probability ``pt`` a row is flagged positive iff its score
``>= pt``. The net benefit of that policy is::

    NB(pt) = TP/n - FP/n * (pt / (1 - pt))

where the odds factor ``pt/(1-pt)`` is the harm-to-benefit exchange rate
implied by choosing ``pt`` as the action threshold. Reference policies:

* treat-all:  NB_all(pt) = prevalence - (1 - prevalence) * pt/(1-pt)
* treat-none: NB_none(pt) = 0 (a flat line on the x-axis)

A useful model's curve sits ABOVE both references over a ``pt`` range; a
useless model's curve hugs treat-none (NB ~ 0) and never beats treat-all.

EFFICIENCY: every TP(pt)/FP(pt) for all <=200 ``pt`` points comes from ONE
descending score sort + a single cumulative-sum pass, then a vectorised
``searchsorted`` maps each ``pt`` onto the sweep -- no per-pt re-scan of the
data, no Python row loop. The spec carries only the <=200-point curves, never
length-n arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from mlframe.reporting.spec import FigureSpec, LinePanelSpec

# DCA is read over a plausible action-threshold WINDOW, not the full [0,1]: above ~0.6 the odds factor explodes and every
# curve collapses to noise, below a few percent treat-all trivially dominates. 200 points resolve the window smoothly.
DEFAULT_N_THRESHOLDS: int = 200
DEFAULT_PT_RANGE: Tuple[float, float] = (0.01, 0.60)


@dataclass(frozen=True)
class DecisionCurveResult:
    """Net-benefit curves + the spec + a deployability verdict.

    ``pt`` is the threshold-probability grid; ``net_benefit`` / ``treat_all`` / ``treat_none`` are the three curves on
    that grid. ``best_pt_advantage`` is the max over ``pt`` of (model NB - max(treat_all NB, 0)); ``useful`` is True iff
    the model strictly beats BOTH references somewhere in the window (the headline DCA verdict the biz_value test pins).
    """

    figure: FigureSpec
    pt: np.ndarray
    net_benefit: np.ndarray
    treat_all: np.ndarray
    treat_none: np.ndarray
    best_pt_advantage: float
    useful: bool


def _finite_binary(y_true, y_score) -> Tuple[np.ndarray, np.ndarray]:
    """Finite (y in {0,1}, score) pairs as int8 / float64; rows with non-finite score or off-{0,1} label dropped."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    mask = np.isfinite(ys) & np.isfinite(yt) & ((yt == 0.0) | (yt == 1.0))
    return yt[mask].astype(np.int8), ys[mask]


def compute_net_benefit(
    y_true,
    y_score,
    *,
    pt_grid: Optional[np.ndarray] = None,
    pt_range: Tuple[float, float] = DEFAULT_PT_RANGE,
    n_thresholds: int = DEFAULT_N_THRESHOLDS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Net-benefit of the model + treat-all + treat-none over a threshold-probability grid.

    Returns ``(pt, nb_model, nb_treat_all, nb_treat_none)``. All TP(pt)/FP(pt) come from one descending score sort plus
    one cumulative-sum pass: ``cum_pos[i]`` / ``cum_neg[i]`` are the TP / FP when flagging the top ``i`` scored rows. A
    row is flagged iff ``score >= pt``, so ``searchsorted`` on the ascending unique scores maps each ``pt`` to the count
    of rows at or above it -- vectorised, O(n log n + P log n), never P full scans.
    """
    yt, ys = _finite_binary(y_true, y_score)
    n = ys.size
    if pt_grid is None:
        pt = np.linspace(pt_range[0], pt_range[1], n_thresholds)
    else:
        pt = np.asarray(pt_grid, dtype=np.float64).ravel()
    pt = np.clip(pt, 1e-6, 1.0 - 1e-6)
    nb_none = np.zeros_like(pt)
    if n == 0:
        return pt, np.zeros_like(pt), np.zeros_like(pt), nb_none

    prevalence = float(yt.mean())
    odds = pt / (1.0 - pt)
    nb_all = prevalence - (1.0 - prevalence) * odds

    # One ascending sort; cum_pos_at_or_above[j] = positives among rows whose score >= sorted_scores[j].
    order = np.argsort(ys, kind="quicksort")
    s_sorted = ys[order]
    pos_sorted = yt[order].astype(np.float64)
    total_pos = float(pos_sorted.sum())
    # Suffix sums: positives / count among rows at index >= j (i.e. score >= s_sorted[j]).
    suffix_pos = total_pos - np.concatenate(([0.0], np.cumsum(pos_sorted)[:-1]))
    suffix_cnt = n - np.arange(n, dtype=np.float64)

    # For each pt: first sorted index whose score >= pt. side='left' so ties (score == pt) are flagged positive.
    # idx == n means pt exceeds every score (nothing flagged): TP = FP = 0 there.
    idx = np.searchsorted(s_sorted, pt, side="left")
    in_range = idx < n
    safe_idx = np.where(in_range, idx, 0)
    flagged = np.where(in_range, suffix_cnt[safe_idx], 0.0)
    tp = np.where(in_range, suffix_pos[safe_idx], 0.0)
    fp = flagged - tp
    nb_model = tp / n - (fp / n) * odds
    return pt, nb_model, nb_all, nb_none


def build_decision_curve_spec(
    y_true,
    y_score,
    *,
    pt_range: Tuple[float, float] = DEFAULT_PT_RANGE,
    n_thresholds: int = DEFAULT_N_THRESHOLDS,
    model_label: str = "model",
    title: str = "Decision-curve analysis (net benefit)",
    figsize: Tuple[float, float] = (8.0, 5.0),
) -> DecisionCurveResult:
    """Decision-curve analysis FigureSpec: model net-benefit vs treat-all / treat-none.

    The model curve is drawn over treat-all (sloped) and treat-none (flat at 0); where the model line sits above both
    references, acting on its predictions is the better policy at that action threshold. The verdict ``useful`` is True
    iff the model strictly beats BOTH references somewhere in ``pt_range`` (margin > 1e-4 to ignore FP noise), and
    ``best_pt_advantage`` quantifies the largest net-benefit gain over the better reference.
    """
    pt, nb_model, nb_all, nb_none = compute_net_benefit(y_true, y_score, pt_range=pt_range, n_thresholds=n_thresholds)

    # A useless model coincides with treat-all at low pt and treat-none at high pt but never rises above the UPPER
    # ENVELOPE of the two references; usefulness is "clears that envelope by a non-noise margin somewhere in pt_range".
    ref_best = np.maximum(nb_all, nb_none)
    advantage = nb_model - ref_best
    best_pt_advantage = float(np.nanmax(advantage)) if advantage.size else float("nan")
    useful = bool(advantage.size and np.nanmax(advantage) > 1e-3)

    # y-axis floor: treat-all dives steeply negative at high pt and would crush the informative region near 0; clip the
    # display window to a bit below the model/treat-none band so the "above the references" gap stays readable.
    finite_nb = nb_model[np.isfinite(nb_model)]
    y_lo = float(min(0.0, finite_nb.min())) if finite_nb.size else 0.0
    finite_all = np.concatenate([arr[np.isfinite(arr)] for arr in (nb_model, nb_all, nb_none)])
    y_hi = float(finite_all.max()) if finite_all.size else 1.0
    ylim = (y_lo, y_hi + 0.05 * max(y_hi - y_lo, 1e-9))

    line = LinePanelSpec(
        x=pt,
        y=(nb_model, nb_all, nb_none),
        series_labels=(
            f"{model_label} (max gain={best_pt_advantage:.3g})",
            "treat all",
            "treat none",
        ),
        line_styles=("-", "--", ":"),
        colors=("#1f77b4", "#d62728", "#7f7f7f"),
        title=title + (" -- USEFUL" if useful else " -- not better than trivial policies"),
        xlabel="Threshold probability p_t",
        ylabel="Net benefit",
        fill_to_baseline=(False, False, False),
        ylim=ylim,
    )
    fig = FigureSpec(suptitle="", panels=((line,),), figsize=figsize)
    return DecisionCurveResult(fig, pt, nb_model, nb_all, nb_none, best_pt_advantage, useful)


__all__ = [
    "DecisionCurveResult",
    "compute_net_benefit",
    "build_decision_curve_spec",
    "DEFAULT_N_THRESHOLDS",
    "DEFAULT_PT_RANGE",
]
