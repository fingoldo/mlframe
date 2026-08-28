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
from typing import Any, Optional, Tuple

import numpy as np

from mlframe.reporting.spec import AnnotationPanelSpec, FigureSpec, LinePanelSpec, PanelSpec

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


def effective_binary_n(y_true: Any, y_score: Any) -> int:
    """Rows ``compute_net_benefit`` actually scores: finite score, finite label, label in {0,1}.

    Exposed because every sample-size-scaled threshold on this chart must be fed THIS count, not
    ``len(y_true)``. Feeding the raw length made a 150-usable-row sample inherit a 200000-row noise bar.
    """
    return int(_finite_binary(y_true, y_score)[0].size)


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


# Floor for the usefulness margin: below this a "gain" is numerically indistinguishable from FP rounding
# no matter how large the sample, so an enormous n must not drive the bar to zero.
_MIN_USEFULNESS_MARGIN = 1e-3
# ~95% one-sided normal quantile. Net benefit is a difference of per-row rates, so its sampling error shrinks
# as 1/sqrt(n); 0.5/sqrt(n) bounds the standard error of a proportion (worst case p=0.5) and z scales it to a
# confidence bar. Not an exact test -- the reference curve is estimated from the same rows -- but it is the
# right ORDER, which a flat constant is not.
_USEFULNESS_Z = 1.96


def _usefulness_margin(n_rows: int) -> float:
    """Net-benefit advantage a model must clear before the chart calls it useful, scaled to sample size.

    A flat 1e-3 bar treated a 2000-row sample and a 2M-row sample identically. At n=2000 the net-benefit
    curve of a PURELY RANDOM score wanders ~0.01 above the reference envelope on noise alone -- ten times the
    old bar -- so random predictions were labelled "USEFUL" on the figure. Verified by simulation across n
    (see ``tests/reporting/test_decision_curve_adaptive_margin.py``): random scores now fail at every size
    tested, while a genuinely informative score still passes.
    """
    if n_rows <= 0:
        return _MIN_USEFULNESS_MARGIN
    return max(_MIN_USEFULNESS_MARGIN, _USEFULNESS_Z * 0.5 / float(np.sqrt(n_rows)))


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
    iff the model strictly beats BOTH references somewhere in ``pt_range`` by more than ``_usefulness_margin(n)``
    (a sample-size-scaled bar, not a flat constant -- see that function), and
    ``best_pt_advantage`` quantifies the largest net-benefit gain over the better reference.
    """
    pt, nb_model, nb_all, nb_none = compute_net_benefit(y_true, y_score, pt_range=pt_range, n_thresholds=n_thresholds)
    n_eff = effective_binary_n(y_true, y_score)
    n_raw = int(np.size(np.asarray(y_true).ravel()))
    n_dropped = max(0, n_raw - n_eff)
    if n_eff == 0:
        panel: PanelSpec = AnnotationPanelSpec(
            text=(
                f"Decision curve unavailable: none of the {n_raw:,} supplied rows carry both a finite score and a "
                "label in {0, 1}. DCA is defined for binary outcomes only -- for multiclass, build one curve per "
                "one-vs-rest binarisation."
            ),
            title=title,
        )
        empty = np.zeros_like(pt)
        return DecisionCurveResult(FigureSpec(suptitle="", panels=((panel,),), figsize=figsize), pt, empty, empty, empty, float("nan"), False)

    # A useless model coincides with treat-all at low pt and treat-none at high pt but never rises above the UPPER
    # ENVELOPE of the two references; usefulness is "clears that envelope by a non-noise margin somewhere in pt_range".
    ref_best = np.maximum(nb_all, nb_none)
    advantage = nb_model - ref_best
    best_pt_advantage = float(np.nanmax(advantage)) if advantage.size else float("nan")
    useful_margin = _usefulness_margin(n_eff)
    useful = bool(advantage.size and np.nanmax(advantage) > useful_margin)

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
    # How-to-read footnote (``FigureSpec.caption``, rendered small + dim under the axes). A decision curve is
    # not self-explanatory: "net benefit" is a unit nobody reads off other charts, and the actionable content
    # is a COMPARISON against two reference policies rather than the curve's own shape. Spelling out the
    # decision rule -- and, when the model loses, saying so in the operator's terms -- is what turns this from
    # a plot into an answer. The verdict sentence is data-dependent, so it reports THIS model, not the generic case.
    _pt_lo, _pt_hi = (float(pt[0]), float(pt[-1])) if pt.size else (0.0, 1.0)
    if useful:
        _best_pt = float(pt[int(np.nanargmax(advantage))]) if advantage.size else float("nan")
        _verdict = (
            f"VERDICT: acting on this model beats both trivial policies, best at p_t={_best_pt:.2f} "
            f"(+{best_pt_advantage:.3g} net benefit -- i.e. {best_pt_advantage * 100:.2f} extra true positives, "
            f"net of false positives, per 100 cases screened)."
        )
    else:
        _verdict = (
            "VERDICT: this model never clears the better of the two trivial policies anywhere in the scanned "
            "threshold range, so at every threshold you would do at least as well by treating everyone (left of "
            "the crossing) or no one (right of it). Ranking quality (ROC/PR) can still be non-trivial -- decision "
            "curves also punish miscalibration, so check the calibration chart before concluding the model is useless."
        )
    how_to_read = (
        f"x = the action threshold p_t you would deploy at (a case is acted on when predicted probability >= p_t); "
        f"choosing p_t encodes your cost ratio: p_t/(1-p_t) is how many false positives you accept per true positive. "
        f"y = net benefit, true positives minus false positives weighted by that ratio, per unit of population -- "
        f"higher is better, and only the VERTICAL GAP to the reference lines is meaningful. Compare against 'treat all' "
        f"(act on everyone) and 'treat none' (act on no one, flat at 0): use the model only at thresholds where its "
        f"line sits ABOVE both. Scanned p_t in [{_pt_lo:.2f}, {_pt_hi:.2f}] on {n_eff:,} usable rows"
        + (f" ({n_dropped:,} of {n_raw:,} dropped: non-finite score or a label outside {{0, 1}})" if n_dropped else "")
        + f". {_verdict}"
    )
    fig = FigureSpec(suptitle="", panels=((line,),), figsize=figsize, caption=how_to_read)
    return DecisionCurveResult(fig, pt, nb_model, nb_all, nb_none, best_pt_advantage, useful)


__all__ = [
    "DecisionCurveResult",
    "compute_net_benefit",
    "effective_binary_n",
    "build_decision_curve_spec",
    "DEFAULT_N_THRESHOLDS",
    "DEFAULT_PT_RANGE",
]
