"""Report the decision threshold the classification metrics were computed at, and the one the data actually supports.

Every crisp metric the suite prints -- accuracy, precision, recall, F1, the confusion matrix -- describes a
decision rule at 0.5. Nobody chose 0.5; it is the default of the function that produced the predictions. On the
production run that motivated this block the base rate was 2.6%, where 0.5 is close to "predict nobody", and the
printed precision/recall pair therefore described a rule no operator would deploy.

Two deliberate limits:

- the threshold is selected on OOF, never on VAL (already spent on early stopping, so its optimum is
  optimistically biased) and never on TEST (selecting there converts the honest estimate into a fitted one).
- the block REPORTS; it does not rewrite the predictions. Changing what "positive" means downstream is the
  caller's decision, and it is made by passing ``decision_costs``, not by a diagnostic silently taking it.

Without an explicit cost ratio the objective falls back to F1, which is reported as informational only: F1
assumes a false positive and a false negative cost the same, which at a 2.6% base rate is almost never true.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Below this many minority-class rows the tuned threshold is dominated by which handful of positives landed in the
# fold, and its bootstrap interval spans most of the score range -- reported, but never presented as a recommendation.
MIN_POSITIVES_FOR_A_RECOMMENDATION = 50


def _counts_at(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, int]:
    """Confusion counts for ``p >= thr``."""
    pred = p >= thr
    pos = y > 0.5
    return {
        "tp": int(np.count_nonzero(pred & pos)),
        "fp": int(np.count_nonzero(pred & ~pos)),
        "tn": int(np.count_nonzero(~pred & ~pos)),
        "fn": int(np.count_nonzero(~pred & pos)),
    }


def _summary_at(y: np.ndarray, p: np.ndarray, thr: float, fp_cost: float, fn_cost: float) -> Dict[str, Any]:
    """Precision / recall / F1 / average cost of one threshold, so the two rules can be read side by side."""
    c = _counts_at(y, p, thr)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n = y.shape[0]
    return {
        "threshold": float(thr),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_cost": (fp * fp_cost + fn * fn_cost) / n if n else float("nan"),
        "predicted_positive": int(tp + fp),
        **c,
    }


def decision_threshold_block(
    model_entry: Any,
    *,
    oof_probs: Optional[np.ndarray],
    oof_target: Optional[np.ndarray],
    decision_costs: Optional[Dict[str, float]] = None,
    n_boot: int = 200,
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """Compare the 0.5 rule against the OOF-tuned one; never applies either.

    ``decision_costs`` is ``{"fp": float, "fn": float}``. Given, the objective is the average cost per row and the
    result is a recommendation; absent, the objective is F1 and the result is explicitly informational.
    """
    if oof_probs is None or oof_target is None:
        # Name the knob. OOF predictions are off by default (``oof_n_splits=0``), so on a stock run this block
        # reports "skipped" every time and reads like a defect rather than a setting -- a production log showed
        # exactly that. Tuning on val or test instead is not an option: val already drove early stopping, and
        # selecting on test converts the one honest estimate into a fitted one.
        return {
            "status": "skipped",
            "reason": "no OOF probabilities / target -- set oof_n_splits>=2 to compute them; the threshold must "
            "not be tuned on val (already spent on early stopping) or test (the only honest estimate)",
        }
    p = np.asarray(oof_probs, dtype=np.float64).ravel()
    y = np.asarray(oof_target, dtype=np.float64).ravel()
    n = min(p.shape[0], y.shape[0])
    p, y = p[:n], y[:n]
    if n < 4:
        return {"status": "skipped", "reason": f"aligned OOF row count {n} < 4"}
    _uniq = np.unique(y[np.isfinite(y)])
    if _uniq.shape[0] != 2:
        return {"status": "skipped", "reason": "decision-threshold tuning is binary-only"}

    fp_cost = float((decision_costs or {}).get("fp", 1.0))
    fn_cost = float((decision_costs or {}).get("fn", 1.0))
    metric = "cost" if decision_costs else "f1"

    from mlframe.metrics.classification import optimal_threshold, optimal_threshold_bootstrap_ci

    thr, _score = optimal_threshold(y, p, metric=metric, fp_cost=fp_cost, fn_cost=fn_cost)
    lo, hi = optimal_threshold_bootstrap_ci(
        y, p, metric=metric, fp_cost=fp_cost, fn_cost=fn_cost, n_boot=n_boot, random_state=rng_seed,
    )
    n_pos = int(np.count_nonzero(y > 0.5))
    return {
        "status": "ok",
        "objective": metric,
        "fitted_on": "oof",
        "n_oof": int(n),
        "n_positives": n_pos,
        "decision_costs": {"fp": fp_cost, "fn": fn_cost} if decision_costs else None,
        # Reported, NOT applied: every metric elsewhere in the report is still the 0.5 rule.
        "applied": False,
        "recommended": bool(decision_costs) and n_pos >= MIN_POSITIVES_FOR_A_RECOMMENDATION,
        "threshold_ci": [lo, hi],
        "default": _summary_at(y, p, 0.5, fp_cost, fn_cost),
        "tuned": _summary_at(y, p, thr, fp_cost, fn_cost),
    }


def format_decision_threshold_line(key: str, block: Dict[str, Any]) -> str:
    """One log line per model: both rules, and what the interval says about trusting the tuned one."""
    if block.get("status") != "ok":
        return f"  [threshold] {key}: skipped ({block.get('reason')})"
    d, t = block["default"], block["tuned"]
    lo, hi = block["threshold_ci"]
    _note = "" if block.get("recommended") else " -- INFORMATIONAL, not a recommendation"
    if not block.get("decision_costs"):
        _note += " (no decision_costs configured; objective fell back to F1, which prices FP and FN equally)"
    elif block["n_positives"] < MIN_POSITIVES_FOR_A_RECOMMENDATION:
        _note += f" (only {block['n_positives']} OOF positives)"
    return (
        f"  [threshold] {key}: @0.5 P={d['precision']:.1%} R={d['recall']:.1%} cost={d['avg_cost']:.4f}"
        f" | @{t['threshold']:.4f} (OOF {block['objective']}, 95% CI [{lo:.4f}, {hi:.4f}])"
        f" P={t['precision']:.1%} R={t['recall']:.1%} cost={t['avg_cost']:.4f}."
        f" Reported only, metrics elsewhere still use 0.5{_note}"
    )


__all__ = ["MIN_POSITIVES_FOR_A_RECOMMENDATION", "decision_threshold_block", "format_decision_threshold_line"]
