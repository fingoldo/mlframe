"""The optimism of a selector's own score: what it reported at selection time versus what held up.

Every arm that chooses a subset by maximising something has already used the data twice -- once to pick the
winner, once to report how good the winner is. The reported number is therefore the maximum of a noisy
sample, and the maximum of a noisy sample is biased upward. That bias is what "winner's curse" names, and it
is not a small effect for a wrapper searching hundreds of candidate subsets on a few thousand rows.

This module makes it a column rather than a caveat. For every arm that reports its own optimum, the report
carries `selection_score - honest_holdout`, averaged over seeds. A large positive value means the arm's own
number cannot be quoted; that indicts the arm's REPORTING, not necessarily its selection, and the two are
kept separate because an arm can pick well and describe itself badly.

The comparison is only fair when both sides measure the same quantity. An arm whose internal optimum is an
accuracy while the holdout is an AUC would show a difference that is a unit mismatch, not optimism, so the
metric is named in the output and an arm whose internal metric is unknown is reported as such rather than
differenced against whatever happens to be at hand.

Arms with no reported optimum -- most filters, which never search -- are absent from the table by design.
Absence here means "did not claim a score", not "was not optimistic".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ._matched_k import SELF_CHOSEN_K

logger = logging.getLogger(__name__)

__all__ = ["OptimismRow", "optimism_rows", "optimism_table"]


@dataclass(frozen=True)
class OptimismRow:
    """One arm's own reported optimum beside the honest holdout, and the gap when the two are comparable."""

    arm: str
    scenario: str
    n_seeds: int
    mean_selection_score: float
    mean_holdout: float
    selection_metric: Optional[str]
    comparable: bool
    optimism: Optional[float]


def _holdout_score(record: Dict[str, Any], model: str, metric: str, k_label: str) -> Optional[float]:
    """Return the honest holdout score for one cell at the arm's own chosen cardinality."""
    block = (record.get("scores") or {}).get(k_label)
    if not isinstance(block, dict):
        return None
    scores = (block.get("models") or {}).get(model)
    if not isinstance(scores, dict) or metric not in scores:
        return None
    return float(scores[metric])


def optimism_rows(
    records: Sequence[Dict[str, Any]],
    model: str = "lightgbm",
    metric: str = "roc_auc",
    k_label: str = SELF_CHOSEN_K,
) -> List[OptimismRow]:
    """Return one row per (scenario, arm) that reported an internal optimum.

    The holdout is read at the arm's SELF-chosen cardinality, because that is the subset whose score the arm
    reported. Reading it at a matched K would compare the arm's claim about one subset with the performance
    of a different one.
    """
    paired: Dict[Any, List[Any]] = {}
    for record in records:
        if record.get("status") != "ok" or record.get("selection_score") is None:
            continue
        holdout = _holdout_score(record, model=model, metric=metric, k_label=k_label)
        if holdout is None:
            continue
        key = (str(record["scenario"]), str(record["arm"]), record.get("selection_metric"))
        paired.setdefault(key, []).append((float(record["selection_score"]), holdout))

    out: List[OptimismRow] = []
    for (scenario, arm, selection_metric), pairs in sorted(paired.items(), key=lambda item: (item[0][0], item[0][1])):
        claimed = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        honest = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        # Subtracting across a metric mismatch produces a unit error wearing the costume of optimism: this
        # RFECV scores internally on a probabilistic error while the report reads AUC, and the difference
        # between the two says nothing about how optimistic the arm was.
        comparable = selection_metric is not None and str(selection_metric) == metric
        out.append(
            OptimismRow(
                arm=arm,
                scenario=scenario,
                n_seeds=len(pairs),
                mean_selection_score=float(claimed.mean()),
                mean_holdout=float(honest.mean()),
                selection_metric=None if selection_metric is None else str(selection_metric),
                comparable=comparable,
                optimism=float((claimed - honest).mean()) if comparable else None,
            )
        )
    return out


def optimism_table(records: Sequence[Dict[str, Any]], model: str = "lightgbm", metric: str = "roc_auc") -> List[str]:
    """Render the winner's-curse block, or a single honest line when no arm reported an optimum."""
    rows = optimism_rows(records, model=model, metric=metric)
    lines = [
        "",
        "=" * 100,
        f"WINNER'S CURSE -- an arm's own reported optimum minus its honest holdout ({model}/{metric} at self-chosen K)",
        "=" * 100,
        "  A large positive value indicts the arm's REPORTING, not necessarily its selection.",
        "  Arms absent from this table did not claim a score; absence is not evidence of honesty.",
    ]
    if not rows:
        lines.append("  no arm in this run reported an internal optimum, so there is nothing to compare")
        return lines

    comparable = [row for row in rows if row.comparable]
    mismatched = [row for row in rows if not row.comparable]
    for row in sorted(comparable, key=lambda item: -(item.optimism or 0.0)):
        lines.append(
            f"  {row.arm:<28} [{row.scenario}] claimed={row.mean_selection_score:+.4f}  "
            f"honest={row.mean_holdout:+.4f}  optimism={row.optimism:+.4f}  m={row.n_seeds}"
        )
    if mismatched:
        lines.append("")
        lines.append(f"  NOT COMPARABLE -- these arms score internally on a different metric than {metric!r}, so no difference is taken:")
        for row in sorted(mismatched, key=lambda item: (item.scenario, item.arm)):
            named = row.selection_metric or "unnamed metric"
            lines.append(
                f"  {row.arm:<28} [{row.scenario}] claimed={row.mean_selection_score:+.4f} ({named})  "
                f"honest={row.mean_holdout:+.4f} ({metric})  m={row.n_seeds}"
            )
    return lines
