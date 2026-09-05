"""Cost against quality: which arms are on the frontier, and which are paying for nothing.

A leaderboard ranked on quality alone answers a question nobody has. Every arm here costs model fits, and
the wrapper arms cost two orders of magnitude more than the filters -- so "best" without a cost axis
silently recommends spending a thousand fits to buy a gain a reader might not accept at ten.

The frontier is computed on the two axes the pre-registration names: `n_model_fits`, which is deterministic
and immune to machine load, and the paired advantage over `all-features`, which is the primary outcome. An
arm is DOMINATED when another arm is at least as good and no more expensive; the dominating arm is named,
because "dominated" without a witness is an assertion rather than a finding.

Two rules keep the table honest:

* an arm whose cost was never measured is listed as unpriced, never as free. Cheapness is the most valuable
  thing an arm can claim here, and a missing measurement must not hand it that claim.
* the null hypothesis sits on the frontier by definition at zero fits and zero advantage. Any arm that fails
  to beat it is dominated BY IT, which is the modal outcome on the real beds and should read as one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._leaderboard import NULL_ARM, extract_long_rows
from ._paired_stats import average_over_cv_seed, paired_differences, paired_t_test

logger = logging.getLogger(__name__)

__all__ = ["ParetoPoint", "pareto_points", "pareto_table", "frontier_arms"]


@dataclass(frozen=True)
class ParetoPoint:
    """One arm's (cost, advantage) position, and what dominates it if anything does."""

    arm: str
    advantage: float
    cost: Optional[float]
    n_seeds: int
    on_frontier: bool
    dominated_by: Optional[str] = None


def _mean_cost(records: Sequence[Dict[str, Any]], arm: str, scenario: str) -> Optional[float]:
    """Return the mean measured `n_model_fits` for one (arm, scenario), or ``None`` when unmeasured."""
    values = [
        float(record["n_model_fits"])
        for record in records
        if record.get("arm") == arm and record.get("scenario") == scenario and record.get("n_model_fits") is not None
    ]
    return float(np.mean(values)) if values else None


def pareto_points(
    records: Sequence[Dict[str, Any]],
    scenario: str,
    model: str,
    k_label: str,
    metric: str = "roc_auc",
    null_arm: str = NULL_ARM,
) -> List[ParetoPoint]:
    """Return every arm's position, with domination resolved, for one (scenario, model, K).

    The null hypothesis is included at zero cost and zero advantage: it is the option of not selecting at
    all, and leaving it off the axes would hide the most common way an arm loses.
    """
    rows = average_over_cv_seed(extract_long_rows(records, model=model, k_label=k_label, metric=metric))
    arms = sorted({str(row["arm"]) for row in rows if str(row["arm"]) != null_arm and str(row["scenario"]) == scenario})

    raw: List[ParetoPoint] = [ParetoPoint(arm=null_arm, advantage=0.0, cost=0.0, n_seeds=0, on_frontier=True)]
    for arm in arms:
        deltas = paired_differences(rows, arm=arm, null_arm=null_arm, scenario=scenario)
        if not deltas:
            continue
        stat = paired_t_test(deltas)
        raw.append(
            ParetoPoint(
                arm=arm,
                advantage=float(stat.mean_delta),
                cost=_mean_cost(records, arm, scenario),
                n_seeds=stat.m,
                on_frontier=False,
            )
        )

    priced = [point for point in raw if point.cost is not None]
    unpriced = [point for point in raw if point.cost is None]

    resolved: List[ParetoPoint] = []
    for point in priced:
        # An unpriced arm can never dominate: without a measured cost, "no more expensive" is unknown, and
        # treating unknown as cheap is exactly the claim this table refuses to hand out for free.
        dominators = [
            other
            for other in priced
            if other.arm != point.arm
            and other.cost is not None
            and point.cost is not None
            and other.cost <= point.cost
            and other.advantage >= point.advantage
            and (other.cost < point.cost or other.advantage > point.advantage)
        ]
        best = min(dominators, key=lambda candidate: (candidate.cost or 0.0, -candidate.advantage)) if dominators else None
        resolved.append(
            ParetoPoint(
                arm=point.arm,
                advantage=point.advantage,
                cost=point.cost,
                n_seeds=point.n_seeds,
                on_frontier=best is None,
                dominated_by=best.arm if best else None,
            )
        )
    resolved.extend(unpriced)
    return sorted(resolved, key=lambda point: (point.cost if point.cost is not None else float("inf"), -point.advantage))


def pareto_table(records: Sequence[Dict[str, Any]], models: Sequence[str], k_label: str, metric: str = "roc_auc") -> List[str]:
    """Render the frontier per (scenario, model) at one K label."""
    scenarios = sorted({str(record["scenario"]) for record in records if record.get("status") == "ok"})
    lines = [
        "",
        "=" * 100,
        f"COST vs QUALITY -- Pareto frontier at {k_label} (cost = mean n_model_fits, quality = paired delta vs the null)",
        "=" * 100,
        "  An arm is dominated when another is at least as good and no more expensive; the witness is named.",
        "  An arm with no measured cost is UNPRICED, never free, and can never dominate another.",
    ]
    for scenario in scenarios:
        for model in models:
            points = pareto_points(records, scenario=scenario, model=model, k_label=k_label, metric=metric)
            if len(points) <= 1:
                continue
            lines.append(f"[{scenario}] model={model}")
            for point in points:
                cost = "unpriced" if point.cost is None else f"{point.cost:8.1f}"
                verdict = "FRONTIER" if point.on_frontier else f"dominated by {point.dominated_by}"
                lines.append(f"    {point.arm:<28} fits={cost}  delta={point.advantage:+.4f}  m={point.n_seeds:<3} -> {verdict}")
    return lines


def frontier_arms(records: Sequence[Dict[str, Any]], model: str, k_label: str, metric: str = "roc_auc") -> Dict[str, Tuple[str, ...]]:
    """Return ``{scenario: arms on the frontier}``, for a caller that wants the summary without the table."""
    out: Dict[str, Tuple[str, ...]] = {}
    for scenario in sorted({str(record["scenario"]) for record in records if record.get("status") == "ok"}):
        points = pareto_points(records, scenario=scenario, model=model, k_label=k_label, metric=metric)
        out[scenario] = tuple(point.arm for point in points if point.on_frontier)
    return out
