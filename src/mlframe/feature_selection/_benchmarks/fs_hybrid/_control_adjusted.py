"""Skill measured against the matched-cardinality control, not against the null alone.

`all-features` answers "is selecting better than not selecting". It does not answer "is THIS selector better
than picking the same number of columns at random", and on a bed where most columns are noise those are very
different questions: on madelon under a linear model the random control already beats the null, so a raw
delta credits every arm with skill that needed no target at all.

Two controls, and they measure different things:

- `random-<k>` is the cardinality control. Its delta is what any subset of that size buys.
- `variance-sort` is the unsupervised-ranking control. Its delta is what marginal scale alone buys, which on
  a bed whose construction leaks relevance into variance can be most of the apparent win.

An arm's adjusted skill is its delta minus the control's, on the same bed, model and K, paired over the same
seeds. Reported alongside the raw delta rather than replacing it: a reader who wants "better than doing
nothing" still has it, and the gap between the two numbers is itself the finding.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ._leaderboard import NULL_ARM, extract_long_rows
from ._paired_stats import average_over_cv_seed

logger = logging.getLogger(__name__)

VARIANCE_CONTROL = "variance-sort"


def _random_control_name(arms: Iterable[str]) -> Optional[str]:
    """The `random-<k>` arm present in these records; its `k` varies per bed, so it cannot be hardcoded."""
    for arm in sorted(arms):
        if arm.startswith("random-"):
            return arm
    return None


def _paired_series(rows: Sequence[Dict[str, Any]], arm: str, scenario: str) -> Dict[int, float]:
    """Return `{dataset_seed: value}` for one arm on one bed, already averaged over the nuisance seed."""
    return {int(r["dataset_seed"]): float(r["value"]) for r in rows if r["arm"] == arm and r["scenario"] == scenario}


def control_adjusted_table(
    records: Sequence[Dict[str, Any]],
    model: str,
    k_label: str,
    metric: str = "roc_auc",
) -> List[Dict[str, Any]]:
    """Per (bed, arm): raw delta vs the null, and the delta net of each control, over the shared seeds.

    Args:
        records: Cell records.
        model: Panel member to score on.
        k_label: The K label to read.
        metric: Metric name inside each cell's model block.

    Returns:
        One row per (scenario, arm) carrying `delta_vs_null`, `delta_vs_random`, `delta_vs_variance` and the
        seed count each was computed over. A control's own row is included so the reader can see its size.
    """
    rows = average_over_cv_seed(extract_long_rows(records, model=model, k_label=k_label, metric=metric))
    scenarios = sorted({str(r["scenario"]) for r in rows})
    arms = sorted({str(r["arm"]) for r in rows})

    out: List[Dict[str, Any]] = []
    for scenario in scenarios:
        # Resolved per SCENARIO: the cardinality control is named after this bed's own k, so one globally
        # resolved name yields an empty series on every bed whose k differs -- and the column silently
        # reads as "not applicable" instead of "not computed".
        random_arm = _random_control_name({str(r["arm"]) for r in rows if r["scenario"] == scenario})
        null = _paired_series(rows, NULL_ARM, scenario)
        rnd = _paired_series(rows, random_arm, scenario) if random_arm else {}
        var = _paired_series(rows, VARIANCE_CONTROL, scenario)
        if not null:
            continue
        for arm in arms:
            if arm == NULL_ARM:
                continue
            series = _paired_series(rows, arm, scenario)
            shared = sorted(set(series) & set(null))
            if not shared:
                continue
            raw = float(np.mean([series[s] - null[s] for s in shared]))
            row: Dict[str, Any] = {
                "scenario": scenario,
                "arm": arm,
                "model": model,
                "k_label": k_label,
                "m": len(shared),
                "delta_vs_null": raw,
                "delta_vs_random": None,
                "delta_vs_variance": None,
            }
            for key, ctrl, ctrl_name in (("delta_vs_random", rnd, random_arm), ("delta_vs_variance", var, VARIANCE_CONTROL)):
                if arm == ctrl_name or not ctrl:
                    continue
                both = sorted(set(series) & set(null) & set(ctrl))
                if both:
                    row[key] = float(np.mean([(series[s] - null[s]) - (ctrl[s] - null[s]) for s in both]))
            out.append(row)
    return out


def format_control_block(records: Sequence[Dict[str, Any]], model: str, k_label: str, top: int = 5) -> List[str]:
    """Render the adjusted table for one (model, K), best-by-adjusted-skill first."""
    table = control_adjusted_table(records, model=model, k_label=k_label)
    if not table:
        return []
    lines = ["", f"[{model} K={k_label}] skill net of the matched-cardinality and unsupervised-ranking controls"]
    by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for row in table:
        by_scenario.setdefault(str(row["scenario"]), []).append(row)
    for scenario in sorted(by_scenario):
        rows = sorted(by_scenario[scenario], key=lambda r: (r["delta_vs_random"] is None, -(r["delta_vs_random"] or -9e9)))
        lines.append(f"  [{scenario}]")
        for row in rows[:top]:
            def fmt(value: Optional[float]) -> str:
                """Render a delta, or a dash when the arm IS that control."""
                return "     -   " if value is None else f"{value:+8.4f}"

            lines.append(
                f"    {row['arm']:<26} vs_null={row['delta_vs_null']:+8.4f}  "
                f"vs_random={fmt(row['delta_vs_random'])}  vs_variance={fmt(row['delta_vs_variance'])}  m={row['m']}"
            )
    return lines


def controls_that_beat_the_null(records: Sequence[Dict[str, Any]], model: str, k_label: str) -> List[Tuple[str, str, float]]:
    """`(scenario, control, delta)` wherever a control itself beats the null, which caps every arm's real credit."""
    hits: List[Tuple[str, str, float]] = []
    for row in control_adjusted_table(records, model=model, k_label=k_label):
        if row["arm"] in (VARIANCE_CONTROL,) or str(row["arm"]).startswith("random-"):
            if float(row["delta_vs_null"]) > 0:
                hits.append((str(row["scenario"]), str(row["arm"]), float(row["delta_vs_null"])))
    return sorted(hits, key=lambda t: -t[2])
