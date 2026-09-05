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

# An empty report cell has exactly two honest meanings and they are NOT the same claim. Rendering both as
# a bare dash is the defect this vocabulary exists to prevent: the reader could not tell "this question
# does not apply to this row" from "nobody answered this question".
STATUS_COMPUTED = "computed"
STATUS_NOT_APPLICABLE = "not_applicable"
STATUS_NOT_COMPUTED = "not_computed"

#: The token each status renders as. Distinct strings, on purpose, and never reused for anything else.
EMPTY_CELL_TOKENS: Dict[str, str] = {STATUS_NOT_APPLICABLE: "n/a(undefined)", STATUS_NOT_COMPUTED: "NOT-COMPUTED"}

#: Printed above every table that can contain one of the tokens above, so the tokens are self-describing.
EMPTY_CELL_LEGEND = (
    f"  legend: {EMPTY_CELL_TOKENS[STATUS_NOT_APPLICABLE]} = the quantity does not exist for this row "
    f"(e.g. the arm IS that control, or the statistic is undefined); {EMPTY_CELL_TOKENS[STATUS_NOT_COMPUTED]} "
    "= it exists but was NOT computed here (reason printed on the row). Different claims, never one token."
)


def _random_control_name(arms: Iterable[str]) -> Optional[str]:
    """The `random-<k>` arm present in these records; its `k` varies per bed, so it cannot be hardcoded."""
    for arm in sorted(arms):
        if arm.startswith("random-"):
            return arm
    return None


def _paired_series(rows: Sequence[Dict[str, Any]], arm: str, scenario: str) -> Dict[int, float]:
    """Return `{dataset_seed: value}` for one arm on one bed, already averaged over the nuisance seed."""
    return {int(r["dataset_seed"]): float(r["value"]) for r in rows if r["arm"] == arm and r["scenario"] == scenario}


def _adjusted_cell(
    series: Dict[int, float],
    null: Dict[int, float],
    control: Dict[int, float],
    arm: str,
    control_name: Optional[str],
    control_label: str,
) -> Tuple[Optional[float], str, str]:
    """One control-adjusted cell as `(value, status, reason)`, never an unexplained empty.

    Args:
        series: `{dataset_seed: value}` for the arm being scored.
        null: `{dataset_seed: value}` for the null-hypothesis arm on the same bed.
        control: `{dataset_seed: value}` for the control, empty when the control did not run here.
        arm: The arm being scored.
        control_name: The control's arm id on this bed, or `None` when it is absent entirely.
        control_label: Human-readable control name for the reason string.

    Returns:
        `(value, STATUS_COMPUTED, "")` when the adjusted delta exists; otherwise `(None, status, reason)`
        where `status` separates "does not apply" from "applies but was not answered".
    """
    if control_name is not None and arm == control_name:
        return None, STATUS_NOT_APPLICABLE, f"this arm IS the {control_label} control"
    if control_name is None:
        return None, STATUS_NOT_COMPUTED, f"no {control_label} control arm is present on this bed at all"
    if not control:
        return None, STATUS_NOT_COMPUTED, f"{control_label} ran but produced no usable cell at this model/K"
    both = sorted(set(series) & set(null) & set(control))
    if not both:
        return None, STATUS_NOT_COMPUTED, f"no dataset_seed shared by the arm, the null and {control_label}"
    return float(np.mean([(series[s] - null[s]) - (control[s] - null[s]) for s in both])), STATUS_COMPUTED, ""


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
        # Resolved from the RECORDS too, not from the scored rows: an arm that ran but produced no cell at this
        # (model, K) must read as "ran, no usable cell here", not as "was never on the bed" -- the
        # cardinality control has `score_kind='none'` and therefore no matched-K row at all, which is a
        # different fact from its absence and is the one a reader needs in order to act.
        random_arm = _random_control_name({str(r.get("arm")) for r in records if str(r.get("scenario")) == scenario})
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
            }
            for key, ctrl, ctrl_name, label in (
                ("delta_vs_random", rnd, random_arm, "random-<k>"),
                ("delta_vs_variance", var, VARIANCE_CONTROL, VARIANCE_CONTROL),
            ):
                value, status, reason = _adjusted_cell(series, null, ctrl, arm=arm, control_name=ctrl_name, control_label=label)
                row[key] = value
                row[f"{key}_status"] = status
                row[f"{key}_reason"] = reason
            out.append(row)
    return out


def _fmt_cell(row: Dict[str, Any], key: str) -> str:
    """Render one adjusted-delta cell, using the status token rather than a bare dash when it is empty."""
    if row.get(f"{key}_status") == STATUS_COMPUTED:
        return f"{float(row[key]):+13.4f}"
    return f"{EMPTY_CELL_TOKENS[str(row.get(f'{key}_status'))]:>13}"


def format_control_block(records: Sequence[Dict[str, Any]], model: str, k_label: str, top: int = 5) -> List[str]:
    """Render the adjusted table for one (model, K), best-by-adjusted-skill first."""
    table = control_adjusted_table(records, model=model, k_label=k_label)
    if not table:
        return []
    lines = ["", f"[{model} K={k_label}] skill net of the matched-cardinality and unsupervised-ranking controls", EMPTY_CELL_LEGEND]
    by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for row in table:
        by_scenario.setdefault(str(row["scenario"]), []).append(row)
    for scenario in sorted(by_scenario):
        rows = sorted(by_scenario[scenario], key=lambda r: (r["delta_vs_random"] is None, -(r["delta_vs_random"] or -9e9)))
        shown = rows[:top]
        # The denominator is stated on the header, not left to be inferred from the number of rows printed.
        lines.append(f"  [{scenario}] showing {len(shown)} of {len(rows)} arm(s), best adjusted skill first")
        for row in shown:
            reasons = [f"{key}: {row[f'{key}_reason']}" for key in ("delta_vs_random", "delta_vs_variance") if row[f"{key}_status"] == STATUS_NOT_COMPUTED]
            lines.append(
                f"    {row['arm']:<26} vs_null={row['delta_vs_null']:+8.4f}  "
                f"vs_random={_fmt_cell(row, 'delta_vs_random')}  vs_variance={_fmt_cell(row, 'delta_vs_variance')}  m={row['m']}"
            )
            for reason in reasons:
                lines.append(f"        why empty -> {reason}")
    return lines


def controls_that_beat_the_null(records: Sequence[Dict[str, Any]], model: str, k_label: str) -> List[Tuple[str, str, float]]:
    """`(scenario, control, delta)` wherever a control itself beats the null, which caps every arm's real credit."""
    hits: List[Tuple[str, str, float]] = []
    for row in control_adjusted_table(records, model=model, k_label=k_label):
        if row["arm"] in (VARIANCE_CONTROL,) or str(row["arm"]).startswith("random-"):
            if float(row["delta_vs_null"]) > 0:
                hits.append((str(row["scenario"]), str(row["arm"]), float(row["delta_vs_null"])))
    return sorted(hits, key=lambda t: -t[2])
