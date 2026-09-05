"""The decision rule: every arm is a paired contrast against the `all-features` null hypothesis.

`all-features` is not a baseline line on a chart -- it is the thing every arm must beat, on the same
holdout, seed by seed. A scenario where no arm clears it is not an empty result: it is reported as its own
leaderboard row, `FS does not pay here`. `wrappers/_noise_floor.py` records why this is the expected modal
outcome rather than a pessimistic hedge (madelon: all-features lgbm 0.872 against RFECV-at-251 0.868).

Rows are produced per `(scenario, panel model, K setting)`. The matched-`K` settings (`1k`, `2k`, `5k`)
are the primary outcome; `self` -- the arm's own chosen cardinality -- is reported separately, because
comparing arms at whatever cardinality each chose measures the stopping rule rather than the ranking.

The selector-by-model interaction is reported explicitly: an arm winning under `logistic` and losing under
`lightgbm` is a result, not noise to be averaged away.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ._paired_stats import PairedTResult, average_over_cv_seed, paired_differences, paired_t_test, reliability

logger = logging.getLogger(__name__)

__all__ = [
    "NULL_ARM",
    "NO_PAY_ROW",
    "ArmVerdict",
    "ScenarioVerdict",
    "extract_long_rows",
    "scenario_verdict",
    "leaderboard",
    "selector_by_model_interaction",
    "reliability_table",
    "cost_table",
    "COST_NOT_MEASURED",
    "DIRECTIONAL_VERDICTS",
]

# The pre-registered null hypothesis. Its name is the arm id the runner writes for the no-selection cell.
NULL_ARM = "all-features"
NO_PAY_ROW = "FS does not pay here"


@dataclass(frozen=True)
class ArmVerdict:
    """One arm's paired contrast against the null hypothesis in one cell of the report."""

    arm: str
    stat: PairedTResult
    # "beats_null" | "loses_to_null" | "indistinguishable" | "identical_to_null" |
    # "beats_null_deterministic" | "loses_to_null_deterministic" | "insufficient_seeds"
    verdict: str


@dataclass(frozen=True)
class ScenarioVerdict:
    """Every arm's contrast in one `(scenario, model, K setting)`, plus the headline row."""

    scenario: str
    model: str
    k_label: str
    arms: Tuple[ArmVerdict, ...]
    headline: str  # winning arm name, or NO_PAY_ROW

    def fs_pays(self) -> bool:
        """True when at least one arm beat the null hypothesis here."""
        return self.headline != NO_PAY_ROW


def extract_long_rows(
    records: Iterable[Dict[str, Any]],
    model: str,
    k_label: str,
    metric: str = "roc_auc",
) -> List[Dict[str, Any]]:
    """Flatten cell records into `{arm, scenario, dataset_seed, cv_seed, value}` rows.

    Only `ok` cells contribute a value; failed cells are counted by `reliability_table` and charged the
    base rate by the intention-to-treat aggregate, never dropped silently here and forgotten.
    """
    out: List[Dict[str, Any]] = []
    for rec in records:
        if rec.get("status") != "ok":
            continue
        block = (rec.get("scores") or {}).get(k_label)
        if not isinstance(block, dict):
            continue
        scores = (block.get("models") or {}).get(model)
        if not isinstance(scores, dict) or metric not in scores:
            continue
        out.append(
            {
                "arm": rec["arm"],
                "scenario": rec["scenario"],
                "dataset_seed": int(rec["dataset_seed"]),
                "cv_seed": int(rec.get("cv_seed", 0)),
                "value": float(scores[metric]),
            }
        )
    return out


# Verdicts that state a direction; a disagreement among them across panel members is an interaction.
DIRECTIONAL_VERDICTS = ("beats_null", "loses_to_null", "beats_null_deterministic", "loses_to_null_deterministic")


def _classify(stat: PairedTResult, alpha: float) -> str:
    """Turn a paired-`t` result into a verdict label.

    A zero-variance difference has no `t` statistic but is not an absent result: every seed moved by the
    same amount, so it is labelled `identical_to_null` (delta exactly 0, e.g. an arm that kept every
    column) or `*_deterministic`, never conflated with too few seeds to test.
    """
    if stat.m < 2:
        return "insufficient_seeds"
    if stat.p_value is None:
        if stat.mean_delta == 0.0:
            return "identical_to_null"
        return "beats_null_deterministic" if stat.mean_delta > 0 else "loses_to_null_deterministic"
    if stat.p_value >= alpha:
        return "indistinguishable"
    return "beats_null" if stat.mean_delta > 0 else "loses_to_null"


def scenario_verdict(
    long_rows: Sequence[Dict[str, Any]],
    scenario: str,
    model: str,
    k_label: str,
    null_arm: str = NULL_ARM,
    alpha: float = 0.05,
) -> ScenarioVerdict:
    """Score every arm in one scenario against the null hypothesis on the paired per-seed differences.

    `cv_seed` is averaged away first, so the statistical function receives exactly one row per
    `(arm, scenario, dataset_seed)`.
    """
    collapsed = average_over_cv_seed([r for r in long_rows if r["scenario"] == scenario])
    arms = sorted({str(r["arm"]) for r in collapsed} - {null_arm})
    verdicts: List[ArmVerdict] = []
    for arm in arms:
        deltas = paired_differences(collapsed, arm=arm, null_arm=null_arm, scenario=scenario)
        stat = paired_t_test(deltas, alpha=alpha)
        verdicts.append(ArmVerdict(arm=arm, stat=stat, verdict=_classify(stat, alpha)))

    winners = [v for v in verdicts if v.verdict in ("beats_null", "beats_null_deterministic")]
    headline = max(winners, key=lambda v: v.stat.mean_delta).arm if winners else NO_PAY_ROW
    return ScenarioVerdict(scenario=scenario, model=model, k_label=k_label, arms=tuple(verdicts), headline=headline)


def leaderboard(
    records: Sequence[Dict[str, Any]],
    models: Sequence[str],
    k_labels: Sequence[str],
    metric: str = "roc_auc",
    null_arm: str = NULL_ARM,
    alpha: float = 0.05,
) -> List[ScenarioVerdict]:
    """Return one `ScenarioVerdict` per `(scenario, model, K setting)` across the whole result set."""
    out: List[ScenarioVerdict] = []
    for model in models:
        for k_label in k_labels:
            rows = extract_long_rows(records, model=model, k_label=k_label, metric=metric)
            for scenario in sorted({str(r["scenario"]) for r in rows}):
                out.append(scenario_verdict(rows, scenario, model, k_label, null_arm=null_arm, alpha=alpha))
    return out


def selector_by_model_interaction(verdicts: Sequence[ScenarioVerdict]) -> List[Dict[str, Any]]:
    """Report, per `(arm, scenario, K setting)`, the verdict under each panel model.

    A row whose verdicts disagree across models is flagged `interaction=True`: that arm's usefulness
    depends on the downstream learner, which a single-model design would have reported as one number.
    """
    grouped: Dict[Tuple[str, str, str], Dict[str, ArmVerdict]] = defaultdict(dict)
    for sv in verdicts:
        for av in sv.arms:
            grouped[(av.arm, sv.scenario, sv.k_label)][sv.model] = av

    out: List[Dict[str, Any]] = []
    for (arm, scenario, k_label), by_model in sorted(grouped.items()):
        labels = {m: v.verdict for m, v in by_model.items()}
        deltas = {m: v.stat.mean_delta for m, v in by_model.items()}
        # Compare DIRECTIONS, not labels: "beats_null" and "beats_null_deterministic" agree.
        directional = {("up" if lbl.startswith("beats_null") else "down") for lbl in labels.values() if lbl in DIRECTIONAL_VERDICTS}
        out.append(
            {
                "arm": arm,
                "scenario": scenario,
                "k_label": k_label,
                "verdict_by_model": labels,
                "mean_delta_by_model": deltas,
                "interaction": len(directional) > 1,
            }
        )
    return out


def reliability_table(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per `(arm, scenario)`: the fraction of cells that completed, with the per-status breakdown."""
    buckets: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for rec in records:
        buckets[(str(rec.get("arm")), str(rec.get("scenario")))].append(str(rec.get("status")))
    out: List[Dict[str, Any]] = []
    for (arm, scenario), statuses in sorted(buckets.items()):
        row: Dict[str, Any] = {"arm": arm, "scenario": scenario}
        row.update(reliability(statuses))
        out.append(row)
    return out


#: Why an arm's `n_model_fits` cell is empty. A mean over 3 of 140 cells and a mean over 140 of 140 are
#: not the same number, and "no cell reported a count" is not "this arm is free" -- both distinctions are
#: carried explicitly rather than left to a blank column.
COST_NOT_MEASURED = "NOT-MEASURED"


def cost_table(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per arm: total and mean `n_model_fits`, the primary (deterministic) cost axis, WITH its denominator.

    The mean is meaningless without the count it was taken over: the wrapper arms are exactly the ones
    whose cells fail most often, so their cost mean can be an average over a handful of surviving cells
    while a cheap arm's is an average over all of them. Every row therefore carries `n_cells`,
    `n_cells_measured` and, when nothing was measured, an explicit reason -- never a bare empty cell that
    reads as "free".

    Wall-clock is carried alongside as advisory only; every figure using it must state that the host was
    contended.
    """
    fits: Dict[str, List[float]] = defaultdict(list)
    wall: Dict[str, List[float]] = defaultdict(list)
    cells: Dict[str, int] = defaultdict(int)
    unmeasured_status: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in records:
        arm = str(rec.get("arm"))
        cells[arm] += 1
        if rec.get("n_model_fits") is not None:
            fits[arm].append(float(rec["n_model_fits"]))
        else:
            unmeasured_status[arm][str(rec.get("status"))] += 1
        if rec.get("wall_time_s") is not None:
            wall[arm].append(float(rec["wall_time_s"]))
    out: List[Dict[str, Any]] = []
    for arm in sorted(cells):
        f = fits.get(arm, [])
        w = wall.get(arm, [])
        by_status = dict(unmeasured_status.get(arm, {}))
        out.append(
            {
                "arm": arm,
                "n_cells": cells[arm],
                "n_cells_measured": len(f),
                "n_model_fits_total": sum(f) if f else None,
                "n_model_fits_mean": (sum(f) / len(f)) if f else None,
                "n_model_fits_reason": None if f else f"no cell of this arm reported a fit count (statuses: {by_status})",
                "unmeasured_by_status": by_status,
                "wall_time_s_mean_advisory": (sum(w) / len(w)) if w else None,
            }
        )
    return out


def _fmt_optional(value: Optional[float], digits: int = 4) -> str:
    """Format an optional float for a text table."""
    return "n/a" if value is None else f"{value:.{digits}f}"
