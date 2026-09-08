"""Scoring the pre-registration's own predictions, so declaring them costs something.

Every bed declares, before the run, which arms it expects to defeat. Until this module those declarations
were write-only: they satisfied a coverage meta-test and were never checked against what happened. A
prediction nobody scores is not a prediction, and in a benchmark run by the author of one of its arms it is
worse than nothing -- it looks like a commitment while carrying no risk.

The scoring is deliberately blunt. An arm "broke" on a bed when it failed to beat the `all-features` null
there; that is what a bed designed to defeat an arm claims will happen. Four cells follow:

* **confirmed** -- predicted to break, and broke. The bed did its job.
* **prediction failed** -- predicted to break, and paid anyway. The interesting cell: our prior about the
  method was wrong, and that is a finding about us, not about the arm.
* **unpredicted failure** -- not predicted, and broke. A bed defeating an arm nobody expected it to is
  either a discovery or a bug in the bed, and both deserve a look.
* **as expected** -- not predicted, and paid.

The headline is the hit rate over the predicted cells. A suite whose predictions are always right is
suspiciously well-tuned to its author's beliefs; one whose predictions are always wrong is not measuring
what it thinks. Both readings need the number to exist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

from ._leaderboard import NULL_ARM, extract_long_rows
from ._paired_stats import average_over_cv_seed, paired_differences, paired_t_test

logger = logging.getLogger(__name__)

__all__ = ["ForecastRow", "forecast_rows", "forecast_table", "OUTCOMES"]

OUTCOMES: Tuple[str, ...] = ("confirmed", "prediction failed", "unpredicted failure", "as expected")


@dataclass(frozen=True)
class ForecastRow:
    """One (bed, arm) prediction and what happened to it."""

    scenario: str
    arm: str
    predicted_to_break: bool
    broke: bool
    delta: float
    p_value: Optional[float]

    @property
    def outcome(self) -> str:
        """Return which of the four cells this row lands in."""
        if self.predicted_to_break:
            return "confirmed" if self.broke else "prediction failed"
        return "unpredicted failure" if self.broke else "as expected"


def _predictions(records: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Tuple[str, ...]], str]:
    """Return ``({scenario: arms it declared it would defeat}, where those declarations came from)``.

    The cells themselves are the preferred source: a prediction that travelled with the run cannot have been
    edited between the run and the scoring. A run written before predictions were persisted falls back to
    the committed scenario registry, which is the pre-registered artefact rather than a fresh opinion -- but
    the fallback is NAMED in the report, because "declared before this run" and "declared before some run"
    are different claims.
    """
    from_cells: Dict[str, Tuple[str, ...]] = {}
    for record in records:
        scenario = str(record.get("scenario", ""))
        # Presence of the field, not its truthiness: a bed that predicts NOTHING will break is making a
        # prediction, and reading its empty list as "no prediction recorded" would drop the bed from the
        # scorecard entirely -- including any arm that broke there against expectation.
        if scenario and "expected_to_break" in record and scenario not in from_cells:
            from_cells[scenario] = tuple(str(arm) for arm in (record["expected_to_break"] or ()))
    if from_cells:
        return from_cells, "the cells themselves"

    out: Dict[str, Tuple[str, ...]] = {}
    present = {str(record.get("scenario", "")) for record in records}
    try:
        from .adversarial_scenarios import ADVERSARIAL_SCENARIOS

        for name in sorted(present & set(ADVERSARIAL_SCENARIOS)):
            _x, _y, truth = ADVERSARIAL_SCENARIOS[name](0)
            declared = cast(Sequence[Any], truth.get("expected_to_break") or ())
            out[name] = tuple(str(arm) for arm in declared)
    except Exception as exc:  # a missing generator must not take the whole report down
        logger.warning("cannot read adversarial predictions for the forecast table: %s", exc)
    try:
        from mlframe.data.datasets import scenarios as scm_scenarios

        for name in sorted(present & set(scm_scenarios.names())):
            out[name] = tuple(scm_scenarios.get(name).expected_to_break)
    except Exception as exc:
        logger.warning("cannot read SCM predictions for the forecast table: %s", exc)
    return {name: arms for name, arms in out.items() if arms}, "the committed registry (this run predates persisted predictions)"


def forecast_rows(
    records: Sequence[Dict[str, Any]],
    model: str = "lightgbm",
    k_label: str = "1k",
    metric: str = "roc_auc",
    alpha: float = 0.05,
) -> List[ForecastRow]:
    """Score every (bed, arm) pair a bed made a prediction about, plus the pairs it did not.

    "Broke" means the arm did not beat the null on that bed: either the paired difference is not positive,
    or it is positive but the paired test does not separate it from zero. An arm that gains without
    separating has not demonstrated a gain, and treating it as one would let noise refute a prediction.
    """
    predictions, _source = _predictions(records)
    if not predictions:
        return []

    rows = average_over_cv_seed(extract_long_rows(records, model=model, k_label=k_label, metric=metric))
    scenarios = sorted({str(row["scenario"]) for row in rows})
    arms = sorted({str(row["arm"]) for row in rows if str(row["arm"]) != NULL_ARM})

    out: List[ForecastRow] = []
    for scenario in scenarios:
        declared = set(predictions.get(scenario, ()))
        for arm in arms:
            deltas = paired_differences(rows, arm=arm, null_arm=NULL_ARM, scenario=scenario)
            if len(deltas) < 2:
                continue
            stat = paired_t_test(deltas)
            # A `None` p-value has two causes and they are opposites. Fewer than two paired seeds means the
            # test could not run; a zero-variance difference means every seed moved by exactly the same
            # amount, which is the STRONGEST evidence available, not the weakest. Reading both as "did not
            # separate" would score a perfectly consistent gain as a broken arm.
            deterministic = stat.m >= 2 and stat.sd_delta == 0.0
            separated = deterministic if stat.p_value is None else stat.p_value < alpha
            beat_the_null = stat.mean_delta > 0 and separated
            out.append(
                ForecastRow(
                    scenario=scenario,
                    arm=arm,
                    predicted_to_break=arm in declared,
                    broke=not beat_the_null,
                    delta=float(stat.mean_delta),
                    p_value=stat.p_value,
                )
            )
    return out


def forecast_table(records: Sequence[Dict[str, Any]], model: str = "lightgbm", k_label: str = "1k") -> List[str]:
    """Render the forecast scorecard: the hit rate, then every prediction that failed, then the surprises."""
    rows = forecast_rows(records, model=model, k_label=k_label)
    _predicted, source = _predictions(records)
    lines = [
        "",
        "=" * 100,
        f"PRE-REGISTERED PREDICTIONS, SCORED -- {model} @ {k_label}",
        "=" * 100,
        "  Each bed declared before the run which arms it expects to defeat. 'Broke' = did not beat the null here.",
        f"  predictions read from: {source}",
    ]
    if not rows:
        lines.append("  no bed in this run carries a prediction, so there is nothing to score")
        return lines

    predicted = [row for row in rows if row.predicted_to_break]
    confirmed = [row for row in predicted if row.broke]
    failed = [row for row in predicted if not row.broke]
    surprises = [row for row in rows if not row.predicted_to_break and row.broke]

    if predicted:
        lines.append(f"  hit rate: {len(confirmed)}/{len(predicted)} predictions held ({len(confirmed) / len(predicted):.0%})")
    else:
        # A rate over zero predictions is not 0% and not 100%; printing `nan%` invites a reader to treat it
        # as one of them. No bed named an arm, so there is nothing to score and the line says exactly that.
        lines.append("  no bed in this run named an arm it expects to defeat, so there is nothing to score")
    lines.append(f"  unpredicted failures: {len(surprises)} of {len(rows) - len(predicted)} unpredicted pairs")

    if failed:
        lines += ["", "  PREDICTION FAILED -- the bed expected to defeat these arms and they paid anyway:"]
        for row in sorted(failed, key=lambda item: -item.delta):
            p_text = "n/a" if row.p_value is None else f"{row.p_value:.4g}"
            lines.append(f"    {row.arm:<24} [{row.scenario}] delta={row.delta:+.4f} p={p_text}")
    if surprises:
        lines += ["", "  UNPREDICTED FAILURE -- these arms lost on beds that did not name them:"]
        for row in sorted(surprises, key=lambda item: item.delta):
            lines.append(f"    {row.arm:<24} [{row.scenario}] delta={row.delta:+.4f}")
    return lines
