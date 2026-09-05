"""Aggregate the Phase 0 JSONL cells into the pre-registered decision tables.

Disclaimer, reproduced on every report this module prints: this benchmark was designed and run by the
author of one of the arms it judges, and the scenario distribution is not a sample from any real problem
population.

What gets printed, in the order the pre-registration prescribes:

1. **Headline** -- per `(scenario, panel model, matched K)`, every arm as a paired per-`dataset_seed`
   difference against the `all-features` null hypothesis, tested with the paired `t`
   (`SE = sd(delta)/sqrt(m)`, `m-1` df). Scenarios where nothing clears the null print the
   `FS does not pay here` row explicitly.
2. **Self-chosen K**, reported separately: it measures the stopping rule, not the ranking.
3. **Control-adjusted skill**: the same delta net of the matched-cardinality (`random-<k>`) and
   unsupervised-ranking (`variance-sort`) controls, with a loud line wherever a control itself beats the
   null -- that delta is the ceiling on the credit any arm on that bed can honestly claim.
4. **Selector-by-model interaction**, flagged wherever an arm's verdict flips between panel members.
5. **Reliability and intention-to-treat**: the fraction of cells completed per arm and scenario, plus an
   aggregate charging every crashed cell the base rate.
6. **Cost**, on `n_model_fits`; wall-clock is printed with an explicit contention caption.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Sequence

from ._cell_store import JsonlCellStore
from ._leaderboard import (
    COST_NOT_MEASURED,
    NO_PAY_ROW,
    NULL_ARM,
    ScenarioVerdict,
    cost_table,
    extract_long_rows,
    leaderboard,
    reliability_table,
    selector_by_model_interaction,
)
from ._control_adjusted import (
    EMPTY_CELL_LEGEND,
    EMPTY_CELL_TOKENS,
    STATUS_NOT_APPLICABLE,
    STATUS_NOT_COMPUTED,
    controls_that_beat_the_null,
    format_control_block,
)
from .adversarial_scenarios import GATE_SCENARIOS
from ._matched_k import SELF_CHOSEN_K
from ._paired_stats import average_over_cv_seed, intention_to_treat_mean
from ._panel import PANEL_MEMBERS
from .run_experiment import RESULTS_PATH

logger = logging.getLogger(__name__)

__all__ = ["MATCHED_K_LABELS", "matched_k_labels_present", "DISCLAIMER", "WALL_CLOCK_CAPTION", "format_report", "main"]

MATCHED_K_LABELS: Sequence[str] = ("1k", "2k", "5k")

# The multiplier labels above only exist on a bed that declares a target set. A real bed uses the absolute
# grid (`k5`, `k10`, ...), so a hardcoded label list silently renders the PRIMARY outcome as empty there --
# which is how the first confirmatory run came back with a self-chosen-K section and nothing else.
def matched_k_labels_present(records: Sequence[Dict[str, Any]]) -> List[str]:
    """Every non-self K label actually present in `records`, multiplier labels first then the absolute grid."""
    labels: set = set()
    for rec in records:
        labels.update(str(k) for k in (rec.get("scores") or {}) if str(k) != SELF_CHOSEN_K)
    ordered = [lab for lab in MATCHED_K_LABELS if lab in labels]
    absolute = sorted((lab for lab in labels if lab.startswith("k") and lab[1:].isdigit()), key=lambda lab: int(lab[1:]))
    return ordered + absolute

DISCLAIMER = (
    "This benchmark was designed and run by the author of one of the arms it judges (MRMR). The scenario "
    "distribution is not a sample from any real problem population."
)
WALL_CLOCK_CAPTION = (
    "Wall-clock is ADVISORY ONLY: the host was contended (routinely 100+ concurrent python processes). "
    "The primary cost axis is n_model_fits, which is deterministic."
)


def _fmt(value: Any, digits: int = 4) -> str:
    """Format a possibly-`None` number for a text table."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_stat(value: Any, stat: Any, digits: int = 4) -> str:
    """Format a test statistic, naming WHY it is absent instead of printing one ambiguous placeholder.

    A missing `t` or `p` has two unrelated causes here and they are not the same claim: fewer than two
    paired seeds means the test was NOT COMPUTED, while a zero-variance difference means the statistic does
    not exist at all -- every seed moved by the same amount, which is a result, not a gap in the data.
    """
    if value is not None:
        return _fmt(value, digits)
    if stat.m < 2:
        return EMPTY_CELL_TOKENS[STATUS_NOT_COMPUTED]
    return EMPTY_CELL_TOKENS[STATUS_NOT_APPLICABLE]


def _headline_block(verdicts: Sequence[ScenarioVerdict]) -> List[str]:
    """Render the paired-contrast leaderboard, including the explicit no-pay rows."""
    lines: List[str] = ["", "=" * 100, "HEADLINE: paired per-dataset_seed delta vs the all-features NULL HYPOTHESIS", "=" * 100, EMPTY_CELL_LEGEND]
    for sv in verdicts:
        lines.append(f"\n[{sv.scenario}] model={sv.model} K={sv.k_label}")
        if not sv.arms:
            lines.append(f"    {NO_PAY_ROW} (no arm produced a paired value here)")
            continue
        for av in sorted(sv.arms, key=lambda a: -(a.stat.mean_delta if a.stat.m else 0.0)):
            st = av.stat
            lines.append(
                f"    {av.arm:<28} delta={_fmt(st.mean_delta)}  se={_fmt(st.se)}  "
                f"t={_fmt_stat(st.t_stat, st, 2)}  df={st.df}  m={st.m}  p={_fmt_stat(st.p_value, st, 5)}  "
                f"ci=[{_fmt_stat(st.ci_low, st)}, {_fmt_stat(st.ci_high, st)}]  -> {av.verdict}"
            )
        lines.append(f"    HEADLINE: {sv.headline}")
    return lines


def _interaction_block(verdicts: Sequence[ScenarioVerdict]) -> List[str]:
    """Render the selector-by-model interaction table."""
    rows = selector_by_model_interaction(verdicts)
    lines = ["", "=" * 100, "SELECTOR x MODEL INTERACTION (a flipped verdict across the panel is a result, not noise)", "=" * 100]
    flagged = [r for r in rows if r["interaction"]]
    for row in rows:
        mark = "  <== INTERACTION" if row["interaction"] else ""
        verdict_text = ", ".join(f"{m}={v}" for m, v in sorted(row["verdict_by_model"].items()))
        lines.append(f"  {row['arm']:<28} [{row['scenario']}] K={row['k_label']}  {verdict_text}{mark}")
    lines.append(f"  ({len(flagged)} of {len(rows)} arm-scenario-K rows show a selector-by-model interaction)")
    return lines


def _reliability_block(records: Sequence[Dict[str, Any]]) -> List[str]:
    """Render the reliability table and the intention-to-treat aggregate."""
    lines = ["", "=" * 100, "RELIABILITY (a crashed cell is NOT missing at random) + INTENTION-TO-TREAT", "=" * 100]
    for row in reliability_table(records):
        lines.append(
            f"  {row['arm']:<28} [{row['scenario']}] completed={row['reliability']:.3f} "
            f"({row['n_ok']}/{row['n_cells']})  statuses={row['by_status']}"
        )

    lines.append("")
    # Gate beds are excluded from the aggregate on purpose. On a null bed `y` is independent of `X`, so ~0.5 is
    # the CORRECT answer and an arm that selects nothing earns it; averaging that in with the signal beds
    # penalises exactly the behaviour the gate rewards. Gate beds are judged on their own metric (how much an
    # arm selects from pure noise), which the headline block reports separately.
    signal_records = [r for r in records if str(r.get("scenario")) not in GATE_SCENARIOS]
    skipped = sorted({str(r.get("scenario")) for r in records} & set(GATE_SCENARIOS))
    lines.append("  intention-to-treat mean roc_auc over SIGNAL beds only (crashed cells charged the base rate 0.5):")
    if skipped:
        lines.append(f"    gate beds excluded from this mean (a low score there is the correct answer): {', '.join(skipped)}")
    for model in PANEL_MEMBERS:
        rows = extract_long_rows(signal_records, model=model, k_label=SELF_CHOSEN_K)
        collapsed = average_over_cv_seed(rows)
        by_arm: Dict[str, List[Any]] = {}
        for rec in signal_records:
            by_arm.setdefault(str(rec.get("arm")), [])
        for row in collapsed:
            by_arm.setdefault(str(row["arm"]), []).append(row["value"])
        totals = {arm: sum(1 for r in signal_records if str(r.get("arm")) == arm) for arm in by_arm}
        for arm in sorted(by_arm):
            values: List[Any] = list(by_arm[arm])
            n_scored = len(values)
            values += [None] * max(0, totals[arm] - len(values))
            # The denominator is printed, never inferred: an ITT mean whose n is invisible cannot be told
            # apart from a mean over a handful of surviving cells.
            lines.append(
                f"    {model:<10} {arm:<28} itt={intention_to_treat_mean(values, base_rate_value=0.5):.4f} "
                f"(n={len(values)} cells, of which {n_scored} scored and {len(values) - n_scored} charged the base rate)"
            )
    return lines


def _control_adjusted_block(records: Sequence[Dict[str, Any]], models: Sequence[str], k_labels: Sequence[str]) -> List[str]:
    """Render, per `(model, K)`, each arm's delta net of the matched-cardinality and variance controls.

    The headline block credits an arm for beating `all-features`. That is not the same as skill: on a bed
    where most columns are noise, a random subset of the SAME cardinality already beats the null, and on
    `madelon` under `logistic` it does so by +0.0171 (p=0.001). Whatever a control earns without using the
    target is not the arm's; printing the raw delta alone overstates every arm on that bed.
    """
    lines = [
        "",
        "=" * 100,
        "CONTROL-ADJUSTED SKILL -- the raw delta vs the null MINUS what a target-free control already earns",
        "=" * 100,
    ]
    for model in models:
        for k_label in k_labels:
            beaters = controls_that_beat_the_null(records, model=model, k_label=k_label)
            for scenario, control, delta in beaters:
                # Loud on purpose: this number is the CEILING on the credit any arm on this bed can claim.
                lines.append(
                    f"  !! CONTROL BEATS THE NULL: [{scenario}] model={model} K={k_label} control={control} "
                    f"delta_vs_null={delta:+.4f} -- every arm's raw delta here is inflated by at least this much"
                )
            lines += format_control_block(records, model=model, k_label=k_label)
    if len(lines) == 4:
        lines.append("  (no (model, K) produced a control-adjusted row; the controls did not run on these cells)")
    return lines


def _cost_block(records: Sequence[Dict[str, Any]]) -> List[str]:
    """Render the cost table on the deterministic axis, with the wall-clock caption."""
    lines = ["", "=" * 100, "COST -- primary axis n_model_fits", "=" * 100, f"  {WALL_CLOCK_CAPTION}"]
    lines.append(f"  {COST_NOT_MEASURED} means no cell of that arm reported a count -- it does NOT mean the arm is free.")
    for row in cost_table(records):
        if row["n_cells_measured"]:
            cost = f"fits_total={row['n_model_fits_total']:.0f}  fits_mean={row['n_model_fits_mean']:.1f}"
        else:
            cost = f"fits_total={COST_NOT_MEASURED}  fits_mean={COST_NOT_MEASURED}"
        # The denominator rides with the mean: a wrapper arm whose cells mostly crash would otherwise show
        # a cost mean over its few survivors next to a cheap arm's mean over all of them.
        lines.append(
            f"  {row['arm']:<28} {cost}  measured_on={row['n_cells_measured']}/{row['n_cells']} cells  "
            f"wall_s_mean(advisory)={row['wall_time_s_mean_advisory']}"
        )
        if row["n_model_fits_reason"]:
            lines.append(f"        why empty -> {row['n_model_fits_reason']}")
        elif row["unmeasured_by_status"]:
            lines.append(f"        cells with no count -> {row['unmeasured_by_status']}")
    return lines


def format_report(records: Sequence[Dict[str, Any]], models: Sequence[str] = PANEL_MEMBERS) -> str:
    """Build the full text report for a set of cell records."""
    lines: List[str] = [DISCLAIMER, "", f"null hypothesis: {NULL_ARM}", f"cells: {len(records)}"]
    k_labels = matched_k_labels_present(records)
    matched = leaderboard(records, models=models, k_labels=k_labels)
    lines += _headline_block(matched)

    self_k = leaderboard(records, models=models, k_labels=[SELF_CHOSEN_K])
    lines += ["", "=" * 100, "SELF-CHOSEN K -- reported SEPARATELY: this measures the stopping rule, not the ranking", "=" * 100]
    # Drop the block's own banner (3 rule/title lines plus the legend) -- the section header above replaces it.
    lines += _headline_block(self_k)[5:]

    # `self` is included here even though the headline reports it separately: the cardinality control
    # declares `score_kind='none'` and so has NO matched-K row at all, making `self` the only K at which
    # the vs-random column can be computed. Omitting it would leave the whole column empty.
    lines += _control_adjusted_block(records, models=models, k_labels=[*k_labels, SELF_CHOSEN_K])
    lines += _interaction_block(matched + self_k)
    lines += _reliability_block(records)
    lines += _cost_block(records)
    return "\n".join(lines)


def main() -> None:
    """Print the report for the default results file."""
    path = os.environ.get("FS_HYBRID_RESULTS", RESULTS_PATH)
    records = JsonlCellStore(path).load()
    if not records:
        print(f"no records at {path}")
        return
    print(format_report(records))


if __name__ == "__main__":
    main()
