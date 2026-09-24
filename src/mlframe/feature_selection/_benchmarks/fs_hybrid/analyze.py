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
from ._leaderboard import NULL_ARM, leaderboard
from ._matched_k import SELF_CHOSEN_K
from ._panel import PANEL_MEMBERS
from .run_experiment import RESULTS_PATH
from ._report_blocks import (
    WALL_CLOCK_CAPTION,
    headline_block as _headline_block,
    interaction_block as _interaction_block,
    reliability_block as _reliability_block,
    control_adjusted_block as _control_adjusted_block,
    cost_block as _cost_block,
    declaration_block as _declaration_block,
    pooled_block as _pooled_block,
    variance_block as _variance_block,
)

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


def format_report(records: Sequence[Dict[str, Any]], models: Sequence[str] = PANEL_MEMBERS) -> str:
    """Build the full text report for a set of cell records."""
    lines: List[str] = [DISCLAIMER, "", f"null hypothesis: {NULL_ARM}", f"cells: {len(records)}"]
    lines += _declaration_block(records)
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
    lines += _pooled_block(records, models=models, k_labels=k_labels)
    lines += _variance_block(records, models=models, k_labels=k_labels)
    if k_labels:
        from ._stability import recovery_table, stability_table

        lines += ["", "=" * 100, "WHAT THE ARM CHOSE (independent of how it scored)", "=" * 100]
        lines += stability_table(records, k_label=k_labels[0])
        lines += recovery_table(records, k_label=k_labels[0])
    lines += _interaction_block(matched + self_k)
    lines += _reliability_block(records)
    lines += _cost_block(records)
    if k_labels:
        from ._pareto import pareto_table

        lines += pareto_table(records, models=models, k_label=k_labels[0])
    from ._forecast import forecast_table

    if k_labels:
        lines += forecast_table(records, model=models[0] if models else "lightgbm", k_label=k_labels[0])
    from ._winners_curse import optimism_table

    lines += optimism_table(records, model=models[0] if models else "lightgbm")
    return "\n".join(lines)


def main() -> None:
    """Print the report for the default results file, optionally with the arms blinded.

    `FS_HYBRID_BLIND=<salt>` relabels every arm before the report is built and writes the mapping beside the
    results. The point is to read the tables before knowing which row is one's own method: any explanation
    found while blind applies to whichever arm it turns out to be. Reveal with `_blinding.unblind_text` once
    the report is committed.
    """
    path = os.environ.get("FS_HYBRID_RESULTS", RESULTS_PATH)
    records = JsonlCellStore(path).load()
    if not records:
        print(f"no records at {path}")
        return

    salt = os.environ.get("FS_HYBRID_BLIND", "").strip()
    if salt:
        from ._blinding import apply_blinding, blind_labels, write_mapping

        mapping = blind_labels({str(record.get("arm", "")) for record in records}, salt=salt)
        mapping_path = write_mapping(path, mapping)
        records = apply_blinding(records, mapping)
        print(f"BLINDED: arm names replaced; mapping written to {mapping_path}")
        print()

    print(format_report(records))


if __name__ == "__main__":
    main()
