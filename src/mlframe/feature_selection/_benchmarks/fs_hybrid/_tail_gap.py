"""Scoring the tail-isolation bed, where set recovery is the wrong measurement.

Every column on the tail-isolation bed is a genuine cause of the target, so an arm that selects all of them
has "recovered the answer key" and told us nothing. The bed's question is narrower and set recovery cannot
express it: the tail-dependent pairs carry three times the signal of the correlation-matched ones despite
being declared identically, so does an arm RANK them higher?

The measurement is a rank gap. Take each arm's selection, count how many of its slots went to columns from
tail-dependent pairs against how many went to the matched control pairs, and normalise. Zero means the arm
split its budget evenly between two groups that are equal on every rank statistic and unequal only in the
tail -- which is what a method reading rank correlation, linear correlation or a ten-bin mutual information
must do, because those statistics are identical for the two groups BY CONSTRUCTION. A positive gap means
the arm resolved the tail.

This is the only measurement in the suite that isolates tail dependence rather than non-monotonicity. The
existing t-copula bed and its Gaussian control cannot do it: they separate the roster identically, because
a symmetric gate at the eightieth percentile fires only about 1.1 times as often under a t copula as under
a correlation-matched Gaussian one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ._leaderboard import NULL_ARM

logger = logging.getLogger(__name__)

__all__ = ["TAIL_ISOLATION_BED", "TailGapRow", "column_groups", "tail_gap_rows", "tail_gap_table", "tail_gap_from_results"]

#: The bed this analysis is about. Named rather than inferred: applying a rank gap to a bed whose columns
#: do not come in matched pairs would produce a number with no meaning attached to it.
TAIL_ISOLATION_BED = "tail_isolation_clayton_vs_gaussian"


@dataclass(frozen=True)
class TailGapRow:
    """One arm's rank gap on the tail-isolation bed."""

    arm: str
    n_seeds: int
    tail_selected: float
    control_selected: float
    gap: float
    resolved: bool

    def as_dict(self) -> Dict[str, Any]:
        """Return the row as a plain mapping, for a report that stores rather than prints it."""
        return {"arm": self.arm, "n_seeds": self.n_seeds, "tail_selected": self.tail_selected, "control_selected": self.control_selected, "gap": self.gap, "resolved": self.resolved}


def column_groups(columns: Sequence[str]) -> Tuple[Set[str], Set[str]]:
    """Split the bed's columns into ``(tail_dependent, matched_control)`` by their declared naming.

    The bed names its Clayton columns ``c<i>a``/``c<i>b`` and its Gaussian ones ``g<i>a``/``g<i>b``. Reading
    the group from the name rather than re-deriving it from the data is deliberate: re-deriving would mean
    estimating tail dependence inside the scorer, using an estimator whose own blind spots are exactly what
    is under test.
    """
    tail = {name for name in columns if name.startswith("c") and name[1:2].isdigit()}
    control = {name for name in columns if name.startswith("g") and name[1:2].isdigit()}
    return tail, control


def _selected_columns(record: Dict[str, Any], k_label: str) -> Optional[List[str]]:
    """Return the columns one cell selected at ``k_label``, or ``None`` when it stored none.

    A selection recorded as ``all_features`` or ``omitted_too_large`` is not an absence of data but a
    deliberate non-storage, and both are unusable here for the same reason: the gap is about WHICH columns
    were chosen, and neither form says.
    """
    payload = record.get("selected")
    if not isinstance(payload, dict):
        return None
    block = payload.get(k_label)
    if not isinstance(block, dict) or block.get("status") != "ok":
        return None
    columns = block.get("columns")
    return [str(name) for name in columns] if isinstance(columns, list) else None


def tail_gap_rows(records: Sequence[Dict[str, Any]], k_label: str = "1k", bed: str = TAIL_ISOLATION_BED) -> List[TailGapRow]:
    """Return each arm's rank gap on the tail-isolation bed, averaged over seeds.

    The gap is ``(tail_share - control_share)``, where each share is the fraction of the arm's selection
    drawn from that group. It is bounded in ``[-1, 1]``, is zero for an arm that cannot tell the groups
    apart, and does not depend on how many columns the arm chose -- which matters, because the arms here
    select very different numbers.
    """
    here = [record for record in records if str(record.get("scenario")) == bed and record.get("status") == "ok" and str(record.get("arm")) != NULL_ARM]
    if not here:
        return []

    by_arm: Dict[str, List[Tuple[float, float]]] = {}
    for record in here:
        columns = _selected_columns(record, k_label)
        if not columns:
            continue
        tail, control = column_groups(columns)
        # Groups are read from the SELECTION, so an arm that picked nothing from either group contributes a
        # zero gap rather than being dropped: declining to choose is an answer to this bed's question.
        total = float(len(columns))
        by_arm.setdefault(str(record["arm"]), []).append((len(tail) / total, len(control) / total))

    out: List[TailGapRow] = []
    for arm, shares in sorted(by_arm.items()):
        tail_share = float(np.mean([pair[0] for pair in shares]))
        control_share = float(np.mean([pair[1] for pair in shares]))
        gap = tail_share - control_share
        # A gap has to beat the seed-to-seed spread of itself before it is a finding rather than a draw.
        gaps = [pair[0] - pair[1] for pair in shares]
        spread = float(np.std(gaps, ddof=1)) if len(gaps) > 1 else float("inf")
        out.append(TailGapRow(arm=arm, n_seeds=len(shares), tail_selected=round(tail_share, 4), control_selected=round(control_share, 4), gap=round(gap, 4), resolved=bool(gap > spread)))
    return sorted(out, key=lambda row: row.gap, reverse=True)


def tail_gap_table(records: Sequence[Dict[str, Any]], k_label: str = "1k") -> List[str]:
    """Render the rank gap as a table, with the reading of a zero stated rather than left to the reader."""
    rows = tail_gap_rows(records, k_label=k_label)
    lines = ["", "=" * 100, f"TAIL DEPENDENCE, ISOLATED -- rank gap on {TAIL_ISOLATION_BED} @ {k_label}", "=" * 100]
    if not rows:
        lines.append("no stored selections on this bed, so the gap cannot be computed (it needs WHICH columns, not how many)")
        return lines
    lines.append("")
    lines.append("The two column groups have matching rank correlation, matching marginals and matching link weights.")
    lines.append("They differ only in lower-tail dependence, so a gap of zero is what every rank statistic must produce.")
    lines.append("")
    lines.append(f"{'arm':<22}{'tail share':>12}{'control share':>15}{'gap':>9}  verdict")
    for row in rows:
        verdict = "resolves the tail" if row.resolved else "cannot tell the groups apart"
        lines.append(f"{row.arm:<22}{row.tail_selected:>12.3f}{row.control_selected:>15.3f}{row.gap:>9.3f}  {verdict}")
    return lines


def tail_gap_from_results(path: str, k_label: str = "1k") -> List[str]:
    """Read a results file and render the table, for a caller that has a path rather than records."""
    from ._cell_store import JsonlCellStore

    records = JsonlCellStore(path).load()
    logger.info("read %d records from %s for the tail gap table", len(records), path)
    return tail_gap_table(records, k_label=k_label)
