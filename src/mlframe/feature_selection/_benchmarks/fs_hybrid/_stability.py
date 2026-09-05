"""Selection stability and support recovery -- what the arm CHOSE, as opposed to how the choice scored.

Downstream score answers whether a selection was useful. It does not answer either of the questions a
practitioner asks next, and the two are independent of it and of each other:

* **Stability.** Re-run the arm on a fresh draw from the same generating process: does it choose the same
  columns? An arm whose selection turns over completely between seeds may still score well on average while
  being useless as an explanation, and the instability is invisible in a mean.
* **Support recovery.** On a bed with declared truth, did it choose the columns that actually drive the
  target? An arm can score well by picking correlated proxies and badly by picking the causes, so this is
  reported beside the score and never in place of it.

Stability uses the Nogueira-Brown index, not mean pairwise Jaccard. The reason is not taste: the index is
corrected for the selection size, so a method that selects 200 of 500 columns at random scores ~0 instead of
the ~0.4 that pairwise Jaccard would report and a reader would mistake for structure. It also has a known
null distribution, which turns "A is more stable than B" from an impression into a testable statement.

Both are computed from the selections the runner persists per cell. A cell that recorded no selection is
reported as such rather than being treated as an empty one: the difference between "this arm selected
nothing" and "nobody wrote down what it selected" is exactly the sort of collapse this benchmark's
meta-tests exist to prevent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "SelectionSet",
    "StabilityResult",
    "RecoveryResult",
    "selection_sets",
    "nogueira_stability",
    "support_recovery",
    "stability_table",
    "recovery_table",
]


@dataclass(frozen=True)
class SelectionSet:
    """One cell's stored selection at one `K` label."""

    arm: str
    scenario: str
    dataset_seed: int
    columns: Tuple[str, ...]


@dataclass(frozen=True)
class StabilityResult:
    """Nogueira-Brown stability for one (arm, scenario, K), with what it was computed from."""

    n_sets: int
    mean_size: float
    n_features_seen: int
    phi: Optional[float]


@dataclass(frozen=True)
class RecoveryResult:
    """Support recovery against a scenario's declared relevant set."""

    n_sets: int
    precision: float
    recall: float
    f1: float
    n_relevant: int


def selection_sets(records: Iterable[Dict[str, Any]], arm: str, scenario: str, k_label: str) -> List[SelectionSet]:
    """Return the stored selections for one (arm, scenario, K), one per dataset seed.

    Cells whose selection was not stored -- an arm with no ranking, the all-features null, or a selection
    too wide for the runner's storage cap -- are skipped here and counted by the caller through `n_sets`,
    never silently folded in as empty selections.
    """
    out: List[SelectionSet] = []
    for rec in records:
        if rec.get("status") != "ok" or rec.get("arm") != arm or rec.get("scenario") != scenario:
            continue
        block = (rec.get("selected") or {}).get(k_label)
        if not isinstance(block, dict) or block.get("status") != "ok":
            continue
        out.append(
            SelectionSet(
                arm=str(arm),
                scenario=str(scenario),
                dataset_seed=int(rec["dataset_seed"]),
                columns=tuple(str(c) for c in block.get("columns", ())),
            )
        )
    return out


def nogueira_stability(sets: Sequence[SelectionSet], n_features: Optional[int] = None) -> StabilityResult:
    """Return the Nogueira-Brown stability index over selections drawn from the same process.

    The index is ``1 - mean_f(s_f^2) / (kbar/d * (1 - kbar/d))``, where ``s_f^2`` is the unbiased variance of
    feature ``f``'s selection indicator across the sets and ``kbar`` the mean selection size. It is 1 for
    identical selections and 0 in expectation for independent selections of the same size -- the
    size-correction pairwise Jaccard lacks.

    ``n_features`` defaults to the number of distinct columns seen. That is a floor, not the true
    dimensionality: columns no run ever selected are invisible here, which biases the denominator and, with
    it, the index. Pass the bed's real width whenever it is known.
    """
    if len(sets) < 2:
        return StabilityResult(n_sets=len(sets), mean_size=float(np.mean([len(s.columns) for s in sets])) if sets else 0.0, n_features_seen=0, phi=None)

    universe = sorted({c for s in sets for c in s.columns})
    seen = len(universe)
    d = int(n_features) if n_features else seen
    if d < seen:
        # More distinct columns were selected than the caller says the bed has. Trusting the argument would
        # index past the end of the matrix; trusting what was actually selected is the only defensible floor.
        logger.warning("n_features=%s is below the %s distinct columns selected; using the larger", d, seen)
        d = seen
    if d <= 0:
        return StabilityResult(n_sets=len(sets), mean_size=0.0, n_features_seen=seen, phi=None)

    index = {name: i for i, name in enumerate(universe)}
    matrix = np.zeros((len(sets), d), dtype=np.float64)
    for row, s in enumerate(sets):
        for column in s.columns:
            matrix[row, index[column]] = 1.0

    m = matrix.shape[0]
    sizes = matrix.sum(axis=1)
    k_bar = float(sizes.mean())
    p_hat = matrix.mean(axis=0)
    # Unbiased per-feature Bernoulli variance across the m draws.
    variance = (m / (m - 1.0)) * p_hat * (1.0 - p_hat)
    denominator = (k_bar / d) * (1.0 - k_bar / d)
    if denominator <= 0.0:
        # Every set is empty, or every set is the whole space: stability is undefined, not perfect.
        return StabilityResult(n_sets=m, mean_size=k_bar, n_features_seen=seen, phi=None)
    phi = 1.0 - float(variance.mean()) / denominator
    return StabilityResult(n_sets=m, mean_size=k_bar, n_features_seen=seen, phi=phi)


def support_recovery(sets: Sequence[SelectionSet], relevant: Sequence[str]) -> Optional[RecoveryResult]:
    """Return mean precision, recall and F1 of the selections against the declared relevant set.

    Averaged over sets rather than pooled: pooling would let one seed's wide selection carry the others,
    and the per-seed selection is the object a practitioner would actually receive.
    """
    truth = {str(c) for c in relevant}
    if not truth or not sets:
        return None

    precisions, recalls, f1s = [], [], []
    for s in sets:
        chosen = set(s.columns)
        hit = len(chosen & truth)
        precision = hit / len(chosen) if chosen else 0.0
        recall = hit / len(truth)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return RecoveryResult(
        n_sets=len(sets),
        precision=float(np.mean(precisions)),
        recall=float(np.mean(recalls)),
        f1=float(np.mean(f1s)),
        n_relevant=len(truth),
    )


def _declared_relevant(records: Sequence[Dict[str, Any]], scenario: str) -> List[str]:
    """Return the relevant set a scenario's cells declare, empty when the bed declares no truth."""
    for rec in records:
        if rec.get("scenario") == scenario and rec.get("truth_relevant"):
            return [str(c) for c in rec["truth_relevant"]]
    return []


def _bed_width(records: Sequence[Dict[str, Any]], scenario: str) -> Optional[int]:
    """Return the bed's feature count, taken from the null arm's own selection size."""
    for rec in records:
        if rec.get("scenario") != scenario:
            continue
        block = (rec.get("selected") or {}).get("self")
        if isinstance(block, dict) and block.get("status") == "all_features":
            return int(block.get("n", 0)) or None
    return None


def stability_table(records: Sequence[Dict[str, Any]], k_label: str) -> List[str]:
    """Render the stability block: one row per (scenario, arm) with a computable index."""
    scenarios = sorted({str(r["scenario"]) for r in records if r.get("status") == "ok"})
    arms = sorted({str(r["arm"]) for r in records if r.get("status") == "ok"})

    lines = [
        "",
        f"SELECTION STABILITY across dataset seeds -- {k_label} (Nogueira-Brown, size-corrected)",
        "",
        "| scenario | arm | seeds | mean |S| | phi |",
        "|---|---|---|---|---|",
    ]
    any_row = False
    for scenario in scenarios:
        width = _bed_width(records, scenario)
        for arm in arms:
            sets = selection_sets(records, arm=arm, scenario=scenario, k_label=k_label)
            result = nogueira_stability(sets, n_features=width)
            if result.phi is None:
                continue
            any_row = True
            lines.append(f"| {scenario} | `{arm}` | {result.n_sets} | {result.mean_size:.1f} | {result.phi:+.3f} |")
    if not any_row:
        lines.append("| (no cell carried a stored selection at this K) | | | | |")
    return lines


def recovery_table(records: Sequence[Dict[str, Any]], k_label: str) -> List[str]:
    """Render support recovery against declared truth; empty on a bed roster that declares none."""
    scenarios = sorted({str(r["scenario"]) for r in records if r.get("status") == "ok"})
    arms = sorted({str(r["arm"]) for r in records if r.get("status") == "ok"})

    lines = [
        "",
        f"SUPPORT RECOVERY against declared truth -- {k_label}",
        "",
        "| scenario | arm | seeds | |relevant| | precision | recall | F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    any_row = False
    for scenario in scenarios:
        relevant = _declared_relevant(records, scenario)
        if not relevant:
            continue
        for arm in arms:
            sets = selection_sets(records, arm=arm, scenario=scenario, k_label=k_label)
            result = support_recovery(sets, relevant)
            if result is None:
                continue
            any_row = True
            lines.append(
                f"| {scenario} | `{arm}` | {result.n_sets} | {result.n_relevant} | " f"{result.precision:.3f} | {result.recall:.3f} | {result.f1:.3f} |"
            )
    if not any_row:
        lines.append("| (no bed in this run declares a relevant set) | | | | | | |")
    return lines
