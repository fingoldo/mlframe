"""Where the spread in a comparison actually comes from: the method, the data draw, the split, or the rows.

The leaderboard answers "is arm A better than arm B". It cannot answer the question a reader of that
leaderboard asks next, which is whether the choice of method matters more than the roll of the data. Both
questions are answered by the same cells, and until now only the first one was being asked of them.

The decomposition runs on the paired DIFFERENCE between two arms rather than on their scores. Differencing
removes every effect shared by the pair -- the bed's intrinsic difficulty, the prevalence, the ceiling --
which are large, uninteresting here, and would otherwise dominate every component. What survives is exactly
the four sources the pre-registration names:

* **arm x scenario** -- the advantage depends on the bed. This is the scientific product: a large share here
  means "which method is better" has no answer that is not qualified by "on what".
* **dataset_seed** -- the advantage moves with an independent regeneration of the data-generating process.
  This is the noise the headline paired `t` is computed against, so it sets what any seed count can resolve.
* **cv_seed** -- the advantage moves when only the inner split changes, the training data and the holdout
  being identical. Selection instability, and a nuisance rather than a replication axis.
* **rows** -- the advantage moves with the finite holdout. Supplied per cell where a row-level variance was
  measured; when it is absent the component is reported as `None` rather than as zero, because an
  unmeasured source and a source measured to be absent are not the same statement.

Estimation is the method of moments on a nested design, which is what the shape of the data supports: the
cells are unbalanced (a failed cell leaves a hole), and a likelihood fit would need an iterative solver and
a distributional assumption to buy a small efficiency gain on four numbers that are read as proportions.
A method-of-moments component can come out NEGATIVE when the true value is near zero; both the raw estimate
and the zero-clamped one are reported, because clamping silently is how a component the data says is absent
gets published as a small positive number.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["VarianceComponents", "decompose_paired_variance", "format_components", "variance_table", "ROW_VARIANCE_KEY"]

#: Key of the per-cell row-level variance of the metric, when a cell measured one.
ROW_VARIANCE_KEY = "row_variance"


@dataclass(frozen=True)
class VarianceComponents:
    """The four sources of spread in one pair of arms' paired difference, as variances and as shares.

    The `*_raw` fields are the method-of-moments estimates as computed; the shares are taken over the
    clamped values, since a proportion of a negative number is not readable. `row` is `None` when no cell
    supplied one.
    """

    arm_by_scenario: float
    dataset_seed: float
    cv_seed: float
    row: Optional[float]
    arm_by_scenario_raw: float
    dataset_seed_raw: float
    n_scenarios: int
    n_dataset_seeds: int
    n_cells: int
    negative_components: Tuple[str, ...]

    @property
    def total(self) -> float:
        """The sum of the clamped components, which is what the shares are taken against."""
        return float(self.arm_by_scenario + self.dataset_seed + self.cv_seed + (self.row if self.row is not None else 0.0))

    def shares(self) -> Dict[str, float]:
        """Return each clamped component as a fraction of the total, or an empty mapping when the total is zero."""
        total = self.total
        if total <= 0.0:
            return {}
        out = {
            "arm_by_scenario": float(self.arm_by_scenario / total),
            "dataset_seed": float(self.dataset_seed / total),
            "cv_seed": float(self.cv_seed / total),
        }
        if self.row is not None:
            out["row"] = float(self.row / total)
        return out

    def method_matters_more_than_the_draw(self) -> Optional[bool]:
        """Whether the bed-dependent part of the advantage exceeds the data draw, or `None` when nothing is measurable."""
        if self.total <= 0.0:
            return None
        return bool(self.arm_by_scenario > self.dataset_seed)


def _paired_by_split(
    rows: Sequence[Dict[str, Any]],
    arm: str,
    null_arm: str,
    value_key: str,
) -> Tuple[Dict[Tuple[str, int, int], float], List[float]]:
    """Return the paired difference per `(scenario, dataset_seed, cv_seed)` and the row variances that came with it.

    A cell contributes only when BOTH arms ran it: the pairing is what removes the bed's own difficulty, and
    an unpaired cell carries that difficulty instead of a difference.
    """
    by_key: Dict[Tuple[str, int, int, str], float] = {}
    row_var: Dict[Tuple[str, int, int, str], float] = {}
    for row in rows:
        row_arm = str(row["arm"])
        if row_arm not in (arm, null_arm):
            continue
        value = row.get(value_key)
        if value is None or not np.isfinite(float(value)):
            continue
        key = (str(row["scenario"]), int(row["dataset_seed"]), int(row.get("cv_seed", 0)), row_arm)
        by_key[key] = float(value)
        measured = row.get(ROW_VARIANCE_KEY)
        if measured is not None and np.isfinite(float(measured)):
            row_var[key] = float(measured)

    deltas: Dict[Tuple[str, int, int], float] = {}
    variances: List[float] = []
    for (scenario, seed, cv_seed, row_arm), value in by_key.items():
        if row_arm != arm:
            continue
        other = by_key.get((scenario, seed, cv_seed, null_arm))
        if other is None:
            continue
        deltas[(scenario, seed, cv_seed)] = value - other
        left = row_var.get((scenario, seed, cv_seed, arm))
        right = row_var.get((scenario, seed, cv_seed, null_arm))
        if left is not None and right is not None:
            # The two arms are scored on the SAME holdout rows, so their row-level errors are positively
            # correlated and the difference's variance is not their sum. Without the covariance the honest
            # move is the larger of the two, which bounds the paired variance from above.
            variances.append(max(left, right))
    return deltas, variances


def _pooled_within(groups: Sequence[np.ndarray]) -> Tuple[float, float]:
    """Return the pooled within-group variance and the mean group size, which de-biases the level above it."""
    usable = [g for g in groups if g.size > 1]
    sizes = [float(g.size) for g in groups]
    mean_size = float(np.mean(sizes)) if sizes else 1.0
    if not usable:
        return 0.0, mean_size
    ss = float(sum(float(((g - g.mean()) ** 2).sum()) for g in usable))
    df = int(sum(g.size - 1 for g in usable))
    return (ss / df if df > 0 else 0.0), mean_size


def decompose_paired_variance(
    rows: Sequence[Dict[str, Any]],
    arm: str,
    null_arm: str,
    value_key: str = "value",
) -> VarianceComponents:
    """Split the variance of one arm pair's paired difference across scenario, dataset seed, split and rows.

    Args:
        rows: Raw cell records, one per `(arm, scenario, dataset_seed, cv_seed)`. Unlike every function in
            `_paired_stats`, this one WANTS the un-collapsed frame: `cv_seed` is a component here, not a
            nuisance to be averaged away before the statistics see it.
        arm: The arm whose advantage is being decomposed.
        null_arm: The arm it is compared against, normally the `all-features` null hypothesis.
        value_key: The metric column.

    Returns:
        The four components. A component the design cannot estimate -- one scenario, or one split per seed --
        comes back as zero, and the counts on the result say which case it was.

    Raises:
        ValueError: When no `(scenario, dataset_seed, cv_seed)` cell carries both arms, so there is nothing
            paired to decompose.
    """
    deltas, row_variances = _paired_by_split(rows, arm=arm, null_arm=null_arm, value_key=value_key)
    if not deltas:
        raise ValueError(f"no cell carries both {arm!r} and {null_arm!r}, so there is no paired difference to decompose")

    by_seed: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for (scenario, seed, _cv_seed), delta in deltas.items():
        by_seed[(scenario, seed)].append(delta)

    sigma_cv, mean_splits_per_seed = _pooled_within([np.asarray(v, dtype=np.float64) for v in by_seed.values()])

    by_scenario: Dict[str, List[float]] = defaultdict(list)
    for (scenario, _seed), values in by_seed.items():
        by_scenario[scenario].append(float(np.mean(values)))

    seed_level, mean_seeds_per_scenario = _pooled_within([np.asarray(v, dtype=np.float64) for v in by_scenario.values()])
    # Each seed mean already carries sigma_cv / (splits per seed) of split noise; leaving it in would count
    # the same variation twice, once as the split and once as the draw.
    sigma_seed_raw = seed_level - (sigma_cv / mean_splits_per_seed if mean_splits_per_seed > 0 else 0.0)

    scenario_means = np.asarray([float(np.mean(v)) for v in by_scenario.values()], dtype=np.float64)
    if scenario_means.size > 1:
        scenario_level = float(scenario_means.var(ddof=1))
        sigma_scenario_raw = scenario_level - (seed_level / mean_seeds_per_scenario if mean_seeds_per_scenario > 0 else 0.0)
    else:
        sigma_scenario_raw = 0.0

    negative: List[str] = []
    if sigma_scenario_raw < 0.0:
        negative.append("arm_by_scenario")
    if sigma_seed_raw < 0.0:
        negative.append("dataset_seed")
    if negative:
        logger.debug("negative moment estimate for %s on %s vs %s; clamped to zero", negative, arm, null_arm)

    return VarianceComponents(
        arm_by_scenario=float(max(sigma_scenario_raw, 0.0)),
        dataset_seed=float(max(sigma_seed_raw, 0.0)),
        cv_seed=float(sigma_cv),
        row=float(np.mean(row_variances)) if row_variances else None,
        arm_by_scenario_raw=float(sigma_scenario_raw),
        dataset_seed_raw=float(sigma_seed_raw),
        n_scenarios=len(by_scenario),
        n_dataset_seeds=len(by_seed),
        n_cells=len(deltas),
        negative_components=tuple(negative),
    )


def format_components(components: VarianceComponents, arm: str, null_arm: str) -> str:
    """Render one decomposition as the report line it is read as, shares first and the caveats named."""
    shares = components.shares()
    if not shares:
        return f"{arm} vs {null_arm}: every component is zero across {components.n_cells} cells -- the difference does not move at all"
    parts = ", ".join(f"{name} {share:.1%}" for name, share in shares.items())
    line = f"{arm} vs {null_arm} ({components.n_scenarios} beds, {components.n_dataset_seeds} bed-seeds, {components.n_cells} cells): {parts}"
    if components.row is None:
        line += "; rows unmeasured"
    if components.negative_components:
        line += f"; negative moment estimate clamped for {', '.join(components.negative_components)}"
    return line


def variance_table(records: Sequence[Dict[str, Any]], model: str, k_label: str) -> List[str]:
    """Return the report block: one decomposition row per arm, on normalized skill for one (model, K).

    Skill rather than a raw metric for the same reason the pooled fit uses it -- the components are summed
    and compared across beds, and a raw AUC difference means something different on each of them.
    """
    from ._bayes import NULL_ARM, extract_skill_rows

    rows = extract_skill_rows(records, model=model, k_label=k_label)
    arms = sorted({str(r["arm"]) for r in rows if str(r["arm"]) != NULL_ARM})
    if not arms:
        return []
    out = [
        "",
        "=" * 100,
        f"WHERE THE ADVANTAGE MOVES -- {model} @ {k_label}",
        "=" * 100,
        "",
        "Variance of the paired difference against the null, split across its sources. A large bed share",
        "means the arm's advantage has no answer that is not qualified by which bed it was measured on.",
        "",
        "| arm | beds | cells | bed | data draw | split | rows | bed sd | bed > draw |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        try:
            components = decompose_paired_variance(rows, arm=arm, null_arm=NULL_ARM)
        except ValueError:
            continue
        shares = components.shares()
        if not shares:
            out.append(f"| `{arm}` | {components.n_scenarios} | {components.n_cells} | flat | flat | flat | flat | 0.0000 | n/a |")
            continue
        verdict = components.method_matters_more_than_the_draw()
        row_share = f"{shares['row']:.1%}" if "row" in shares else "n/m"
        out.append(
            f"| `{arm}` | {components.n_scenarios} | {components.n_cells} | {shares['arm_by_scenario']:.1%} | "
            f"{shares['dataset_seed']:.1%} | {shares['cv_seed']:.1%} | {row_share} | "
            f"{float(np.sqrt(components.arm_by_scenario)):.4f} | {'yes' if verdict else 'no'} |"
        )
    return out
