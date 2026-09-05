"""Inference for the Phase 0 protocol: paired per-`dataset_seed` differences against `all-features`.

Three commitments from the pre-registration are enforced here rather than left to discipline.

1. **The headline test is the paired `t` on the `m` per-`dataset_seed` differences**, with
   `SE = sd(delta)/sqrt(m)` and `m-1` degrees of freedom. `dataset_seed` replicates are independent
   regenerations of the data-generating process, so the correlated-`t` / Nadeau-Bengio family (which
   exists to patch reuse of one finite dataset) does not apply and `rho = 0`.

2. **Row-level bootstrap on a 10k holdout is descriptive only.** At `Delta = 0.005` and `n = 10k` the
   row-wise paired bootstrap gives `Delta/SE` around 8, so it returns `p < 1e-6` for every effect anyone
   cares about; it is a formality, not a test. `row_level_descriptive_ci` returns a result explicitly
   stamped `descriptive_only=True` and no verdict function accepts it.

3. **`cv_seed` is a nuisance axis, not a replication axis.** Within a `dataset_seed` the training data
   and the holdout are identical and only the inner split moves. It is averaged away *first*, and its
   spread is reported separately as selection instability. `assert_one_row_per_cell` is the guard: any
   statistical function receiving more than one row per `(arm, scenario, dataset_seed)` understates the
   standard error by a factor of `sqrt(c)`.

`evaluation/noise_band.py` is deliberately not used for the headline. It bands the standard error of `K`
inner-CV fold means (models fit on `(K-1)/K` of train, scored on `n_train/K` rows) while the headline is
a holdout number from a model fit on the full train; the band carries 4 degrees of freedom, so the
threshold deciding every verdict is itself random with roughly 35% relative spread; and its
`n_comparisons` Bonferroni *widens* the band, which is correct for controlling false *differences* and
anti-conservative for claiming *equivalence* -- it makes an equivalence claim easier rather than harder.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "CellIndex",
    "PairedTResult",
    "DescriptiveCI",
    "assert_one_row_per_cell",
    "average_over_cv_seed",
    "paired_differences",
    "paired_t_test",
    "row_level_descriptive_ci",
    "reliability",
    "intention_to_treat_mean",
]

CellIndex = Tuple[str, str, int]  # (arm, scenario, dataset_seed)


@dataclass(frozen=True)
class PairedTResult:
    """The headline statistic: paired `t` over per-`dataset_seed` differences."""

    m: int
    mean_delta: float
    sd_delta: float
    se: float
    t_stat: Optional[float]
    df: int
    p_value: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]

    def beats_null(self, alpha: float = 0.05) -> Optional[bool]:
        """True when the difference is positive and the two-sided `t` test rejects at `alpha`."""
        if self.p_value is None:
            return None
        return bool(self.mean_delta > 0 and self.p_value < alpha)


@dataclass(frozen=True)
class DescriptiveCI:
    """A row-level bootstrap interval. Descriptive only -- never an input to a verdict."""

    low: float
    high: float
    n_boot: int
    descriptive_only: bool = True


def _index(row: Dict[str, Any]) -> CellIndex:
    """Return the `(arm, scenario, dataset_seed)` index of a result row."""
    return (str(row["arm"]), str(row["scenario"]), int(row["dataset_seed"]))


def assert_one_row_per_cell(rows: Sequence[Dict[str, Any]]) -> None:
    """Raise when any `(arm, scenario, dataset_seed)` appears more than once.

    This is the meta-guard the pre-registration demands: reaching a statistical function with `c` rows
    per cell (one per `cv_seed`) deflates the standard error by `sqrt(c)`.
    """
    counts: Dict[CellIndex, int] = defaultdict(int)
    for row in rows:
        counts[_index(row)] += 1
    dupes = {k: v for k, v in counts.items() if v > 1}
    if dupes:
        sample = sorted(dupes.items())[:5]
        raise ValueError(
            f"{len(dupes)} (arm, scenario, dataset_seed) cells carry more than one row -- cv_seed must be "
            f"averaged away before any statistical function sees the frame. First offenders: {sample}"
        )


def average_over_cv_seed(rows: Iterable[Dict[str, Any]], value_key: str = "value") -> List[Dict[str, Any]]:
    """Collapse `cv_seed` by averaging, returning one row per `(arm, scenario, dataset_seed)`.

    The spread across `cv_seed` is preserved as `selection_instability_sd` (population sd over the
    collapsed rows) so it can be reported in its own right instead of leaking into the headline error.
    """
    buckets: Dict[CellIndex, List[float]] = defaultdict(list)
    for row in rows:
        val = row.get(value_key)
        if val is None or not np.isfinite(float(val)):
            continue
        buckets[_index(row)].append(float(val))

    out: List[Dict[str, Any]] = []
    for (arm, scenario, seed), vals in sorted(buckets.items()):
        arr = np.asarray(vals, dtype=np.float64)
        out.append(
            {
                "arm": arm,
                "scenario": scenario,
                "dataset_seed": seed,
                value_key: float(arr.mean()),
                "n_cv_seeds": int(arr.size),
                "selection_instability_sd": float(arr.std(ddof=0)) if arr.size > 1 else 0.0,
            }
        )
    return out


def paired_differences(
    rows: Sequence[Dict[str, Any]],
    arm: str,
    null_arm: str,
    scenario: str,
    value_key: str = "value",
) -> List[float]:
    """Return the per-`dataset_seed` differences `arm - null_arm` within one scenario.

    Only seeds where both arms produced a value contribute; the pairing is what makes the comparison
    exact, so an unpaired seed is dropped here and accounted for by `reliability` instead.
    """
    assert_one_row_per_cell([r for r in rows if r["scenario"] == scenario and r["arm"] in (arm, null_arm)])
    by_arm: Dict[str, Dict[int, float]] = {arm: {}, null_arm: {}}
    for row in rows:
        if row["scenario"] != scenario:
            continue
        name = str(row["arm"])
        if name in by_arm and row.get(value_key) is not None:
            by_arm[name][int(row["dataset_seed"])] = float(row[value_key])
    shared = sorted(set(by_arm[arm]) & set(by_arm[null_arm]))
    return [by_arm[arm][s] - by_arm[null_arm][s] for s in shared]


def paired_t_test(deltas: Sequence[float], alpha: float = 0.05) -> PairedTResult:
    """Paired `t` over the per-seed differences: `SE = sd(delta)/sqrt(m)`, `m-1` degrees of freedom."""
    arr = np.asarray([d for d in deltas if np.isfinite(d)], dtype=np.float64)
    m = int(arr.size)
    if m == 0:
        return PairedTResult(0, float("nan"), float("nan"), float("nan"), None, 0, None, None, None)
    mean = float(arr.mean())
    if m == 1:
        return PairedTResult(1, mean, 0.0, float("nan"), None, 0, None, None, None)

    sd = float(arr.std(ddof=1))
    se = sd / math.sqrt(m)
    df = m - 1
    if se <= 0.0:
        # Every seed moved by exactly the same amount: the difference is real but the t statistic is
        # undefined; report it without a p-value rather than emitting an infinite one.
        return PairedTResult(m, mean, sd, se, None, df, None, mean, mean)

    from scipy import stats as _stats

    t_stat = mean / se
    p_value = float(2.0 * _stats.t.sf(abs(t_stat), df))
    half = float(_stats.t.ppf(1.0 - alpha / 2.0, df)) * se
    return PairedTResult(m, mean, sd, se, float(t_stat), df, p_value, mean - half, mean + half)


def row_level_descriptive_ci(
    paired_row_deltas: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> DescriptiveCI:
    """Row-level paired bootstrap interval on a single holdout. Descriptive only, never a verdict.

    Kept because it is genuinely informative about within-holdout noise, and stamped so it cannot be
    mistaken for the inferential result: on a 10k holdout it declares `p < 1e-6` for every effect of
    practical interest.
    """
    arr = np.asarray(paired_row_deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = arr.size
    if n == 0:
        return DescriptiveCI(float("nan"), float("nan"), n_boot)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return DescriptiveCI(float(low), float(high), n_boot)


def reliability(statuses: Sequence[str]) -> Dict[str, Any]:
    """Return the completed fraction and the per-status breakdown for a group of cells."""
    total = len(statuses)
    counts: Dict[str, int] = defaultdict(int)
    for status in statuses:
        counts[str(status)] += 1
    return {
        "n_cells": total,
        "n_ok": counts.get("ok", 0),
        "reliability": (counts.get("ok", 0) / total) if total else float("nan"),
        "by_status": dict(sorted(counts.items())),
    }


def intention_to_treat_mean(values: Sequence[Optional[float]], base_rate_value: float) -> float:
    """Mean over all cells, charging every missing (crashed/timed-out/OOM) cell the base-rate score.

    A crashed cell is not missing at random: an arm that dies on `p > n` is worse than one returning
    garbage, and complete-case averaging over a grid whose hardest scenarios kill the weakest arms is
    survivorship bias.
    """
    filled = [base_rate_value if (v is None or not np.isfinite(float(v))) else float(v) for v in values]
    if not filled:
        return float("nan")
    return float(np.mean(filled))
