"""The four sizes a run comes in, and the estimate that says what one will cost before it is paid.

A benchmark with one size is a benchmark that gets run once. The grid this suite can express is large
enough that the full version takes days, so without smaller versions the only options are "the whole
thing" or "nothing" -- and the repository already contains two hundred `bench_*` scripts that nobody
runs, plus a `_results/` directory in `.gitignore`, which is what that choice looks like after a year.

Four tiers, each with a job the others cannot do:

* **smoke** -- seconds. Two arms, two beds, one seed. It answers "does the harness still run", which is
  the question a pull request needs answered and the only one it can afford.
* **nightly** -- hours. Every arm, a third of the beds, three seeds. Enough to notice a regression in an
  arm, not enough to publish.
* **weekly** -- a day or two. Every arm, every bed, twenty seeds, which is the pre-registered floor for
  the paired test to resolve the effects this suite cares about.
* **manual** -- whatever the caller asks for. Explicit rather than implicit, so a one-off wide run is
  visible as a one-off in the manifest instead of being mistaken for the weekly.

`estimate` is what makes the choice informed. It multiplies the grid by the median per-cell cost measured
in an earlier run and reports the predicted wall-clock BEFORE the run starts, because the alternative is
discovering that a configuration was a two-day job forty hours into it. With no history it says so rather
than guessing: a prediction from no data is worse than no prediction, since only one of the two gets
believed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["Tier", "TIERS", "TIER_NAMES", "PREDICTION_SEEDS", "SOURCES", "get_tier", "scenarios_for", "median_cell_seconds", "estimate", "format_estimate", "Estimate"]


@dataclass(frozen=True)
class Tier:
    """One named run size."""

    name: str
    #: Which bed library the tier runs over. The legs answer different questions and are kept in separate
    #: results files, so a tier has to name one rather than inheriting whatever the environment last said.
    source: str
    dataset_seeds: Tuple[int, ...]
    cv_seeds: Tuple[int, ...]
    #: Beds to run, or ``None`` for every registered bed.
    scenarios: Optional[Tuple[str, ...]]
    #: Arms to run, or ``None`` for the whole roster.
    arms: Optional[Tuple[str, ...]]
    purpose: str

    def cell_count(self, available_scenarios: int, available_arms: int) -> int:
        """Return how many cells this tier will run against a grid of the given size."""
        scenarios = len(self.scenarios) if self.scenarios is not None else available_scenarios
        arms = len(self.arms) if self.arms is not None else available_arms
        return scenarios * arms * len(self.dataset_seeds) * len(self.cv_seeds)


#: The development seed range. The report-only range [1000..1099] is reserved and a tier never reaches into
#: it by default: a threshold tuned against report seeds is the failure the reservation exists to prevent.
_DEV_SEEDS: Tuple[int, ...] = tuple(range(20))

TIERS: Dict[str, Tier] = {
    "smoke": Tier(
        name="smoke",
        source="scm",
        dataset_seeds=(0,),
        cv_seeds=(0,),
        scenarios=("null_p100", "linear_k5_p50"),
        arms=("all-features", "skb-f", "variance-sort"),
        purpose="does the harness still run: a pull request can afford this and nothing larger",
    ),
    "nightly": Tier(
        name="nightly",
        source="scm",
        dataset_seeds=(0, 1, 2),
        cv_seeds=(0,),
        scenarios=None,
        arms=None,
        purpose="notice a regression in an arm; too few seeds to publish a difference",
    ),
    "weekly": Tier(
        name="weekly",
        source="scm",
        dataset_seeds=_DEV_SEEDS,
        cv_seeds=(0,),
        scenarios=None,
        arms=None,
        purpose="the pre-registered floor of twenty paired seeds, which is what the headline test needs",
    ),
    "manual": Tier(
        name="manual",
        source="scm",
        dataset_seeds=(0, 1, 2),
        cv_seeds=(0,),
        scenarios=None,
        arms=None,
        purpose="a one-off, visible as one in the manifest rather than mistaken for the weekly",
    ),
}


#: Seeds for the `predictions` tier: the development range, as for every other tier. Scoring a forecast is not
#: tuning a threshold, so nothing here is fitted against these seeds and the report range stays untouched.
PREDICTION_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)


def _predictions_tier() -> Tier:
    """Build the tier that tests the section 2e forecasts: those methods, on the beds that name them.

    Derived from the registry rather than listed, because a hand-written bed list is the thing that drifts: a
    prediction added to a bed would then never be run. Built on request, not at import, since resolving the
    registry costs more than every other tier together and most runs never ask for this one.
    """
    from mlframe.data.datasets.scenarios import SCENARIOS

    from ._roster import PREREGISTERED_2E_ARMS

    forecast = set(PREREGISTERED_2E_ARMS)
    beds = list(SCENARIOS.values()) if isinstance(SCENARIOS, dict) else list(SCENARIOS)
    named = tuple(sorted(bed.name for bed in beds if forecast & set(bed.expected_to_break)))
    return Tier(
        name="predictions",
        source="scm",
        dataset_seeds=PREDICTION_SEEDS,
        cv_seeds=(0,),
        scenarios=named,
        arms=tuple(PREREGISTERED_2E_ARMS),
        purpose="score the section 2e break predictions: each forecast method on every bed that predicts it breaks",
    )


#: Tiers built on request rather than at import.
_LAZY_TIERS: Dict[str, Callable[[], Tier]] = {"predictions": _predictions_tier}

#: Every tier name the command line accepts, eager and lazy alike.
TIER_NAMES: Tuple[str, ...] = tuple(sorted(set(TIERS) | set(_LAZY_TIERS)))

#: Bed libraries a tier may name, and what each leg is for. Kept here rather than resolved from an
#: environment variable: a run whose bed list depended on a variable somebody exported earlier is a run
#: nobody can reproduce from its own manifest.
SOURCES: Tuple[str, ...] = ("scm", "adversarial", "real", "default")


def scenarios_for(source: str, n_samples: int = 0) -> List[Tuple[str, Any]]:
    """Return ``[(name, generator)]`` for one bed library.

    Raises:
        ValueError: On an unknown source, so a typo cannot quietly fall through to the smallest library.
    """
    if source == "scm":
        from ._scm_beds import SCM_BED_ROWS, scm_bed_scenarios

        return list(scm_bed_scenarios(include_null=True, n_samples=int(n_samples) or SCM_BED_ROWS))
    if source == "adversarial":
        from .scenarios import ADVERSARIAL_SCENARIOS, make

        return [(name, (lambda nm: (lambda seed: make(nm, seed)))(name)) for name in ADVERSARIAL_SCENARIOS]
    if source == "real":
        from ._real_beds import real_bed_scenarios

        return list(real_bed_scenarios())
    if source == "default":
        from .synth import make_dataset

        return [("default", lambda seed: make_dataset(n_samples=5000, seed=seed))]
    raise ValueError(f"unknown scenario source {source!r}; expected one of {SOURCES}")


def get_tier(name: str) -> Tier:
    """Return one tier by name.

    Raises:
        KeyError: On an unknown name, listing what exists. A typo'd tier that silently fell back to a
            default would run a different experiment from the one asked for and label it with the name
            that was asked for.
    """
    if name in _LAZY_TIERS:
        return _LAZY_TIERS[name]()
    if name not in TIERS:
        raise KeyError(f"unknown tier {name!r}; available: {list(TIER_NAMES)}")
    return TIERS[name]


def median_cell_seconds(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Return ``{arm: median wall seconds}`` from an earlier run's records.

    The median rather than the mean, and per ARM rather than pooled: the roster spans four orders of
    magnitude in cost, so a pooled figure predicts nothing about a grid whose arm mix differs. Cells whose
    timing was invalidated -- a failed memo drain -- are excluded, because a dictionary lookup is not a
    cost and including it would make the estimate optimistic exactly where the run is slowest.
    """
    by_arm: Dict[str, List[float]] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        if record.get("memo_drained") is False:
            continue
        wall = record.get("wall_time_s")
        if wall is None:
            continue
        by_arm.setdefault(str(record.get("arm")), []).append(float(wall))
    return {arm: float(np.median(values)) for arm, values in sorted(by_arm.items()) if values}


@dataclass(frozen=True)
class Estimate:
    """What a tier is predicted to cost, and how much of that prediction rests on measurement."""

    tier: str
    n_cells: int
    n_cells_priced: int
    predicted_seconds: Optional[float]
    per_arm_seconds: Dict[str, float]
    unpriced_arms: Tuple[str, ...]


def estimate(tier: Tier, scenarios: Sequence[str], arms: Sequence[str], history: Sequence[Dict[str, Any]], workers: int = 1) -> Estimate:
    """Predict a tier's wall-clock from an earlier run's per-arm medians.

    Args:
        tier: The tier to price.
        scenarios: Beds this run will cover.
        arms: Arms this run will cover.
        history: Records from an earlier run, used only for their per-arm medians.
        workers: Parallel workers, which divide the total. An optimistic divisor -- the slowest arm still
            bounds the tail -- and it is labelled as such rather than corrected by a fudge factor.

    Returns:
        An :class:`Estimate`. ``predicted_seconds`` is ``None`` when no arm in the grid has ever been
        timed: a prediction from no data is worse than none, because only one of the two is believed.
    """
    chosen_scenarios = list(tier.scenarios) if tier.scenarios is not None else list(scenarios)
    chosen_arms = list(tier.arms) if tier.arms is not None else list(arms)
    medians = median_cell_seconds(history)
    per_seed = len(tier.dataset_seeds) * len(tier.cv_seeds) * len(chosen_scenarios)

    total = 0.0
    priced = 0
    unpriced: List[str] = []
    for arm in chosen_arms:
        if arm in medians:
            total += medians[arm] * per_seed
            priced += per_seed
        else:
            unpriced.append(arm)

    n_cells = len(chosen_arms) * per_seed
    predicted = (total / max(1, int(workers))) if priced else None
    return Estimate(
        tier=tier.name,
        n_cells=n_cells,
        n_cells_priced=priced,
        predicted_seconds=predicted,
        per_arm_seconds={arm: medians[arm] for arm in chosen_arms if arm in medians},
        unpriced_arms=tuple(sorted(unpriced)),
    )


def format_estimate(value: Estimate) -> List[str]:
    """Render an estimate as the lines a `--dry-run` prints before anybody commits a machine to it."""
    lines = ["", "=" * 100, f"DRY RUN -- tier {value.tier!r}", "=" * 100, "", f"cells: {value.n_cells}"]
    if value.predicted_seconds is None:
        lines.append("predicted wall-clock: UNKNOWN -- no arm in this grid has ever been timed, and a guess would be believed")
        return lines

    hours = value.predicted_seconds / 3600.0
    lines.append(f"predicted wall-clock: {hours:.1f} h ({value.predicted_seconds / 60.0:.0f} min), from the per-arm medians of an earlier run")
    coverage = value.n_cells_priced / value.n_cells if value.n_cells else 0.0
    lines.append(f"priced: {value.n_cells_priced} of {value.n_cells} cells ({coverage:.0%})")
    if value.unpriced_arms:
        lines.append(f"NOT priced, so the figure above is a FLOOR: {', '.join(value.unpriced_arms)}")
    lines.append("")
    lines.append("the slowest arms, which bound the tail regardless of how many workers run:")
    for arm, seconds in sorted(value.per_arm_seconds.items(), key=lambda item: item[1], reverse=True)[:5]:
        lines.append(f"  {arm:<20}{seconds:>8.1f} s per cell")
    return lines
