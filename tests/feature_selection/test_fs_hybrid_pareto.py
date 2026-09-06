"""Tests for the cost-versus-quality frontier.

Domination is a claim about two arms at once, so the cases that matter are the ones where a naive
implementation gets it subtly wrong: ties on one axis, an arm with no measured cost, and the null
hypothesis, which must be able to dominate everything that fails to beat it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mlframe.feature_selection._benchmarks.fs_hybrid._pareto import frontier_arms, pareto_points, pareto_table


def _record(arm: str, seed: int, auc: float, fits: Optional[int], scenario: str = "bed") -> Dict[str, Any]:
    """One ok cell with a lightgbm AUC at k5 and a cost."""
    return {
        "status": "ok",
        "arm": arm,
        "scenario": scenario,
        "dataset_seed": seed,
        "cv_seed": 0,
        "n_model_fits": fits,
        "scores": {"k5": {"models": {"lightgbm": {"roc_auc": auc}}}},
    }


def _grid(spec: Dict[str, Any], seeds: int = 5) -> List[Dict[str, Any]]:
    """Build records from ``{arm: (auc, fits)}``, with the null at 0.70."""
    out: List[Dict[str, Any]] = []
    for seed in range(seeds):
        out.append(_record("all-features", seed, 0.70, 10))
        for arm, (auc, fits) in spec.items():
            out.append(_record(arm, seed, auc, fits))
    return out


class TestDomination:
    """The core relation."""

    def test_a_cheaper_and_better_arm_dominates(self) -> None:
        """The witness is named, so the claim can be checked rather than taken on trust."""
        points = {p.arm: p for p in pareto_points(_grid({"cheap": (0.76, 20), "dear": (0.74, 200)}), "bed", "lightgbm", "k5")}
        assert points["cheap"].on_frontier
        assert not points["dear"].on_frontier and points["dear"].dominated_by == "cheap"

    def test_a_dearer_but_better_arm_stays_on_the_frontier(self) -> None:
        """Paying more for more is a trade-off, not a defeat; the frontier is where that trade-off lives."""
        points = {p.arm: p for p in pareto_points(_grid({"cheap": (0.72, 20), "dear": (0.80, 200)}), "bed", "lightgbm", "k5")}
        assert points["cheap"].on_frontier and points["dear"].on_frontier

    def test_the_null_dominates_an_arm_that_does_not_beat_it(self) -> None:
        """Not selecting costs nothing and is the modal winner on real beds; the table must say so."""
        points = {p.arm: p for p in pareto_points(_grid({"loser": (0.65, 50)}), "bed", "lightgbm", "k5")}
        assert points["loser"].dominated_by == "all-features"

    def test_an_equal_arm_at_a_higher_cost_is_dominated(self) -> None:
        """Same quality, more fits: nothing is bought, so this must not read as a tie on the frontier."""
        points = {p.arm: p for p in pareto_points(_grid({"a": (0.76, 20), "b": (0.76, 60)}), "bed", "lightgbm", "k5")}
        assert points["a"].on_frontier
        assert not points["b"].on_frontier and points["b"].dominated_by == "a"

    def test_identical_cost_and_quality_do_not_dominate_each_other(self) -> None:
        """Two arms that agree on both axes are tied, and a strict comparison must keep both."""
        points = {p.arm: p for p in pareto_points(_grid({"a": (0.76, 20), "b": (0.76, 20)}), "bed", "lightgbm", "k5")}
        assert points["a"].on_frontier and points["b"].on_frontier


class TestUnpricedArms:
    """Cheapness is the most valuable claim here, and it must be earned by a measurement."""

    def test_an_unmeasured_cost_reads_as_unpriced_not_free(self) -> None:
        """A missing count would otherwise sort first and look like the cheapest arm in the table."""
        points = {p.arm: p for p in pareto_points(_grid({"unpriced": (0.76, None)}), "bed", "lightgbm", "k5")}
        assert points["unpriced"].cost is None

    def test_an_unpriced_arm_never_dominates_another(self) -> None:
        """ "No more expensive" is unknown without a cost, and unknown must not be treated as cheap."""
        records = _grid({"unpriced": (0.90, None), "priced": (0.76, 30)})
        points = {p.arm: p for p in pareto_points(records, "bed", "lightgbm", "k5")}
        assert points["priced"].on_frontier
        assert points["priced"].dominated_by is None

    def test_unpriced_arms_sort_last(self) -> None:
        """The table is read by cost, so an arm without one belongs at the end, not at the top."""
        points = pareto_points(_grid({"unpriced": (0.99, None), "priced": (0.76, 30)}), "bed", "lightgbm", "k5")
        assert points[-1].arm == "unpriced"


class TestRendering:
    """What the report shows."""

    def test_the_table_names_the_dominating_arm(self) -> None:
        """A verdict of 'dominated' with no witness is an assertion; the reader must be able to check it."""
        rendered = "\n".join(pareto_table(_grid({"cheap": (0.80, 20), "dear": (0.74, 200)}), ["lightgbm"], "k5"))
        assert "dominated by cheap" in rendered
        assert "FRONTIER" in rendered

    def test_the_frontier_summary_matches_the_table(self) -> None:
        """The two views are the same computation, so they cannot disagree about who is on the frontier."""
        records = _grid({"cheap": (0.80, 20), "dear": (0.74, 200)})
        summary = frontier_arms(records, model="lightgbm", k_label="k5")
        points = pareto_points(records, "bed", "lightgbm", "k5")
        assert set(summary["bed"]) == {point.arm for point in points if point.on_frontier}
