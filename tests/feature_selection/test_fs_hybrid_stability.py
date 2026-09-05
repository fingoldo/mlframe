"""Behavioural tests for selection stability and support recovery.

The stability index is checked against the two anchors that define it -- identical selections score 1, and
independent selections of the same size score ~0 -- because those are exactly what distinguishes it from
mean pairwise Jaccard, which reports a large positive number for the second case and would let a random
selector look structured.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid._stability import (
    SelectionSet,
    nogueira_stability,
    recovery_table,
    selection_sets,
    stability_table,
    support_recovery,
)


def _sets(columns_per_seed: List[List[str]], arm: str = "arm", scenario: str = "bed") -> List[SelectionSet]:
    """Build one ``SelectionSet`` per seed from explicit column lists."""
    return [SelectionSet(arm=arm, scenario=scenario, dataset_seed=i, columns=tuple(cols)) for i, cols in enumerate(columns_per_seed)]


def _record(scenario: str, arm: str, seed: int, columns: List[str], k_label: str = "k5", relevant: List[str] | None = None) -> Dict[str, Any]:
    """Build an ok cell record carrying a stored selection at one K label."""
    rec: Dict[str, Any] = {
        "status": "ok",
        "arm": arm,
        "scenario": scenario,
        "dataset_seed": seed,
        "cv_seed": 0,
        "selected": {k_label: {"status": "ok", "columns": list(columns)}},
    }
    if relevant is not None:
        rec["truth_relevant"] = list(relevant)
    return rec


class TestNogueiraIndex:
    """The two anchors that define the index, and the cases where it must decline to answer."""

    def test_identical_selections_score_one(self) -> None:
        """A selector that picks the same columns on every seed is perfectly stable."""
        result = nogueira_stability(_sets([["a", "b", "c"]] * 5), n_features=20)
        assert result.phi == pytest.approx(1.0)

    def test_independent_selections_of_equal_size_score_about_zero(self) -> None:
        """Random selections of a fixed size are the index's null, and it must not reward them."""
        rng = np.random.default_rng(0)
        names = [f"f{i}" for i in range(200)]
        draws = [list(rng.choice(names, size=20, replace=False)) for _ in range(40)]
        result = nogueira_stability(_sets(draws), n_features=200)
        assert result.phi is not None
        assert abs(result.phi) < 0.05

    def test_pairwise_jaccard_would_have_been_fooled_by_the_same_data(self) -> None:
        """The size correction is the point: raw overlap on the null is clearly positive, the index is not."""
        rng = np.random.default_rng(1)
        names = [f"f{i}" for i in range(50)]
        draws = [set(rng.choice(names, size=25, replace=False)) for _ in range(30)]
        jaccard = float(np.mean([len(a & b) / len(a | b) for i, a in enumerate(draws) for b in list(draws)[i + 1 :]]))
        result = nogueira_stability(_sets([sorted(d) for d in draws]), n_features=50)
        assert jaccard > 0.2
        assert result.phi is not None and abs(result.phi) < 0.1

    def test_partial_overlap_lands_between_the_anchors(self) -> None:
        """A selector sharing a stable core and churning the rest is neither perfectly stable nor random."""
        draws = [["a", "b", "c", f"x{i}", f"y{i}"] for i in range(10)]
        result = nogueira_stability(_sets(draws), n_features=100)
        assert result.phi is not None
        assert 0.2 < result.phi < 0.95

    def test_a_width_below_what_was_selected_does_not_index_past_the_matrix(self) -> None:
        """A wrong width is corrected to what the selections themselves prove, not trusted into a crash."""
        result = nogueira_stability(_sets([["a", "b", "c"], ["a", "d", "e"]]), n_features=2)
        assert result.phi is not None
        assert result.n_features_seen == 5

    def test_one_set_has_no_stability(self) -> None:
        """Stability is a property of repeated draws; a single selection has none to report."""
        assert nogueira_stability(_sets([["a", "b"]])).phi is None

    def test_degenerate_sizes_report_undefined_rather_than_perfect(self) -> None:
        """Empty selections everywhere leave the index undefined; reporting 1.0 would claim stability."""
        assert nogueira_stability(_sets([[], [], []]), n_features=10).phi is None


class TestSupportRecovery:
    """Recovery against a declared relevant set."""

    def test_perfect_selection_scores_one(self) -> None:
        """Choosing exactly the relevant columns is precision and recall of one."""
        result = support_recovery(_sets([["a", "b"]] * 3), relevant=["a", "b"])
        assert result is not None
        assert (result.precision, result.recall, result.f1) == (1.0, 1.0, 1.0)

    def test_precision_and_recall_move_independently(self) -> None:
        """A wide selection keeps recall and loses precision, which is the whole point of reporting both."""
        result = support_recovery(_sets([["a", "b", "n1", "n2"]] * 3), relevant=["a", "b"])
        assert result is not None
        assert result.recall == pytest.approx(1.0)
        assert result.precision == pytest.approx(0.5)

    def test_averaged_per_seed_not_pooled(self) -> None:
        """One seed's wide selection cannot carry the others; the mean is over per-seed scores."""
        result = support_recovery(_sets([["a", "b"], ["n1", "n2"]]), relevant=["a", "b"])
        assert result is not None
        assert result.recall == pytest.approx(0.5)
        assert result.precision == pytest.approx(0.5)

    def test_a_bed_without_declared_truth_returns_nothing(self) -> None:
        """A real bed declares no relevant set, and that is reported as absence, not as zero recovery."""
        assert support_recovery(_sets([["a"]]), relevant=[]) is None


class TestFromRecords:
    """Reading selections back out of cell records, including what must NOT be counted."""

    def test_only_stored_selections_are_read(self) -> None:
        """Cells whose selection was omitted or unrankable contribute nothing, rather than an empty set."""
        records = [_record("bed", "arm", 0, ["a", "b"]), _record("bed", "arm", 1, ["a", "c"])]
        records.append({"status": "ok", "arm": "arm", "scenario": "bed", "dataset_seed": 2, "selected": {"k5": {"status": "omitted_too_large", "n": 900}}})
        records.append({"status": "ok", "arm": "arm", "scenario": "bed", "dataset_seed": 3, "selected": {"k5": {"status": "no_ranking"}}})
        records.append({"status": "ok", "arm": "all-features", "scenario": "bed", "dataset_seed": 0, "selected": {"k5": {"status": "all_features", "n": 50}}})
        got = selection_sets(records, arm="arm", scenario="bed", k_label="k5")
        assert [s.dataset_seed for s in got] == [0, 1]

    def test_failed_cells_are_not_read(self) -> None:
        """A crashed cell's record may carry stale fields; it is not a selection."""
        bad = _record("bed", "arm", 5, ["a", "b"])
        bad["status"] = "error"
        assert selection_sets([bad], arm="arm", scenario="bed", k_label="k5") == []

    def test_tables_say_so_when_nothing_was_stored(self) -> None:
        """A run predating selection storage renders an explicit note, never an empty table read as zero."""
        records = [{"status": "ok", "arm": "arm", "scenario": "bed", "dataset_seed": 0}]
        assert any("no cell carried a stored selection" in line for line in stability_table(records, k_label="k5"))
        assert any("declares a relevant set" in line for line in recovery_table(records, k_label="k5"))

    def test_recovery_table_uses_the_declared_truth_from_the_records(self) -> None:
        """Truth travels with the cells, so the report needs no second source to score recovery against."""
        records = [_record("bed", "arm", seed, ["a", "b"], relevant=["a", "b", "c"]) for seed in range(3)]
        rendered = "\n".join(recovery_table(records, k_label="k5"))
        assert "`arm`" in rendered
        assert "0.667" in rendered
