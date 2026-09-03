"""Behavioural tests for the Phase 0 pre-registered benchmark protocol (fs_hybrid).

Each test pins one commitment `docs/BENCHMARK_PREREGISTRATION.md` makes: the `all-features` null
hypothesis, matched-`K` scoring, the two-member downstream panel, paired per-`dataset_seed` inference with
`cv_seed` averaged away first, reliability with distinguishable failure statuses, `n_model_fits` as the
cost axis, and a resumable JSONL store keyed by a sort-key-stable digest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid._cell_store import JsonlCellStore
from mlframe.feature_selection._benchmarks.fs_hybrid._leaderboard import (
    NO_PAY_ROW,
    NULL_ARM,
    cost_table,
    extract_long_rows,
    leaderboard,
    reliability_table,
    scenario_verdict,
    selector_by_model_interaction,
)
from mlframe.feature_selection._benchmarks.fs_hybrid._matched_k import (
    K_MULTIPLIERS,
    SELF_CHOSEN_K,
    cut_at_k,
    matched_k_grid,
    ranking_from_arm_result,
)
from mlframe.feature_selection._benchmarks.fs_hybrid._paired_stats import (
    assert_one_row_per_cell,
    average_over_cv_seed,
    intention_to_treat_mean,
    paired_differences,
    paired_t_test,
    reliability,
    row_level_descriptive_ci,
)
from mlframe.feature_selection._benchmarks.fs_hybrid._panel import (
    PANEL_MEMBERS,
    assert_wrapper_estimator_differs,
    base_rate_scores,
    normalized_skill,
    score_predictions,
)
from mlframe.feature_selection._benchmarks.fs_hybrid._protocol_types import (
    CELL_STATUSES,
    CellSpec,
    canonical_json,
    cell_key,
    classify_exception,
)


@dataclass(frozen=True)
class _FakeArmResult:
    """Stand-in for the shared `ArmResult` dataclass, exposing the same attribute contract."""

    support: np.ndarray
    score: Optional[np.ndarray]
    score_kind: str
    ranked_prefix: Optional[Tuple[int, ...]] = None


class _LegacyAdapter:
    """A legacy fs_hybrid adapter: a selected set with no internal ordering."""

    def __init__(self, selected: Any) -> None:
        self.raw_selected_ = list(selected)


# ------------------------------------------------------------------ cell identity and the JSONL store
def test_cell_key_is_insertion_order_independent() -> None:
    """The digest must come from a sorted-key encoding, or a resume silently re-runs the whole grid."""
    a = {"arm": "mrmr", "scenario": "madelon", "dataset_seed": 3}
    b = {"dataset_seed": 3, "scenario": "madelon", "arm": "mrmr"}
    assert cell_key(a) == cell_key(b)
    assert canonical_json(a) == canonical_json(b)


def test_cell_key_changes_when_the_spec_changes() -> None:
    """A different cell must get a different key."""
    base = CellSpec(scenario="s", arm="a", dataset_seed=0, cv_seed=0)
    assert base.key() != CellSpec(scenario="s", arm="a", dataset_seed=1, cv_seed=0).key()
    assert base.key() != CellSpec(scenario="s", arm="a", dataset_seed=0, cv_seed=0, config={"k": 1}).key()


def test_store_round_trips_and_resumes(tmp_path: Any) -> None:
    """Completed keys come back from the file, so a resumed run skips exactly what it finished."""
    store = JsonlCellStore(tmp_path / "results.jsonl")
    store.append({"cell_key": "aaa", "status": "ok"})
    store.append({"cell_key": "bbb", "status": "error"})
    assert store.completed_keys() == {"aaa", "bbb"}
    assert store.completed_keys(statuses={"ok"}) == {"aaa"}
    assert len(store.load()) == 2


def test_store_tolerates_a_truncated_trailing_line(tmp_path: Any) -> None:
    """A process killed mid-write must not make the whole resume state unreadable."""
    path = tmp_path / "results.jsonl"
    store = JsonlCellStore(path)
    store.append({"cell_key": "aaa", "status": "ok"})
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"cell_key": "bb')
    assert store.completed_keys() == {"aaa"}


def test_failed_cell_statuses_are_distinguishable() -> None:
    """`error` / `timeout` / `crashed` / `oom` must not collapse into one bucket."""
    assert classify_exception(MemoryError()) == "oom"
    assert classify_exception(TimeoutError()) == "timeout"
    assert classify_exception(OSError("The paging file is too small")) == "oom"
    assert classify_exception(ValueError("p > n")) == "error"
    assert set(CELL_STATUSES) == {"ok", "error", "timeout", "crashed", "oom"}


# ------------------------------------------------------------------ matched K
def test_matched_k_grid_is_one_two_and_five_times_the_target_size() -> None:
    """The pre-registered grid, capped at the feature count."""
    assert K_MULTIPLIERS == (1, 2, 5)
    assert matched_k_grid(4, n_features=100) == {"1k": 4, "2k": 8, "5k": 20}
    assert matched_k_grid(4, n_features=10)["5k"] == 10


def test_continuous_score_produces_a_ranking_and_cuts_at_k() -> None:
    """A continuous arm ranks by score descending, ties broken by column order."""
    res = _FakeArmResult(support=np.array([True, False, True, False]), score=np.array([0.1, 0.9, 0.5, 0.9]), score_kind="continuous")
    ranking = ranking_from_arm_result(res, ["a", "b", "c", "d"])
    assert ranking.order == ("b", "d", "c", "a")
    assert cut_at_k(ranking, 2) == ["b", "d"]
    assert ranking.selected == ("a", "c")


def test_selection_order_ranking_uses_the_greedy_prefix() -> None:
    """`selection_order` arms rank by pick order; unpicked features have no score at all."""
    res = _FakeArmResult(support=np.array([True, True, False]), score=None, score_kind="selection_order", ranked_prefix=(1, 0))
    ranking = ranking_from_arm_result(res, ["a", "b", "c"])
    assert ranking.order == ("b", "a")
    assert cut_at_k(ranking, 5) == ["b", "a"]  # truncated, never padded
    assert ranking.coverage == pytest.approx(2 / 3)


def test_continuous_arm_without_a_score_is_fatal() -> None:
    """Synthesising a pseudo-score would silently compute a different statistic for this arm."""
    res = _FakeArmResult(support=np.array([True, False]), score=None, score_kind="continuous")
    with pytest.raises(ValueError, match="refusing to invent a ranking"):
        ranking_from_arm_result(res, ["a", "b"])


def test_unrankable_arm_gets_no_matched_k_row() -> None:
    """`score_kind='none'` yields no matched-K cut, rather than a fabricated one."""
    ranking = ranking_from_arm_result(_LegacyAdapter(["a"]), ["a", "b"])
    assert ranking.score_kind == "none"
    assert cut_at_k(ranking, 1) is None
    assert SELF_CHOSEN_K == "self"


# ------------------------------------------------------------------ panel
def test_panel_has_at_least_logistic_and_a_gbm() -> None:
    """The minimum panel is mandated; a single-model design cannot see the interaction."""
    assert {"logistic", "lightgbm"} <= set(PANEL_MEMBERS)


def test_wrapper_arm_optimising_the_only_panel_member_is_refused() -> None:
    """A wrapper scored solely on its own internal objective is a tautology."""
    assert assert_wrapper_estimator_differs("rfecv_lgbm", "lightgbm") is None
    with pytest.raises(ValueError, match="its own objective"):
        assert_wrapper_estimator_differs("rfecv_lgbm", "lightgbm", panel=("lightgbm",))


def test_scores_come_from_the_mlframe_fast_kernels() -> None:
    """A perfectly separating score must reach AUC 1.0 through `mlframe.metrics`."""
    y = np.array([0, 0, 1, 1])
    scores = score_predictions(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert scores["roc_auc"] == pytest.approx(1.0)
    assert scores["brier"] < 0.05
    assert set(scores) == {"roc_auc", "average_precision", "brier", "log_loss"}


def test_normalized_skill_is_one_at_the_bayes_floor_and_zero_at_the_base_rate() -> None:
    """The pre-registered ROPE scale."""
    assert normalized_skill(0.0, brier_base_rate=0.25) == pytest.approx(1.0)
    assert normalized_skill(0.25, brier_base_rate=0.25) == pytest.approx(0.0)
    assert normalized_skill(0.1, brier_base_rate=0.0) is None


def test_base_rate_scores_are_the_crashed_cell_charge() -> None:
    """A constant prevalence predictor has no ranking skill."""
    out = base_rate_scores(np.array([0, 1, 1, 1]), np.array([0, 1, 1, 1]))
    assert out["roc_auc"] == 0.5
    assert out["brier"] > 0.0


# ------------------------------------------------------------------ inference
def test_cv_seed_must_be_averaged_away_before_any_statistic() -> None:
    """More than one row per (arm, scenario, dataset_seed) deflates the SE by sqrt(c)."""
    rows = [
        {"arm": "a", "scenario": "s", "dataset_seed": 0, "cv_seed": 0, "value": 0.8},
        {"arm": "a", "scenario": "s", "dataset_seed": 0, "cv_seed": 1, "value": 0.9},
    ]
    with pytest.raises(ValueError, match="more than one row"):
        assert_one_row_per_cell(rows)

    collapsed = average_over_cv_seed(rows)
    assert len(collapsed) == 1
    assert collapsed[0]["value"] == pytest.approx(0.85)
    assert collapsed[0]["selection_instability_sd"] == pytest.approx(0.05)
    assert_one_row_per_cell(collapsed)


def test_paired_t_matches_scipy_on_the_per_seed_differences() -> None:
    """The headline statistic is exactly the paired t: SE = sd(delta)/sqrt(m), m-1 df."""
    scipy_stats = pytest.importorskip("scipy.stats")
    deltas = [0.01, 0.02, -0.005, 0.015, 0.004]
    result = paired_t_test(deltas)
    expected = scipy_stats.ttest_1samp(deltas, 0.0)
    assert result.m == 5
    assert result.df == 4
    assert result.t_stat == pytest.approx(float(expected.statistic))
    assert result.p_value == pytest.approx(float(expected.pvalue))
    assert result.se == pytest.approx(float(np.std(deltas, ddof=1) / np.sqrt(5)))


def test_paired_differences_pairs_on_dataset_seed() -> None:
    """Only seeds where both arms produced a value contribute."""
    rows = [
        {"arm": "x", "scenario": "s", "dataset_seed": 0, "value": 0.80},
        {"arm": NULL_ARM, "scenario": "s", "dataset_seed": 0, "value": 0.75},
        {"arm": "x", "scenario": "s", "dataset_seed": 1, "value": 0.70},
    ]
    assert paired_differences(rows, arm="x", null_arm=NULL_ARM, scenario="s") == pytest.approx([0.05])


def test_row_level_bootstrap_is_labelled_descriptive_only() -> None:
    """It returns p<1e-6 for every effect of interest; it may never drive a verdict."""
    ci = row_level_descriptive_ci(np.random.default_rng(0).normal(0.01, 0.1, 500), n_boot=200)
    assert ci.descriptive_only is True
    assert ci.low < ci.high


def test_intention_to_treat_charges_a_crashed_cell_the_base_rate() -> None:
    """Complete-case averaging over a grid whose hard scenarios kill weak arms is survivorship bias."""
    assert intention_to_treat_mean([0.9, None], base_rate_value=0.5) == pytest.approx(0.7)
    assert intention_to_treat_mean([0.9, 0.9], base_rate_value=0.5) == pytest.approx(0.9)


def test_reliability_reports_the_completed_fraction_and_the_statuses() -> None:
    """Reliability is per arm and scenario, with the failure kinds kept apart."""
    out = reliability(["ok", "ok", "oom", "error"])
    assert out["reliability"] == pytest.approx(0.5)
    assert out["by_status"] == {"error": 1, "ok": 2, "oom": 1}


# ------------------------------------------------------------------ decision rule
def _record(arm: str, scenario: str, seed: int, values: Any, status: str = "ok", k_label: str = "1k") -> Any:
    """Build one minimal cell record with the given per-model scores."""
    return {
        "arm": arm,
        "scenario": scenario,
        "dataset_seed": seed,
        "cv_seed": 0,
        "status": status,
        "n_model_fits": 2,
        "wall_time_s": 1.0,
        "scores": {k_label: {"models": {model: {"roc_auc": val} for model, val in values.items()}}},
    }


def test_scenario_with_no_arm_clearing_the_null_reports_fs_does_not_pay() -> None:
    """The modal outcome gets its own leaderboard row instead of an empty table."""
    jitter = [0.000, 0.003, -0.002, 0.001, -0.001, 0.002]
    records = []
    for seed in range(6):
        records.append(_record(NULL_ARM, "bed", seed, {"lightgbm": 0.870 + 0.001 * seed}))
        records.append(_record("mrmr", "bed", seed, {"lightgbm": 0.860 + 0.001 * seed + jitter[seed]}))
    rows = extract_long_rows(records, model="lightgbm", k_label="1k")
    verdict = scenario_verdict(rows, "bed", "lightgbm", "1k")
    assert verdict.headline == NO_PAY_ROW
    assert verdict.fs_pays() is False
    assert verdict.arms[0].verdict == "loses_to_null"


def test_an_arm_beating_the_null_on_every_seed_wins_the_headline() -> None:
    """A consistent paired gain must be picked up as `beats_null`."""
    jitter = [0.000, 0.003, -0.002, 0.001, -0.001, -0.001]
    records = []
    for seed in range(6):
        records.append(_record(NULL_ARM, "bed", seed, {"lightgbm": 0.870 + 0.002 * seed}))
        records.append(_record("noise_floor", "bed", seed, {"lightgbm": 0.940 + 0.002 * seed + jitter[seed]}))
    rows = extract_long_rows(records, model="lightgbm", k_label="1k")
    verdict = scenario_verdict(rows, "bed", "lightgbm", "1k")
    assert verdict.headline == "noise_floor"
    assert verdict.arms[0].stat.mean_delta == pytest.approx(0.07, abs=0.002)


def test_selector_by_model_interaction_is_flagged() -> None:
    """An arm winning for the linear model and losing for the GBM is a reported result."""
    jitter = [0.000, 0.003, -0.002, 0.001, -0.001, 0.002]
    records = []
    for seed in range(6):
        records.append(_record(NULL_ARM, "bed", seed, {"logistic": 0.70 + 0.002 * seed, "lightgbm": 0.90 + 0.002 * seed}))
        records.append(
            _record(
                "arm_x",
                "bed",
                seed,
                {"logistic": 0.80 + 0.002 * seed + jitter[seed], "lightgbm": 0.85 + 0.002 * seed + jitter[seed]},
            )
        )
    verdicts = leaderboard(records, models=("logistic", "lightgbm"), k_labels=("1k",))
    rows = selector_by_model_interaction(verdicts)
    flagged = [r for r in rows if r["arm"] == "arm_x"]
    assert len(flagged) == 1
    assert flagged[0]["interaction"] is True
    assert flagged[0]["verdict_by_model"] == {"logistic": "beats_null", "lightgbm": "loses_to_null"}


def test_reliability_and_cost_tables_include_failed_cells() -> None:
    """A failed cell writes a status and still shows up in the reliability denominator."""
    records = [
        _record("arm_x", "bed", 0, {"lightgbm": 0.8}),
        {"arm": "arm_x", "scenario": "bed", "dataset_seed": 1, "cv_seed": 0, "status": "oom", "n_model_fits": 0},
    ]
    rel = reliability_table(records)
    assert rel[0]["reliability"] == pytest.approx(0.5)
    assert rel[0]["by_status"]["oom"] == 1
    cost = cost_table(records)
    assert cost[0]["n_model_fits_total"] == pytest.approx(2.0)


def test_failed_cells_are_never_silently_dropped_from_the_file(tmp_path: Any) -> None:
    """Every cell, including a crash, produces exactly one JSONL object."""
    store = JsonlCellStore(tmp_path / "r.jsonl")
    store.append({"cell_key": "k1", "status": "ok"})
    store.append({"cell_key": "k2", "status": "crashed", "error": "boom"})
    text = (tmp_path / "r.jsonl").read_bytes().decode()
    assert len([line for line in text.splitlines() if line.strip()]) == 2
    assert json.loads(text.splitlines()[1])["status"] == "crashed"


def test_zero_variance_delta_is_not_reported_as_too_few_seeds() -> None:
    """An arm identical to the null on every seed has no t statistic but is a result, not an absence."""
    records = []
    for seed in range(5):
        records.append(_record(NULL_ARM, "bed", seed, {"lightgbm": 0.80 + 0.01 * seed}))
        records.append(_record("clone", "bed", seed, {"lightgbm": 0.80 + 0.01 * seed}))
    rows = extract_long_rows(records, model="lightgbm", k_label="1k")
    verdict = scenario_verdict(rows, "bed", "lightgbm", "1k")
    assert verdict.arms[0].verdict == "identical_to_null"
    assert verdict.headline == NO_PAY_ROW
