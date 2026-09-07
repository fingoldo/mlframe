"""One quality frame per run, and a fair verdict when two runs are compared.

Comparing runs that differ only in features or hyperparameters is not obvious when the suite trains
heterogeneous targets: NDCG@10 = 0.81 against RMSE = 3.2 has no common scale, and averaging them is not a
comparison. The properties tested here are the ones that make the aggregate defensible rather than merely
computable -- each is a way the naive version would quietly lie.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from mlframe.training.targets_performance import (
    AGGREGATE_ROW,
    DEFAULT_SPLIT,
    compare_targets_performance,
    targets_performance_frame,
)


def _entry(name: str, split: str = "test", **metrics):
    """A stand-in for a trained model entry: the suite reads ``.metrics[split][name]`` off exactly this."""
    return types.SimpleNamespace(model_name=name, metrics={split: dict(metrics)})


def _run(scale: float = 1.0):
    """Four target KINDS at once, which is the case a single scale cannot express."""
    models = {
        "learning_to_rank": {"clicks": [_entry("lgb", **{"NDCG@10": 0.80 * scale, "MAP": 0.60 * scale})]},
        "binary_classification": {
            "churn": [_entry("cb", ROC_AUC=0.70 * scale, log_loss=0.50 / scale), _entry("xgb", ROC_AUC=0.68 * scale, log_loss=0.55 / scale)]
        },
        "regression": {"revenue": [_entry("lgb", RMSE=3.0 / scale, MAE=2.0 / scale)]},
        "quantile_regression": {"latency": [_entry("cb", pinball=1.2 / scale)]},
    }
    metadata = {
        "dummy_baselines": {
            "learning_to_rank": {"clicks": {"primary_metric": "val_NDCG@10"}},
            "binary_classification": {"churn": {"primary_metric": "val_ROC_AUC"}},
            "regression": {"revenue": {"primary_metric": "val_RMSE"}},
            "quantile_regression": {"latency": {"primary_metric": "val_pinball"}},
        }
    }
    return models, metadata


# ---------------------------------------------------------------------------
# The per-run frame
# ---------------------------------------------------------------------------


def test_the_frame_has_one_row_per_target_sorted_by_name():
    """Sorted by NAME, not by kind: that is what makes two runs' frames line up for a diff."""
    frame = targets_performance_frame(*_run())
    targets = [t for t in frame["target_name"] if t != AGGREGATE_ROW]
    assert targets == sorted(targets) == ["churn", "clicks", "latency", "revenue"]


def test_the_frame_ends_with_exactly_one_aggregate_row():
    """A second aggregate row would be double-counted by anything that sums the frame."""
    frame = targets_performance_frame(*_run())
    assert list(frame["target_name"]).count(AGGREGATE_ROW) == 1
    assert frame["target_name"].iloc[-1] == AGGREGATE_ROW


def test_a_metric_that_does_not_apply_to_a_target_is_absent_not_zero():
    """An LtR target has no RMSE. Filling that with 0.0 would make it look like a perfect regression."""
    frame = targets_performance_frame(*_run()).set_index("target_name")
    assert np.isnan(frame.loc["clicks", "RMSE"]), "a metric the target never reported was filled in"
    assert frame.loc["clicks", "NDCG@10"] == pytest.approx(0.80)


def test_the_best_model_is_picked_by_the_targets_own_primary_metric_direction():
    """``churn`` has two models; higher ROC_AUC wins, and the row must carry that model's numbers."""
    frame = targets_performance_frame(*_run()).set_index("target_name")
    assert frame.loc["churn", "best_model"] == "cb"
    assert frame.loc["churn", "ROC_AUC"] == pytest.approx(0.70)
    assert frame.loc["churn", "n_models"] == 2


def test_the_aggregate_row_reports_coverage_not_a_quality_score():
    """A single run has nothing to normalise against, so a quality number there would be invented."""
    frame = targets_performance_frame(*_run())
    aggregate = frame[frame["target_name"] == AGGREGATE_ROW].iloc[0]
    assert np.isnan(aggregate["primary_value"]), "the aggregate row carries a quality value for one run"
    assert "metrics scoreable" in str(aggregate["primary_metric"])
    assert aggregate["n_models"] == 5  # 1 + 2 + 1 + 1


def test_the_frame_defaults_to_the_honest_split():
    """``val`` drove early stopping and is optimistically biased; ``test`` is the untouched estimate."""
    assert DEFAULT_SPLIT == "test"
    assert set(targets_performance_frame(*_run())["split"]) == {"test"}


def test_an_empty_run_produces_an_empty_frame_rather_than_raising():
    """A suite that trained nothing still has to return a table, not blow up at the last step."""
    assert targets_performance_frame({}, {}).empty


# ---------------------------------------------------------------------------
# The comparison, and the properties that make its aggregate fair
# ---------------------------------------------------------------------------


def test_the_better_run_wins_across_four_target_kinds():
    """The whole point: one verdict over LtR, binary, regression and quantile at once."""
    result = compare_targets_performance({"base": _run(1.0), "better": _run(1.05), "worse": _run(0.9)})
    assert result.winner == "better", f"{result.scores.to_dict('records')}"
    assert list(result.scores["run"]) == ["better", "base", "worse"], "runs are not ordered by score"


def test_lower_is_better_metrics_are_inverted_before_scoring():
    """Without the direction lookup, a run with a WORSE RMSE would score higher on it."""
    good = ({"regression": {"a": [_entry("m", RMSE=1.0)]}}, {})
    bad = ({"regression": {"a": [_entry("m", RMSE=9.0)]}}, {})
    result = compare_targets_performance({"good": good, "bad": bad})
    assert result.winner == "good"
    assert result.frame.set_index("run").loc["good", "higher_is_better"] is np.False_ or not result.frame.set_index("run").loc["good", "higher_is_better"]


def test_a_run_cannot_improve_its_score_by_reporting_an_extra_metric():
    """The comparison uses the INTERSECTION of metrics; otherwise adding a metric is a free win."""
    plain = ({"regression": {"a": [_entry("m", RMSE=3.0)]}, "binary_classification": {"b": [_entry("m", ROC_AUC=0.7)]}}, {})
    rich = ({"regression": {"a": [_entry("m", RMSE=3.0, MAE=1.0)]}, "binary_classification": {"b": [_entry("m", ROC_AUC=0.7)]}}, {})
    result = compare_targets_performance({"plain": plain, "rich": rich})
    assert result.winner is None and "tied" in result.reason, f"the extra metric moved the score: {result.scores.to_dict('records')}"


def test_a_metric_rich_target_cannot_outvote_a_metric_poor_one():
    """The second scaling: metrics collapse into one score per target BEFORE targets are averaged.

    Here one target reports eleven metrics and the other reports one. The eleven are identical across
    runs, so the single differing metric on the other target must decide the winner outright.
    """
    shared = {f"MAE_{i}": 1.0 for i in range(10)}
    good = ({"regression": {"a": [_entry("m", RMSE=3.0, **shared)]}, "binary_classification": {"b": [_entry("m", ROC_AUC=0.9)]}}, {})
    bad = ({"regression": {"a": [_entry("m", RMSE=3.0, **shared)]}, "binary_classification": {"b": [_entry("m", ROC_AUC=0.5)]}}, {})
    result = compare_targets_performance({"good_b": good, "bad_b": bad})
    assert result.winner == "good_b"
    gap = float(result.scores.set_index("run").loc["good_b", "score"] - result.scores.set_index("run").loc["bad_b", "score"])
    assert gap == pytest.approx(0.5), f"one of two targets flipping should move the score half a unit, moved {gap}"


def test_a_metric_with_an_unknown_direction_is_excluded_and_named():
    """Higher might be better or worse; scoring it either way is a guess, and a silent one is worse."""
    x = ({"regression": {"a": [_entry("m", RMSE=3.0, my_custom_score=42.0)]}}, {})
    y = ({"regression": {"a": [_entry("m", RMSE=4.0, my_custom_score=99.0)]}}, {})
    result = compare_targets_performance({"x": x, "y": y})
    assert result.excluded_metrics == ("my_custom_score",)
    assert sorted(set(result.frame["metric"])) == ["RMSE"], "the unscoreable metric reached the aggregate"
    assert result.winner == "x", "the known metric still decides"
    assert "my_custom_score" in targets_performance_frame(*x).columns, "the metric vanished from the frame as well"


def test_a_single_run_yields_no_winner_and_says_why():
    """Normalisation is across runs, so one run has no frame of reference at all."""
    result = compare_targets_performance({"only": _run()})
    assert result.winner is None
    assert "nothing to normalise against" in result.reason
    assert result.scores.empty


def test_runs_with_no_shared_target_yield_no_winner_and_say_why():
    """Comparing runs that trained different things is not a comparison."""
    a = ({"regression": {"alpha": [_entry("m", RMSE=1.0)]}}, {})
    b = ({"regression": {"beta": [_entry("m", RMSE=1.0)]}}, {})
    result = compare_targets_performance({"a": a, "b": b})
    assert result.winner is None and result.reason == "the runs share no target"


def test_only_the_shared_targets_are_scored():
    """A target one run never trained cannot count for or against either of them."""
    both = {"regression": {"shared": [_entry("m", RMSE=2.0)]}}
    extra = {"regression": {"shared": [_entry("m", RMSE=3.0)], "extra": [_entry("m", RMSE=0.1)]}}
    result = compare_targets_performance({"small": (both, {}), "big": (extra, {})})
    assert result.common_targets == (("regression", "shared"),)
    assert result.winner == "small", "the unshared target leaked into the verdict"


def test_identical_runs_tie_rather_than_picking_one():
    """Breaking a genuine tie by argmax order would report a difference that is not there."""
    result = compare_targets_performance({"a": _run(1.0), "b": _run(1.0)})
    assert result.winner is None and "tied" in result.reason


@pytest.mark.parametrize("normalisation", ["rank", "minmax"])
def test_both_normalisations_agree_on_a_clear_winner(normalisation):
    """They weigh gaps differently; they must not disagree about who is ahead on a monotone case."""
    result = compare_targets_performance({"base": _run(1.0), "better": _run(1.2)}, normalisation=normalisation)
    assert result.winner == "better"


def test_an_unknown_normalisation_is_rejected():
    """A typo must not silently fall back to a scaling the caller did not choose."""
    with pytest.raises(ValueError, match="normalisation must be one of"):
        compare_targets_performance({"a": _run(), "b": _run(1.1)}, normalisation="magic")


def test_minmax_reflects_the_size_of_the_gap_where_rank_does_not():
    """The reason both exist: rank says only "ahead", minmax says "ahead by this much"."""
    runs = {
        "low": ({"regression": {"a": [_entry("m", RMSE=10.0)]}}, {}),
        "mid": ({"regression": {"a": [_entry("m", RMSE=9.9)]}}, {}),
        "high": ({"regression": {"a": [_entry("m", RMSE=1.0)]}}, {}),
    }
    ranked = compare_targets_performance(runs, normalisation="rank").scores.set_index("run")["score"]
    scaled = compare_targets_performance(runs, normalisation="minmax").scores.set_index("run")["score"]
    assert ranked["mid"] == pytest.approx(0.5), "rank should place the middle run midway regardless of the gap"
    assert scaled["mid"] < 0.05, f"minmax should show mid as nearly as bad as low, got {scaled['mid']}"


# ---------------------------------------------------------------------------
# Suite integration
# ---------------------------------------------------------------------------


def test_the_suite_phase_stores_the_frame_on_the_metadata():
    """Every run has to leave the frame behind, or there is nothing to compare later."""
    from mlframe.training.core._phase_targets_performance import render_targets_performance

    models, metadata = _run()
    render_targets_performance(models, metadata)
    frame = metadata["targets_performance"]
    assert isinstance(frame, pd.DataFrame) and len(frame) == 5  # four targets + the aggregate row


def test_the_suite_phase_never_raises_on_a_malformed_run():
    """Diagnostic, never load-bearing: a broken entry must not take the suite down at the last step."""
    from mlframe.training.core._phase_targets_performance import render_targets_performance

    metadata: dict = {}
    render_targets_performance({"regression": {"a": [object()]}}, metadata)
    assert "targets_performance" in metadata, "the phase bailed without leaving anything behind"
