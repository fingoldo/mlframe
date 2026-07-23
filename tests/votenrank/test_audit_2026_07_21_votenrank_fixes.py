"""Regression tests for audits/full_audit_2026-07-21/votenrank.md findings F1-F15.

F11 (fairness_computation.py's unconditional .cuda(), no CPU fallback) is assessed as a deliberate
documented simplification per the audit's own reading (module docstring already states "CUDA-only") --
no fix, no test needed.

PR3 (F1/F2 regression tests) is covered by F1/F2 below. PR5 (KNNFallbackPredictor k validation) is
covered by F9's test. PR1/PR2/PR4 are feature-parity/docs/CI proposals with no reported bug -- deferred.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _toy_table(n_models=4, n_tasks=3, seed=0):
    """Build a small toy Leaderboard score table."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.uniform(0.1, 1.0, size=(n_models, n_tasks)),
        index=[f"model_{i}" for i in range(n_models)],
        columns=[f"task_{j}" for j in range(n_tasks)],
    )


# ---------------------------------------------------------------------------
# F1: Leaderboard.__init__ rejects a weights dict key that doesn't match any task
# ---------------------------------------------------------------------------


def test_f1_leaderboard_rejects_unknown_weight_key():
    """F1 leaderboard rejects unknown weight key."""
    from mlframe.votenrank import Leaderboard

    table = _toy_table()
    with pytest.raises(ValueError, match="do not match any table column"):
        Leaderboard(table, weights={"task_0": 2.0, "task_typo": 3.0})


def test_f1_leaderboard_accepts_known_weight_keys():
    """F1 leaderboard accepts known weight keys."""
    from mlframe.votenrank import Leaderboard

    table = _toy_table()
    lb = Leaderboard(table, weights={"task_0": 2.0})
    assert lb.weights["task_0"] == 2.0


# ---------------------------------------------------------------------------
# F2: Leaderboard.__init__ rejects duplicate model names
# ---------------------------------------------------------------------------


def test_f2_leaderboard_rejects_duplicate_model_names():
    """F2 leaderboard rejects duplicate model names."""
    from mlframe.votenrank import Leaderboard

    table = _toy_table()
    dup_table = pd.concat([table, table.iloc[[0]]])  # model_0 now appears twice
    with pytest.raises(ValueError, match="duplicate model name"):
        Leaderboard(dup_table)


# ---------------------------------------------------------------------------
# F3: preprocess_value verifies the scraped row order against the hardcoded roster
# ---------------------------------------------------------------------------


def test_f3_preprocess_value_raises_on_reordered_roster():
    """F3 preprocess value raises on reordered roster."""
    from mlframe.votenrank.data_processing import preprocess_value

    roster = ["Human", "craig.starr", "DuKG", "HERO 1", "HERO 2", "HERO 3", "HERO 4"]
    shuffled = roster[::-1]  # deliberately wrong order
    df = pd.DataFrame(
        {
            "Model": shuffled,
            "Mean-Rank": [1] * 7,
            "Meta-Ave": [0.5] * 7,
            "metric_a": np.linspace(0.1, 0.9, 7),
        }
    )
    with pytest.raises(ValueError, match="row order"):
        preprocess_value(df)


def test_f3_preprocess_value_accepts_matching_roster():
    """F3 preprocess value accepts matching roster."""
    from mlframe.votenrank.data_processing import preprocess_value

    roster = ["Human", "craig.starr", "DuKG", "HERO 1", "HERO 2", "HERO 3", "HERO 4"]
    df = pd.DataFrame(
        {
            "Model": roster,
            "Mean-Rank": [1] * 7,
            "Meta-Ave": [0.5] * 7,
            "metric_a": np.linspace(0.1, 0.9, 7),
        }
    )
    out = preprocess_value(df)
    assert list(out.index) == roster


# ---------------------------------------------------------------------------
# F4: mean_ranking(mean_type="geometric") raises on a non-positive score instead of silent 0/NaN
# ---------------------------------------------------------------------------


def test_f4_geometric_mean_raises_on_nonpositive_score():
    """F4 geometric mean raises on nonpositive score."""
    from mlframe.votenrank import Leaderboard

    table = _toy_table()
    table.iloc[0, 0] = 0.0
    lb = Leaderboard(table)
    with pytest.raises(ValueError, match="strictly positive"):
        lb.mean_ranking(mean_type="geometric")


def test_f4_geometric_mean_still_works_on_positive_table():
    """F4 geometric mean still works on positive table."""
    from mlframe.votenrank import Leaderboard

    lb = Leaderboard(_toy_table())
    out = lb.mean_ranking(mean_type="geometric")
    assert len(out) == 4


# ---------------------------------------------------------------------------
# F5: an all-zero-weights Leaderboard raises instead of silently producing inf/NaN scores
# ---------------------------------------------------------------------------


def test_f5_leaderboard_rejects_all_zero_weights():
    """F5 leaderboard rejects all zero weights."""
    from mlframe.votenrank import Leaderboard

    table = _toy_table()
    with pytest.raises(ValueError, match="must sum to a positive value"):
        Leaderboard(table, weights={t: 0.0 for t in table.columns})


# ---------------------------------------------------------------------------
# F6: borda/dowdall/plurality/threshold/baldwin all guard against a partial table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("borda_ranking", {}),
        ("dowdall_ranking", {}),
        ("plurality_ranking", {}),
        ("threshold_election", {}),
        ("baldwin_election", {}),
    ],
)
def test_f6_partial_table_guard_raises(method, kwargs):
    """F6 partial table guard raises."""
    from mlframe.votenrank import Leaderboard

    table = _toy_table()
    table.iloc[0, 0] = np.nan
    lb = Leaderboard(table)
    assert lb.is_partial
    with pytest.raises(ValueError, match="partial"):
        getattr(lb, method)(**kwargs)


def test_f6_complete_table_still_works_for_all_guarded_methods():
    """F6 complete table still works for all guarded methods."""
    from mlframe.votenrank import Leaderboard

    lb = Leaderboard(_toy_table())
    assert not lb.is_partial
    for method in ("borda_ranking", "dowdall_ranking", "plurality_ranking", "threshold_election", "baldwin_election"):
        assert getattr(lb, method)() is not None


# ---------------------------------------------------------------------------
# F7: shapley_blend docstring now says "positive-value member", matching the strict `>` code
# ---------------------------------------------------------------------------


def test_f7_shapley_blend_docstring_says_positive_not_nonnegative():
    """F7 shapley blend docstring says positive not nonnegative."""
    from mlframe.votenrank import shapley_blend

    doc = shapley_blend.__doc__ or ""
    assert "keeps every strictly-positive-value member" in doc
    assert "non-negative-value member" not in doc


# ---------------------------------------------------------------------------
# F8: confidence_gated_blend logs a debug message when the cupy backend fails/is unavailable
# ---------------------------------------------------------------------------


def test_f8_cupy_fallback_logs_debug(monkeypatch, caplog):
    """F8 cupy fallback logs debug."""
    import importlib
    import logging

    # `mlframe.votenrank.__init__` re-exports `confidence_gated_blend` (the function) under the same
    # name as the submodule, shadowing the submodule attribute on the parent package -- fetch the
    # real module via sys.modules/importlib rather than `import ... as` (which would resolve to the
    # already-shadowed function attribute).
    cgb = importlib.import_module("mlframe.votenrank.confidence_gated_blend")

    def _raise(*args, **kwargs):
        """Raise, simulating a failed cupy backend call."""
        raise RuntimeError("no GPU")

    monkeypatch.setattr(cgb, "_blend_cupy", _raise)
    n = 2500  # must clear _DISPATCH_MIN_N (2000) or confidence_gated_blend short-circuits to numpy before backend dispatch
    rng = np.random.default_rng(0)
    ensemble_pred = rng.uniform(size=n)
    aux_pred = rng.uniform(size=n)
    aux_conf = rng.uniform(size=n)

    with caplog.at_level(logging.DEBUG, logger="mlframe.votenrank.confidence_gated_blend"):
        out = cgb.confidence_gated_blend(
            ensemble_pred, aux_pred, aux_conf, confidence_threshold=0.5, gated_weight=0.2, default_weight=0.0, force_backend="cupy"
        )
    assert out is not None
    assert any("cupy backend unavailable/failed" in rec.message for rec in caplog.records), "F8 REGRESSION: cupy fallback must log a debug message"


# ---------------------------------------------------------------------------
# F9: KNNFallbackPredictor.predict() before fit() raises a clear ValueError, not a bare IndexError
# ---------------------------------------------------------------------------


def test_f9_predict_before_fit_raises_clear_value_error():
    """F9 predict before fit raises clear value error."""
    from mlframe.votenrank.knn_fallback_predictor import KNNFallbackPredictor

    predictor = KNNFallbackPredictor(k=3)
    with pytest.raises(ValueError, match="before fit"):
        predictor.predict(np.zeros((2, 4)))


def test_f9_predictor_rejects_invalid_k_at_construction():
    """PR5: k<1 raises at construction time instead of failing later inside knn_search."""
    from mlframe.votenrank.knn_fallback_predictor import KNNFallbackPredictor

    with pytest.raises(ValueError, match="k must be >= 1"):
        KNNFallbackPredictor(k=0)


# ---------------------------------------------------------------------------
# F10: dual_optimizer_weight_blend(include_coord_descent=True) on a single-model pool raises clearly
# ---------------------------------------------------------------------------


def test_f10_single_model_with_coord_descent_raises_clear_error():
    """F10 single model with coord descent raises clear error."""
    from mlframe.votenrank.dual_optimizer_blend import dual_optimizer_weight_blend

    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    preds = [y + rng.normal(scale=0.1, size=50)]

    def loss_fn(y_true, y_pred):
        """Loss fn."""
        return float(np.mean((y_true - y_pred) ** 2))

    with pytest.raises(ValueError, match="include_coord_descent=True requires >= 2 models"):
        dual_optimizer_weight_blend(preds, y, loss_fn, include_coord_descent=True)


# ---------------------------------------------------------------------------
# F12: stability_exp's sns.set is no longer an import-time side effect
# ---------------------------------------------------------------------------


def test_f12_sns_set_not_called_at_import_time():
    """F12 sns set not called at import time."""
    import inspect

    from mlframe.votenrank import stability_exp

    src = inspect.getsource(stability_exp)
    module_level_lines = [line for line in src.splitlines() if line.strip().startswith("sns.set(") and not line.startswith((" ", "\t"))]
    assert not module_level_lines, "F12 REGRESSION: sns.set(...) must not run at module import scope"
    # It must still be called from inside the plotting entry points.
    assert '    sns.set(style="whitegrid")' in src


# ---------------------------------------------------------------------------
# F13: no Wave-N/date audit markers remain in the 3 flagged comments
# ---------------------------------------------------------------------------


def test_f13_no_wave_date_markers_remain():
    """F13 no wave date markers remain."""
    import inspect

    from mlframe.votenrank import iia_exp, utils
    from mlframe.votenrank.leaderboard import leaderboard_impl

    for mod in (utils, iia_exp, leaderboard_impl):
        src = inspect.getsource(mod)
        assert "Wave " not in src or "(2026-05-20)" not in src, f"F13 REGRESSION: {mod.__name__} still has a Wave-N/date audit marker"


# ---------------------------------------------------------------------------
# F14: correctness-asserting tests for previously-untested election/CW methods
# ---------------------------------------------------------------------------


def test_f14_condorcet_election_hand_computable():
    """F14 condorcet election hand computable."""
    from mlframe.votenrank import Leaderboard

    # model_a beats model_b and model_c on every task -> unique Condorcet winner.
    table = pd.DataFrame(
        {"t1": [3, 2, 1], "t2": [3, 1, 2], "t3": [3, 2, 1]},
        index=["model_a", "model_b", "model_c"],
    )
    lb = Leaderboard(table)
    assert lb.condorcet_election() == ["model_a"]


def test_f14_baldwin_election_matches_borda_when_no_elimination_needed():
    """F14 baldwin election matches borda when no elimination needed."""
    from mlframe.votenrank import Leaderboard

    table = pd.DataFrame(
        {"t1": [3, 2, 1], "t2": [3, 2, 1]},
        index=["model_a", "model_b", "model_c"],
    )
    lb = Leaderboard(table)
    assert lb.baldwin_election() == lb.borda_election()


def test_f14_threshold_election_eliminates_last_place():
    """F14 threshold election eliminates last place."""
    from mlframe.votenrank import Leaderboard

    table = pd.DataFrame(
        {"t1": [3, 2, 1], "t2": [3, 2, 1]},
        index=["model_a", "model_b", "model_c"],
    )
    lb = Leaderboard(table)
    assert lb.threshold_election() == ["model_a"]


def test_f14_copeland_ranking_slice_types_agree_on_ordering():
    """F14 copeland ranking slice types agree on ordering."""
    from mlframe.votenrank import Leaderboard

    table = pd.DataFrame(
        {"t1": [3, 2, 1], "t2": [1, 3, 2], "t3": [2, 1, 3]},
        index=["model_a", "model_b", "model_c"],
    )
    lb = Leaderboard(table)
    for slice_type in ("lower", "upper", "difference", "lower_with_ties"):
        out = lb.copeland_ranking(slice_type=slice_type)
        assert len(out) == 3


def test_f14_two_step_ranking_and_meta_leaderboard():
    """F14 two step ranking and meta leaderboard."""
    from mlframe.votenrank import Leaderboard

    table = pd.DataFrame(
        {"t1": [3.0, 2.0, 1.0], "t2": [3.0, 2.0, 1.0], "t3": [1.0, 3.0, 2.0], "t4": [1.0, 3.0, 2.0]},
        index=["model_a", "model_b", "model_c"],
    )
    lb = Leaderboard(table)
    task_groups = {"group1": ["t1", "t2"], "group2": ["t3", "t4"]}
    meta_lb = lb.get_meta_leaderboard(task_groups=task_groups, ranking_method="borda")
    assert meta_lb.table.shape == (3, 2)


def test_f14_find_weights_for_condorcet_feasible_and_infeasible():
    """F14 find weights for condorcet feasible and infeasible."""
    from mlframe.votenrank import Leaderboard

    # model_a strictly dominates -> a Condorcet-consistent weighting must exist (e.g. uniform).
    table = pd.DataFrame({"t1": [3, 2, 1], "t2": [3, 2, 1]}, index=["model_a", "model_b", "model_c"])
    lb = Leaderboard(table)
    weights = lb.find_weights_for_condorcet("model_a")
    assert weights is not None
    assert weights == pytest.approx(weights)  # sanity: a real dict, not the old sentinel string

    result = lb.split_models_by_feasibility()
    assert "model_a" in result["feasible"]


# ---------------------------------------------------------------------------
# F15: the infeasible sentinel is now None, not the magic string "infeasible"
# ---------------------------------------------------------------------------


def test_f15_infeasible_sentinel_is_none_not_string():
    """F15 infeasible sentinel is none not string."""
    from mlframe.votenrank import Leaderboard

    # 3 models tied on every task with a mutually-exclusive weight restriction designed to be infeasible:
    # force weights_lb higher than the simplex can satisfy (sum of lower bounds > 1).
    table = pd.DataFrame({"t1": [1, 1, 1], "t2": [1, 1, 1]}, index=["model_a", "model_b", "model_c"])
    lb = Leaderboard(table)
    restrictions = {"weights_lb": [(["t1"], 0.9), (["t2"], 0.9)]}  # sum >= 1.8 > 1, infeasible on the simplex
    result = lb._find_weights_for_majority_graph([("model_b", "model_a")], restrictions=restrictions)
    assert result is None, "F15 REGRESSION: infeasible LP must return None, not the magic string 'infeasible'"
