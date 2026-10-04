"""Wave 58 (2026-05-20): stable-sort tie non-determinism — cluster follow-up
to wave 57. Closes the remaining 10 P1 sites in the same audit class.

Same bug shape as wave 57: sorted/argsort with ties on score silently flips
output across runs when input order differs. This commit closes the per-
filter / per-helper sites that wave 57's commit deferred:

  1. feature_selection/filters/hermite_fe.py:1390 (history sort)
     Secondary key on bf_idx -- kept[0] no longer drifts.

  2. feature_selection/filters/hermite_fe.py:2237 (results.sort)
     Secondary key on (deg_a, deg_b, bf_name).

  3. feature_selection/filters/composition.py:80 (single_mi)
     Secondary key on feature index.

  4. feature_selection/filters/composition.py:99 (pair_scores)
     Secondary key on (i, j).

  5. feature_selection/filters/fe_baselines.py:140 + :226 (trivial / triplet
     baselines)
     Secondary key on name; next(iter(...)) winner now deterministic.

  6. feature_selection/wrappers/_helpers.py:196 (knockoff FDR selected)
     Secondary key on feature name; tied |W| no longer drifts downstream
     [:topN] slicing.

  7. feature_selection/filters/estimators.py:173 (perm-MI significant order)
     np.lexsort with significant-index tiebreak.

  8. training/baseline_diagnostics.py:602 (ablation top-K by raw_fi)
     np.lexsort with feature-index tiebreak.

  9. feature_engineering/numerical.py:692 (top-N modes)
     np.lexsort with value as tiebreak; tied counts -> deterministic mode.

  10. feature_selection/filters/cat_interactions.py:2505 (top-K pairs)
      np.lexsort replaces argpartition (impl-defined tie); deterministic.

  11. feature_selection/filters/cat_interactions.py:3137 (k-way results)
      Secondary key on var-index tuple.

  12. evaluation/reports.py:446 (precision@top-decile)
      np.lexsort with row-position tiebreak.

Total post-wave-57+58: all 25 wave-57 audit sites are closed. FE-transformer
top-K row-pickers (active_virtual / pseudo_smote / tree_path_boolean / etc.)
follow the same pattern but operate on research-grade transformers; left as
documented known pattern -- the user-facing impact is per-feature drift not
suite-level selection drift.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_hermite_history_uses_secondary_bf_idx() -> None:
    """Tied top scores in the Hermite history resolve to the lowest bf_idx whatever the iteration order.

    Without the secondary key a stable sort keeps input order, so ``kept[0]`` follows the order the history
    was accumulated in.
    """
    import numpy as np

    from mlframe.feature_selection.filters._hermite_fe_diverse import _select_diverse_topm

    def _entry(score, bf_idx, a, b):
        """One (score, raw_mi, bf_idx, coef_a, coef_b) history tuple."""
        return (score, score, bf_idx, np.array(a, dtype=np.float64), np.array(b, dtype=np.float64))

    history = [_entry(0.5, 7, [1.0, 0.0], [0.0, 1.0]), _entry(0.5, 2, [0.0, 1.0], [1.0, 0.0]), _entry(0.5, 4, [1.0, 1.0], [0.0, 0.0]), _entry(0.1, 0, [0.3, 0.2], [0.1, 0.0])]
    for order in (history, history[::-1], [history[2], history[0], history[3], history[1]]):
        kept = _select_diverse_topm(list(order), top_m=1)
        assert [e[2] for e in kept] == [2], f"tied top-score winner depends on history order: {[e[2] for e in order]}"


def test_hermite_results_uses_secondary_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tied-MI Hermite results come back ordered by bin-function name whatever order the bin functions were registered in."""
    from mlframe.feature_selection.filters import _hermite_fe_optimise as mod

    coef_len = 3
    history = [
        (0.5, 0.5, 0, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        (0.5, 0.5, 1, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])),
    ]
    assert len(history[0][3]) == coef_len
    monkeypatch.setattr(mod, "_run_cma_search", lambda **_kw: (None, None, None, None, None, list(history)))
    monkeypatch.setattr(mod, "_baseline_mi_pair", lambda *_a, **_kw: 0.0)
    rng = np.random.default_rng(0)
    x_a, x_b = rng.normal(size=400), rng.normal(size=400)
    y = (x_a > 0).astype(np.int64)

    def _bf(a, b):
        """Elementwise sum of the two polynomial legs."""
        return a + b

    results = mod.optimise_pair_multimode(x_a, x_b, y, top_m=2, min_l2_distance=0.1, bin_funcs={"zeta": _bf, "alpha": _bf}, sweep_degrees=False, max_degree=2)
    assert [r.bin_func_name for r in results] == ["alpha", "zeta"]


def _compose_pair_selection(monkeypatch: pytest.MonkeyPatch, column_mi: dict[int, float], n_cols: int, top_k: int) -> list[tuple[int, int]]:
    """Run one compose_pair_fe round with a fake MI estimator and return the pairs it selected."""
    from mlframe.feature_selection.filters import composition, fe_baselines, hermite_fe

    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, n_cols))
    key_to_col = {round(float(X[0, j]), 9): j for j in range(n_cols)}

    def _fake_mi(x, y, **_kw):
        """Per-column MI from the lookup table; any derived (pair) feature scores a constant."""
        col = key_to_col.get(round(float(x[0]), 9))
        return column_mi[col] if col is not None else 0.3

    monkeypatch.setattr(fe_baselines, "_mi_1d", _fake_mi)
    monkeypatch.setattr(hermite_fe, "optimise_hermite_pair", lambda *_a, **_kw: None)
    out = composition.compose_pair_fe(X, rng.integers(0, 2, 300), n_rounds=1, top_k_per_round=top_k)
    assert len(out["rounds"]) == 1
    return out["rounds"][0]["selected_pairs"]


def test_composition_single_mi_secondary_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Features tied on single-feature MI enter the pair pool lowest column index first, and a weaker column stays out."""
    pairs = _compose_pair_selection(monkeypatch, {0: 0.1, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}, n_cols=5, top_k=2)
    assert pairs == [(1, 2), (1, 3)]


def test_composition_pair_scores_secondary_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pairs tied on joint MI are selected in ascending (i, j) order."""
    pairs = _compose_pair_selection(monkeypatch, {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}, n_cols=4, top_k=2)
    assert pairs == [(0, 1), (0, 2)]


def test_fe_baselines_secondary_key_on_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both baseline rankers list tied-MI features in name order, so the first key is the alphabetically smallest."""
    from mlframe.feature_selection.filters import fe_baselines
    from mlframe.feature_selection.filters.hermite_fe import shared

    rng = np.random.default_rng(3)
    x_a, x_b, x_c = (rng.uniform(1.0, 2.0, 200) for _ in range(3))
    y = rng.integers(0, 2, 200)

    monkeypatch.setattr(fe_baselines, "_mi_1d", lambda *_a, **_kw: 0.25)
    triplet = fe_baselines.score_triplet_baselines(x_a, x_b, x_c, y)
    assert len(triplet) > 1
    assert list(triplet) == sorted(triplet)

    monkeypatch.setattr(shared, "plugin_mi_classif_batch_njit", lambda X, *_a, **_kw: np.full(X.shape[1], 0.25))
    trivial = fe_baselines.score_trivial_baselines(x_a, x_b, y)
    assert len(trivial) > 1
    assert list(trivial) == sorted(trivial)


def test_knockoff_helper_secondary_key_on_name() -> None:
    """Features with identical W statistics are selected in name order, independent of the dict's insertion order."""
    from mlframe.feature_selection.wrappers._knockoffs import select_features_fdr

    names = ["zeta", "mid", "alpha", "beta", "omega"]
    W = {n: 2.0 for n in names}
    W["noise"] = -0.01
    assert select_features_fdr(W, q=0.5) == sorted(names)
    assert select_features_fdr(dict(reversed(list(W.items()))), q=0.5) == sorted(names)


def test_estimators_perm_mi_uses_lexsort(monkeypatch: pytest.MonkeyPatch) -> None:
    """Significant features are ordered by MI descending, with tied MIs broken by feature index."""
    from mlframe.feature_selection.filters import estimators

    observed = np.array([0.5, 0.9, 0.5, 0.9, 0.0])
    calls = {"n": 0}

    def _fake_ksg(X, y, feature_indices, **_kw):
        """First call is the observed MI; the permutation-null calls score zero."""
        calls["n"] += 1
        return observed.copy() if calls["n"] == 1 else np.zeros(len(feature_indices))

    monkeypatch.setattr(estimators, "ksg_mi_with_target", _fake_ksg)
    X = np.random.default_rng(0).normal(size=(60, 5))
    y = np.random.default_rng(1).integers(0, 2, 60)
    mi, p, support = estimators.ksg_mi_with_significance(X, y, list(range(5)), n_permutations=40, alpha=0.1, n_jobs=1)
    assert np.array_equal(mi, observed)
    assert p[4] > 0.1
    assert list(support) == [1, 3, 0, 2]


def test_baseline_diagnostics_ablation_uses_lexsort() -> None:
    """With every feature importance tied, the ablation drops the first top_k features by column position."""
    from types import SimpleNamespace

    from mlframe.training.baselines._baseline_diagnostics_ablation import _run_ablation

    import pandas as pd

    cols = ["f3", "f2", "f1", "f0"]
    X = pd.DataFrame(np.zeros((8, 4)), columns=cols)
    dropped: list[str] = []

    def _fit_quick_and_score(X_drop, y, kept, cat_kept, target_type, metric_name, inner_n_jobs):
        """Record which feature the ablation left out and report a fixed metric."""
        dropped.append(next(c for c in cols if c not in kept))
        return 0.5, None

    owner = SimpleNamespace(config=SimpleNamespace(ablation_top_k=2), _fit_quick_and_score=_fit_quick_and_score)
    entries = _run_ablation(owner, X, np.zeros(8), cols, [], "binary", np.full(4, 0.5), 0.6, "auc", True)
    assert sorted(dropped) == ["f2", "f3"]
    assert [e.feature for e in entries] == ["f3", "f2"]


def test_numerical_top_modes_uses_lexsort() -> None:
    """With max_modes=1 and tied counts the smallest tied value is the reported mode, on both the NaN-aware and the fused path."""
    from mlframe.feature_engineering.numerical import compute_nunique_modes_quantiles_numpy

    arr = np.array([3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 5.0])
    for data in (np.append(arr, np.nan), arr):
        res = compute_nunique_modes_quantiles_numpy(data, max_modes=1)
        modes_min, modes_max, modes_qty = res[1], res[2], res[4]
        assert (modes_min, modes_max, modes_qty) == (1.0, 1.0, 1)


def test_cat_interactions_topk_uses_lexsort() -> None:
    """Pairs tied on score are kept lowest-index first and a weaker eligible pair is cut."""
    from mlframe.feature_selection.filters._cat_kway_materialize import _select_top_k_pairs
    from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig

    cfg = CatFEConfig(enable=True, top_k_pairs=2, min_interaction_information=0.0, select_on="synergy")
    ii = np.array([0.1, 0.5, 0.5, 0.5, 0.5])
    idx = _select_top_k_pairs(ii, np.arange(5), np.arange(5) + 1, cfg, n_samples=1000)
    assert list(idx) == [1, 2]


def test_cat_interactions_kway_secondary_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Of many k-way candidates tied on joint MI, the lexicographically smallest index tuple survives the top_k cut."""
    from itertools import combinations

    from mlframe.feature_selection.filters import cat_interactions
    from mlframe.feature_selection.filters._cat_interactions_step import run_cat_interaction_step
    from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig
    from mlframe.feature_selection.filters.info_theory import merge_vars

    rng = np.random.default_rng(13)
    n = 2500
    x = [rng.integers(0, 2, n).astype(np.int32) for _ in range(3)]
    y = (x[0] ^ x[1] ^ x[2]).astype(np.int32)
    noise = [rng.integers(0, 4, n).astype(np.int32) for _ in range(2)]
    data = np.column_stack([*x, *noise, y]).astype(np.int32)
    nbins = np.array([2, 2, 2, 4, 4, 2], dtype=np.int64)
    cols = ["x1", "x2", "x3", "n0", "n1", "y"]
    cls_y, fq_y, _ = merge_vars(factors_data=data, vars_indices=np.array([5], dtype=np.int64), var_is_nominal=None, factors_nbins=nbins, dtype=np.int32)

    triples = sorted(combinations(range(5), 3))
    state = {"calls": 0}

    def _fake_expand(*, factors_data, seed_indices, **_kw):
        """First seed batch yields nothing so the all-pairs fallback runs; later seeds yield tied triples in descending tuple order."""
        state["calls"] += 1
        if state["calls"] == 1:
            return None
        idx_tuple = triples[len(triples) - 1 - ((state["calls"] - 2) % len(triples))]
        classes, _, n_uniq = merge_vars(factors_data=factors_data, vars_indices=np.array(idx_tuple, dtype=np.int64), var_is_nominal=None, factors_nbins=nbins, dtype=np.int32)
        return idx_tuple, classes, int(n_uniq), 0.3

    monkeypatch.setattr(cat_interactions, "_greedy_expand_one_seed", _fake_expand)
    cfg = CatFEConfig(enable=True, top_k_pairs=1, max_kway_order=3, min_interaction_information=-0.05, full_npermutations=0, fwer_correction="none")
    _, _, _, st = run_cat_interaction_step(
        data=data, cols=cols, nbins=nbins, target_indices=np.array([5], dtype=np.int64), classes_y=cls_y, classes_y_safe=cls_y, freqs_y=fq_y,
        categorical_vars=[0, 1, 2, 3, 4], cfg=cfg, dtype=np.int32,
    )
    assert state["calls"] > 2
    kway = [r for r in st.recipes if r.extra.get("kway_order") == 3]
    assert [r.src_names for r in kway] == [("x1", "x2", "x3")]


def test_evaluation_reports_precision_at_decile_uses_lexsort() -> None:
    """With all predictions tied the top decile is the first rows by position, whatever the labels there are."""
    from mlframe.evaluation.reports import _precision_at_top_decile

    preds = np.full(20, 0.5)
    y_front = np.array([1, 1] + [0] * 18)
    y_back = np.array([0] * 18 + [1, 1])
    assert _precision_at_top_decile(y_front, preds) == 1.0
    assert _precision_at_top_decile(y_back, preds) == 0.0
