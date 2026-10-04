"""Wave 57 (2026-05-20): stable-sort tie non-determinism in feature ranking,
model leaderboards, ensemble member selection, and metric computation.

Audit class: sorted(...) / np.argsort(...) calls where the sort key has ties
and the result depends on upstream input order -- silently flipping feature
selection / member ranking / metric values across runs when the input order
changes (dict iteration, pandas stride, fold order).

7 P0 + 5 high-impact P1 fixes applied in the original commit (listed below).

Follow-up pass RESOLVED (audit2 repro-P2-3): the once-"remaining" FE-transformer /
cat_interactions / composition / knockoff sites were re-enumerated and adjudicated
against the CURRENT tree:
  - composition.py / cat_interactions.py: no ranking-sort sites remain (eliminated by
    later refactors).
  - wrappers/_knockoffs.py:186 sorts ``set(abs_W>0)`` (unique values) -> no ties -> safe.
  - Transformer sites hardened with a deterministic tiebreak: spectral_attention.py
    (kind="stable" on degenerate eigenvalues), rf_proximity.py (kind="stable" top-k
    order), fca_closed_concepts.py (secondary content key on the intent tuple so
    equal-extent-size concepts do not depend on the ``concepts`` lib iteration order).
  - Remaining transformer value-sorts (quantile/CDF/IQR/KS/kmeans-distance, Spearman
    ranks) are tie-order-invariant aggregates or documented <=1-ULP proxies -> safe.
The original numbered fixes:

  P0 (7 sites):
    1. feature_selection/wrappers/_rfecv.py:361,365 (SFFS swap_out/swap_in)
       Secondary key on feature name -- tied zero-FI features no longer
       drift selection across runs.
    2. feature_selection/wrappers/_rfecv.py:2007 (stability_selection top-K)
       np.lexsort with feature-index tiebreaker; the public support_mask is
       now reproducible.
    3. feature_selection/wrappers/_rfecv.py:2045 (logged top-10 by frequency)
       np.lexsort for deterministic log output.
    4. feature_selection/wrappers/_rfecv.py:2272 (Jaccard/Dice stability metric)
       Secondary key on feature name; selection_stability_ stays stable.
    5. feature_selection/importance.py:216 (FI bar plot top-N)
       np.lexsort with column-position tiebreaker; bar contents reproducible.
    6. metrics/core.py (7 sites: ROC, PR, NDCG, average_precision_score body)
       kind="stable" on all argsort by y_score so AUC/PR-AUC/precision@K
       stay reproducible when input row order changes.
    7. metrics/ranking.py (4 sites: NDCG order, DCG ideal, per-group MAP/MRR)
       Same kind="stable" treatment.

  P1 high-impact (5 sites):
    8. training/composite_ensemble.py:1229 (component trim by |weight|)
       np.lexsort + component-index tiebreak; tied weights no longer flip
       which components survive across stack-row orderings.
    9. training/composite_discovery.py:2626 (aggregated-score top-M)
       np.lexsort with spec name; tied RMSE no longer makes top-M selection
       depend on dict iteration.
   10. training/composite_discovery.py:617 (mi_gain top-K)
       Secondary key on spec name.
   11. feature_selection/filters/mrmr.py:1867 (empty-support fallback top-K)
       Secondary key on feature index.
   12. feature_selection/filters/screen.py:678 (expected_gains candidate loop)
       np.lexsort with candidate-index tiebreak.
   13. training/core/_phase_train_one_target.py:290 (ensemble flavour winner)
       Secondary key on ensemble name; tied val metric -> deterministic winner.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

MLFRAME_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe"


def _read(rel: str) -> str:
    """Read a module source by relative path under src/mlframe.

    Monolith-split compat: when the requested file is one of the parents
    whose code moved to siblings, append every matching sibling so source-
    pattern sensors that pre-date the splits still match.
    """
    primary = (MLFRAME_ROOT / rel).read_text(encoding="utf-8")
    if rel == "training/core/_phase_train_one_target.py":
        _core = MLFRAME_ROOT / "training" / "core"
        for _sib_name in (
            "_phase_train_one_target_body.py",
            "_phase_train_one_target_ensembling.py",
            "_phase_train_one_target_polars_fastpath.py",
            "_phase_train_one_target_pre_screen.py",
            "_phase_train_one_target_model_setup.py",
        ):
            _sib_path = _core / _sib_name
            if _sib_path.exists():
                primary = primary + "\n" + _sib_path.read_text(encoding="utf-8")
    elif rel == "feature_selection/filters/mrmr/_mrmr_class.py":
        # mrmr subpackage split: MRMR class body in mrmr/_mrmr_class.py; the rest of the surface lives in
        # _mrmr_{fingerprints,fit_impl,fe_step,validate_transform}.py + the mrmr/__init__.py facade.
        _dir = MLFRAME_ROOT / "feature_selection" / "filters"
        for nm in (
            "mrmr/__init__.py",
            "_mrmr_fingerprints.py",
            "_mrmr_fit_impl/_fit_impl_core.py",
            "_mrmr_fit_impl/_helpers.py",
            "_mrmr_fe_step/_step_core.py",
            "_mrmr_fe_step/_helpers.py",
            "_mrmr_validate_transform.py",
        ):
            sibling = _dir / nm
            if sibling.exists():
                primary = primary + "\n" + sibling.read_text(encoding="utf-8")
    elif rel == "feature_selection/filters/screen.py":
        # 2026-05-22 split: screen_predictors moved to _screen_predictors.py.
        sibling = MLFRAME_ROOT / "feature_selection" / "filters" / "_screen_predictors.py"
        if sibling.exists():
            primary = primary + "\n" + sibling.read_text(encoding="utf-8")
    elif rel == "feature_selection/wrappers/rfecv/__init__.py":
        # RFECV.fit + ._fit_stability_selection + .select_optimal_nfeatures_
        # live in sibling submodules of the rfecv/ subpackage.
        _dir = MLFRAME_ROOT / "feature_selection" / "wrappers" / "rfecv"
        for nm in (
            "_fit.py",
            "_stability_select.py",
            "_diagnostics.py",
            "_fit_fold.py",
            "_fit_outer_loop.py",
            "_finalize.py",
            "_sffs.py",
        ):
            sibling = _dir / nm
            if sibling.exists():
                primary = primary + "\n" + sibling.read_text(encoding="utf-8")
    return primary


# ---------------------------------------------------------------------------
# P0 source-level sensors
# ---------------------------------------------------------------------------


def _sffs_tried_designs(monkeypatch, best_set, original_features, fi):
    """Run one SFFS swap pass with CV scores no swap can beat; return the column lists of the swapped designs, in the order tried."""
    import pandas as pd
    import sklearn.model_selection as model_selection
    from sklearn.dummy import DummyClassifier

    from mlframe.feature_selection.wrappers.rfecv._sffs import _sffs_swap_pass

    tried: list = []

    def fake_cross_val_score(estimator, X, y, **kwargs):
        """Record the swapped design and return a score that never improves on the reference."""
        tried.append(list(X.columns))
        return np.array([0.1, 0.1])

    monkeypatch.setattr(model_selection, "cross_val_score", fake_cross_val_score)
    X = pd.DataFrame(np.zeros((6, len(original_features))), columns=list(original_features))
    y = np.array([0, 1, 0, 1, 0, 1])
    owner = SimpleNamespace(swap_top_k=2, _fit_sample_weight_=None)
    _sffs_swap_pass(
        owner, X, y, DummyClassifier(), 2, "accuracy", len(best_set), 0.9, {len(best_set): list(best_set)}, {"run": fi}, list(original_features), {}, {}, 0, 3
    )
    return tried


def test_rfecv_sffs_swap_uses_secondary_name_key(monkeypatch) -> None:
    """SFFS swaps the tied-lowest kept features for the tied-highest dropped ones in name order, whatever the input order."""
    fi = {"d": 1.0, "b": 1.0, "c": 1.0, "e": 0.5, "a": 0.5, "f": 0.5}
    expected = [["a", "c", "d"], ["b", "d", "e"]]
    for best_set, original in ((["d", "b", "c"], ["e", "d", "c", "b", "a", "f"]), (["c", "d", "b"], ["f", "a", "b", "c", "d", "e"])):
        fi_in_order = {k: fi[k] for k in original}
        tried = _sffs_tried_designs(monkeypatch, best_set, original, fi_in_order)
        assert [sorted(t) for t in tried] == expected
        tried = _sffs_tried_designs(monkeypatch, best_set, original, dict(reversed(list(fi_in_order.items()))))
        assert [sorted(t) for t in tried] == expected


def test_rfecv_stability_topk_uses_lexsort() -> None:
    """Stability selection over tied importances picks the lowest-index features, so the public support mask is reproducible."""
    import pandas as pd
    from sklearn.base import BaseEstimator, ClassifierMixin

    from mlframe.feature_selection.wrappers.rfecv import RFECV

    class TiedImportanceClassifier(BaseEstimator, ClassifierMixin):
        """Classifier whose every feature has the same importance."""

        def fit(self, X, y, sample_weight=None):
            """Record equal importances."""
            self.classes_ = np.unique(y)
            self.feature_importances_ = np.ones(X.shape[1])
            return self

        def predict(self, X):
            """Constant prediction."""
            return np.zeros(len(X), dtype=int)

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 8)), columns=[f"f{i}" for i in range(8)])
    y = (rng.random(40) > 0.5).astype(int)
    selector = RFECV(
        estimator=TiedImportanceClassifier(),
        importance_getter="feature_importances_",
        stability_selection=True,
        stability_n_bootstrap=6,
        stability_top_k=3,
        stability_threshold=0.6,
        random_state=0,
        verbose=0,
    )
    selector.fit(X, y)
    assert np.asarray(selector.support_).tolist() == [True, True, True, False, False, False, False, False]


def test_rfecv_per_fold_top_uses_secondary_key() -> None:
    """Per-fold top-N sets break importance ties on feature name, so stability does not depend on dict insertion order."""
    from mlframe.feature_selection.wrappers.rfecv._diagnostics import selection_stability_, stability_vs_n_curve_

    fis = {
        "2_0": {"c": 0.0, "b": 0.0, "a": 0.0},
        "2_1": {"a": 0.0, "b": 0.0, "c": 0.0},
        "2_2": {"z": 5.0, "c": 0.0, "b": 0.0, "a": 0.0},
    }
    owner = SimpleNamespace(feature_importances_=fis, n_features_=2, n_features_in_=4, feature_names_in_=["a", "b", "c", "z"])
    # top-2 sets: {a, b}, {a, b}, {z, a} -> pair Jaccards 1, 1/3, 1/3
    assert selection_stability_(owner) == pytest.approx(5.0 / 9.0)
    assert stability_vs_n_curve_(owner) == {2: pytest.approx(5.0 / 9.0)}


def test_importance_topn_uses_lexsort(monkeypatch, tmp_path) -> None:
    """The FI bar plot keeps the lowest-position features among tied magnitudes."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    from mlframe.feature_selection.importance import plot_feature_importance

    labels: list = []
    original_set = Axes.set

    def recording_set(self, **kwargs):
        """Capture the y tick labels the plot assigns."""
        if "yticklabels" in kwargs:
            labels.append([str(x) for x in kwargs["yticklabels"]])
        return original_set(self, **kwargs)

    monkeypatch.setattr(Axes, "set", recording_set)
    fi = np.ones(20)
    fi[7] = -3.0
    plot_feature_importance(fi, [f"f{i}" for i in range(20)], kind="tie", n=6, show_plots=False, plot_file=str(tmp_path / "fi.png"), log_fi=False)
    assert len(labels) == 1
    assert set(labels[0]) == {"f7", "f0", "f1", "f2", "f3", "f4"}


def _has_stable_kind(src: str, base: str, count: int = 1) -> bool:
    """Either ``kind="stable"`` or ``kind="mergesort"`` (the numpy alias that
    guarantees stability) is acceptable - both deliver the tie-determinism
    the wave-57 fix targeted."""
    stable_hits = src.count(f'{base}, kind="stable")')
    merge_hits = src.count(f'{base}, kind="mergesort")')
    return (stable_hits + merge_hits) >= count


def test_metrics_core_uses_stable_argsort(monkeypatch) -> None:
    """ROC/PR AUC (overall and per group) on heavily tied scores do not depend on row order.

    The metric kernels accumulate only at tie-run boundaries, so the default unstable argsort is
    exact; the MLFRAME_METRICS_STABLE_SORT=1 opt-in must still give the stable descending order."""
    from sklearn.metrics import roc_auc_score

    from mlframe.metrics._auc_per_group import fast_aucs_per_group
    from mlframe.metrics._core_auc_brier import _argsort_desc_for_metrics, fast_aucs

    rng = np.random.default_rng(1)
    n = 400
    score = rng.integers(0, 5, n).astype(float) / 4  # five distinct values: every score is tied
    y = (rng.random(n) < 0.3 + 0.4 * score).astype(np.int64)
    groups = rng.integers(0, 3, n)

    base = fast_aucs(y, score)
    base_grouped = fast_aucs_per_group(y, score, groups)
    assert base[0] == pytest.approx(roc_auc_score(y, score))
    for _ in range(5):
        perm = rng.permutation(n)
        assert fast_aucs(y[perm], score[perm]) == base
        assert fast_aucs_per_group(y[perm], score[perm], groups[perm]) == base_grouped

    monkeypatch.setenv("MLFRAME_METRICS_STABLE_SORT", "1")
    np.testing.assert_array_equal(_argsort_desc_for_metrics(score), np.argsort(score, kind="stable")[::-1])


def test_metrics_ranking_uses_stable_argsort() -> None:
    """NDCG/DCG-ideal/per-group MAP-MRR argsort sites use a stable kind."""
    src = _read("metrics/ranking.py")
    assert _has_stable_kind(src, "np.argsort(-y_score_q", count=3)
    assert _has_stable_kind(src, "np.argsort(-y_sc", count=1)


# ---------------------------------------------------------------------------
# P1 source-level sensors
# ---------------------------------------------------------------------------


def test_composite_ensemble_trim_uses_lexsort() -> None:
    """Trimming to the top-N components keeps the lowest-index components among tied weights."""
    from mlframe.training.composite.ensemble._cross_target import CompositeCrossTargetEnsemble

    weights = np.full(24, 0.1)
    weights[5] = 0.3
    weights[17] = 0.3
    ensemble = CompositeCrossTargetEnsemble([object()] * 24, [f"m{i}" for i in range(24)], weights, "nnls_stack", is_convex=False)
    capped = ensemble.cap_inference_components(4)
    assert capped.component_names == ["m0", "m1", "m5", "m17"]
    assert capped.notes["dropped_components"] == [f"m{i}" for i in range(2, 24) if i not in (5, 17)]


def _tied_specs():
    """Three specs, two tied on score, in an input order that is not the name order."""
    from types import SimpleNamespace

    return [SimpleNamespace(name=n, transform_name="diff", v=v) for n, v in (("c", 1.0), ("b", 0.5), ("a", 1.0))]


def test_composite_discovery_aggregated_score_uses_lexsort() -> None:
    """The top-M rerank orders by aggregated RMSE ascending and breaks ties on spec name, whatever the input order.

    It ranks through ``rank_specs`` with its default name tiebreak (the lexsort it replaced did the same); tested on
    the behaviour rather than on the source line, which moved when the rankings were unified.
    """
    from mlframe.training.composite.discovery._score import Score, rank_specs

    for specs in (_tied_specs(), _tied_specs()[::-1]):
        ranked = rank_specs(specs, lambda s: Score(s.v, "y_rmse", "tiny_rerank", "tiny_consensus", s.transform_name))
        assert [s.name for s in ranked] == ["b", "a", "c"]


def test_composite_discovery_mi_gain_uses_secondary_name() -> None:
    """The mi_gain top-K sort orders by gain descending and breaks ties on spec name, whatever the input order."""
    from mlframe.training.composite.discovery._score import Score, rank_specs

    for specs in (_tied_specs(), _tied_specs()[::-1]):
        ranked = rank_specs(specs, lambda s: Score(s.v, "mi_nats", "screen", "mi_gain", s.transform_name), descending=True)
        assert [s.name for s in ranked] == ["a", "c", "b"]


def test_mrmr_empty_fallback_uses_secondary_index() -> None:
    """The empty-support rescue ranks tied MI scores by feature index."""
    from mlframe.feature_selection.filters._mrmr_fit_impl._finalise import _rank_raw_candidates

    names = [f"f{i}" for i in range(24)]
    owner = SimpleNamespace(feature_names_in_=names, n_features_in_=24, factors_names_to_use=None, factors_to_use=None)
    cached = {(i,): 0.2 for i in range(24)}
    cached[(9,)] = 0.7
    ranked, allowed = _rank_raw_candidates(owner, {n: i for i, n in enumerate(names)}, cached)
    assert allowed is None
    assert [r[0] for r in ranked] == [9, *[i for i in range(24) if i != 9]]
    assert [r[1] for r in ranked] == [0.7] + [0.2] * 23


def test_screen_expected_gains_uses_lexsort() -> None:
    """Two exact copies of the signal column tie on gain; the winner is the name-first one in either column order."""
    import pandas as pd

    from mlframe.feature_selection.filters import MRMR

    rng = np.random.default_rng(3)
    n = 500
    signal = rng.normal(size=n)
    y = (signal + 0.2 * rng.normal(size=n) > 0).astype(np.int32)
    noise = rng.normal(size=(n, 3))
    frames = {
        "a_first": pd.DataFrame({"a": signal, "b": signal, "n0": noise[:, 0], "n1": noise[:, 1], "n2": noise[:, 2]}),
        "b_first": pd.DataFrame({"b": signal, "a": signal, "n0": noise[:, 0], "n1": noise[:, 1], "n2": noise[:, 2]}),
    }
    picked = {}
    for label, frame in frames.items():
        MRMR._FIT_CACHE.clear()
        selector = MRMR(full_npermutations=10, baseline_npermutations=5, n_jobs=1, verbose=0, random_seed=7)
        selector.fit(frame, y)
        picked[label] = sorted(frame.columns[np.asarray(selector.support_)])
        MRMR._FIT_CACHE.clear()
    assert picked["a_first"] == picked["b_first"]
    assert picked["a_first"] == ["a"]


def test_phase_train_ensemble_flavour_uses_secondary_name() -> None:
    """Tied validation metrics give the name-first flavour as the winner, for lower-is-better and higher-is-better metrics, in any insertion order."""
    from mlframe.training.core._ensemble_chooser import _choose_ensemble_flavour

    def result(**val_metrics):
        """Ensemble result carrying only validation metrics."""
        return SimpleNamespace(metrics={"val": dict(val_metrics)})

    for metric, best, worse in (("rmse", 1.0, 2.0), ("roc_auc", 0.9, 0.8)):
        flavours = {f"m{i:02d}": result(**{metric: worse}) for i in range(20)}
        flavours["m17"] = result(**{metric: best})
        flavours["m05"] = result(**{metric: best})
        flavours["m11"] = result(**{metric: best})
        assert _choose_ensemble_flavour(flavours) == "m05"
        assert _choose_ensemble_flavour(dict(reversed(list(flavours.items())))) == "m05"


# ---------------------------------------------------------------------------
# Behavioural sensor: secondary-key tiebreak gives deterministic output.
# ---------------------------------------------------------------------------


def test_lexsort_tiebreak_returns_same_top_k_across_input_permutations() -> None:
    """Demonstrate the bug-class invariant: lexsort with content-based secondary
    key gives the same top-K regardless of input row ordering."""
    scores = np.array([0.5, 0.5, 0.5, 0.3, 0.5])
    names = np.array(["alpha", "beta", "gamma", "delta", "epsilon"])

    # Permute input order; both should yield same top-3 by (-score, name).
    perm1 = np.arange(5)
    perm2 = np.array([4, 2, 0, 3, 1])

    def top3(perm):
        """Top-3 names by (-score, name) lexsort for the given row permutation."""
        s, n = scores[perm], names[perm]
        order = np.lexsort((n, -s))
        return tuple(sorted(n[order[:3]].tolist()))

    assert top3(perm1) == top3(perm2), "Lexsort with content-based tiebreaker must yield identical top-K across input permutations."


# ---------------------------------------------------------------------------
# Follow-up pass sensors (audit2 repro-P2-3): the FE-transformer sites
# ---------------------------------------------------------------------------


def test_spectral_attention_eig_sort_is_stable(monkeypatch) -> None:
    """Degenerate eigenvalues keep the solver's column order, so which eigenvector becomes feature-k does not drift."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as sparse_linalg

    from mlframe.feature_engineering.transformer.spectral_attention import _eigvecs_from_graph

    n_eigvecs = 30
    k = n_eigvecs + 1
    vals = np.array([0.5, 0.9, 0.5, 1.0, 0.9] * 7)[:k]
    vecs = np.eye(60)[:, :k]
    monkeypatch.setattr(sparse_linalg, "eigsh", lambda A, k, which: (vals, vecs))
    eigvals, eigvecs, _ = _eigvecs_from_graph(sp.csr_matrix(np.ones((60, 60), dtype=np.float32)), n_eigvecs)
    order = sorted(range(k), key=lambda i: -vals[i])
    assert eigvecs.argmax(axis=0).tolist() == order[1 : n_eigvecs + 1]
    np.testing.assert_allclose(eigvals, vals[order][1 : n_eigvecs + 1])


def test_rf_proximity_topk_sort_is_stable() -> None:
    """Top-k neighbours come back descending by similarity, as a valid top-k, with tied similarities in a stable order."""
    import scipy.sparse as sp

    from mlframe.feature_engineering.transformer.rf_proximity import _topk_proximity

    rng = np.random.default_rng(0)
    bank = sp.csr_matrix((rng.random((50, 6)) < 0.5).astype(np.float32))
    query = sp.csr_matrix((rng.random((4, 6)) < 0.5).astype(np.float32))
    k = 30
    ids, sims = _topk_proximity(query, bank, k)
    sim_dense = (query @ bank.T).toarray()
    assert ids.shape == sims.shape == (4, k)
    for row in range(4):
        assert len(set(ids[row].tolist())) == k
        np.testing.assert_array_equal(sims[row], sim_dense[row, ids[row]])
        assert np.all(np.diff(sims[row]) <= 0)
        excluded = np.setdiff1d(np.arange(50), ids[row])
        assert sims[row].min() >= sim_dense[row, excluded].max()
        assert len(np.unique(sims[row])) < k
        partition = np.argpartition(-sim_dense, kth=k - 1, axis=1)[row, :k]
        expected = partition[np.argsort(-sim_dense[row, partition], kind="stable")]
        np.testing.assert_array_equal(ids[row], expected)


def test_fca_closed_concepts_topk_uses_content_tiebreak() -> None:
    """The emitted concept features do not depend on the order the training rows (and so the lattice) arrive in."""
    from mlframe.feature_engineering.transformer.fca_closed_concepts import compute_fca_closed_concepts_features

    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 6)).astype(np.float32)
    y = rng.normal(size=60).astype(np.float32)
    Xq = rng.normal(size=(15, 6)).astype(np.float32)
    base = compute_fca_closed_concepts_features(X, y, Xq, seed=1, top_k=8)
    assert base.shape == (15, 10)
    assert base["fca_n_concepts"].min() > 0
    for _ in range(3):
        perm = rng.permutation(60)
        shuffled = compute_fca_closed_concepts_features(X[perm], y[perm], Xq, seed=1, top_k=8)
        assert shuffled.equals(base)


def test_fca_concept_topk_selection_is_permutation_invariant() -> None:
    """Behavioural: the (-extent_size, intent) key selects the SAME top_k concepts regardless of the order
    the lattice yields equal-size concepts in. Mirrors the sorted() the transformer performs."""
    # Three concepts of extent-size 3 (a tie) + one of size 2. Content keys are the intent tuples.
    concepts_a = [((0, 1, 2), ("f1", "f3")), ((3, 4, 5), ("f0", "f2")), ((6, 7, 8), ("f2", "f4")), ((9, 10), ("f5",))]
    concepts_b = [concepts_a[2], concepts_a[0], concepts_a[3], concepts_a[1]]  # different lattice order

    def top2(concepts):
        """Top-2 intents by (-extent_size, intent) sort for the given concept ordering."""
        c = list(concepts)
        c.sort(key=lambda x: (-len(x[0]), tuple(sorted(x[1]))))
        return [x[1] for x in c[:2]]

    assert top2(concepts_a) == top2(concepts_b), "content tiebreak must give order-independent top_k concepts"
