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
        for nm in ("mrmr/__init__.py", "_mrmr_fingerprints.py", "_mrmr_fit_impl/_fit_impl_core.py", "_mrmr_fit_impl/_helpers.py", "_mrmr_fe_step/_step_core.py", "_mrmr_fe_step/_helpers.py", "_mrmr_validate_transform.py"):
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


def test_rfecv_sffs_swap_uses_secondary_name_key() -> None:
    src = _read("feature_selection/wrappers/rfecv/__init__.py")
    assert "key=lambda f: (fi_mean.get(f, 0.0), str(f))" in src
    assert "key=lambda f: (-fi_mean.get(f, 0.0), str(f))" in src


def test_rfecv_stability_topk_uses_lexsort() -> None:
    import re
    # Monolith split: the stability-selection top-k logic moved out of __init__.py into the
    # sibling _stability_select.py; the site is now a single line, not the indented multi-line
    # shape from before the split.
    src = _read("feature_selection/wrappers/rfecv/_stability_select.py")
    pattern = re.compile(
        r"np\.lexsort\(\(np\.arange\(len\(per_feature_score_sum\)\), "
        r"-per_feature_score_sum\)\)"
    )
    assert pattern.search(src) is not None


def test_rfecv_per_fold_top_uses_secondary_key() -> None:
    src = _read("feature_selection/wrappers/rfecv/__init__.py")
    assert "key=lambda k: (-fi[k], str(k))" in src


def test_importance_topn_uses_lexsort() -> None:
    src = _read("feature_selection/importance.py")
    assert "_abs_order_full = np.lexsort((np.arange(len(_abs_fi)), -_abs_fi))" in src


def _has_stable_kind(src: str, base: str, count: int = 1) -> bool:
    """Either ``kind="stable"`` or ``kind="mergesort"`` (the numpy alias that
    guarantees stability) is acceptable - both deliver the tie-determinism
    the wave-57 fix targeted."""
    stable_hits = src.count(f'{base}, kind="stable")')
    merge_hits = src.count(f'{base}, kind="mergesort")')
    return (stable_hits + merge_hits) >= count


def test_metrics_core_uses_stable_argsort() -> None:
    # When ``metrics/core.py`` was split into siblings, the sites this sensor
    # pins moved: ``np.argsort(y_score`` lives in ``_core_auc_brier.py``
    # (the fast_roc_auc / fast_aucs kernels + the central
    # ``_argsort_desc_for_metrics`` dispatcher), per-group AUC scans live
    # in ``_auc_per_group.py``, the ``-y_p`` argsort lives in the
    # classification-report binning helper in
    # ``_classification_report.py``. Concatenate all module sources so
    # the count assertions still pin the post-fix totals.
    src = (
        _read("metrics/core.py")
        + _read("metrics/_core_auc_brier.py")
        + _read("metrics/_auc_per_group.py")
        + _read("metrics/classification/_classification_report.py")
    )
    # The pre-fix shape was 3 inline ``np.argsort(y_score, kind="stable")``
    # call-sites; the refactor consolidated them into ONE stable-sort
    # branch inside ``_argsort_desc_for_metrics`` plus per-callsite uses
    # of the dispatcher. Accept EITHER shape:
    #   * 3+ inline stable/mergesort sites (the legacy pattern), OR
    #   * 1 stable branch inside _argsort_desc_for_metrics + 2+ uses of
    #     that dispatcher (the post-consolidation pattern).
    _inline_stable = (
        src.count('np.argsort(y_score, kind="stable")')
        + src.count('np.argsort(y_score, kind="mergesort")')
    )
    _dispatch_uses = src.count("_argsort_desc_for_metrics")
    assert _inline_stable >= 3 or (_inline_stable >= 1 and _dispatch_uses >= 2), (
        f"y_score sort determinism: need either >=3 inline stable argsorts "
        f"or >=1 stable dispatcher + >=2 dispatcher uses; got inline={_inline_stable}, "
        f"dispatcher refs={_dispatch_uses}"
    )
    # group_y_score sites (now split across _auc_per_group.py: one stable,
    # one mergesort variant).
    assert _has_stable_kind(src, "np.argsort(group_y_score", count=1)
    # The old y_p site at the end of the fast_numba_aucs body was hoisted out to the caller in
    # _classification_report.py's batch ICE/AUC kernel as `np.argsort(-y_pred_NK, axis=0)`
    # (deliberately unstable/quicksort, not stable/mergesort): that kernel's docstring proves the
    # walk only accumulates at tie-run boundaries, so it's provably invariant to within-tie order
    # -- a stable sort there would just be wasted work, not a correctness requirement. Re-framed
    # 2026-07-13 per the "validated improvement changed the contract" rule rather than asserting
    # a now-nonexistent stable-kind call.
    assert "np.argsort(-y_pred_NK, axis=0)" in src


def test_metrics_ranking_uses_stable_argsort() -> None:
    src = _read("metrics/ranking.py")
    assert _has_stable_kind(src, "np.argsort(-y_score_q", count=3)
    assert _has_stable_kind(src, "np.argsort(-y_sc", count=1)


# ---------------------------------------------------------------------------
# P1 source-level sensors
# ---------------------------------------------------------------------------


def test_composite_ensemble_trim_uses_lexsort() -> None:
    """The lexsort pattern moved from composite_ensemble.py to the sibling
    _composite_cross_target_ensemble.py during the cross-target-ensemble
    monolith split. Either location is acceptable - pin the pattern
    in whichever module currently houses it."""
    facade_src = _read("training/composite/ensemble/__init__.py")
    sibling_src = _read("training/composite/ensemble/_cross_target.py")
    pattern = "order = np.lexsort((np.arange(len(_abs_w)), -_abs_w))"
    assert pattern in facade_src or pattern in sibling_src


def test_composite_discovery_aggregated_score_uses_lexsort() -> None:
    # ``_tiny_model_rerank`` moved to the ``_composite_discovery_tiny_rerank.py``
    # sibling when ``composite_discovery.py`` was split below 1k LOC.
    src = _read("training/composite/discovery/_tiny_rerank.py")
    assert "order = np.lexsort((_names, agg_scores))" in src


def test_composite_discovery_mi_gain_uses_secondary_name() -> None:
    # ``fit`` (where the mi_gain top-K sort lives) moved to
    # ``_composite_discovery_fit.py``.
    src = _read("training/composite/discovery/_fit.py")
    assert "key=lambda s: (-s.mi_gain, getattr(s, \"name\", \"\"))" in src


def test_mrmr_empty_fallback_uses_secondary_index() -> None:
    # The empty-RAW-support fallback rescue was carved out of the giant ``_fit_impl``
    # body into the ``_mrmr_fit_impl/_finalise.py`` sibling when ``_fit_impl_core.py``
    # was split toward the 1k LOC ceiling; the stable secondary-index sort lives there now.
    src = _read("feature_selection/filters/_mrmr_fit_impl/_finalise.py")
    assert "_raw_mi.sort(key=lambda kv: (-kv[1], kv[0]))" in src


def test_screen_expected_gains_uses_lexsort() -> None:
    # Confirm-step body moved to ``_confirm_predictor.py`` during the
    # post-1k-LOC monolith split; concat so the source-presence assertion
    # matches regardless of which sibling currently houses the literal.
    #
    # The tiebreak key was subsequently STRENGTHENED: the original wave-57 fix
    # used a positional index tiebreak ``np.arange(len(expected_gains))`` which
    # is NOT column-order invariant; the candidate-confirmation refactor
    # replaced it with a NAME-derived ``_name_rank`` (contiguous int rank of the
    # candidate name, invariant under column reordering). Both are stable
    # lexsorts on descending ``expected_gains``; pin the current name-rank shape
    # while still accepting the legacy index shape so the intent (deterministic,
    # input-order-invariant gain ranking) is what's asserted.
    src = _read("feature_selection/filters/screen.py")
    _confirm = MLFRAME_ROOT / "feature_selection" / "filters" / "_confirm_predictor.py"
    if _confirm.exists():
        src += "\n" + _confirm.read_text(encoding="utf-8")
    assert (
        "np.lexsort((_name_rank, -np.asarray(expected_gains)))" in src
        or "np.lexsort((np.arange(len(expected_gains)), -np.asarray(expected_gains)))" in src
    )


def test_phase_train_ensemble_flavour_uses_secondary_name() -> None:
    """``_choose_ensemble_flavour`` (where the _scored.sort lives) moved
    to sibling ``_ensemble_chooser.py``; concat so the source sensor
    matches the post-carve layout."""
    src = _read("training/core/_phase_train_one_target.py")
    _sib = MLFRAME_ROOT / "training" / "core" / "_ensemble_chooser.py"
    if _sib.exists():
        src += "\n" + _sib.read_text(encoding="utf-8")
    assert "_scored.sort(key=lambda kv: (kv[1], kv[0]))" in src
    assert "_scored.sort(key=lambda kv: (-kv[1], kv[0]))" in src


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
        s, n = scores[perm], names[perm]
        order = np.lexsort((n, -s))
        return tuple(sorted(n[order[:3]].tolist()))

    assert top3(perm1) == top3(perm2), (
        "Lexsort with content-based tiebreaker must yield identical top-K "
        "across input permutations."
    )


# ---------------------------------------------------------------------------
# Follow-up pass sensors (audit2 repro-P2-3): the FE-transformer sites
# ---------------------------------------------------------------------------


def test_spectral_attention_eig_sort_is_stable() -> None:
    src = (MLFRAME_ROOT / "feature_engineering" / "transformer" / "spectral_attention.py").read_text(encoding="utf-8")
    assert 'np.argsort(-eigvals_A, kind="stable")' in src, (
        "spectral_attention eigenvalue order must use kind='stable' so degenerate eigenvalues do not "
        "flip which eigenvector becomes feature-k"
    )


def test_rf_proximity_topk_sort_is_stable() -> None:
    src = (MLFRAME_ROOT / "feature_engineering" / "transformer" / "rf_proximity.py").read_text(encoding="utf-8")
    assert 'np.argsort(-part_sims, axis=1, kind="stable")' in src, (
        "rf_proximity top-k neighbour order must use kind='stable' (quantised proximities tie often)"
    )


def test_fca_closed_concepts_topk_uses_content_tiebreak() -> None:
    src = (MLFRAME_ROOT / "feature_engineering" / "transformer" / "fca_closed_concepts.py").read_text(encoding="utf-8")
    assert "key=lambda x: (-len(x[0]), tuple(sorted(x[1])))" in src, (
        "fca top_k concept selection must break equal-extent-size ties on intent content, not on the "
        "concepts-lib lattice iteration order"
    )


def test_fca_concept_topk_selection_is_permutation_invariant() -> None:
    """Behavioural: the (-extent_size, intent) key selects the SAME top_k concepts regardless of the order
    the lattice yields equal-size concepts in. Mirrors the sorted() the transformer performs."""
    # Three concepts of extent-size 3 (a tie) + one of size 2. Content keys are the intent tuples.
    concepts_a = [((0, 1, 2), ("f1", "f3")), ((3, 4, 5), ("f0", "f2")), ((6, 7, 8), ("f2", "f4")), ((9, 10), ("f5",))]
    concepts_b = [concepts_a[2], concepts_a[0], concepts_a[3], concepts_a[1]]  # different lattice order

    def top2(concepts):
        c = list(concepts)
        c.sort(key=lambda x: (-len(x[0]), tuple(sorted(x[1]))))
        return [x[1] for x in c[:2]]

    assert top2(concepts_a) == top2(concepts_b), "content tiebreak must give order-independent top_k concepts"
