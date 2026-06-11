"""Wave 57 (2026-05-20): stable-sort tie non-determinism in feature ranking,
model leaderboards, ensemble member selection, and metric computation.

Audit class: sorted(...) / np.argsort(...) calls where the sort key has ties
and the result depends on upstream input order -- silently flipping feature
selection / member ranking / metric values across runs when the input order
changes (dict iteration, pandas stride, fold order).

7 P0 + 5 high-impact P1 fixes applied this commit; remaining 13 P1 sites
across FE transformers / cat_interactions / composition.py / knockoff helpers
follow the same pattern (secondary content key on tied score) and are
addressed in a follow-up pass:

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
        for nm in ("mrmr/__init__.py", "_mrmr_fingerprints.py", "_mrmr_fit_impl.py", "_mrmr_fe_step/_step_core.py", "_mrmr_fe_step/_helpers.py", "_mrmr_validate_transform.py"):
            sibling = _dir / nm
            if sibling.exists():
                primary = primary + "\n" + sibling.read_text(encoding="utf-8")
    elif rel == "feature_selection/filters/screen.py":
        # 2026-05-22 split: screen_predictors moved to _screen_predictors.py.
        sibling = MLFRAME_ROOT / "feature_selection" / "filters" / "_screen_predictors.py"
        if sibling.exists():
            primary = primary + "\n" + sibling.read_text(encoding="utf-8")
    elif rel == "feature_selection/wrappers/_rfecv.py":
        # 2026-05-21 split: RFECV.fit + ._fit_stability_selection +
        # .select_optimal_nfeatures_ moved to sibling files.
        _dir = MLFRAME_ROOT / "feature_selection" / "wrappers"
        for nm in (
            "_rfecv_fit.py",
            "_rfecv_stability_select.py",
            "_rfecv_diagnostics.py",
            "_rfecv_fit_fold.py",
            "_rfecv_fit_outer_loop.py",
            "_rfecv_finalize.py",
        ):
            sibling = _dir / nm
            if sibling.exists():
                primary = primary + "\n" + sibling.read_text(encoding="utf-8")
    return primary


# ---------------------------------------------------------------------------
# P0 source-level sensors
# ---------------------------------------------------------------------------


def test_rfecv_sffs_swap_uses_secondary_name_key() -> None:
    src = _read("feature_selection/wrappers/_rfecv.py")
    assert "key=lambda f: (fi_mean.get(f, 0.0), str(f))" in src
    assert "key=lambda f: (-fi_mean.get(f, 0.0), str(f))" in src


def test_rfecv_stability_topk_uses_lexsort() -> None:
    import re
    src = _read("feature_selection/wrappers/_rfecv.py")
    # Indent-tolerant: the body was dedented by 4 spaces during the rfecv
    # monolith split (class-method extraction). Match the lexsort + tiebreak
    # tuple shape rather than the literal whitespace.
    pattern = re.compile(
        r"np\.lexsort\(\s*\n\s+\(np\.arange\(len\(per_feature_score_sum\)\), "
        r"-per_feature_score_sum\)"
    )
    assert pattern.search(src) is not None


def test_rfecv_per_fold_top_uses_secondary_key() -> None:
    src = _read("feature_selection/wrappers/_rfecv.py")
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
    # The y_p site at the end of the fast_numba_aucs body (moved to
    # _classification_report.py).
    assert _has_stable_kind(src, "np.argsort(-y_p", count=1)


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
    src = _read("feature_selection/filters/mrmr/_mrmr_class.py")
    assert "_raw_mi.sort(key=lambda kv: (-kv[1], kv[0]))" in src


def test_screen_expected_gains_uses_lexsort() -> None:
    # Confirm-step body moved to ``_confirm_predictor.py`` during the
    # post-1k-LOC monolith split; concat so the source-presence assertion
    # matches regardless of which sibling currently houses the literal.
    src = _read("feature_selection/filters/screen.py")
    _confirm = MLFRAME_ROOT / "feature_selection" / "filters" / "_confirm_predictor.py"
    if _confirm.exists():
        src += "\n" + _confirm.read_text(encoding="utf-8")
    assert "np.lexsort((np.arange(len(expected_gains)), -np.asarray(expected_gains)))" in src


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
