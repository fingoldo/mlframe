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

from pathlib import Path

import numpy as np


MLFRAME_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe"


def _read(rel: str) -> str:
    """Read a source file under src/mlframe.

    Monolith-split compat: append matching siblings so source-pattern
    sensors that pre-date the splits still match.
    """
    _path = MLFRAME_ROOT / rel
    if not _path.exists() and _path.suffix == ".py":
        # Monolith-split compat: the flat module became a subpackage
        # (``X.py`` -> ``X/__init__.py`` + submodules). Read __init__ + every submodule.
        _pkg = _path.with_suffix("")
        _init = _pkg / "__init__.py"
        if _init.exists():
            _parts = [_init.read_text(encoding="utf-8")]
            for _sub in sorted(_pkg.glob("*.py")):
                if _sub.name != "__init__.py":
                    _parts.append(_sub.read_text(encoding="utf-8"))
            primary = "\n".join(_parts)
        else:
            primary = _path.read_text(encoding="utf-8")
    else:
        primary = _path.read_text(encoding="utf-8")
    if rel == "feature_selection/filters/hermite_fe.py":
        _dir = MLFRAME_ROOT / "feature_selection" / "filters"
        for nm in ("_hermite_fe_optimise.py", "_hermite_fe_optimise_pair.py", "_hermite_fe_mi.py"):
            sibling = _dir / nm
            if sibling.exists():
                primary = primary + "\n" + sibling.read_text(encoding="utf-8")
    elif rel == "feature_selection/filters/cat_interactions.py":
        # 2026-05-22 split: run_cat_interaction_step (which contains the
        # kway_results.sort site) moved to _cat_interactions_step.py.
        sibling = MLFRAME_ROOT / "feature_selection" / "filters" / "_cat_interactions_step.py"
        if sibling.exists():
            primary = primary + "\n" + sibling.read_text(encoding="utf-8")
    return primary


def test_hermite_history_uses_secondary_bf_idx() -> None:
    src = _read("feature_selection/filters/hermite_fe.py")
    assert "sorted(history, key=lambda r: (-r[0], r[2]))" in src


def test_hermite_results_uses_secondary_keys() -> None:
    src = _read("feature_selection/filters/hermite_fe.py")
    assert "results.sort(key=lambda r: (-r.mi" in src


def test_composition_single_mi_secondary_key() -> None:
    src = _read("feature_selection/filters/composition.py")
    assert "single_mi.sort(key=lambda kv: (-kv[1], kv[0]))" in src


def test_composition_pair_scores_secondary_key() -> None:
    src = _read("feature_selection/filters/composition.py")
    assert "pair_scores.sort(key=lambda kv: (-kv[2], kv[0], kv[1]))" in src


def test_fe_baselines_secondary_key_on_name() -> None:
    src = _read("feature_selection/filters/fe_baselines.py")
    assert src.count("sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))") >= 2


def test_knockoff_helper_secondary_key_on_name() -> None:
    # Knockoff helpers moved to sibling _knockoffs.py during the wrappers
    # /_helpers.py monolith split (kept under 1k LOC). Concat so the source
    # sensor matches the post-carve layout regardless of which sibling
    # currently houses the literal.
    src = _read("feature_selection/wrappers/_helpers.py")
    _sibling = MLFRAME_ROOT / "feature_selection" / "wrappers" / "_knockoffs.py"
    if _sibling.exists():
        src += "\n" + _sibling.read_text(encoding="utf-8")
    assert "selected.sort(key=lambda kv: (-kv[1], kv[0]))" in src


def test_estimators_perm_mi_uses_lexsort() -> None:
    src = _read("feature_selection/filters/estimators.py")
    assert "significant[np.lexsort((significant, -_obs_sig))]" in src


def test_baseline_diagnostics_ablation_uses_lexsort() -> None:
    """The ablation block moved to sibling _baseline_diagnostics_ablation.py
    when baseline_diagnostics.py was split below 1k LOC; concat so the
    source sensor matches the post-carve layout."""
    src = _read("training/baselines/diagnostics.py")
    _sibling = MLFRAME_ROOT / "training" / "_baseline_diagnostics_ablation.py"
    if _sibling.exists():
        src += "\n" + _sibling.read_text(encoding="utf-8")
    assert "order = np.lexsort((np.arange(len(raw_fi)), -raw_fi))[:top_k]" in src


def test_numerical_top_modes_uses_lexsort() -> None:
    src = _read("feature_engineering/numerical.py")
    assert "modes_indices = np.lexsort((vals, -counts))[:max_modes]" in src


def test_cat_interactions_topk_uses_lexsort() -> None:
    # ``_select_top_k_pairs`` moved to the ``_cat_kway_materialize.py``
    # sibling when ``cat_interactions.py`` was split below 1k LOC.
    src = _read("feature_selection/filters/_cat_kway_materialize.py")
    assert "top_idx = np.lexsort((np.arange(len(masked_score)), -masked_score))" in src


def test_cat_interactions_kway_secondary_key() -> None:
    src = _read("feature_selection/filters/cat_interactions.py")
    assert "kway_results.sort(key=lambda r: (-r[3]," in src


def test_evaluation_reports_precision_at_decile_uses_lexsort() -> None:
    src = _read("evaluation/reports.py")
    assert "idx = np.lexsort((np.arange(n), -preds))[:k]" in src
