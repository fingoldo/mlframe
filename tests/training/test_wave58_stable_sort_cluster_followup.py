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
    return (MLFRAME_ROOT / rel).read_text(encoding="utf-8")


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
    src = _read("feature_selection/wrappers/_helpers.py")
    assert "selected.sort(key=lambda kv: (-kv[1], kv[0]))" in src


def test_estimators_perm_mi_uses_lexsort() -> None:
    src = _read("feature_selection/filters/estimators.py")
    assert "significant[np.lexsort((significant, -_obs_sig))]" in src


def test_baseline_diagnostics_ablation_uses_lexsort() -> None:
    src = _read("training/baseline_diagnostics.py")
    assert "order = np.lexsort((np.arange(len(raw_fi)), -raw_fi))[:top_k]" in src


def test_numerical_top_modes_uses_lexsort() -> None:
    src = _read("feature_engineering/numerical.py")
    assert "modes_indices = np.lexsort((vals, -counts))[:max_modes]" in src


def test_cat_interactions_topk_uses_lexsort() -> None:
    src = _read("feature_selection/filters/cat_interactions.py")
    assert "top_idx = np.lexsort((np.arange(len(masked_score)), -masked_score))" in src


def test_cat_interactions_kway_secondary_key() -> None:
    src = _read("feature_selection/filters/cat_interactions.py")
    assert "kway_results.sort(key=lambda r: (-r[3]," in src


def test_evaluation_reports_precision_at_decile_uses_lexsort() -> None:
    src = _read("evaluation/reports.py")
    assert "idx = np.lexsort((np.arange(n), -preds))[:k]" in src
