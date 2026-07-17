"""Wave 62 (2026-05-20): close the deferred FE-transformer top-K row pickers
from wave 58.

In wave 58 I documented 9 FE-transformer top-K row-picker sites as "known-
pattern follow-up" and left them. Per user pushback (no deferred items,
"половину под ковер" is exactly the anti-pattern called out in
feedback_no_padding_parametric_pins + the standing "не deferred" directive),
closing them now with the same uniform pattern: np.lexsort with row-index
secondary key for tie-determinism.

9 sites fixed:

  1. feature_engineering/transformer/active_virtual.py:93,98 (uncertainty top-K)
  2. feature_engineering/transformer/pseudo_smote.py:95,99 (proba/pred top-K)
  3. feature_engineering/transformer/tree_path_boolean.py:47 (leaf-score top-K)
  4. feature_engineering/transformer/apriori_itemsets.py:98 (lift top-K)
  5. feature_engineering/transformer/multi_threshold_ordinal.py:61 (LGB FI top-3)
  6. feature_engineering/transformer/multi_baseline_hard_row.py:97 (within-subset top-K)
  7. feature_engineering/transformer/class_balanced_hard_row.py:82 (within-subset top-K)
  8. feature_engineering/transformer/multi_temp_cbhr.py:69 (within-subset top-K)
  9. feature_engineering/transformer/hard_row_attention.py:120,125 (hardest rows
     -- replaces argpartition impl-defined tie-break with deterministic lexsort)

Combined with waves 57+58, the stable-sort tie audit class is now FULLY closed
across mlframe's entire surface (production + FE transformers).
"""

from __future__ import annotations

from pathlib import Path



MLFRAME_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe"


def _read(rel: str) -> str:
    """Read."""
    return (MLFRAME_ROOT / rel).read_text(encoding="utf-8")


def test_active_virtual_uncertainty_uses_lexsort() -> None:
    """Active virtual uncertainty uses lexsort."""
    src = _read("feature_engineering/transformer/active_virtual.py")
    assert src.count("np.lexsort((np.arange(len(uncertainty)), -uncertainty))") >= 2


def test_pseudo_smote_topk_uses_lexsort() -> None:
    """Pseudo smote topk uses lexsort."""
    src = _read("feature_engineering/transformer/pseudo_smote.py")
    assert "np.lexsort((np.arange(len(proba)), -proba))" in src
    assert "np.lexsort((np.arange(len(pred)), -pred))" in src


def test_tree_path_boolean_uses_lexsort() -> None:
    """Tree path boolean uses lexsort."""
    src = _read("feature_engineering/transformer/tree_path_boolean.py")
    assert "np.lexsort((np.arange(len(_scores_arr)), -_scores_arr))[:top_k]" in src


def test_apriori_itemsets_uses_lexsort() -> None:
    """Apriori itemsets uses lexsort."""
    src = _read("feature_engineering/transformer/apriori_itemsets.py")
    assert "np.lexsort((np.arange(len(_lifts_arr)), -_lifts_arr))[:top_k]" in src


def test_multi_threshold_ordinal_uses_lexsort() -> None:
    """Multi threshold ordinal uses lexsort."""
    src = _read("feature_engineering/transformer/multi_threshold_ordinal.py")
    assert "np.lexsort((-np.arange(len(_imp)), _imp))[-3:]" in src


def test_multi_baseline_hard_row_uses_lexsort() -> None:
    """Multi baseline hard row uses lexsort."""
    src = _read("feature_engineering/transformer/multi_baseline_hard_row.py")
    assert "np.lexsort((np.arange(len(sub_values)), -sub_values))[:k_eff]" in src


def test_class_balanced_hard_row_uses_lexsort() -> None:
    """Class balanced hard row uses lexsort."""
    src = _read("feature_engineering/transformer/class_balanced_hard_row.py")
    assert "np.lexsort((np.arange(len(sub_values)), -sub_values))[:k_eff]" in src


def test_multi_temp_cbhr_uses_lexsort() -> None:
    """Multi temp cbhr uses lexsort."""
    src = _read("feature_engineering/transformer/multi_temp_cbhr.py")
    assert "np.lexsort((np.arange(len(sub_values)), -sub_values))[:k_eff]" in src


def test_hard_row_attention_uses_lexsort_replaces_argpartition() -> None:
    """Hard row attention uses lexsort replaces argpartition."""
    src = _read("feature_engineering/transformer/hard_row_attention.py")
    assert "np.lexsort((np.arange(len(abs_residuals)), -abs_residuals))[:k_eff]" in src
    assert "np.lexsort((np.arange(len(abs_residuals)), -abs_residuals))[:n_hard]" in src
    # The pre-fix argpartition/argsort chain must be gone.
    assert "np.argpartition(abs_residuals, -n_hard)" not in src
