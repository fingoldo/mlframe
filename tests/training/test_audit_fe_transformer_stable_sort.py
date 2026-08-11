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

import numpy as np

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


def _assert_topk_within_subset_breaks_ties_by_index(topk_within_subset) -> None:
    """Behavioral check for the ``*_hard_row``/``*_cbhr`` top-K-within-subset picker's tie-break
    contract: when several rows in the subset share the exact same value, the picker must be a
    STABLE selection over the subset's original (ascending index) order, not an arbitrary/impl-
    defined one (e.g. a plain ``argpartition``/``argsort`` without a secondary key would be free to
    return any of the tied rows in any order, which would silently make FE features non-reproducible
    across otherwise-identical runs)."""
    # 6 candidate rows, all tied at the same value: correct top-3 by "stable ascending index"
    # tie-break is exactly rows [0, 1, 2], in that order.
    values = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    subset_idx = np.arange(6)
    top3 = topk_within_subset(values, subset_idx, 3)
    assert list(top3) == [0, 1, 2]

    # Mixed ties: rows {1, 3} tie for the top value, {0, 2} tie for a lower value, row 4 is
    # strictly lowest. Requesting top-3 must resolve the top-value tie by ascending index (1
    # before 3) before spilling into the next-highest tied group (0 before 2).
    values2 = np.array([2.0, 9.0, 2.0, 9.0, 1.0])
    subset_idx2 = np.arange(5)
    top3_mixed = topk_within_subset(values2, subset_idx2, 3)
    assert list(top3_mixed) == [1, 3, 0]

    # A non-trivial subset (not 0..n-1) must still tie-break by the subset's own row order, and
    # the returned indices must be positions into the ORIGINAL `values` array, not the subset.
    values3 = np.array([0.0, 5.0, 5.0, 0.0, 5.0, 0.0])
    subset_idx3 = np.array([1, 2, 4])  # all three tied at 5.0
    top2_subset = topk_within_subset(values3, subset_idx3, 2)
    assert list(top2_subset) == [1, 2]

    # Empty subset returns empty, never raises.
    assert topk_within_subset(values, np.array([], dtype=np.int64), 3).size == 0


def test_multi_baseline_hard_row_uses_lexsort() -> None:
    """multi_baseline_hard_row's top-K-within-subset picker breaks value ties deterministically by
    ascending original index (the property the retired getsource/lexsort-string check was a proxy
    for)."""
    from mlframe.feature_engineering.transformer.multi_baseline_hard_row import _topk_within_subset

    _assert_topk_within_subset_breaks_ties_by_index(_topk_within_subset)


def test_class_balanced_hard_row_uses_lexsort() -> None:
    """class_balanced_hard_row's top-K-within-subset picker breaks value ties deterministically by
    ascending original index (the property the retired getsource/lexsort-string check was a proxy
    for)."""
    from mlframe.feature_engineering.transformer.class_balanced_hard_row import _topk_within_subset

    _assert_topk_within_subset_breaks_ties_by_index(_topk_within_subset)


def test_multi_temp_cbhr_uses_lexsort() -> None:
    """multi_temp_cbhr's top-K-within-subset picker breaks value ties deterministically by
    ascending original index (the property the retired getsource/lexsort-string check was a proxy
    for)."""
    from mlframe.feature_engineering.transformer.multi_temp_cbhr import _topk_within_subset

    _assert_topk_within_subset_breaks_ties_by_index(_topk_within_subset)


def test_hard_row_attention_uses_lexsort_replaces_argpartition() -> None:
    """Hard row attention uses lexsort replaces argpartition."""
    src = _read("feature_engineering/transformer/hard_row_attention.py")
    assert "np.lexsort((np.arange(len(abs_residuals)), -abs_residuals))[:k_eff]" in src
    assert "np.lexsort((np.arange(len(abs_residuals)), -abs_residuals))[:n_hard]" in src
    # The pre-fix argpartition/argsort chain must be gone.
    assert "np.argpartition(abs_residuals, -n_hard)" not in src
