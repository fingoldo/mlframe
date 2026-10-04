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

import numpy as np


def _stump_inputs(side: float, n_virtual: int = 50):
    """Train data a one-split model separates on feature 0, and virtual rows that all land on one side of it, marked by feature 1."""
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=400)
    X = np.column_stack([x0, rng.normal(size=400)])
    virtuals = np.column_stack([np.full(n_virtual, side), np.arange(n_virtual, dtype=float)])
    return X, (x0 > 0).astype(int), virtuals


def test_active_virtual_uncertainty_uses_lexsort() -> None:
    """When nothing passes the boundary filter, tied uncertainty keeps the lowest-index virtuals, for binary and regression."""
    from mlframe.feature_engineering.transformer.active_virtual import _filter_boundary_virtuals

    X, y, virtuals = _stump_inputs(2.0)
    for task, target in (("binary", y), ("regression", y.astype(float))):
        kept = _filter_boundary_virtuals(X, target, virtuals, task, 1, 1e-9, n_estimators=1, max_depth=1)
        np.testing.assert_array_equal(kept, virtuals[:10])


def test_pseudo_smote_topk_uses_lexsort() -> None:
    """When nothing passes the confidence filter, tied predictions keep the lowest-index virtuals, for binary and regression."""
    from mlframe.feature_engineering.transformer.pseudo_smote import _fit_aux_lgb_and_filter

    X, y, high_virtuals = _stump_inputs(2.0)
    kept = _fit_aux_lgb_and_filter(X, y, high_virtuals, "binary", 1, 1.1, n_estimators=1, max_depth=1)
    np.testing.assert_array_equal(kept, high_virtuals[:10])
    X, y, low_virtuals = _stump_inputs(-2.0)
    kept = _fit_aux_lgb_and_filter(X, y.astype(float), low_virtuals, "regression", 1, 0.0, n_estimators=1, max_depth=1)
    np.testing.assert_array_equal(kept, low_virtuals[:10])


class _FakeBooster:
    """Booster exposing only ``dump_model``."""

    def __init__(self, dump):
        """Hold the model dump."""
        self._dump = dump

    def dump_model(self):
        """Return the canned dump."""
        return self._dump


def _comb_tree(n_leaves: int, special_leaf: int = -1) -> dict:
    """Tree whose leaf ``i`` is reached by splitting on feature ``i``; every leaf scores the same except ``special_leaf``."""

    def leaf(i):
        """One leaf."""
        return {"leaf_value": 2.0 if i == special_leaf else 1.0, "leaf_count": 10}

    node = leaf(n_leaves - 1)
    for i in reversed(range(n_leaves - 1)):
        node = {"split_feature": i, "threshold": 0.5, "left_child": leaf(i), "right_child": node}
    return {"tree_info": [{"tree_structure": node}]}


def test_tree_path_boolean_uses_lexsort() -> None:
    """Tied leaf scores give the first paths in tree order, and a higher-scoring leaf goes first."""
    from mlframe.feature_engineering.transformer.tree_path_boolean import _extract_top_paths

    paths = _extract_top_paths(_FakeBooster(_comb_tree(40)), np.zeros((1, 40)), top_k=8)
    assert [p[-1][0] for p in paths] == list(range(8))
    paths = _extract_top_paths(_FakeBooster(_comb_tree(40, special_leaf=30)), np.zeros((1, 40)), top_k=8)
    assert [p[-1][0] for p in paths] == [30, 0, 1, 2, 3, 4, 5, 6]


def test_apriori_itemsets_uses_lexsort(monkeypatch) -> None:
    """Itemsets with tied lift fill the top-k slots in mining order."""
    import mlxtend.frequent_patterns as frequent_patterns
    import pandas as pd

    from mlframe.feature_engineering.transformer.apriori_itemsets import compute_apriori_itemsets_features

    d = 12
    itemsets = [frozenset({f"f{j}_b{b}"}) for j in range(d) for b in range(2)]
    monkeypatch.setattr(frequent_patterns, "fpgrowth", lambda df, **kwargs: pd.DataFrame({"support": 0.5, "itemsets": itemsets}))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, d)).astype(np.float32)
    query = np.array([[100.0 if j % 3 == 0 else -100.0 for j in range(d)]], dtype=np.float32)
    out = compute_apriori_itemsets_features(X, np.ones(100), query, seed=1, task="binary", n_bins=2, top_k=20)
    top_bin = [1 if j % 3 == 0 else 0 for j in range(d)]
    expected = [float(k % 2 == top_bin[k // 2]) for k in range(20)]
    assert [float(out[f"apri_itemset{k}"][0]) for k in range(20)] == expected
    assert float(out["apri_n_frequent_total"][0]) == len(itemsets)


def test_multi_threshold_ordinal_uses_lexsort(monkeypatch) -> None:
    """The three sub-population splits use the top-3 importance features, with tied importances resolved to the lowest feature index."""
    import lightgbm

    from mlframe.feature_engineering.transformer.multi_threshold_ordinal import compute_multi_threshold_ordinal_features

    fits: list = []
    importances: list = []

    class StubClassifier:
        """LightGBM stand-in recording each fitted target and reporting canned importances."""

        def __init__(self, **kwargs):
            """Ignore the hyper-parameters."""

        def fit(self, X, y):
            """Record the target."""
            fits.append(np.asarray(y).copy())
            self.feature_importances_ = np.asarray(importances[0])
            return self

        def predict_proba(self, X):
            """Uninformative probabilities."""
            return np.full((len(X), 2), 0.5)

    monkeypatch.setattr(lightgbm, "LGBMClassifier", StubClassifier)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8)).astype(np.float32)
    y = (rng.random(60) > 0.4).astype(np.float32)
    for importance, expected_columns in (([5] * 8, [2, 1, 0]), ([1, 5, 5, 5, 5, 0, 0, 9], [2, 1, 7])):
        fits.clear()
        importances[:] = [importance]
        compute_multi_threshold_ordinal_features(X, y, X[:5], seed=1, task="binary", standardize=False)
        assert len(fits) == 4
        expected_targets = [((y > 0.5) & (X[:, j] > float(np.median(X[:, j])))).astype(np.int32) for j in expected_columns]
        assert any(not np.array_equal(expected_targets[0], t) for t in expected_targets[1:])
        for fitted, expected in zip(fits[1:], expected_targets):
            np.testing.assert_array_equal(fitted, expected)


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


def test_hard_row_attention_uses_lexsort_replaces_argpartition(monkeypatch) -> None:
    """Rows with tied absolute residual become anchors in row order, and a larger residual goes first."""
    from mlframe.feature_engineering.transformer import hard_row_attention

    monkeypatch.setattr(hard_row_attention, "_fit_baseline_predict", lambda Xt, y_t, **kwargs: np.zeros(len(y_t)))
    X = np.column_stack([np.arange(30.0), np.zeros(30)]).astype(np.float32)
    n_hard = 20
    for hardest_row in (None, 7):
        y = np.ones(30)
        anchors = list(range(n_hard))
        if hardest_row is not None:
            y[hardest_row] = 5.0
            anchors = [hardest_row] + [i for i in range(n_hard) if i != hardest_row]
        out = hard_row_attention.compute_hard_row_attention_features(X, y, X, seed=1, n_hard=n_hard, standardize=False)
        sq = (X[:, 0][:, None] - np.asarray(anchors, dtype=np.float32)[None, :]) ** 2
        np.testing.assert_array_equal(out["hrattn_min_dist"].to_numpy(), sq.min(axis=1))
        np.testing.assert_array_equal(out["hrattn_best_hard"].to_numpy(), sq.argmin(axis=1))
