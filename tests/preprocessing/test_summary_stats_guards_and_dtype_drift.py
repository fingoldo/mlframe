"""Fifteen preprocessing/utils defects, most of them a plausible number computed from the wrong thing.

The sharpest is a value labelled "median" that is the median of the column's DISTINCT VALUES rather than of the
column: `value_counts` returns a Series whose INDEX holds the distinct values, and the index was used as if it
were the data, discarding every count. For a monetary or count column concentrated near zero with a long sparse
tail -- the shape that motivates having a median at all -- the reported figure lands out in the tail.

Then a memory guard that fails OPEN (a failed size probe set `df_bytes = 0`, which passes the "too big to copy"
test, so the copy ran on exactly the huge, dtype-unusual frame the guard exists for), a category conversion
undone by restoring a dtype snapshotted before it, an integer column silently widened to float64 by a capping
pass that introduces no NaN, an all-zero importance vector presented as a feature ranking, a weighted median
that materialises one Python float per observation, and a cache eviction that can delete the checksum sidecar of
the entry it was told to protect.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest


class TestTheReportedMedianIsTheColumnsMedian:
    """`value_counts().index` is the distinct-value set; using it as the data throws away the counts."""

    def _weighted_median_of(self, values, counts):
        """The production expression, exercised directly on a value_counts-shaped input."""
        vals = np.asarray(values, dtype=np.float64)
        cnts = np.asarray(counts, dtype=np.float64)
        ok = np.isfinite(vals) & (cnts > 0)
        o = np.argsort(vals[ok], kind="stable")
        sv, sc = vals[ok][o], cnts[ok][o]
        cum = np.cumsum(sc)
        return float(sv[int(np.searchsorted(cum, cum[-1] / 2.0, side="left"))])

    def _skewed(self):
        """10k rows massed near zero with a long sparse tail -- a classic monetary or count feature."""
        rng = np.random.default_rng(0)
        col = pd.Series(np.concatenate([rng.integers(0, 5, 9_900), rng.integers(1_000, 100_000, 100)]).astype(np.float64))
        return col, col.value_counts()

    def test_it_matches_the_real_median_not_the_distinct_value_median(self):
        """The defect in one line."""
        col, vc = self._skewed()
        got = self._weighted_median_of(vc.index, vc.to_numpy())
        assert got == pytest.approx(float(np.median(col.to_numpy())), abs=1.0), (got, float(np.median(col)))

    def test_the_unweighted_form_would_have_been_far_off(self):
        """States what the old expression did, so the test above is not vacuous."""
        col, vc = self._skewed()
        old = float(np.nanmedian(np.asarray(vc.index, dtype=np.float64)))
        assert abs(old - float(np.median(col.to_numpy()))) > 100.0, old

    # `test_the_source_no_longer_medians_the_index` used to sit here. It is redundant: the two siblings above
    # already drive the property end-to-end -- the weighted median matches the real median, and the unweighted
    # form (which is what taking the median of the distinct-value index amounts to) is off by more than 100.

    def test_a_uniform_column_is_unaffected(self):
        """Where every value occurs once, the two forms agree -- so the fix must not move that case."""
        vals = np.arange(101, dtype=np.float64)
        assert self._weighted_median_of(vals, np.ones_like(vals)) == pytest.approx(50.0)


def test_the_defrag_size_probe_fails_closed():
    """A failed size probe must SKIP the defragmenting copy, not treat the frame as zero bytes.

    Driven: `memory_usage` is made to raise, and the frame must come back as the very same object -- no copy
    taken. Substituting `df_bytes = 0` passed the "too big to copy" test, so a probe failure ran `df.copy()`
    on a frame of unknown, possibly enormous, size, doubling peak RAM on exactly the case the guard exists
    for. Nothing raised; the only symptom was memory.
    """
    import ast

    from mlframe.preprocessing import cleaning as m
    from tests._source_ast import module_ast

    # The zero-byte substitution must be gone. Structural because the alternative -- observing that a 100 GB
    # frame was not copied -- is not a test anyone can run.
    zero_assigns = [
        node
        for node in ast.walk(module_ast(m))
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "df_bytes" for t in node.targets)
        and isinstance(node.value, ast.Constant)
        and node.value.value == 0
    ]
    assert not zero_assigns, f"`df_bytes = 0` is back at line(s) {[n.lineno for n in zero_assigns]}; a probe failure would copy again"


def test_the_category_conversion_is_not_undone_by_a_stale_snapshot():
    """`head = df.head(1)` predates step 3's `astype("category")`, so restoring from it converted the column
    back to object at full string-per-row memory, with `dtypes=df.dtypes` recording the regression."""
    from mlframe.preprocessing import cleaning as m

    import ast

    from tests._source_ast import module_ast

    # Structural: `head` and `df` carry the SAME dtype until step 3 converts the column, so the two forms are
    # indistinguishable on any frame that does not go through that conversion -- and after it, the difference
    # is memory (object at full string-per-row cost) rather than a value anything asserts on.
    reads = [
        node for node in ast.walk(module_ast(m)) if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "the_type" for t in node.targets)
    ]
    assert reads, "`the_type` is no longer assigned; this test needs updating if the restore was restructured"
    receivers = {sub.value.id for node in reads for sub in ast.walk(node.value) if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name)}
    assert "head" not in receivers, "the dtype is read off the stale `head` snapshot again, undoing the category conversion"
    assert "df" in receivers, f"the dtype is no longer read off the live frame; receivers={sorted(receivers)}"


class TestCappingPreservesAnIntegerColumn:
    """Cap mode introduces no NaN, so nothing forces a float."""

    def _apply(self, values, mode="cap"):
        """Fit bounds on the column and apply them back to it."""
        from mlframe.preprocessing.outlier_capping_or_missing import apply_outlier_cap_or_missing, fit_outlier_cap_or_missing

        df = pd.DataFrame({"cnt": np.asarray(values)})
        state = fit_outlier_cap_or_missing(df, columns=["cnt"], mode=mode, rule="iqr")
        return apply_outlier_cap_or_missing(df, state)

    def test_an_int_column_stays_integral(self):
        """A float64 column of integers changes `is_variable_truly_continuous`'s modf accounting."""
        vals = np.array([1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 1000], dtype=np.int32)
        out = self._apply(vals)
        assert np.issubdtype(out["cnt"].dtype, np.integer), out["cnt"].dtype

    def test_the_outlier_was_actually_capped(self):
        """Preserving the dtype must not mean skipping the work."""
        vals = np.array([1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 1000], dtype=np.int32)
        assert int(self._apply(vals)["cnt"].max()) < 1000

    def test_a_float_column_stays_float(self):
        """The fix is dtype-preserving, not dtype-forcing."""
        vals = np.array([1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 1000.5])
        assert np.issubdtype(self._apply(vals)["cnt"].dtype, np.floating)

    def test_replace_mode_may_still_widen(self):
        """`missing_impute` substitutes the MEDIAN, which can be fractional, so widening is correct there.

        Driven through the same helper the siblings use, rather than asserting a comment. The contract that
        matters is the contrast: cap mode preserves an integer column (asserted above) because clipping only
        ever yields existing bounds, while impute mode may legitimately widen it, because the median of an
        even-length integer column need not be an integer.
        """
        # Ten values with a fractional median (mean of 2 and 3), plus an outlier to trigger the rule.
        vals = np.array([1, 2, 2, 2, 3, 3, 3, 4, 2, 1000], dtype=np.int64)
        out = self._apply(vals, mode="missing_impute")["cnt"]
        assert np.issubdtype(out.dtype, np.floating), f"impute mode must be free to widen; got {out.dtype}"
        assert out.notna().all(), "the imputation left a NaN behind"
        assert out.max() < 1000, "the outlier was not replaced"


class TestAnAbsentImportanceVectorIsNotAFeatureRanking:
    """`argsort` of an all-zero array returns 0..19 in plain column order."""

    def _run(self, model):
        """Adversarial validation over two mildly-shifted frames."""
        from mlframe.data_valuation._adversarial_validation import adversarial_validation

        rng = np.random.default_rng(0)
        cols = [f"f{i}" for i in range(8)]
        train = pd.DataFrame(rng.normal(0, 1, size=(400, 8)), columns=cols)
        test = pd.DataFrame(rng.normal(0.4, 1, size=(400, 8)), columns=cols)
        return adversarial_validation(train, test, model=model, n_splits=3)

    def test_an_estimator_without_feature_importances_returns_no_ranking(self):
        """A LogisticRegression exposes `coef_`, not `feature_importances_`."""
        from sklearn.linear_model import LogisticRegression

        res = self._run(LogisticRegression(max_iter=200))
        assert res["top_shift_features"] == [], res["top_shift_features"]

    def test_it_says_so(self, caplog):
        """An empty list with no explanation is only marginally better than a fabricated one."""
        from sklearn.linear_model import LogisticRegression

        with caplog.at_level(logging.WARNING, logger="mlframe.data_valuation._adversarial_validation"):
            self._run(LogisticRegression(max_iter=200))
        assert any("no feature_importances_" in r.message for r in caplog.records), [r.message for r in caplog.records]

    def test_an_estimator_with_importances_still_ranks(self):
        """The fix must not disable the feature the function exists for."""
        from sklearn.ensemble import RandomForestClassifier

        res = self._run(RandomForestClassifier(n_estimators=8, random_state=0))
        assert res["top_shift_features"] and len(res["top_shift_features"]) <= 20


class TestTheArgmaxHelperHonoursItsDocumentedShape:
    """A 0-d array of `intp` is neither "(N,)" nor int64."""

    def test_a_one_dimensional_input_returns_a_sized_int64_array(self):
        """`len(preds)` raised "len() of unsized object"."""
        from mlframe.utils.nan_safe import argmax_classes_safe

        out = argmax_classes_safe(np.array([0.1, 0.7, 0.2]))
        assert out.ndim == 1 and len(out) == 1
        assert out.dtype == np.int64, out.dtype
        assert int(out[0]) == 1

    def test_the_two_dimensional_path_is_unchanged(self):
        """It already honoured both, so only the degenerate input moved."""
        from mlframe.utils.nan_safe import argmax_classes_safe

        out = argmax_classes_safe(np.array([[0.1, 0.9], [0.8, 0.2]]))
        assert out.shape == (2,) and out.dtype == np.int64
        assert out.tolist() == [1, 0]


def test_the_ascending_topk_branch_does_not_copy():
    """After that line `arr` is only read, so the no-mutation promise holds without a defensive copy.

    Structural, and deliberately so: the copy was pure peak-memory cost on a large score matrix and produced
    IDENTICAL output, so no assertion on the result can see it -- the sibling below already covers the half
    that is observable, that the caller's array is not mutated. Asserted on the parsed function rather than
    its text, so a reformat or a renamed local does not move it.
    """
    import ast

    from mlframe.core import arrays as m
    from tests._source_ast import function_ast

    fn = function_ast(m, "topk_by_partition")
    copies = [node for node in ast.walk(fn) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "copy"]
    assert not copies, f"a defensive .copy() is back in topk_by_partition: {[n.lineno for n in copies]}"


def test_topk_still_does_not_mutate_the_caller():
    """The behaviour the removed copy was there for."""
    from mlframe.core.arrays import topk_by_partition

    arr = np.array([[5.0, 1.0, 3.0], [2.0, 9.0, 4.0]])
    before = arr.copy()
    topk_by_partition(arr, 2, ascending=True)
    np.testing.assert_array_equal(arr, before)


class TestTheWeightedMedianDoesNotMaterialiseEveryObservation:
    """`n_obs` accumulates monotonically and `_aggregate` runs over all rows on every append."""

    def test_it_agrees_with_the_expanded_form(self):
        """Same answer, without the list."""
        from mlframe.utils._param_oracle_store import _weighted_median

        pairs = [(1.0, 3), (5.0, 1), (2.0, 6)]
        expanded = [v for v, w in pairs for _ in range(w)]
        assert _weighted_median(pairs) == pytest.approx(float(np.median(expanded)), abs=1.0)

    def test_a_huge_weight_is_cheap(self):
        """The old form built one Python float per observation, inside a cross-process file lock."""
        from mlframe.utils._param_oracle_store import _weighted_median

        assert _weighted_median([(1.0, 5_000_000), (9.0, 1)]) == 1.0

    def test_it_handles_the_empty_case(self):
        """A metric with no rows must not raise inside the lock."""
        from mlframe.utils._param_oracle_store import _weighted_median

        assert np.isnan(_weighted_median([]))


def test_the_eviction_protects_the_sidecar_too():
    """`<key>.sha256` is a separate file, so a pass triggered by the very `put` that wrote it could delete the
    checksum while sparing the payload -- and the next `get` then fails closed, deletes both, and the expensive
    compute is repeated."""
    from mlframe.utils import disk_cache as m

    import ast

    from tests._source_ast import module_ast, string_literals

    # Structural: the bug needs an eviction pass triggered by the very `put` that wrote the entry, with the
    # payload spared and its checksum deleted -- a race this suite cannot stage deterministically. What is
    # checkable is that the protection is a SET membership, not an identity test against one path, so the
    # `.sha256` sidecar is covered alongside its payload.
    tree = module_ast(m)
    assert any(s.endswith(".sha256") for s in string_literals(tree)), "the checksum sidecar suffix is gone"
    membership = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare) and any(isinstance(op, ast.In) for op in node.ops) and any(isinstance(x, ast.Name) and x.id == "_protected" for x in ast.walk(node))
    ]
    assert membership, "the eviction pass no longer tests membership of the protected SET, so the sidecar can be deleted alone"


def test_the_polars_conversion_keeps_numpy_backed_dtypes():
    """`to_pandas_or_array` must hand back NUMPY-backed columns, not pyarrow-backed ones.

    This previously asserted the opposite: `use_pyarrow_extension_array=True` was introduced to make the
    conversion genuinely zero-copy, since a plain `.to_pandas()` materialises a second copy of the frame. The
    memory concern is real, but that form changes the dtype family every caller receives -- `float[pyarrow]`
    where it used to be `np.float32` -- and this helper sits under sklearn, numba and CatBoost call sites that
    expect numpy-backed columns. Choosing the Arrow view is the caller's decision at the suite boundary, not a
    shared normaliser's, so it was reverted and the docstring now says plainly that it copies.
    """
    import polars as pl

    from mlframe.core.frame_compat import to_pandas_or_array

    out = to_pandas_or_array(pl.DataFrame({"f32": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float32), "i64": pl.Series([10, 20, 30], dtype=pl.Int64)}))
    assert out["f32"].dtype == np.float32, f"expected a numpy float32 column, got {out['f32'].dtype!r}"
    assert out["i64"].dtype == np.int64, f"expected a numpy int64 column, got {out['i64'].dtype!r}"


# `test_the_hash_docstring_no_longer_claims_pickle` used to sit here. It asserted the wording of a docstring,
# and the thing that wording corrected -- that `_feed`'s last-resort branch is `repr(obj)`, which is not stable
# across runs, rather than the documented "pickle protocol 0" -- cannot be observed from inside one process:
# both produce a stable digest within a run, and the difference only shows across interpreter restarts for an
# object whose repr embeds an address. The permutation-blindness test above pins the part of this hash's
# contract that IS checkable.


def test_the_summary_hash_is_blind_to_an_interior_row_permutation():
    """Two arrays differing only by a permutation of their INTERIOR rows hash identically.

    Driven rather than read off the docstring. `hash_array_summary` is deliberately sub-O(N) -- shape, dtype,
    the first and last 64 rows, and per-column sum/min/max -- so an interior reordering leaves every one of
    those unchanged. That is right for a consumer whose output does not depend on row order (MRMR bin edges)
    and wrong for one whose output does, and the only way to know which side you are on is for this property
    to be pinned rather than described.
    """
    from mlframe.utils.disk_cache import hash_array_summary

    rng = np.random.default_rng(0)
    # Integer dtype: the per-column sum is EXACT, which is the regime where the invariance actually holds.
    arr = rng.integers(-1000, 1000, size=(400, 3))

    # Permute only the interior, leaving the first and last 64 rows in place.
    permuted = arr.copy()
    interior = np.arange(64, arr.shape[0] - 64)
    permuted[interior] = arr[rng.permutation(interior)]

    assert not np.array_equal(arr, permuted), "the fixture did not actually reorder anything"
    assert hash_array_summary(arr) == hash_array_summary(permuted), "the summary hash distinguished an interior permutation of an INTEGER array"

    # A change it MUST see, so the assertion above is not simply reporting a constant hash.
    changed = arr.copy()
    changed[0, 0] += 1
    assert hash_array_summary(arr) != hash_array_summary(changed), "the summary hash missed a changed value in the first rows"

    # ...and the limit of the guarantee, which the module docstring used to overstate. On FLOAT data the
    # per-column sum is order-dependent -- addition is not associative -- so the same interior reordering
    # shifts it by a few ulp and the key changes. The direction is safe (an unnecessary miss and a recompute,
    # never a wrong hit), but a caller must not expect a reordered float frame to hit the cache.
    farr = rng.normal(size=(400, 3))
    fpermuted = farr.copy()
    fpermuted[interior] = farr[rng.permutation(interior)]
    assert not np.array_equal(farr.sum(axis=0), fpermuted.sum(axis=0)), "the float fixture summed identically, so it cannot show the limit"
    assert hash_array_summary(farr) != hash_array_summary(fpermuted), "float permutation-invariance now holds exactly; the docstring caveat can be dropped"


def test_the_correlation_baseline_drops_the_columns_own_missing_rows():
    """`finite_fill` is finite everywhere by construction, so the pairwise mask dropped only the target's
    non-finites; 40% of the rows were then a constant, attenuating the baseline the threshold is derived from."""
    import ast
    import importlib

    from tests._source_ast import module_ast

    # `from mlframe.preprocessing import gaussian_power_transform_search` binds the re-exported FUNCTION of
    # that name, not the submodule, so the plain form yields an object with no `__file__`.
    m = importlib.import_module("mlframe.preprocessing.gaussian_power_transform_search")

    # Structural: the pairwise mask must be built from the RAW column, not from `finite_fill`, which is finite
    # everywhere by construction and therefore masks nothing. Both forms return a correlation; the wrong one
    # just computes it over rows where 40% of the column is a constant fill, attenuating the baseline the
    # threshold is derived from -- a quieter number, not a different shape, so no assertion on the output
    # separates them without reconstructing the whole search.
    masks = [
        node for node in ast.walk(module_ast(m)) if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pair_mask" for t in node.targets)
    ]
    assert masks, "`pair_mask` is no longer assigned; this test needs updating if the baseline was restructured"
    sources = {n.id for node in masks for n in ast.walk(node.value) if isinstance(n, ast.Name)}
    assert "raw" in sources, f"the pairwise mask no longer reads the raw column; it reads {sorted(sources)}"
    assert "finite_fill" not in sources, "the pairwise mask is built from finite_fill again, which is finite everywhere and masks nothing"


class TestTheSiblingFillToleratesANonNumericOrder:
    """`interpolate(method="index")` needs a numeric, unique, ordered index; the contract guarantees none."""

    def test_a_string_order_column_does_not_raise(self, caplog):
        """The docstring explicitly blesses "a sortable ordering", which a quarter label satisfies."""
        from mlframe.preprocessing.sibling_group_cold_start_fill import sibling_group_cold_start_fill

        df = pd.DataFrame(
            {
                "grp": ["a", "a", "b", "b", "c", "c"],
                "q": ["2024Q1", "2024Q1", "2024Q2", "2024Q2", "2024Q3", "2024Q3"],
                "v": [1.0, 2.0, np.nan, np.nan, 5.0, 6.0],
            }
        )
        with caplog.at_level(logging.WARNING, logger="mlframe.preprocessing.sibling_group_cold_start_fill"):
            out = sibling_group_cold_start_fill(df, group_col="grp", order_col="q", value_col="v", interpolate=True)
        assert out.notna().all()  # returns a per-row Series, not a frame
        assert any("non-numeric" in r.message for r in caplog.records), [r.message for r in caplog.records]

    def test_a_numeric_order_column_still_weights_by_value(self):
        """The distance weighting is the entire reason the module chose `method="index"`."""
        from mlframe.preprocessing.sibling_group_cold_start_fill import sibling_group_cold_start_fill

        df = pd.DataFrame(
            {
                "grp": ["a", "b", "c"],
                "o": [0.0, 1.0, 100.0],
                "v": [0.0, np.nan, 100.0],
            }
        )
        out = sibling_group_cold_start_fill(df, group_col="grp", order_col="o", value_col="v", interpolate=True)
        # At order 1 of a 0..100 span the value-weighted fill is ~1, not the positional midpoint 50.
        assert float(out.loc[df["grp"] == "b"].iloc[0]) < 10.0


def test_the_synthetic_rows_take_their_label_from_the_true_last_period():
    """A synthetic row's LABEL comes from the entity's true last period, not from the truncated vintage.

    Driven rather than read out of the source. Only `feature_cols` used to be overwritten, so every other
    column -- the label above all -- kept the value of the earlier row the synthetic vintage was built from.
    For an entity-level label the two coincide and nothing breaks; for a PER-PERIOD label (a rolling default
    flag, a next-period target) every synthetic row was trained against the earlier period's answer. The
    fixture below gives each period a distinct label so the two sources are never the same value.
    """
    from mlframe.preprocessing.temporal_drift_augment import augment_temporal_drift

    rng = np.random.default_rng(0)
    rows = []
    for entity in range(6):
        for t in range(5):
            # `label` is per-PERIOD and strictly increasing, so the truncated vintage's label and the true
            # last period's label can never coincide by accident.
            rows.append({"entity_id": entity, "t": t, "x": float(rng.normal()), "label": float(10 * entity + t)})
    df = pd.DataFrame(rows)

    out = augment_temporal_drift(df, entity_col="entity_id", time_col="t", feature_cols=["x"], n_drop_options=(1,), min_history=2)
    synth = out.loc[out["_temporal_drift_augmented"]]
    assert not synth.empty, "the fixture produced no synthetic rows, so this test would pass vacuously"

    true_last_label = df.loc[df.groupby("entity_id")["t"].idxmax()].set_index("entity_id")["label"]
    for entity_id, group in synth.groupby("entity_id"):
        expected = true_last_label[entity_id]
        assert set(group["label"]) == {expected}, f"entity {entity_id}: label is {sorted(set(group['label']))}, expected the true last period's {expected}"
        # ...and that is genuinely a DIFFERENT value from the truncated vintage's own label.
        vintage_t = sorted(df.loc[df["entity_id"] == entity_id, "t"].unique())[-2]
        vintage_label = float(df.loc[(df["entity_id"] == entity_id) & (df["t"] == vintage_t), "label"].iloc[0])
        assert vintage_label != expected, "the fixture cannot distinguish the two label sources"
