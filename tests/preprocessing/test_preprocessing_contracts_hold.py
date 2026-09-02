"""Four preprocessing modules promised something their code did not do.

  * `outlier_cap_or_missing` told callers to "apply the returned bounds' equivalent transform to test data"
    while returning only a DataFrame. There were no bounds to reuse, so the only options were re-running the
    whole function on the test frame -- recomputing the thresholds from the test distribution, the exact
    pattern the advice exists to avoid -- or reimplementing a private helper at the call site.
  * `train_test_support_screen` validated `target_col` and then never read it, so with the default column
    list the target was screened as a categorical feature and the output recommended an encoding for it.
  * `impute_with_missing_indicator` promised `group_col` "itself is never imputed by this call", but with the
    default column list a nullable grouping column landed in it: grouped by itself, the NaN group's own stat
    is NaN, so the apply step filled those rows from the global fallback and silently altered the key.
  * `apply_rare_category_collapse` used `Series.where`, which raises on a pandas `category` column -- the
    natural input dtype here, since `cleaning.analyse_and_clean_features` produces it automatically. The fit
    half succeeded, so the crash surfaced only at apply time, potentially on the inference frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.preprocessing.outlier_capping_or_missing import (
    apply_outlier_cap_or_missing,
    fit_outlier_cap_or_missing,
    outlier_cap_or_missing,
)
from mlframe.preprocessing.rare_count_pruning import apply_rare_category_collapse


class TestOutlierCappingHasAFitApplyPair:
    """The thresholds have to be reusable, or the leakage advice is unfollowable."""

    @pytest.fixture
    def frames(self):
        """A train frame with outliers and a test frame on a deliberately different scale."""
        rng = np.random.default_rng(0)
        train = pd.DataFrame({"x": np.concatenate([rng.normal(0, 1, 500), [40.0, -40.0]])})
        test = pd.DataFrame({"x": np.concatenate([rng.normal(0, 1, 200) * 5.0, [80.0]])})
        return train, test

    def test_the_bounds_are_returned(self, frames):
        """There was nothing to hand to a test frame at all."""
        train, _ = frames
        state = fit_outlier_cap_or_missing(train)
        assert np.isfinite(state["columns"]["x"]["lower"]) and np.isfinite(state["columns"]["x"]["upper"])

    def test_the_test_frame_is_capped_at_the_train_bounds(self, frames):
        """Re-fitting on the test frame would derive its thresholds from the rows being filtered."""
        train, test = frames
        state = fit_outlier_cap_or_missing(train)
        out = apply_outlier_cap_or_missing(test, state)
        assert out["x"].max() <= state["columns"]["x"]["upper"] + 1e-9

    def test_that_differs_from_refitting_on_the_test_frame(self, frames):
        """The discriminating half: the test frame is on a wider scale, so its own bounds are wider."""
        train, test = frames
        replayed = apply_outlier_cap_or_missing(test, fit_outlier_cap_or_missing(train))
        refitted = outlier_cap_or_missing(test)
        assert not np.allclose(replayed["x"].to_numpy(), refitted["x"].to_numpy())

    def test_the_impute_median_is_persisted(self, frames):
        """`missing_impute` recomputed its median from the frame being transformed."""
        train, test = frames
        state = fit_outlier_cap_or_missing(train, mode="missing_impute")
        assert np.isfinite(state["columns"]["x"]["median"])
        out = apply_outlier_cap_or_missing(test, state)
        assert out["x"].notna().all()

    def test_the_single_frame_wrapper_is_unchanged(self, frames):
        """`outlier_cap_or_missing(df)` must still be exactly apply(df, fit(df))."""
        train, _ = frames
        assert np.allclose(
            outlier_cap_or_missing(train)["x"].to_numpy(),
            apply_outlier_cap_or_missing(train, fit_outlier_cap_or_missing(train))["x"].to_numpy(),
        )

    def test_a_column_absent_from_the_test_frame_is_skipped(self, frames):
        """Schema-forgiving, matching the sibling apply functions."""
        train, _ = frames
        state = fit_outlier_cap_or_missing(train)
        assert apply_outlier_cap_or_missing(pd.DataFrame({"other": [1.0, 2.0]}), state).shape == (2, 1)


class TestTheTargetIsNotScreenedAsAFeature:
    """`target_col` was validated and then ignored."""

    def _screen(self, **kw):
        """Run the support screen on a frame whose test half carries the label column."""
        from mlframe.preprocessing.category_support import train_test_support_screen

        rng = np.random.default_rng(0)
        train = pd.DataFrame({"cat": rng.choice(list("abc"), 300), "y": rng.integers(0, 2, 300)})
        test = pd.DataFrame({"cat": rng.choice(list("abx"), 200), "y": rng.integers(0, 2, 200)})
        return train_test_support_screen(train, test, target_col="y", **kw)

    def test_the_target_gets_no_row(self):
        """The output used to recommend an encoding for the target itself."""
        assert "y" not in set(self._screen()["column"])

    def test_the_feature_is_still_screened(self):
        """Excluding the target must not empty the report."""
        assert "cat" in set(self._screen()["column"])

    def test_an_explicit_list_containing_the_target_is_still_refused(self):
        """The documented constraint, which was the parameter's only effect before."""
        from mlframe.preprocessing.category_support import train_test_support_screen

        rng = np.random.default_rng(1)
        train = pd.DataFrame({"cat": rng.choice(list("abc"), 100), "y": rng.integers(0, 2, 100)})
        with pytest.raises(ValueError):
            train_test_support_screen(train, train, target_col="y", categorical_cols=["cat", "y"])


class TestTheGroupingKeyIsNotImputed:
    """The apply step was rewriting the column it grouped by."""

    def _frame(self):
        """A frame whose grouping column itself has nulls."""
        return pd.DataFrame(
            {
                "region": ["a", "b", None, "a", "b", None, "a", "b"],
                "v": [1.0, 2.0, 3.0, None, 5.0, 6.0, None, 8.0],
            }
        )

    def test_the_group_column_is_excluded_from_the_default_list(self):
        """It landed there whenever it had a null, which is the only case that matters."""
        from mlframe.preprocessing.missing_indicator_pairing import fit_missing_indicator_imputation

        state = fit_missing_indicator_imputation(self._frame(), strategy="median", group_col="region")
        assert "region" not in state.get("columns", state)

    def test_the_group_column_survives_the_transform_unchanged(self):
        """The user-visible consequence: the grouping key silently altered."""
        from mlframe.preprocessing.missing_indicator_pairing import apply_missing_indicator_imputation, fit_missing_indicator_imputation

        df = self._frame()
        state = fit_missing_indicator_imputation(df, strategy="median", group_col="region")
        out = apply_missing_indicator_imputation(df, state)
        assert out["region"].isna().sum() == df["region"].isna().sum()

    def test_no_indicator_column_appears_for_the_group_key(self):
        """The docstring never promised one."""
        from mlframe.preprocessing.missing_indicator_pairing import apply_missing_indicator_imputation, fit_missing_indicator_imputation

        df = self._frame()
        out = apply_missing_indicator_imputation(df, fit_missing_indicator_imputation(df, strategy="median", group_col="region"))
        assert not any(c.startswith("region_was_missing") for c in out.columns)

    def test_an_explicit_list_naming_the_group_key_is_refused(self):
        """Silently dropping it from an EXPLICIT list would be its own surprise."""
        from mlframe.preprocessing.missing_indicator_pairing import fit_missing_indicator_imputation

        with pytest.raises(ValueError, match="cannot be imputed"):
            fit_missing_indicator_imputation(self._frame(), strategy="median", group_col="region", columns=["region", "v"])

    def test_the_other_column_is_still_imputed(self):
        """Excluding the key must not disable the imputation itself."""
        from mlframe.preprocessing.missing_indicator_pairing import apply_missing_indicator_imputation, fit_missing_indicator_imputation

        df = self._frame()
        out = apply_missing_indicator_imputation(df, fit_missing_indicator_imputation(df, strategy="median", group_col="region"))
        assert out["v"].notna().all()


class TestRareCollapseWorksOnCategoryDtype:
    """`category` is what this package's own cleaning step produces for fewly-valued object columns."""

    def _frame(self):
        """A category-dtype column with a rare level."""
        return pd.DataFrame({"c": pd.Series(["a", "a", "a", "b", "b", "rare"], dtype="category")})

    def test_it_does_not_raise(self):
        """`TypeError: Cannot setitem on a Categorical with a new category (__other__)`."""
        apply_rare_category_collapse(self._frame(), {"c": ["rare"]})

    def test_the_rare_level_is_collapsed(self):
        """Not raising is not enough; the collapse has to happen."""
        out = apply_rare_category_collapse(self._frame(), {"c": ["rare"]}, other_label="__other__")
        assert "rare" not in set(out["c"].astype(str)) and "__other__" in set(out["c"].astype(str))

    def test_the_column_stays_categorical(self):
        """Silently widening it to object would change every downstream dtype check."""
        out = apply_rare_category_collapse(self._frame(), {"c": ["rare"]})
        assert isinstance(out["c"].dtype, pd.CategoricalDtype)

    def test_an_object_column_is_unaffected(self):
        """The path that already worked."""
        df = pd.DataFrame({"c": ["a", "a", "b", "rare"]})
        out = apply_rare_category_collapse(df, {"c": ["rare"]}, other_label="__other__")
        assert out["c"].tolist() == ["a", "a", "b", "__other__"]

    def test_the_adversarial_rebin_twin_also_works(self):
        """The same construct, same crash, in `_merge_skewed_categories`."""
        from mlframe.preprocessing.adversarial_rebin import _replace_with_label

        s = pd.Series(["a", "b", "rare"], dtype="category")
        out = _replace_with_label(s, s.isin(pd.Index(["rare"])), "__other__")
        assert "__other__" in set(out.astype(str))
