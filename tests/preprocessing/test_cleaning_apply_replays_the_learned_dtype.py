"""Train and apply cast the same column to different dtypes, and integer columns crashed outright.

`analyse_and_clean_features` merges rare values into `default_na_val` (NaN by default) and casts the column to
`default_float_type` -- float32 -- because a NaN cannot live in an integer column. `apply_features_cleaning`
then cast the same column to *the apply frame's own* dtype instead of the learned one.

Two consequences. Silently: a float64 column is fitted on float32-rounded values and scored on float64 ones, so
any downstream binning or edge computation sees a different tie structure on train than on validation. Loudly:
an int64 column raises `pandas.errors.IntCastingNaNError: Cannot convert non-finite values (NA or inf) to
integer`, since the transform introduces the NaN that the cast back to int64 cannot hold. That is the DEFAULT
configuration, so any integer feature with a rare value broke the apply path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.preprocessing.cleaning import analyse_and_clean_features, apply_features_cleaning

N = 20_000


def _frame():
    """An int64 and a float64 column, each with a handful of values rare enough to be merged."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "i": np.concatenate([rng.integers(0, 3, N - 4), [99, 99, 98, 97]]).astype("int64"),
            "f": np.concatenate([rng.normal(0, 1, N - 4).round(1), [1e6, 1e6, 9e5, 8e5]]).astype("float64"),
        }
    )


@pytest.fixture
def learned():
    """The cleaning result learned on a train frame that really does trigger a rare-value merge."""
    res = analyse_and_clean_features(_frame(), update_data=True)
    assert res["features_transforms"], "fixture no longer triggers a rare-value merge; the test would prove nothing"
    return res


class TestTheLearnedDtypeIsRecorded:
    """The apply side cannot replay a decision that was never written down."""

    def test_the_result_carries_the_learned_dtypes(self, learned):
        """A new key, so an old result dict must still be handled -- see the fallback test below."""
        assert learned["features_dtypes"]

    def test_a_merged_integer_column_is_recorded_as_float(self, learned):
        """The cast is forced by the NaN the merge introduces."""
        assert learned["features_dtypes"]["i"].startswith("float")

    def test_every_transformed_column_has_a_dtype(self, learned):
        """A transform with no recorded dtype falls back to the leaky path."""
        assert set(learned["features_transforms"]) <= set(learned["features_dtypes"])


class TestApplyDoesNotCrashOnIntegers:
    """The loud half."""

    def test_applying_to_an_int_column_succeeds(self, learned):
        """`IntCastingNaNError` on the default configuration."""
        apply_features_cleaning(_frame(), learned)

    def test_the_mutating_path_succeeds_too(self, learned):
        """Both branches of `apply_features_cleaning` re-derived the dtype the same way."""
        apply_features_cleaning(_frame(), learned, update_data=True)

    def test_the_merged_values_became_missing(self, learned):
        """The transform must still do its job after the cast is fixed."""
        out = apply_features_cleaning(_frame(), learned)
        assert out["i"].isna().sum() == 4


class TestTrainAndApplyAgreeOnDtype:
    """The silent half."""

    def test_the_applied_dtype_matches_what_was_learned(self, learned):
        """Stated directly: the two must not diverge."""
        out = apply_features_cleaning(_frame(), learned)
        assert out["i"].dtype.name == learned["features_dtypes"]["i"]

    def test_the_apply_frames_own_dtype_does_not_decide(self, learned):
        """Handing the apply side a differently-typed frame must not change the output dtype."""
        wide = _frame()
        wide["i"] = wide["i"].astype("float64")
        assert apply_features_cleaning(wide, learned)["i"].dtype.name == learned["features_dtypes"]["i"]

    def test_an_untransformed_column_is_left_alone(self, learned):
        """Only columns with recorded transforms are cast."""
        out = apply_features_cleaning(_frame(), learned)
        assert out["f"].dtype.name == _frame()["f"].dtype.name or "f" in learned["features_dtypes"]

    def test_a_legacy_result_without_the_key_still_applies(self, learned):
        """Result dicts pickled before `features_dtypes` existed must keep working."""
        legacy = {k: v for k, v in learned.items() if k != "features_dtypes"}
        legacy["features_transforms"] = {c: t for c, t in learned["features_transforms"].items() if not c.startswith("i")}
        apply_features_cleaning(_frame(), legacy)
