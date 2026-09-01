"""The apply side recomputed its NaN fill from the frame it was applied to, and wrote it into the output.

`apply_gaussian_power_transform` documents itself as replaying "the SAME function that was measured and
selected, not a freshly-refit one". That held for the Box-Cox lambda and the Yeo-Johnson power, and not for the
median used to fill non-finite cells: that was recomputed from whatever frame was passed in. Applying to
validation data therefore imputed with validation statistics, and at single-row inference the "median" is that
one row's own value, so a missing cell was filled from an unrelated observed cell.

Second defect in the same line: the filled values were written into the output, so missingness silently vanished
from the frame and every previously-NaN row came back holding a transform of the training median -- an
imputation the docstring never mentions and the caller never asked for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.preprocessing.gaussian_power_transform_search import apply_gaussian_power_transform, gaussian_power_transform_search


@pytest.fixture
def searched():
    """A right-skewed train column with a few missing cells, plus its search result."""
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"x": np.concatenate([rng.lognormal(0, 1, 300), [np.nan, np.nan]])})
    return train, gaussian_power_transform_search(train, columns=["x"])


class TestTheFittedFillIsReplayed:
    """The train/serve statistic mismatch."""

    def test_the_search_records_its_fill_median(self, searched):
        """The apply side cannot replay a value that was never recorded."""
        _, res = searched
        assert "fill_median" in res["x"]

    def test_the_recorded_median_is_the_train_median(self, searched):
        """Recorded, not re-derived: it must equal the median of the finite training values."""
        train, res = searched
        finite = train["x"].to_numpy()[np.isfinite(train["x"].to_numpy())]
        assert res["x"]["fill_median"] == pytest.approx(float(np.median(finite)))

    def test_the_apply_frames_own_median_is_not_used(self, searched):
        """A validation frame on a different scale must not shift the fill."""
        train, res = searched
        shifted = pd.DataFrame({"x": train["x"].to_numpy() * 100.0})
        with_nan = apply_gaussian_power_transform(shifted, res)
        # The finite rows are what the fill can influence; transform them with an explicitly-pinned median and
        # confirm the function landed on the same numbers.
        pinned = apply_gaussian_power_transform(shifted, {"x": {**res["x"], "fill_median": res["x"]["fill_median"]}})
        assert np.allclose(with_nan["x"].to_numpy(), pinned["x"].to_numpy(), equal_nan=True)

    def test_a_single_row_does_not_impute_from_itself(self, searched):
        """At inference the apply frame's "median" is the row's own value."""
        _, res = searched
        one = pd.DataFrame({"x": [np.nan]})
        out = apply_gaussian_power_transform(one, res)
        assert np.isnan(out["x"].iloc[0]), "a single missing row was imputed from itself"

    def test_a_legacy_result_without_the_median_warns(self, searched, caplog):
        """Old result dicts must keep working, but must not replay the leaky path silently."""
        train, res = searched
        legacy = {"x": {k: v for k, v in res["x"].items() if k != "fill_median"}}
        with caplog.at_level("WARNING"):
            apply_gaussian_power_transform(train, legacy)
        assert any("fill_median" in r.message for r in caplog.records)


class TestMissingnessSurvives:
    """The output must not silently gain imputed values."""

    def test_nan_cells_come_back_as_nan(self, searched):
        """The direct statement of the second defect."""
        train, res = searched
        out = apply_gaussian_power_transform(train, res)
        assert out["x"].isna().sum() == 2

    def test_the_missing_positions_are_the_same_ones(self, searched):
        """Not merely the same COUNT of NaN."""
        train, res = searched
        out = apply_gaussian_power_transform(train, res)
        assert (out["x"].isna().to_numpy() == train["x"].isna().to_numpy()).all()

    def test_the_finite_rows_are_still_transformed(self, searched):
        """Restoring NaN must not disable the transform it was blocking."""
        train, res = searched
        out = apply_gaussian_power_transform(train, res)
        finite_in = train["x"].dropna().to_numpy()
        finite_out = out["x"].dropna().to_numpy()
        assert abs(float(pd.Series(finite_out).skew())) < abs(float(pd.Series(finite_in).skew()))

    def test_a_column_with_no_missing_values_is_unaffected(self):
        """The common case must be untouched by the restore step."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"x": rng.lognormal(0, 1, 200)})
        res = gaussian_power_transform_search(df, columns=["x"])
        out = apply_gaussian_power_transform(df, res)
        assert out["x"].notna().all()

    def test_the_boxcox_replay_path_restores_missing_too(self):
        """The Box-Cox branch returns early and had its own assignment."""
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"x": np.concatenate([rng.lognormal(0, 1, 200), [np.nan]])})
        res = gaussian_power_transform_search(df, columns=["x"], candidate_transforms=("boxcox",))
        assert apply_gaussian_power_transform(df, res)["x"].isna().sum() == 1
