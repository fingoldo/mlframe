"""Row-wise extremality was ranked within whatever frame it was handed, which is a train/serve skew.

Measured on the original implementation: one row scored ``[0.808, 0.793, 0.653]`` inside a 50k-row split and
``[0.0, 0.0, 0.0]`` scored on its own, because a single row is its own median. Train, val and test were each
ranked against themselves too, so the model was trained on one definition of the feature and served another.

Ranking against a reference fixed at fit time removes the dependence on which rows are present. It is also
somewhat cheaper on later frames -- measured at n=2M x 20 columns: 17.65s within-batch vs 12.23s against a
fitted reference (1.44x), with a 2.41s one-off fit. The speed is a side effect; the point is the definition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_engineering.row_wise_extremality import (
    _compute_extremality_matrix,
    row_wise_extremality_index,
    row_wise_top_k_extreme_columns,
)
from mlframe.feature_engineering.row_wise_extremality_reference import (
    DEFAULT_MAX_REFERENCE_ROWS,
    extremality_matrix_from_reference,
    fit_extremality_reference,
)


@pytest.fixture(scope="module")
def frame():
    """A plain numeric frame big enough for the rank convention's O(1/n) gap to be small."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((5000, 6)), columns=[f"c{i}" for i in range(6)])


@pytest.fixture(scope="module")
def reference(frame):
    """The fit-time reference built from that frame."""
    return fit_extremality_reference(frame)


class TestTheSkewIsGone:
    """The defect, stated as behaviour."""

    def test_one_row_scores_the_same_alone_as_in_the_batch(self, frame, reference):
        """This is the whole bug: at predict time rows arrive one at a time."""
        single, _ = extremality_matrix_from_reference(frame.iloc[[123]], reference)
        batch, _ = extremality_matrix_from_reference(frame, reference)
        assert single[0] == pytest.approx(batch[123], abs=1e-12)

    def test_within_batch_ranking_really_did_collapse_a_lone_row(self, frame):
        """Guards the test above: without a reference a single row still scores 0 everywhere."""
        lone, _ = _compute_extremality_matrix(frame.iloc[[123]], None)
        assert np.allclose(lone[0], 0.0)

    def test_a_subset_scores_the_same_as_the_full_frame(self, frame, reference):
        """Val and test were each ranked against themselves; now every split shares one definition."""
        rows = [7, 500, 4999]
        subset, _ = extremality_matrix_from_reference(frame.iloc[rows], reference)
        full, _ = extremality_matrix_from_reference(frame, reference)
        assert subset == pytest.approx(full[rows], abs=1e-12)


class TestItAgreesWithTheOldDefinitionOnFittingData:
    """The change must not silently redefine the score on the data it was fitted on."""

    def test_the_two_agree_to_the_rank_convention_gap(self, frame, reference):
        """Both are |percentile - 0.5| * 2; the conventions differ by one half-step, i.e. O(1/n)."""
        old, _ = _compute_extremality_matrix(frame, None)
        new, _ = extremality_matrix_from_reference(frame, reference)
        assert np.nanmax(np.abs(old - new)) < 5.0 / len(frame)

    def test_the_median_row_still_scores_near_zero(self):
        """The anchor of the definition: a value at its column's median is not extreme."""
        df = pd.DataFrame({"c": np.arange(1001, dtype=float)})
        ref = fit_extremality_reference(df)
        mid, _ = extremality_matrix_from_reference(pd.DataFrame({"c": [500.0]}), ref)
        assert mid[0, 0] < 0.01

    def test_an_extreme_value_still_scores_near_one(self):
        """The other anchor: the top of the observed range is maximally extreme."""
        df = pd.DataFrame({"c": np.arange(1001, dtype=float)})
        ref = fit_extremality_reference(df)
        top, _ = extremality_matrix_from_reference(pd.DataFrame({"c": [1000.0]}), ref)
        assert top[0, 0] > 0.99

    def test_a_value_beyond_the_reference_range_saturates(self):
        """Production sees values the fit never did; they must land at the extreme, not wrap or raise."""
        df = pd.DataFrame({"c": np.arange(1001, dtype=float)})
        ref = fit_extremality_reference(df)
        out, _ = extremality_matrix_from_reference(pd.DataFrame({"c": [10_000.0, -10_000.0]}), ref)
        assert out[0, 0] > 0.99
        assert out[1, 0] > 0.99


class TestDegenerateInputs:
    """What the reference does when there is nothing to compare against."""

    def test_nan_values_stay_nan(self, reference):
        """A missing value has no position in a distribution."""
        out, _ = extremality_matrix_from_reference(pd.DataFrame({f"c{i}": [np.nan] for i in range(6)}), reference)
        assert np.isnan(out).all()

    def test_an_all_nan_column_gets_an_empty_reference(self):
        """Nothing finite to describe, so nothing is stored."""
        ref = fit_extremality_reference(pd.DataFrame({"c": [np.nan, np.nan]}))
        assert ref["c"].size == 0

    def test_a_column_with_no_reference_scores_nan_rather_than_being_re_ranked(self, reference):
        """Re-ranking it within the batch would quietly restore the skew for that column."""
        out, _ = extremality_matrix_from_reference(pd.DataFrame({"unseen": [1.0, 2.0, 3.0]}), {})
        assert np.isnan(out).all()

    def test_an_empty_frame_returns_an_empty_matrix(self, reference):
        """Zero rows is a legitimate split shape, not an error."""
        out, _ = extremality_matrix_from_reference(pd.DataFrame({"c0": pd.Series(dtype=float)}), reference)
        assert out.shape[0] == 0

    def test_the_reference_is_capped_in_size(self):
        """A 2M-row fit must not carry 2M values per column around for the life of the model."""
        big = pd.DataFrame({"c": np.arange(DEFAULT_MAX_REFERENCE_ROWS * 2, dtype=float)})
        assert fit_extremality_reference(big)["c"].size == DEFAULT_MAX_REFERENCE_ROWS

    def test_a_capped_reference_still_ranks_correctly(self):
        """Subsampling the sorted values must preserve the quantile grid."""
        n = DEFAULT_MAX_REFERENCE_ROWS * 2
        big = pd.DataFrame({"c": np.arange(n, dtype=float)})
        ref = fit_extremality_reference(big)
        out, _ = extremality_matrix_from_reference(pd.DataFrame({"c": [float(n // 2), 0.0]}), ref)
        assert out[0, 0] < 0.01
        assert out[1, 0] > 0.99


class TestThePublicApiTakesAReference:
    """Both entry points, so the aggregate index and the per-row breakdown stay consistent."""

    def test_the_index_accepts_a_reference(self, frame, reference):
        """The row-mean score has to be servable too."""
        one = row_wise_extremality_index(frame.iloc[[10]], reference=reference)
        many = row_wise_extremality_index(frame, reference=reference)
        assert one.iloc[0] == pytest.approx(many.iloc[10], abs=1e-12)

    def test_top_k_accepts_a_reference(self, frame, reference):
        """This is the one the training pipeline actually calls."""
        one = row_wise_top_k_extreme_columns(frame.iloc[[10]], k=2, reference=reference)
        many = row_wise_top_k_extreme_columns(frame, k=2, reference=reference)
        assert one["top1_score"].iloc[0] == pytest.approx(many["top1_score"].iloc[10], abs=1e-12)

    def test_omitting_the_reference_keeps_the_old_behaviour(self, frame):
        """The batch-relative score stays available for callers who want a descriptive statistic."""
        lone = row_wise_top_k_extreme_columns(frame.iloc[[10]], k=2)
        assert lone["top1_score"].iloc[0] == pytest.approx(0.0)


class TestTheSuiteUsesItByDefault:
    """A fix nobody turns on changes nothing."""

    def test_the_config_defaults_to_the_fitted_reference(self):
        """The old behaviour stays reachable, but it is no longer what a production fit gets."""
        from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig

        assert PreprocessingExtensionsConfig().row_wise_extreme_columns_fit_reference is True


class TestTiedAndLowCardinalityColumns:
    """The blind spot that shipped: the first version was only ever tested on continuous data.

    Counting values STRICTLY below gives every row of a constant or heavily-tied column a percentile of ~0, i.e.
    maximal extremality. A production frame is mostly ratios, counts and mostly-zero columns, so the whole
    feature collapsed to a constant and the suite's zero-variance pre-screen dropped all three
    ``row_extreme_topN_score`` columns -- a feature silently deleted by a bug in its own fix.
    """

    def test_a_constant_column_is_not_extreme(self):
        """Every value IS the median; the old form scored all of them 0.9995."""
        df = pd.DataFrame({"c": np.ones(500)})
        out, _ = extremality_matrix_from_reference(df, fit_extremality_reference(df))
        assert np.nanmax(np.abs(out)) < 1e-9

    def test_a_mostly_zero_column_flags_only_the_non_zeros(self):
        """The 5% that differ are the extreme ones, not all 100%."""
        rng = np.random.default_rng(0)
        vals = np.where(rng.random(2000) < 0.95, 0.0, rng.random(2000) + 1.0)
        df = pd.DataFrame({"c": vals})
        out, _ = extremality_matrix_from_reference(df, fit_extremality_reference(df))
        assert out[vals == 0.0].mean() < out[vals != 0.0].mean()

    def test_a_tied_column_keeps_real_variance(self):
        """Zero variance is exactly what got the feature dropped by the pre-screen."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"c": np.round(rng.random(2000), 1)})
        out, _ = extremality_matrix_from_reference(df, fit_extremality_reference(df))
        assert np.nanstd(out) > 0.2

    def test_the_top_k_scores_keep_real_variance_on_a_realistic_frame(self):
        """End to end, on the column mix a production frame actually has."""
        rng = np.random.default_rng(2)
        n = 2000
        df = pd.DataFrame({
            "const": np.ones(n),
            "mostly_zero": np.where(rng.random(n) < 0.95, 0.0, rng.random(n)),
            "ratio": np.round(rng.random(n), 1),
            "cont": rng.standard_normal(n),
        })
        ref = fit_extremality_reference(df, list(df.columns))
        out = row_wise_top_k_extreme_columns(df, columns=list(df.columns), k=3, reference=ref)
        scores = out[[c for c in out.columns if c.endswith("_score")]]
        assert (scores.std() > 0.01).all(), f"a collapsed score column would be dropped as zero-variance: {scores.std().to_dict()}"

    def test_the_mode_of_a_skewed_column_sits_at_the_median(self):
        """The anchor the tie handling has to respect: the most common value is the least surprising one."""
        df = pd.DataFrame({"c": np.concatenate([np.zeros(900), np.arange(1, 101, dtype=float)])})
        ref = fit_extremality_reference(df)
        at_mode, _ = extremality_matrix_from_reference(pd.DataFrame({"c": [0.0]}), ref)
        at_tail, _ = extremality_matrix_from_reference(pd.DataFrame({"c": [100.0]}), ref)
        assert at_mode[0, 0] < at_tail[0, 0]

    def test_a_lone_tied_row_still_matches_its_batch_score(self):
        """The skew fix has to survive the tie fix."""
        rng = np.random.default_rng(3)
        df = pd.DataFrame({"a": np.round(rng.random(1000), 1), "b": np.zeros(1000)})
        ref = fit_extremality_reference(df, list(df.columns))
        batch, _ = extremality_matrix_from_reference(df, ref, list(df.columns))
        lone, _ = extremality_matrix_from_reference(df.iloc[[42]], ref, list(df.columns))
        assert lone[0] == pytest.approx(batch[42], abs=1e-12)
