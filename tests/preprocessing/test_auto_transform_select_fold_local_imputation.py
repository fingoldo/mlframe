"""The transform-ranking CV imputed missing values from the whole column before splitting.

``_fit_transform_fold`` exists to stop a scaler's fit statistics leaking across the CV split, and its docstring
says exactly that. But the median that filled the column's non-finite cells was computed one line above the fold
loop, over every row -- so each held-out fold's own values informed its own imputation, biasing the score that
decides which transform gets recommended.

A median is a fit statistic like any other. These tests pin that it is fitted on the train slice only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.preprocessing.auto_transform_select import _fill_nonfinite_from_train


class TestTheFillComesFromTheTrainSliceOnly:
    """The discriminating property: a value that exists only in the held-out rows must not move the fill."""

    def test_the_fill_ignores_the_held_out_rows(self):
        """Whole-column imputation would put the test rows' median into the train rows' gaps."""
        x = np.array([1.0, 1.0, 1.0, np.nan, 500.0, 500.0, 500.0])
        train_idx = np.array([0, 1, 2, 3])
        filled = _fill_nonfinite_from_train(x, train_idx)
        assert filled[3] == 1.0, "the gap must be filled from the train rows, whose median is 1.0"

    def test_a_different_fold_gets_a_different_fill(self):
        """If the fill were global it would be identical for every fold, which is the whole tell."""
        x = np.array([1.0, 1.0, np.nan, 500.0, 500.0])
        low = _fill_nonfinite_from_train(x, np.array([0, 1, 2]))
        high = _fill_nonfinite_from_train(x, np.array([2, 3, 4]))
        assert low[2] != high[2], f"both folds imputed {low[2]}; the fill is not fold-local"
        assert (low[2], high[2]) == (1.0, 500.0)

    def test_it_does_not_mutate_the_caller(self):
        """The fold loop calls this once per fold on the same source array."""
        x = np.array([1.0, np.nan, 3.0])
        _fill_nonfinite_from_train(x, np.array([0, 2]))
        assert np.isnan(x[1]), "the source column must still carry its missing values for the next fold"

    @pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
    def test_every_non_finite_kind_is_filled(self, bad):
        """The prior code tested `~np.isfinite`, which covers all three; the replacement must too."""
        x = np.array([2.0, bad, 4.0])
        assert _fill_nonfinite_from_train(x, np.array([0, 2]))[1] == 3.0

    def test_an_all_missing_train_slice_falls_back_to_zero(self):
        """No train statistic exists, so the documented 0.0 fallback stands rather than a NaN propagating."""
        x = np.array([np.nan, np.nan, 5.0])
        assert _fill_nonfinite_from_train(x, np.array([0, 1]))[0] == 0.0


class TestTheRankingStillWorksEndToEnd:
    """The fix moves work inside the loop; the function it serves must still produce a ranking."""

    def test_a_column_with_gaps_still_gets_scored(self):
        """Guards against the per-fold fill breaking the caller rather than just de-biasing it."""
        from mlframe.preprocessing.auto_transform_select import select_column_transforms

        rng = np.random.default_rng(0)
        n = 400
        raw = rng.lognormal(0.0, 1.0, n)
        raw[rng.random(n) < 0.1] = np.nan
        df = pd.DataFrame({"skewed": raw})
        y = (raw > np.nanmedian(raw)).astype(int)
        out = select_column_transforms(df, y, columns=["skewed"], task="classification", n_splits=3, random_state=0)
        assert "skewed" in out
        assert out["skewed"]["all_scores"], "every candidate transform should carry a score"
        assert out["skewed"]["best_transform"] in out["skewed"]["all_scores"]
