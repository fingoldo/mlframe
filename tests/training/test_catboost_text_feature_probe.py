"""One unusable text column must not cost the fit every other text column.

A production run promoted five high-cardinality columns to ``text_features``; CatBoost raised
``Dictionary size is 0`` after 40 seconds of fitting, and the handler dropped ALL FIVE and retried, so the
model trained with no text features at all and nothing said the promotion had been undone.

The handler also asserted the wrong cause -- "too few non-null samples" -- while the same run's log reported
981,873 distinct values in one of those columns. Measured against the installed CatBoost, row counts do not
predict this failure at all (see ``_cb_text_probe``'s table), so the probe asks CatBoost per column instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

catboost = pytest.importorskip("catboost")

from mlframe.training.cb._cb_text_probe import PROBE_MAX_ROWS, unusable_text_features, usable_text_features

N = 400


def _frame(seed: int = 0) -> tuple:
    """A frame carrying one usable text column and three CatBoost cannot build a dictionary from."""
    rng = np.random.default_rng(seed)
    y = (rng.random(N) < 0.4).astype(int)
    df = pd.DataFrame(
        {
            "num": rng.normal(size=N),
            # Three tokens per row, two of which recur -- CatBoost builds a vocabulary from this.
            "usable": [f"tok{i} tok{i % 7} common" for i in range(N)],
            # One token per row: raises regardless of how often the token repeats.
            "single_token": [f"tok{i % 5}" for i in range(N)],
            "empty": [""] * N,
            "whitespace": ["   "] * N,
        }
    )
    return df, y


class TestTheProbeIdentifiesTheRightColumns:
    """The probe's whole job: name the offenders instead of condemning the group."""

    def test_only_the_unusable_columns_are_rejected(self):
        """The defect in one assertion: a good column must survive a bad one sharing the fit."""
        df, y = _frame()
        bad = unusable_text_features(df, y, ["usable", "single_token", "empty", "whitespace"], verbose=False)
        assert set(bad) == {"single_token", "empty", "whitespace"}
        assert "usable" not in bad

    def test_row_count_does_not_predict_the_failure(self):
        """The old message blamed sparsity; a fully-populated, high-cardinality column still fails."""
        df, y = _frame()
        assert df["single_token"].notna().all()
        assert unusable_text_features(df, y, ["single_token"], verbose=False)

    def test_usable_helper_preserves_order(self):
        """Callers pass the surviving list straight back into fit_params, so order has to hold."""
        df, y = _frame()
        assert usable_text_features(df, y, ["usable", "empty"], verbose=False) == ["usable"]

    def test_nulls_are_reported_rather_than_crashing_the_probe(self):
        """CatBoost rejects None in a text feature outright; the probe must name that, not raise."""
        df, y = _frame()
        df["with_nulls"] = [None] * N
        bad = unusable_text_features(df, y, ["with_nulls"], verbose=False)
        # Nulls are normalised to empty strings before probing, so this lands as the dictionary failure.
        assert "with_nulls" in bad

    def test_no_text_features_is_a_no_op(self):
        """An empty request must not construct a probe frame or import catboost."""
        df, y = _frame()
        assert unusable_text_features(df, y, [], verbose=False) == {}
        assert usable_text_features(df, y, None, verbose=False) == []

    def test_a_missing_column_is_skipped_not_rejected(self):
        """A name the frame does not carry is the caller's bookkeeping problem, not an unusable column."""
        df, y = _frame()
        assert unusable_text_features(df, y, ["not_a_column"], verbose=False) == {}

    def test_single_class_target_is_skipped(self):
        """CatBoost raises on a one-class target for reasons unrelated to the text column."""
        df, _y = _frame()
        assert unusable_text_features(df, np.zeros(N, dtype=int), ["single_token"], verbose=False) == {}

    def test_probe_sample_cap_is_conservative_in_the_safe_direction(self):
        """Sampling can only shrink a token's frequency, so a pass on the sample is a pass on the whole."""
        assert PROBE_MAX_ROWS >= 10_000


class TestTheFitSurvivesOneBadColumn:
    """End to end: a real CatBoost fit with a mixed set of text columns keeps the good one."""

    def test_fit_succeeds_on_the_probed_subset(self):
        """Fitting on the probe's verdict must not raise, which is what the retry path relies on."""
        df, y = _frame()
        keep = usable_text_features(df, y, ["usable", "single_token"], verbose=False)
        assert keep == ["usable"]
        catboost.CatBoostClassifier(iterations=2, verbose=0, allow_writing_files=False).fit(df[["num", "usable"]], y, text_features=keep)

    def test_the_unprobed_set_still_raises(self):
        """Pins that the fixture really does reproduce the production failure, not a benign variant."""
        df, y = _frame()
        with pytest.raises(Exception, match="Dictionary size is 0"):
            catboost.CatBoostClassifier(iterations=2, verbose=0, allow_writing_files=False).fit(
                df[["num", "usable", "single_token"]], y, text_features=["usable", "single_token"]
            )


class TestTheUnigramRescue:
    """The root cause, and the remedy that keeps the columns instead of discarding them.

    CatBoost's default text pipeline builds word BIGRAMS. A column whose rows hold a single token cannot
    contribute one, so its dictionary comes out empty and the whole fit aborts. Unigrams fix it -- which means
    the right response to the failure is to change the dictionary, not to drop the caller's text features.
    """

    def test_single_token_column_fails_by_default_and_fits_on_unigrams(self):
        """The discriminating test for the cause: same data, only the dictionary differs."""
        from mlframe.training.cb._cb_text_probe import unigram_text_processing

        df, y = _frame()
        frame = df[["num", "single_token"]]
        with pytest.raises(Exception, match="Dictionary size is 0"):
            catboost.CatBoostClassifier(iterations=2, verbose=0, allow_writing_files=False).fit(frame, y, text_features=["single_token"])
        catboost.CatBoostClassifier(iterations=2, verbose=0, allow_writing_files=False, text_processing=unigram_text_processing()).fit(
            frame, y, text_features=["single_token"]
        )

    def test_unigrams_serve_single_and_multi_token_columns_together(self):
        """A mixed frame is the realistic case; unigrams must cover both or the rescue is not general."""
        from mlframe.training.cb._cb_text_probe import unigram_rescues_text_features

        df, y = _frame()
        assert unigram_rescues_text_features(df, y, ["usable", "single_token"], verbose=False) is True

    def test_the_rescue_declines_when_it_cannot_help(self):
        """An empty column has no tokens at any gram order, so the caller must fall through to dropping."""
        from mlframe.training.cb._cb_text_probe import unigram_rescues_text_features

        df, y = _frame()
        assert unigram_rescues_text_features(df, y, ["empty"], verbose=False) is False

    def test_declaring_both_gram_orders_does_not_work(self):
        """Pins WHY the rescue is unigram-only: any empty dictionary aborts the fit, so bigrams cannot ride along."""
        df, y = _frame()
        both = {
            "tokenizers": [{"tokenizer_id": "Space", "delimiter": " "}],
            "dictionaries": [{"dictionary_id": "U", "gram_order": "1"}, {"dictionary_id": "B", "gram_order": "2"}],
            "feature_processing": {"default": [{"dictionaries_names": ["U", "B"], "feature_calcers": ["BoW"], "tokenizers_names": ["Space"]}]},
        }
        with pytest.raises(Exception, match="Dictionary size is 0"):
            catboost.CatBoostClassifier(iterations=2, verbose=0, allow_writing_files=False, text_processing=both).fit(
                df[["num", "single_token"]], y, text_features=["single_token"]
            )
