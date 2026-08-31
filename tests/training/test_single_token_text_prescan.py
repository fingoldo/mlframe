"""The empty-bigram-dictionary abort was discovered by failing a fit, which on 2.4M rows cost 44.99 seconds.

CatBoost's default text processing builds word BIGRAMS. A column carrying one token per row can never produce
one, the dictionary comes back empty, and the whole fit aborts -- after which the suite probes, switches to a
unigram dictionary and refits. The production log shows both fits: 44.99s wasted, then 374s of real training.

A whitespace scan over a sample answers the same question before the first fit. It is deliberately conservative:
a column is reported only when NOT ONE sampled row has two tokens, because a column that is merely mostly
single-token can still build a dictionary and must be left to the fit.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlframe.training.cb import single_token_text_features


class TestWhichColumnsCannotBuildABigram:
    """The scan's verdict per column shape."""

    def test_a_single_token_column_is_reported(self):
        """The production shape: skills_text with one token per row."""
        df = pd.DataFrame({"skills": ["python", "sql", "excel"] * 10})
        assert single_token_text_features(df, ["skills"]) == ["skills"]

    def test_a_multi_token_column_is_not_reported(self):
        """Free text builds bigrams fine and must not be pushed onto the unigram path."""
        df = pd.DataFrame({"desc": ["senior python developer", "data analyst wanted"] * 10})
        assert single_token_text_features(df, ["desc"]) == []

    def test_one_multi_token_row_is_enough_to_leave_it_alone(self):
        """Conservative by design: a dictionary needs only some bigrams, not all rows to have them."""
        values = ["python"] * 99 + ["python and sql"]
        assert single_token_text_features(pd.DataFrame({"skills": values}), ["skills"]) == []

    def test_only_the_offending_columns_come_back(self):
        """A mixed frame must not drag a healthy column onto the unigram path with the broken one."""
        df = pd.DataFrame({"skills": ["python"] * 20, "desc": ["a long description here"] * 20})
        assert single_token_text_features(df, ["skills", "desc"]) == ["skills"]

    def test_nulls_do_not_count_as_tokens(self):
        """A missing value is not a one-token row and must not be read as evidence either way."""
        df = pd.DataFrame({"skills": [None, "python", None] * 10})
        assert single_token_text_features(df, ["skills"]) == ["skills"]

    def test_an_empty_column_list_is_a_no_op(self):
        """Every non-text fit reaches this, so the empty case must cost nothing and return nothing."""
        assert single_token_text_features(pd.DataFrame({"a": [1, 2]}), []) == []

    def test_a_missing_column_is_skipped_rather_than_raising(self):
        """This runs before a fit; an exception here would abort a run the scan exists to speed up."""
        assert single_token_text_features(pd.DataFrame({"a": [1, 2]}), ["not_there"]) == []

    def test_a_polars_frame_is_handled(self):
        """The CatBoost fastpath hands the fit a polars frame."""
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({"skills": ["python", "sql"] * 10, "desc": ["two words here"] * 20})
        assert single_token_text_features(df, ["skills", "desc"]) == ["skills"]

    def test_a_large_column_is_sampled_not_scanned_whole(self):
        """The point is to be cheap: a 2M-row column must not be read end to end."""
        n = 200_000
        values = ["python"] * n
        out = single_token_text_features(pd.DataFrame({"skills": values}), ["skills"], sample_rows=1000)
        assert out == ["skills"]

    def test_sampling_still_finds_a_multi_token_row(self):
        """A strided sample has to be dense enough to see a column that is genuinely multi-token."""
        values = ["a b c"] * 50_000
        assert single_token_text_features(pd.DataFrame({"d": values}), ["d"], sample_rows=100) == []
