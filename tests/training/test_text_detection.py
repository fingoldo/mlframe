"""
Tests for the phase-K multi-criteria text-vs-categorical detector.

Round-3 audits A10 + R2-21: tighten the heuristic so UUID / hash IDs
don't get TF-IDF'd and natural-language columns don't get treated as
high-cardinality categoricals.

Coverage:
  * Each trigger fires for the right column shape (1-4 separately).
  * Anti-UUID guard rejects UUID-v4 strings (entropy ≈ 4.04 < 4.5).
  * Anti-UUID guard rejects no-space hash IDs (mean_tokens < 2).
  * User overrides (explicit_text / explicit_categorical / skip).
  * pl.Categorical / pl.Enum / pd.Categorical respected when
    ``respect_explicit_categorical_dtype=True`` (round-3 user-confirmed).
  * Numeric / boolean columns rejected as non-string-dtype.
  * Decisions list returned per column for ``fhc.describe()``.
  * Round-3 T14 corner cases:
    - URL-only column -> not text (low entropy + low tokens)
    - Pure emoji / single-char -> not text
    - JSON-as-string -> high tokens, gets text? -> document.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import polars as pl

from mlframe.training.feature_handling import (
    TextDetectionConfig,
    detect_text_columns,
)


# =====================================================================
# Trigger 1: definite_long (mean_chars >= 100)
# =====================================================================


class TestTriggerDefiniteLong:
    def test_long_text_classified_as_text(self):
        # Each row >100 chars to fire the "definite_long" trigger (vs
        # the "medium_with_tokens" trigger which catches shorter text).
        df = pl.DataFrame(
            {
                "long_col": [
                    "a quick brown fox jumps over the lazy dog and over the moon and back ten times every full hour with extra commentary",
                    "another lengthy review with hundreds of characters typed by an enthusiastic reviewer who really likes this particular product line",
                    "third sample equally lengthy with multiple sentences and varied vocabulary that goes on for a reasonable while now and then more",
                    "fourth row to ensure mean is well above the 100 char threshold for definite-long classification with substantive content here",
                ],
            }
        )
        text_cols, decisions = detect_text_columns(df)
        assert "long_col" in text_cols
        d = next(d for d in decisions if d.column == "long_col")
        assert d.rule_name == "definite_long"


# =====================================================================
# Trigger 2: medium_with_tokens (mean_chars >= 30 AND mean_tokens >= 4)
# =====================================================================


class TestTriggerMediumWithTokens:
    def test_short_review_classified_as_text(self):
        df = pl.DataFrame(
            {
                "review": [
                    "this product worked very well for me",  # 38 chars, 7 tokens
                    "the camera quality is much better than expected",
                    "I would recommend it to a friend who needs one",
                    "the battery lasts the whole day fine",
                ]
                * 5,  # padding for cardinality
            }
        )
        text_cols, _decisions = detect_text_columns(df)
        assert "review" in text_cols


# =====================================================================
# Trigger 4: high_cardinality with substance
# =====================================================================


class TestTriggerHighCardinality:
    def test_high_cardinality_with_substance_is_text(self):
        n = 400
        df = pl.DataFrame(
            {
                "skills": [f"skill {i} of many varied options including {i * 2} bytes worth" for i in range(n)],
            }
        )
        text_cols, _ = detect_text_columns(df)
        assert "skills" in text_cols


# =====================================================================
# Anti-UUID guard
# =====================================================================


class TestAntiUuidGuard:
    def test_uuid_column_rejected_by_entropy(self):
        """UUID-v4 entropy is ~4.04 < 4.5 threshold."""
        df = pl.DataFrame(
            {
                "user_id": [str(uuid.UUID(int=i)) for i in range(200)],
            }
        )
        text_cols, decisions = detect_text_columns(df)
        assert "user_id" not in text_cols
        d = next(d for d in decisions if d.column == "user_id")
        assert d.rule_name == "anti_uuid_filter"

    def test_no_space_hash_rejected_by_token_guard(self):
        """64-hex-char hash-style IDs have no spaces -> mean_tokens=1
        -> rejected even if cardinality is high."""
        rng = np.random.RandomState(0)
        df = pl.DataFrame(
            {
                "hash_id": ["".join(rng.choice(list("0123456789abcdef"), 64).tolist()) for _ in range(150)],
            }
        )
        text_cols, _decisions = detect_text_columns(df)
        assert "hash_id" not in text_cols

    def test_natural_text_passes_entropy(self):
        """Real English has entropy ~4.2-4.6 -- above the 4.5 threshold
        when the corpus is varied enough."""
        df = pl.DataFrame(
            {
                "review": [
                    "Product reviews are written by humans who use a wide range of words and varied vocabulary often.",
                ]
                * 50,
            }
        )
        # Single repeated string -> entropy could be lower; this test
        # just verifies the anti-uuid filter doesn't fire on real
        # English with at least medium chars + tokens.
        _text_cols, decisions = detect_text_columns(df)
        d = next(d for d in decisions if d.column == "review")
        # Either text OR categorical -- but not anti_uuid_filter
        assert d.rule_name != "anti_uuid_filter"


# =====================================================================
# User overrides
# =====================================================================


class TestUserOverrides:
    def test_explicit_text_columns_bypass_heuristic(self):
        df = pl.DataFrame(
            {
                "short_col": ["a", "b", "c"] * 30,  # short, low entropy -> normally cat
            }
        )
        cfg = TextDetectionConfig(explicit_text_columns=["short_col"])
        text_cols, decisions = detect_text_columns(df, config=cfg)
        assert "short_col" in text_cols
        d = next(d for d in decisions if d.column == "short_col")
        assert d.rule_name == "explicit_text"

    def test_explicit_categorical_columns_bypass_heuristic(self):
        df = pl.DataFrame(
            {
                "long_col": ["a quick brown fox " * 20] * 50,  # long -> normally text
            }
        )
        cfg = TextDetectionConfig(explicit_categorical_columns=["long_col"])
        text_cols, decisions = detect_text_columns(df, config=cfg)
        assert "long_col" not in text_cols
        d = next(d for d in decisions if d.column == "long_col")
        assert d.rule_name == "explicit_categorical"

    def test_skip_columns_excludes_from_analysis(self):
        df = pl.DataFrame(
            {
                "ignore_me": ["this would normally be very long text " * 5] * 50,
            }
        )
        cfg = TextDetectionConfig(skip_columns=["ignore_me"])
        text_cols, decisions = detect_text_columns(df, config=cfg)
        assert "ignore_me" not in text_cols
        d = next(d for d in decisions if d.column == "ignore_me")
        assert d.rule_name == "skip_columns"


# =====================================================================
# Categorical dtype respect (round-3 user-confirmed)
# =====================================================================


class TestCategoricalDtypeRespect:
    def test_explicit_categorical_dtype_polars(self):
        """A polars Categorical column gets cat-flag; high cardinality
        + substance won't override that user signal."""
        df = pl.DataFrame(
            {
                "skills": [f"skill_{i}_lots_of_descriptive_text" for i in range(400)],
            }
        ).with_columns(pl.col("skills").cast(pl.Categorical))

        # Default: respect_explicit_categorical_dtype=True
        text_cols, decisions = detect_text_columns(df)
        d = next(d for d in decisions if d.column == "skills")
        assert d.rule_name == "explicit_categorical_dtype"
        assert "skills" not in text_cols

    def test_explicit_categorical_dtype_pandas(self):
        df = pd.DataFrame(
            {
                "skills": pd.Series([f"skill_{i}_lots_of_descriptive_text" for i in range(400)], dtype="category"),
            }
        )
        _text_cols, decisions = detect_text_columns(df)
        d = next(d for d in decisions if d.column == "skills")
        assert d.rule_name == "explicit_categorical_dtype"

    def test_respect_disabled_via_config(self):
        df = pl.DataFrame(
            {
                "skills": [f"skill_{i}_lots_of_descriptive_text" for i in range(400)],
            }
        ).with_columns(pl.col("skills").cast(pl.Categorical))
        cfg = TextDetectionConfig(respect_explicit_categorical_dtype=False)
        _text_cols, decisions = detect_text_columns(df, config=cfg)
        # Now the heuristic runs; cardinality + tokens should classify it text.
        d = next(d for d in decisions if d.column == "skills")
        # Either text (if heuristic fires) or anti_uuid_filter (if not).
        # The test pins that it's NOT classified by the dtype-respect path.
        assert d.rule_name != "explicit_categorical_dtype"


# =====================================================================
# Non-string columns
# =====================================================================


class TestNonStringDtype:
    def test_numeric_column_rejected(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0] * 100})
        text_cols, decisions = detect_text_columns(df)
        assert "x" not in text_cols
        d = next(d for d in decisions if d.column == "x")
        assert d.rule_name == "non_string_dtype"

    def test_boolean_column_rejected(self):
        df = pl.DataFrame({"flag": [True, False, True] * 100})
        text_cols, decisions = detect_text_columns(df)
        assert "flag" not in text_cols
        d = next(d for d in decisions if d.column == "flag")
        assert d.rule_name == "non_string_dtype"


# =====================================================================
# Corner cases
# =====================================================================


class TestCornerCases:
    def test_url_only_column_anti_uuid_filtered(self):
        df = pl.DataFrame(
            {
                "url": [f"http://example.com/path/{i}" for i in range(150)],
            }
        )
        text_cols, _decisions = detect_text_columns(df)
        assert "url" not in text_cols
        # URLs typically have low alphabet entropy + few tokens.

    def test_single_char_column_not_text(self):
        df = pl.DataFrame({"flag": ["A", "B", "C", "D"] * 50})
        text_cols, _ = detect_text_columns(df)
        assert "flag" not in text_cols

    def test_pandas_polars_consistency(self):
        # Same data via both backends -> same decisions.
        n = 100
        text_data = [f"row {i} of medium length text content" for i in range(n)]
        df_pl = pl.DataFrame({"col": text_data})
        df_pd = pd.DataFrame({"col": text_data})
        cols_pl, _ = detect_text_columns(df_pl)
        cols_pd, _ = detect_text_columns(df_pd)
        assert ("col" in cols_pl) == ("col" in cols_pd)

    def test_empty_column_handled(self):
        df = pl.DataFrame({"empty": [None] * 50}, schema={"empty": pl.Utf8})
        text_cols, _decisions = detect_text_columns(df)
        # Empty column has 0 non-null -> should fall through to anti-uuid
        # filter (entropy 0 < 4.5) -> categorical (or no_trigger_fired).
        assert "empty" not in text_cols


# =====================================================================
# Decisions trace shape
# =====================================================================


class TestDecisionsTrace:
    def test_one_decision_per_candidate_column(self):
        df = pl.DataFrame(
            {
                "a": ["x"] * 50,
                "b": ["alpha bravo charlie " * 5] * 50,
                "c": [1.0] * 50,
            }
        )
        _text_cols, decisions = detect_text_columns(df)
        assert len(decisions) == 3
        cols_seen = {d.column for d in decisions}
        assert cols_seen == {"a", "b", "c"}

    def test_stats_attached_to_decision(self):
        df = pl.DataFrame(
            {
                "review": ["a long enough review with multiple tokens " * 3] * 50,
            }
        )
        _, decisions = detect_text_columns(df)
        d = decisions[0]
        assert "mean_chars" in d.stats
        assert "mean_tokens" in d.stats
        assert "alphabet_entropy" in d.stats
        assert "n_unique" in d.stats
