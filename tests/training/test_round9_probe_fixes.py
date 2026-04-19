"""Round 9 sensors for three fixes from the round-9 probe.

1. ``training/pipeline.py::apply_preprocessing_extensions`` TF-IDF block:
   pre-fix it only expanded train when a tfidf_column was missing in
   val/test, producing column-count mismatch that surfaced opaquely as
   a scaler shape error downstream. Now: skip the column entirely
   across all splits with a WARN naming the offending split, and a
   separate WARN for tfidf_columns typos (col missing from train too).

2. ``training/strategies.py::is_polars_categorical`` — didn't detect
   ``pl.Enum``. Same class of bug we already fixed in
   ``_auto_detect_feature_types`` (round 4). HGBStrategy's cardinality
   cast branch silently skipped Enum columns, treating them as numeric.
   Now: ``isinstance(dtype, pl.Enum)`` also returns True.

3. ``training/strategies.py::build_pipeline`` — when
   ``requires_encoding=True`` AND there are ``cat_features`` AND
   ``category_encoder is None``, the encoding step was silently skipped
   and sklearn later raised opaquely on raw string categoricals deep
   inside model.fit. Now: WARN naming the strategy class and cat count.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import polars as pl
import pytest


# =============================================================================
# Fix A1 — TF-IDF val/test column mismatch
# =============================================================================


class _TfidfConfig:
    """Minimal duck-typed config matching PolarsPipelineConfig's TF-IDF
    surface. Using a stand-in avoids pulling the full Pydantic stack
    for what's essentially a data bag."""
    def __init__(self, tfidf_columns):
        self.tfidf_columns = tfidf_columns
        self.tfidf_max_features = 10
        self.tfidf_ngram_range = (1, 1)
        # Other required knobs — keep extension pipeline as no-op.
        self.scaler = None
        self.binarization_threshold = None
        self.kbins = None
        self.polynomial_degree = None
        self.nonlinear_features = None
        self.dim_reducer = None
        self.memory_safety_max_features = 100_000


def _call_apply_extensions(train, val, test, tfidf_columns, caplog):
    from mlframe.training.pipeline import apply_preprocessing_extensions
    cfg = _TfidfConfig(tfidf_columns=tfidf_columns)
    with caplog.at_level(logging.WARNING, logger="mlframe.training.pipeline"):
        out = apply_preprocessing_extensions(train, val, test, cfg, None)
    return out


class TestTfidfSplitColumnParity:

    def test_column_parity_after_tfidf_all_splits(self, caplog):
        """Happy path: text col present in all three splits → TF-IDF
        expands all three symmetrically, no WARN, column counts match.

        Note: sklearn's TfidfVectorizer default token_pattern is
        ``r"(?u)\\b\\w\\w+\\b"`` — requires 2+ word characters per
        token. Use realistic multi-char words here."""
        train = pd.DataFrame({"text": ["hello world data", "hello data"], "num": [1.0, 2.0]})
        val = pd.DataFrame({"text": ["world data science", "hello science"], "num": [3.0, 4.0]})
        test = pd.DataFrame({"text": ["science data", "hello world"], "num": [5.0, 6.0]})
        out_train, out_val, out_test, pipes = _call_apply_extensions(
            train, val, test, ["text"], caplog
        )
        assert out_train.shape[1] == out_val.shape[1] == out_test.shape[1]
        warns = [r for r in caplog.records if r.levelname == "WARNING"]
        assert not warns, f"Unexpected WARN on clean TF-IDF: {[r.message for r in warns]}"

    def test_missing_col_in_val_triggers_warn_and_skip(self, caplog):
        """The exact pre-fix scenario: ``text`` present in train, missing
        from val. Pre-fix: train got TF-IDF-expanded, val didn't →
        downstream scaler raised on shape mismatch. Post-fix: WARN +
        skip the column entirely so all splits stay aligned."""
        train = pd.DataFrame({"text": ["hello world", "data science"], "num": [1.0, 2.0]})
        val = pd.DataFrame({"num": [3.0, 4.0]})  # no 'text'!
        test = pd.DataFrame({"text": ["hello", "world"], "num": [5.0, 6.0]})
        out_train, out_val, out_test, pipes = _call_apply_extensions(
            train, val, test, ["text"], caplog
        )
        warns = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("split mismatch" in m.lower()
                   or "missing from a non-train split" in m
                   or ("skipping" in m and "val" in m)
                   for m in warns), warns
        # TF-IDF was skipped entirely -> 'text' stays as-is in train
        # (and val has no 'text' column). No TF-IDF columns appended
        # to any split. Column counts must match between train and val
        # minus the raw 'text' col.

    def test_typo_column_triggers_warn(self, caplog):
        """tfidf_columns lists a name not in train at all (likely typo).
        WARN at 'typo' level, don't attempt the TF-IDF."""
        train = pd.DataFrame({"real_text": ["hello world"], "num": [1.0]})
        out = _call_apply_extensions(train, None, None, ["fake_text"], caplog)
        warns = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("fake_text" in m and ("typo" in m or "not found" in m) for m in warns), warns


# =============================================================================
# Fix C2 — pl.Enum recognized as categorical in strategies
# =============================================================================


class TestIsPolarsCategoricalEnum:

    def test_categorical_dtype_detected(self):
        from mlframe.training.strategies import is_polars_categorical
        assert is_polars_categorical(pl.Categorical)
        assert is_polars_categorical(pl.Utf8)
        assert is_polars_categorical(pl.String)

    def test_enum_dtype_detected(self):
        """Pre-fix: is_polars_categorical(pl.Enum(...)) returned False
        because pl.Enum is an instance-level dtype object, not matching
        the class-level entries. HGBStrategy silently treated Enum
        columns as numeric."""
        from mlframe.training.strategies import is_polars_categorical
        enum_dt = pl.Enum(["a", "b", "c"])
        assert is_polars_categorical(enum_dt), (
            "pl.Enum must be recognised as categorical alongside pl.Categorical"
        )

    def test_numeric_dtypes_rejected(self):
        """False-positive sensor: numeric dtypes must stay non-categorical.
        Recognizing Enum musn't broaden the check too far."""
        from mlframe.training.strategies import is_polars_categorical
        assert not is_polars_categorical(pl.Int32)
        assert not is_polars_categorical(pl.Float32)
        assert not is_polars_categorical(pl.Boolean)

    def test_get_polars_cat_columns_includes_enum(self):
        from mlframe.training.strategies import get_polars_cat_columns
        enum_dt = pl.Enum(["red", "green"])
        df = pl.DataFrame({
            "num": [1.0, 2.0],
            "cat": pl.Series("cat", ["a", "b"]).cast(pl.Categorical),
            "en":  pl.Series("en",  ["red", "green"], dtype=enum_dt),
        })
        cols = get_polars_cat_columns(df)
        assert "cat" in cols
        assert "en" in cols
        assert "num" not in cols


# =============================================================================
# Fix C4 — category_encoder=None + requires_encoding=True WARN
# =============================================================================


class TestBuildPipelineEncoderMissingWarn:

    def test_warn_when_encoder_none_and_cats_present(self, caplog):
        """Strategy that requires encoding + user passes cat_features
        but forgets to pass a category_encoder. Pre-fix: step silently
        skipped; downstream model.fit crashed on raw strings."""
        from mlframe.training.strategies import HGBStrategy
        strat = HGBStrategy()
        assert strat.requires_encoding, "HGBStrategy should require encoding"
        with caplog.at_level(logging.WARNING, logger="mlframe.training.strategies"):
            strat.build_pipeline(
                base_pipeline=None,
                cat_features=["a", "b", "c"],
                category_encoder=None,
                imputer=None,
                scaler=None,
            )
        warns = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("category_encoder" in m and "None" in m for m in warns), warns
        # Must name the strategy class so operators know which model
        # config is misconfigured.
        assert any("HGBStrategy" in m for m in warns), warns

    def test_no_warn_when_encoder_provided(self, caplog):
        from mlframe.training.strategies import HGBStrategy
        from sklearn.preprocessing import OrdinalEncoder
        strat = HGBStrategy()
        with caplog.at_level(logging.WARNING, logger="mlframe.training.strategies"):
            strat.build_pipeline(
                base_pipeline=None,
                cat_features=["a"],
                category_encoder=OrdinalEncoder(),
                imputer=None,
                scaler=None,
            )
        warns = [r for r in caplog.records if r.levelname == "WARNING"]
        assert not warns

    def test_no_warn_when_no_cat_features(self, caplog):
        """If there are no cat_features at all, the missing encoder is
        not a problem — don't WARN (keeps the noise floor low)."""
        from mlframe.training.strategies import HGBStrategy
        strat = HGBStrategy()
        with caplog.at_level(logging.WARNING, logger="mlframe.training.strategies"):
            strat.build_pipeline(
                base_pipeline=None,
                cat_features=[],
                category_encoder=None,
                imputer=None,
                scaler=None,
            )
        warns = [r for r in caplog.records if r.levelname == "WARNING"]
        assert not warns

    def test_tree_strategy_does_not_require_encoding(self, caplog):
        """False-positive sensor: TreeModelStrategy.requires_encoding=False.
        So even cat_features + encoder=None must stay silent — tree
        models handle categoricals natively."""
        from mlframe.training.strategies import TreeModelStrategy
        strat = TreeModelStrategy()
        assert not strat.requires_encoding
        with caplog.at_level(logging.WARNING, logger="mlframe.training.strategies"):
            strat.build_pipeline(
                base_pipeline=None,
                cat_features=["a", "b"],
                category_encoder=None,
                imputer=None,
                scaler=None,
            )
        warns = [r for r in caplog.records if r.levelname == "WARNING"]
        assert not warns
