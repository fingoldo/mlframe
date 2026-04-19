"""Round 7 sensors for two proactive-probe findings:

1. ``training/phases.py::_format_ctx`` used a bare f-string
   ``f"{k}={v}"`` without value truncation. A caller passing a large
   object as a phase context kwarg (e.g. a full DataFrame via
   ``phase("fit", eval_set=val_df)``) blew up the resulting log line to
   MB+ — breaking log rotation and structured-log aggregation (newlines
   in ``repr``). Now truncated to 120 chars per value.

2. ``training/core.py::_validate_input_columns_against_metadata``
   (extracted 2026-04-19 from inline logic in ``predict_mlframe_models_suite``):
   previously only WARN'd on missing columns and proceeded. If the
   missing column was a load-bearing feature (cat/text/embedding),
   the pipeline silently ran on a shape-mismatched frame and either
   crashed opaquely inside sklearn or produced garbage predictions.
   Now: critical missing → ``ValueError``; non-critical missing → WARN.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# _format_ctx — truncation sensor
# ---------------------------------------------------------------------------

class TestFormatCtxTruncation:

    def test_short_values_are_not_truncated(self):
        from mlframe.training.phases import _format_ctx
        out = _format_ctx({"model": "CatBoost", "n_rows": 810_000})
        assert out == "model=CatBoost n_rows=810000"

    def test_none_values_dropped(self):
        from mlframe.training.phases import _format_ctx
        out = _format_ctx({"a": 1, "b": None, "c": 2})
        assert "b" not in out
        assert "a=1" in out and "c=2" in out

    def test_long_string_value_truncated(self):
        """A 5000-char string value must not grow the log line past
        ~120 chars (truncation cap). Pre-fix would let 5000 chars
        through, and on a wide phase context that hit MB+."""
        from mlframe.training.phases import _format_ctx
        huge = "x" * 5000
        out = _format_ctx({"big_arg": huge})
        # The key remains intact + some truncation suffix
        assert out.startswith("big_arg=")
        assert out.endswith("...")
        # Value portion should be ≤ 120 chars (the default max_val_len)
        val_portion = out[len("big_arg="):]
        assert len(val_portion) <= 120

    def test_huge_list_repr_truncated(self):
        """The classic trigger: someone passes a list/array as a
        context kwarg. ``str(list(10_000 items))`` is ~120k chars.
        Truncation caps it so one log line doesn't drown a screen."""
        from mlframe.training.phases import _format_ctx
        out = _format_ctx({"eval_set": list(range(10_000))})
        val_portion = out[len("eval_set="):]
        assert len(val_portion) <= 120

    def test_empty_context_returns_empty_string(self):
        from mlframe.training.phases import _format_ctx
        assert _format_ctx({}) == ""
        assert _format_ctx(None or {}) == ""

    def test_custom_max_val_len(self):
        """Caller can tune max_val_len for a specific site."""
        from mlframe.training.phases import _format_ctx
        out = _format_ctx({"k": "v" * 500}, max_val_len=50)
        val_portion = out[len("k="):]
        assert len(val_portion) <= 50


# ---------------------------------------------------------------------------
# _validate_input_columns_against_metadata — critical-missing sensor
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_metadata():
    """Metadata shape matching what the training flow saves."""
    return {
        "columns": ["a", "b", "cat_x", "text_y"],
        "cat_features": ["cat_x"],
        "text_features": ["text_y"],
        "embedding_features": [],
    }


class TestValidateInputColumnsAgainstMetadata:

    def test_happy_path_identical_columns(self, minimal_metadata):
        from mlframe.training.core import _validate_input_columns_against_metadata
        df = pd.DataFrame({
            "a": [1, 2], "b": [3, 4],
            "cat_x": ["p", "q"], "text_y": ["hello", "world"],
        })
        out = _validate_input_columns_against_metadata(df, minimal_metadata)
        assert list(out.columns) == ["a", "b", "cat_x", "text_y"]

    def test_extra_columns_are_dropped(self, minimal_metadata):
        from mlframe.training.core import _validate_input_columns_against_metadata
        df = pd.DataFrame({
            "a": [1, 2], "b": [3, 4],
            "cat_x": ["p", "q"], "text_y": ["hello", "world"],
            "unused_extra": [0.1, 0.2],  # not in metadata.columns
        })
        out = _validate_input_columns_against_metadata(df, minimal_metadata)
        assert "unused_extra" not in out.columns

    def test_missing_cat_feature_raises(self, minimal_metadata):
        """The critical path: 'cat_x' is declared as cat_feature in
        metadata. If input is missing it, the pipeline+model cannot
        run correctly. Pre-fix: silent WARN + shape-mismatch crash
        later inside sklearn. Post-fix: loud ValueError naming the
        column."""
        from mlframe.training.core import _validate_input_columns_against_metadata
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "text_y": ["x", "y"]})
        with pytest.raises(ValueError, match="cat_x"):
            _validate_input_columns_against_metadata(df, minimal_metadata)

    def test_missing_text_feature_raises(self, minimal_metadata):
        from mlframe.training.core import _validate_input_columns_against_metadata
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "cat_x": ["p", "q"]})
        with pytest.raises(ValueError, match="text_y"):
            _validate_input_columns_against_metadata(df, minimal_metadata)

    def test_missing_non_critical_only_warns(self, minimal_metadata, caplog):
        """If only a 'b' column (plain numeric, not cat/text/embedding)
        is missing, WARN but proceed — some callers drop reconstructable
        derived columns intentionally."""
        from mlframe.training.core import _validate_input_columns_against_metadata
        df = pd.DataFrame({
            "a": [1, 2],
            # 'b' missing
            "cat_x": ["p", "q"], "text_y": ["hello", "world"],
        })
        with caplog.at_level(logging.WARNING, logger="mlframe.training.core"):
            out = _validate_input_columns_against_metadata(df, minimal_metadata)
        warns = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("'b'" in m or "b'" in m or "b\"" in m or "b]" in m for m in warns), warns

    def test_error_message_lists_all_critical_missing(self, minimal_metadata):
        """Diagnostic: all missing critical columns should be named,
        not just the first one. Operator fixing the upstream extraction
        needs the full list."""
        from mlframe.training.core import _validate_input_columns_against_metadata
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError) as excinfo:
            _validate_input_columns_against_metadata(df, minimal_metadata)
        msg = str(excinfo.value)
        assert "cat_x" in msg
        assert "text_y" in msg
        assert "load-bearing" in msg or "cat/text/embedding" in msg

    def test_empty_columns_in_metadata_is_no_op(self):
        """No expected columns recorded → no validation, return df as-is."""
        from mlframe.training.core import _validate_input_columns_against_metadata
        df = pd.DataFrame({"a": [1], "b": [2]})
        out = _validate_input_columns_against_metadata(df, {"columns": []})
        assert list(out.columns) == ["a", "b"]

    def test_embedding_feature_missing_also_raises(self):
        """The critical-features check must cover embedding_features
        too, not just cat+text."""
        from mlframe.training.core import _validate_input_columns_against_metadata
        meta = {
            "columns": ["a", "emb"],
            "cat_features": [],
            "text_features": [],
            "embedding_features": ["emb"],
        }
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="emb"):
            _validate_input_columns_against_metadata(df, meta)
