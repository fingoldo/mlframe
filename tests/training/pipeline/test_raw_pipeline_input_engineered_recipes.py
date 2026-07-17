"""Regression: ``_test_df_is_raw_pipeline_input`` must count a selector's ENGINEERED recipe columns,
not just its base ``support_`` subset, when judging whether a frame is already-transformed.

Pre-fix (fuzz c0018/c0111), the selector-output-width discriminator computed ``len(_out_cols)`` from
``support_`` alone (the selected BASE columns), ignoring any feature-engineering recipes the selector
also appended (MRMR fits both a base subset AND FE recipes). A cache-hit test_df already at its true
output width (base-selected + engineered) then compared wider than the under-counted base-only width,
was misjudged as "still raw", and got double-transformed through the pipeline fit for the RAW input
width -- category_encoders' "Unexpected input dimension" (MRMR selected 2 base + 4 engineered = 6
actual output cols; the support_-only count of 2 made 6 > 2 look raw).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.pipeline._pipeline_helpers import _test_df_is_raw_pipeline_input


class _FakeSelector:
    """Minimal stand-in for a fitted MRMR/RFECV selector: base support_ subset + engineered recipes."""

    def __init__(self, feature_names_in_, support_, n_engineered_recipes):
        self.feature_names_in_ = np.array(feature_names_in_)
        self.support_ = np.array(support_, dtype=bool)
        self._engineered_recipes_ = [object()] * n_engineered_recipes


def test_already_transformed_frame_with_engineered_recipes_is_not_misjudged_as_raw():
    # 2 base columns selected out of 4 input features, plus 4 engineered recipe columns appended --
    # matching the exact fuzz c0018 shapes (support_ len 2, engineered 4, actual output width 6).
    selector = _FakeSelector(
        feature_names_in_=["a", "b", "c", "d"],
        support_=[True, False, True, False],
        n_engineered_recipes=4,
    )
    # Already-transformed test_df: 2 selected base cols + 4 engineered cols = 6 total, matching the
    # selector's TRUE output width. Must be judged NOT raw (no engineered recipes double-transform).
    already_transformed = pd.DataFrame(np.zeros((10, 6)))
    assert not _test_df_is_raw_pipeline_input(selector, already_transformed, passthrough_cols=None, skip_preprocessing=True)


def test_genuinely_raw_frame_wider_than_full_output_is_still_flagged_raw():
    selector = _FakeSelector(
        feature_names_in_=["a", "b", "c", "d"],
        support_=[True, False, True, False],
        n_engineered_recipes=4,
    )
    # A raw 8-col frame (wider than the true 6-col output) must still be flagged for transform.
    raw = pd.DataFrame(np.zeros((10, 8)))
    assert _test_df_is_raw_pipeline_input(selector, raw, passthrough_cols=None, skip_preprocessing=True)


def test_no_engineered_recipes_falls_back_to_support_only_width():
    # A selector with zero FE recipes must behave exactly as before (support_-only width).
    selector = _FakeSelector(
        feature_names_in_=["a", "b", "c", "d"],
        support_=[True, False, True, False],
        n_engineered_recipes=0,
    )
    already_transformed = pd.DataFrame(np.zeros((10, 2)))
    assert not _test_df_is_raw_pipeline_input(selector, already_transformed, passthrough_cols=None, skip_preprocessing=True)
    raw = pd.DataFrame(np.zeros((10, 4)))
    assert _test_df_is_raw_pipeline_input(selector, raw, passthrough_cols=None, skip_preprocessing=True)
