"""``support_mask_from_selector``: strict name->index mapping, and the engineered-name defect it closes.

``extract_selected`` tries ``get_feature_names_out()`` FIRST. A selector fitted with feature engineering
enabled (MRMR with FE) answers with ENGINEERED names that are absent from ``feature_names_in_``, and the
function's contract is to pass them through unchanged. ``compare_selectors`` intersects them away, which
silently shrinks the reported support; ``support_mask_from_selector`` raises instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection import extract_selected, support_mask_from_selector

FEATURES = ["f0", "f1", "f2", "f3"]


class _EngineeredNamesSelector:
    """MRMR-with-FE shape: get_feature_names_out() reports engineered columns, not the input ones."""

    feature_names_in_ = np.array(FEATURES, dtype=object)
    support_ = np.array([0, 2], dtype=np.int64)

    def get_feature_names_out(self):
        """Engineered output names, none of which appear in ``feature_names_in_``."""
        return np.array(["f0", "f2_x_f3_ratio"], dtype=object)


class _IndexSupportSelector:
    """MRMR shape without FE: ``support_`` holds np.int64 INDICES in greedy-selection order, not a mask."""

    feature_names_in_ = np.array(FEATURES, dtype=object)
    support_ = np.array([2, 0], dtype=np.int64)


class _BoolMaskSelector:
    """sklearn shape: ``support_`` is a boolean mask aligned to ``feature_names_in_``."""

    feature_names_in_ = np.array(FEATURES, dtype=object)
    support_ = np.array([True, False, True, False])


def test_extract_selected_passes_engineered_names_through():
    """Documents the raw behaviour: the promised intersection is NOT done in the function body."""
    assert extract_selected(_EngineeredNamesSelector(), FEATURES) == ["f0", "f2_x_f3_ratio"]


def test_support_mask_raises_on_unmappable_engineered_name():
    """An engineered name has no index in the input columns, so the mask must refuse rather than drop it silently."""
    with pytest.raises(ValueError, match="f2_x_f3_ratio"):
        support_mask_from_selector(_EngineeredNamesSelector(), FEATURES)


def test_support_mask_from_int64_index_support():
    """MRMR's ``np.int64`` index ``support_`` must keep resolving through ``feature_names_in_``."""
    got = support_mask_from_selector(_IndexSupportSelector(), FEATURES)
    assert got.dtype == bool
    np.testing.assert_array_equal(got, [True, False, True, False])


def test_support_mask_from_boolean_support():
    """The plain sklearn mask passes through unchanged."""
    np.testing.assert_array_equal(support_mask_from_selector(_BoolMaskSelector(), FEATURES), [True, False, True, False])


def test_support_mask_rejects_duplicate_feature_names():
    """A duplicated column name makes name-to-index resolution ambiguous, so it is rejected, not resolved to the first hit."""
    with pytest.raises(ValueError, match="duplicates"):
        support_mask_from_selector(_BoolMaskSelector(), ["f0", "f0", "f2", "f3"])


def test_support_mask_length_matches_feature_names():
    """The mask is always aligned to the caller's column list, which is what makes it comparable across selectors."""
    assert support_mask_from_selector(_BoolMaskSelector(), FEATURES).shape == (len(FEATURES),)


def test_private_alias_still_points_at_the_promoted_function():
    """Existing call sites use the private name; the alias must stay the same object, not a copy."""
    import importlib

    # ``from mlframe.feature_selection import compare_selectors`` resolves to the re-exported FUNCTION,
    # not the module, so the module has to be imported by its full path.
    mod = importlib.import_module("mlframe.feature_selection.compare_selectors")

    assert mod._extract_selected is mod.extract_selected
