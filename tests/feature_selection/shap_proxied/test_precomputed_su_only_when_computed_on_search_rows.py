"""Precomputed SU(X, y) scores replace the prefilter only when they were computed on exactly the search rows, never on rows the selector later holds out."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxied_fit_steps import _precomputed_su_matches_search_rows


def _aligned(n_fit):
    """A minimal aligned-artifacts mapping."""
    out = {"su_to_target": np.array([0.3, 0.1])}
    if n_fit is not None:
        out["n_samples_at_fit"] = n_fit
    return out


def test_artifacts_from_the_full_frame_are_not_used_on_a_smaller_search_set():
    """MRMR fitted on 1000 rows, the selector searches 800 (200 held out): the SU scores saw the holdout, so they are refused."""
    assert _precomputed_su_matches_search_rows(_aligned(1000), 800) is False


def test_artifacts_computed_on_exactly_the_search_rows_are_used():
    """Same row count: usable (negative control)."""
    assert _precomputed_su_matches_search_rows(_aligned(800), 800) is True


def test_artifacts_without_a_row_count_keep_the_historical_behaviour():
    """No n_samples_at_fit recorded: nothing to compare, so they stay usable."""
    assert _precomputed_su_matches_search_rows(_aligned(None), 800) is True


def test_missing_su_means_no():
    """No su_to_target in the artifacts: the booster prefilter runs."""
    assert _precomputed_su_matches_search_rows({}, 800) is False
    assert _precomputed_su_matches_search_rows(None, 800) is False
