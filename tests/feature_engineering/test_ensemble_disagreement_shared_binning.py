"""Regression: predictor_disagreement_features shares the bin-histogram between entropy + top2_gap.

The builder must compute the per-row equal-width histogram ONCE and feed both entropy and top2_gap, and the
result must stay bit-identical to the standalone public functions (which still recompute the binning each).
Pins the optimization: a future "inline the binning twice again" or a divergent shared-counts path trips this.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering import ensemble_features as ef


@pytest.mark.parametrize("n,k,n_bins", [(2000, 8, 5), (1500, 20, 7), (500, 3, 4)])
def test_builder_entropy_top2_bit_identical_to_standalone(n: int, k: int, n_bins: int) -> None:
    """Builder entropy top2 bit identical to standalone."""
    rng = np.random.default_rng(n + k + n_bins)
    preds = rng.standard_normal((n, k)).astype(np.float64)

    out = ef.predictor_disagreement_features(preds, emit_pairs=False, n_bins=n_bins)
    ent_standalone = ef.predictor_consensus_entropy(preds, n_bins=n_bins)
    top2_standalone = ef.predictor_top2_mode_gap(preds, n_bins=n_bins)

    # Shared-counts builder path must equal the standalone (recompute-the-binning) path exactly.
    assert np.array_equal(out["entropy"], ent_standalone)
    assert np.array_equal(out["top2_gap"], top2_standalone)


def test_shared_counts_helpers_match_full_pipeline() -> None:
    """_entropy_from_counts / _top2_gap_from_counts on shared _bin_counts == public functions."""
    rng = np.random.default_rng(7)
    preds = rng.standard_normal((1000, 12)).astype(np.float64)
    n_bins = 6
    counts = ef._bin_counts(preds, n_bins)
    assert np.array_equal(ef._entropy_from_counts(counts), ef.predictor_consensus_entropy(preds, n_bins=n_bins))
    assert np.array_equal(ef._top2_gap_from_counts(counts), ef.predictor_top2_mode_gap(preds, n_bins=n_bins))


def test_nan_rows_still_handled_in_shared_path() -> None:
    """All-missing row gives NaN histogram features, partial rows stay finite, builder equals standalone."""
    rng = np.random.default_rng(99)
    preds = rng.standard_normal((400, 8)).astype(np.float64)
    preds[3, :] = np.nan
    preds[10, 2] = np.inf
    out = ef.predictor_disagreement_features(preds, emit_pairs=False)
    assert np.isnan(out["entropy"][3]) and np.isnan(out["top2_gap"][3])
    keep = np.arange(preds.shape[0]) != 3
    assert np.isfinite(out["entropy"][keep]).all()
    assert np.isfinite(out["top2_gap"][keep]).all()
    assert np.array_equal(out["entropy"], ef.predictor_consensus_entropy(preds), equal_nan=True)
    assert np.array_equal(out["top2_gap"], ef.predictor_top2_mode_gap(preds), equal_nan=True)


def test_bin_counts_excludes_nan_cells() -> None:
    """Row [0.1, 0.5, 0.9, nan] with 4 bins counts only the 3 valid predictors, not 4 in bin 0."""
    counts = ef._bin_counts(np.array([[0.1, 0.5, 0.9, np.nan]]), 4)
    assert counts.sum() == 3.0
    assert counts[0, 0] == 1.0 and counts[0, 3] == 1.0


def test_consensus_entropy_over_valid_predictors_only() -> None:
    """A missing predictor does not fake consensus: entropy equals that of the valid predictors alone."""
    full = np.array([[0.1, 0.5, 0.9]])
    with_nan = np.array([[0.1, 0.5, 0.9, np.nan]])
    assert ef.predictor_consensus_entropy(with_nan, n_bins=4)[0] == pytest.approx(ef.predictor_consensus_entropy(full, n_bins=4)[0])
    assert ef.predictor_consensus_entropy(with_nan, n_bins=4)[0] > 0.5
