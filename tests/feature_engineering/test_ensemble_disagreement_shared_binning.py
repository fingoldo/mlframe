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
    arr = ef._coerce_preds(preds)
    counts = ef._bin_counts(arr, n_bins)
    assert np.array_equal(ef._entropy_from_counts(counts), ef.predictor_consensus_entropy(preds, n_bins=n_bins))
    assert np.array_equal(ef._top2_gap_from_counts(counts, arr.shape[1]), ef.predictor_top2_mode_gap(preds, n_bins=n_bins))


def test_nan_rows_still_handled_in_shared_path() -> None:
    rng = np.random.default_rng(99)
    preds = rng.standard_normal((400, 8)).astype(np.float64)
    preds[3, :] = np.nan  # all-non-finite row -> coerced to 0.0
    preds[10, 2] = np.inf  # partial -> row median
    out = ef.predictor_disagreement_features(preds, emit_pairs=False)
    assert np.isfinite(out["entropy"]).all()
    assert np.isfinite(out["top2_gap"]).all()
    assert np.array_equal(out["entropy"], ef.predictor_consensus_entropy(preds))
    assert np.array_equal(out["top2_gap"], ef.predictor_top2_mode_gap(preds))
