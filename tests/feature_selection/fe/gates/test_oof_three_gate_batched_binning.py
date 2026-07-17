"""CPX12b regression: leakage-preserving batched train-edge binning in the
three-gate OOF MI scorer.

Pins two invariants of _bin_with_train_edges_batched / _fold_test_bins:

1. IDENTITY -- the batched-across-columns binner is bit-identical to the prior
   per-column _bin_with_train_edges loop, including on low-cardinality / tied
   columns where quantile edges are fragile. This guards both the full OOF MI
   vector and the raw bin codes.

2. LEAKAGE-FREE -- the bin edges depend ONLY on the train rows. Perturbing the
   TEST rows (while leaving train rows untouched) must NOT change a single train
   bin code. This is the property that makes the K-fold OOF estimate honest, and
   it is the exact reason the full-frame batched siblings could not be reused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _orthogonal_three_gate_mi_fe as mod
from mlframe.feature_selection.filters._orthogonal_three_gate_mi_fe import (
    _bin_with_train_edges,
    _bin_with_train_edges_batched,
    score_features_by_kfold_oof_mi,
)


def _make_arr(n, p, seed):
    """Make arr."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, p))
    # Fragile binning cases: a 3-level low-cardinality column, a binary column,
    # and a constant column (all collapse / dedup edges).
    arr[:, 0] = rng.integers(0, 3, size=n).astype(np.float64)
    arr[:, 1] = (rng.standard_normal(n) > 0).astype(np.float64)
    arr[:, 2] = 7.0
    return arr


def test_batched_train_edge_binning_bit_identical_to_scalar():
    """Batched train edge binning bit identical to scalar."""
    arr = _make_arr(800, 12, seed=1)
    rng = np.random.default_rng(0)
    perm = rng.permutation(arr.shape[0])
    test_idx = np.sort(perm[:160])
    train_mask = np.ones(arr.shape[0], dtype=bool)
    train_mask[test_idx] = False
    train_idx = np.where(train_mask)[0]

    tb_b, te_b = _bin_with_train_edges_batched(
        arr[train_idx, :],
        arr[test_idx, :],
        nbins=10,
    )
    for j in range(arr.shape[1]):
        tb_s, te_s = _bin_with_train_edges(
            arr[train_idx, j],
            arr[test_idx, j],
            nbins=10,
        )
        assert np.array_equal(tb_b[:, j], tb_s), f"train bins differ col {j}"
        assert np.array_equal(te_b[:, j], te_s), f"test bins differ col {j}"


def test_oof_mi_identical_across_batch_gate(monkeypatch):
    """Oof mi identical across batch gate."""
    raw = pd.DataFrame(_make_arr(900, 6, seed=2), columns=[f"r{i}" for i in range(6)])
    eng = pd.DataFrame(_make_arr(900, 8, seed=3), columns=[f"r{i}__He2" for i in range(8)])
    y = (raw["r3"].to_numpy() + 0.3 * np.random.default_rng(5).standard_normal(900) > 0).astype(int)

    # Force the BATCHED path (threshold high), then the SCALAR path (threshold 0).
    monkeypatch.setattr(mod, "_OOF_BATCH_BINNING_MAX_TRAIN_ROWS", 10**9)
    df_batched = score_features_by_kfold_oof_mi(raw, eng, y, n_folds=5, seed=7, nbins=10)
    monkeypatch.setattr(mod, "_OOF_BATCH_BINNING_MAX_TRAIN_ROWS", 0)
    df_scalar = score_features_by_kfold_oof_mi(raw, eng, y, n_folds=5, seed=7, nbins=10)

    df_batched = df_batched.sort_values("engineered_col").reset_index(drop=True)
    df_scalar = df_scalar.sort_values("engineered_col").reset_index(drop=True)
    np.testing.assert_array_equal(
        df_batched["engineered_mi_oof"].to_numpy(),
        df_scalar["engineered_mi_oof"].to_numpy(),
    )
    np.testing.assert_array_equal(
        df_batched["baseline_mi_oof"].to_numpy(),
        df_scalar["baseline_mi_oof"].to_numpy(),
    )


def test_train_edges_do_not_leak_test_rows():
    """LEAKAGE SENSOR: changing test rows must not move a single train bin code.

    If the batched binner ever quantiled the whole (train+test) column, the train
    bins would shift when test values change. They must not.
    """
    arr = _make_arr(700, 10, seed=4)
    rng = np.random.default_rng(11)
    perm = rng.permutation(arr.shape[0])
    test_idx = np.sort(perm[:140])
    train_mask = np.ones(arr.shape[0], dtype=bool)
    train_mask[test_idx] = False
    train_idx = np.where(train_mask)[0]

    train_arr = arr[train_idx, :]
    test_a = arr[test_idx, :].copy()
    test_b = test_a.copy()
    # Heavily perturb the test rows only (huge outliers) -- edges must be unmoved.
    test_b[:] = test_b * 50.0 + 1000.0

    tb_a, _ = _bin_with_train_edges_batched(train_arr, test_a, nbins=10)
    tb_b, _ = _bin_with_train_edges_batched(train_arr, test_b, nbins=10)
    assert np.array_equal(tb_a, tb_b), "train bins changed with test rows -> leakage!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
