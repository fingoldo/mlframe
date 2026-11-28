"""Cluster 12 regression: the three fs_hybrid downstream evaluators share one model definition and agree (audit 2026-11-20, finding 12-F8)."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCH_DIR = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "_benchmarks" / "fs_hybrid"
sys.path.insert(0, str(BENCH_DIR))
import _downstream_shared as ds


def _frame(n=300, seed=0):
    """A small separable binary problem split into train and test halves."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = (X["a"] + 0.5 * rng.normal(size=n) > 0).astype(int).to_numpy()
    h = n // 2
    return X.iloc[:h], X.iloc[h:], y[:h], y[h:]


def test_cols_matrix_and_mean_agree_and_empty_input_is_nan():
    """on_cols equals on_matrix of the slice, the mean evaluator equals the rounded mean of the three, and an empty subset gives NaN everywhere."""
    Xtr, Xte, ytr, yte = _frame()
    by_cols = ds.downstream_on_cols(Xtr, Xte, ytr, yte, ["a", "b"])
    by_matrix = ds.downstream_on_matrix(Xtr[["a", "b"]], Xte[["a", "b"]], ytr, yte)
    assert by_cols == by_matrix
    mean = ds.lgb_logit_knn_mean_auc(Xtr, Xte, ytr, yte, ["a", "b"])
    assert abs(mean - np.mean(list(by_matrix.values()))) < 2e-4
    assert math.isnan(ds.lgb_logit_knn_mean_auc(Xtr, Xte, ytr, yte, []))
    assert all(math.isnan(v) for v in ds.downstream_on_cols(Xtr, Xte, ytr, yte, []).values())
