"""Regression: the per-pair FE materialise must not overrun ``final_transformed_vals`` on the CPU fallback path.

``_score_one_pair`` builds GPU op-codes and runs a candidate-sizing loop that advances the column cursor ``i`` to
``_K`` to size the fused GPU materialise+bin launch. When the GPU-binning gate declines (no visible CUDA device, or
below the size crossover) that fused block is skipped WITHOUT raising, so the ``except``-branch ``i = 0`` reset never
ran and the CPU materialise loop started writing at column ``_K`` instead of 0 -> ``IndexError: index K is out of
bounds for axis 1 with size K``. This is the default path on any GPU-less host (and under CUDA_VISIBLE_DEVICES="").

Pre-fix: an MRMR fit that reaches the per-pair FE scan crashes with that IndexError. Post-fix: the cursor is reset
before the CPU loop, so the fit completes.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def _xy(n: int = 240, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({name: rng.standard_normal(n) for name in ("a", "b", "c", "d", "e", "f")})
    # A target with pairwise structure so the FE pair-search engine actually materialises candidate columns.
    y = pd.Series(((X["a"] * X["b"] + X["c"] - X["d"]) > 0).astype(np.int64), name="targ")
    return X, y


def test_fe_pair_cpu_fallback_does_not_overrun_buffer(monkeypatch):
    """A fit that drives the per-pair FE materialise on the CPU fallback must complete, not IndexError."""
    # Force the GPU-materialise prep ON (its default) so the candidate-sizing loop advances ``i``; the test harness
    # runs CUDA-hidden so the GPU-binning gate declines and the CPU fallback path (the buggy one) is exercised.
    monkeypatch.setenv("MLFRAME_FE_GPU_MATERIALISE", "1")
    X, y = _xy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(verbose=0, random_seed=7).fit(X, y)
    # The fit completing at all is the regression assertion (pre-fix it raised IndexError inside _score_one_pair).
    assert hasattr(m, "support_")
    assert int(np.asarray(m.support_).sum()) >= 1
