"""Regression: DCD swap evaluation must not print debug tracing to stdout.

``evaluate_swap_candidate`` (and ``screen_dcd_discover_and_swap``'s caller path) had leftover
``print(f"DEBUG_...")`` statements firing unconditionally on every anchor/pool evaluation -- pure
debug leftovers, not gated behind ``verbose``. Surfaced by profiling/bug_hunt_fuzz_chains.py: a DCD
swap-eligible combo spammed dozens of ``DEBUG_SCREEN:``/``DEBUG_EVAL_ENTRY:``/``DEBUG_DCD:`` lines to
stdout, polluting logs and wasting cycles on f-string formatting of large pool/cluster structures.
"""

from __future__ import annotations

import io
import warnings
from contextlib import redirect_stdout

import numpy as np
import pandas as pd


def _cluster_with_categorical_member(n: int = 1500, seed: int = 0):
    """Anchor + 2 numeric duplicates + 1 categorical member, all clustering via binned codes (reused shape
    from test_dcd_swap_categorical_member.py -- this combo reliably fires evaluate_swap_candidate)."""
    rng = np.random.default_rng(int(seed))
    latent = rng.standard_normal(n)

    def _quantize(x, k=4):
        """Discretize x into up to k quantile-bin integer codes."""
        edges = np.quantile(x, np.linspace(0, 1, k + 1))
        edges = np.unique(edges)
        if edges.size < 3:
            return np.zeros_like(x, dtype=np.int32)
        return np.clip(np.searchsorted(edges[1:-1], x, side="right"), 0, k - 1).astype(np.int32)

    y = (latent + 0.3 * rng.standard_normal(n) > 0).astype(np.int64)
    anchor_raw = latent + 0.5 * rng.standard_normal(n)
    clean_raw = latent + 0.02 * rng.standard_normal(n)
    cat_codes = _quantize(latent)
    cat_labels = np.array(["A", "B", "C", "D"])[cat_codes]

    y_col = y.astype(np.int32)
    anchor_b = _quantize(anchor_raw)
    clean_b = _quantize(clean_raw)
    factors = np.column_stack([y_col, anchor_b, clean_b, cat_codes])
    factors_nbins = np.array(
        [int(y_col.max()) + 1, int(anchor_b.max()) + 1, int(clean_b.max()) + 1, int(cat_codes.max()) + 1],
        dtype=np.int64,
    )
    X_raw = pd.DataFrame({"y": y.astype(float), "anchor": anchor_raw, "clean_member": clean_raw, "cat_member": cat_labels})
    return X_raw, factors, factors_nbins


def test_evaluate_swap_candidate_prints_nothing_to_stdout():
    """A DCD swap-eligible evaluation must not leak debug tracing to stdout."""
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
        DCDState,
        evaluate_swap_candidate,
    )

    X_raw, factors, factors_nbins = _cluster_with_categorical_member()
    state = DCDState(
        pool_pruned_mask=np.zeros(4, dtype=bool),
        X_raw_ref=X_raw,
        factors_data=factors,
        factors_nbins=factors_nbins,
        cols=["y", "anchor", "clean_member", "cat_member"],
        nbins=factors_nbins,
        target_indices=np.array([0], dtype=np.int64),
        quantization_method="quantile",
        quantization_nbins=4,
        quantization_dtype=np.int32,
        cluster_size_threshold=2,
        min_cluster_size=2,
        swap_gain_threshold=0.05,
        tau_cluster=0.5,
        swap_method="pca_pc1",
    )
    state.cluster_anchors[1] = {2, 3}
    state.member_to_anchor[2] = 1
    state.member_to_anchor[3] = 1
    state.pool_pruned_mask[2] = True
    state.pool_pruned_mask[3] = True

    buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(buf):
            evaluate_swap_candidate(state, anchor=1, selected_vars=[1], target_y=np.array([0], dtype=np.int64))

    captured = buf.getvalue()
    assert "DEBUG" not in captured, f"evaluate_swap_candidate leaked debug output to stdout: {captured!r}"


def test_evaluate_swap_candidate_below_threshold_prints_nothing():
    """The early-return path (cluster below threshold) must also stay silent."""
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
        DCDState,
        evaluate_swap_candidate,
    )

    X_raw, factors, factors_nbins = _cluster_with_categorical_member()
    state = DCDState(
        pool_pruned_mask=np.zeros(4, dtype=bool),
        X_raw_ref=X_raw,
        factors_data=factors,
        factors_nbins=factors_nbins,
        cols=["y", "anchor", "clean_member", "cat_member"],
        nbins=factors_nbins,
        target_indices=np.array([0], dtype=np.int64),
        quantization_method="quantile",
        quantization_nbins=4,
        quantization_dtype=np.int32,
        cluster_size_threshold=99,
        min_cluster_size=99,
        swap_gain_threshold=0.05,
        tau_cluster=0.5,
        swap_method="pca_pc1",
    )
    state.cluster_anchors[1] = {2}

    buf = io.StringIO()
    with redirect_stdout(buf):
        decision = evaluate_swap_candidate(state, anchor=1, selected_vars=[1], target_y=np.array([0], dtype=np.int64))

    assert decision.accept is False
    assert buf.getvalue() == ""
