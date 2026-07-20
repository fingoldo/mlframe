"""Regression: DCD swap must not decline outright when a cluster member is categorical (string/object).

THE BUG
-------
``evaluate_swap_candidate`` clusters on DISCRETIZED codes (``factors_data``), so a categorical column can
legitimately cluster with numeric ones (their binned codes correlate) even though its RAW values are strings.
Building the PC1/aggregate candidate reads the RAW columns from ``X_raw`` and blindly casts the whole block to
float64 -- a raw string member made that raise ``ValueError: could not convert string to float``, caught by a
broad except that declined the WHOLE swap even when 2+ numeric members remained (surfaced by
profile_fuzz_chains.py: "DCD swap: PC1 fit failed: ValueError(\"could not convert string to float: 'X'\")" on a
cats=8 combo). Fixed by narrowing to numeric-castable members before the cast.

A second, related bug this pins: the pre-existing NaN/Inf ``finite_mask`` filter already dropped columns from the
working matrix without updating ``members``/``member_names`` in lockstep, so ``recipe_obj["members"]`` (used for
replay + aggregate naming) could end up longer than ``mean``/``std``/``signs`` whenever a member was actually
dropped. Both filters now keep ``members``/``member_names`` in sync with the matrix's surviving columns.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _cluster_with_categorical_member(n: int = 1500, seed: int = 0):
    """anchor (noisy latent) + 2 clean numeric duplicates + 1 categorical member, all correlated via binned codes.

    The categorical column's raw values are strings but its DISCRETIZED codes correlate with the shared latent
    (built from the same quantile bins as the numeric members), so DCD's binned-code clustering can legitimately
    group it with the numeric members -- exactly the real-world shape that crashed PC1 fit.
    """
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
    cat_codes = _quantize(latent)  # correlates with latent -> clusters with the numeric members
    cat_labels = np.array(["A", "B", "C", "D"])[cat_codes]  # raw column is STRING, not numeric

    y_col = y.astype(np.int32)
    anchor_b = _quantize(anchor_raw)
    clean_b = _quantize(clean_raw)
    factors = np.column_stack([y_col, anchor_b, clean_b, cat_codes])
    factors_nbins = np.array(
        [int(y_col.max()) + 1, int(anchor_b.max()) + 1, int(clean_b.max()) + 1, int(cat_codes.max()) + 1],
        dtype=np.int64,
    )
    X_raw = pd.DataFrame(
        {
            "y": y.astype(float),
            "anchor": anchor_raw,
            "clean_member": clean_raw,
            "cat_member": cat_labels,
        }
    )
    return X_raw, factors, factors_nbins


def test_dcd_swap_with_categorical_member_does_not_blanket_decline():
    """A categorical cluster member must not crash PC1 fit and force decline -- the swap should still evaluate
    using the surviving numeric members (anchor + clean_member), and if it fires, recipe_obj's arrays must stay
    length-consistent with the actually-used member set (the categorical index excluded).
    """
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
    state.cluster_anchors[1] = {2, 3}  # anchor=1 (numeric), members 2 (numeric) + 3 (categorical)
    state.member_to_anchor[2] = 1
    state.member_to_anchor[3] = 1
    state.pool_pruned_mask[2] = True
    state.pool_pruned_mask[3] = True
    selected_vars = [1]

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warnings.warn escapes; logger.warning is fine (not a Python warning)
        decision = evaluate_swap_candidate(
            state, anchor=1, selected_vars=selected_vars, target_y=np.array([0], dtype=np.int64),
        )

    # Pre-fix this always declined (branch == "none", accept == False) because PC1 fit crashed on the
    # categorical column regardless of the 2 perfectly-usable numeric members (anchor + clean_member).
    assert decision.accept, "swap must not blanket-decline just because one cluster member is categorical"
    if decision.branch == "aggregate":
        members = decision.recipe_obj["members"]
        assert 3 not in members, "the categorical member must be excluded from the aggregate, not silently included"
        assert (
            len(members) == len(decision.recipe_obj["mean"]) == len(decision.recipe_obj["std"]) == len(decision.recipe_obj["signs"])
        ), "members/mean/std/signs must stay length-consistent after filtering non-numeric/non-finite columns"
