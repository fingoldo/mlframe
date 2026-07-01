"""Biz-value: ``auto_chain_top_k`` bounds how many ``residual x unary`` chains discovery keeps.

``discover_chains`` searches ``residual x tail-unary`` chains and returns up to ``top_k`` that beat both single stages
on tiny-CV y-scale RMSE. On a heavy-tailed residual target (where a tail-compression unary genuinely helps the residual
stage), at least one chain qualifies; ``top_k`` is the cap on how many are returned.

A regression that ignores ``auto_chain_top_k`` (returns all candidates, or hardcodes a different count) breaks the
bound assertion.
"""
from __future__ import annotations

import numpy as np

from mlframe.training.composite.discovery._auto_chain import discover_chains


def _make_heavy_tail_residual(n=1500, seed=0):
    """y has a strong linear-in-base component plus a heavy-tailed feature-driven residual a unary can compress."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, n)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    # cube makes the residual heavy-tailed -> cbrt/yj tail-compression helps the second stage.
    resid = (0.8 * x0 + 0.4 * x1) ** 3
    y = 2.0 * base + resid + 0.05 * rng.normal(size=n)
    x_matrix = np.column_stack([x0, x1])
    return y, base, x_matrix


def test_biz_val_auto_chain_top_k_caps_returned_chains():
    """top_k bounds the number of chains; a larger cap returns at least as many."""
    y, base, x_matrix = _make_heavy_tail_residual()
    chains_1 = discover_chains(
        y=y, base=base, x_matrix=x_matrix,
        cv_folds=3, n_estimators=30, num_leaves=8, random_state=0, top_k=1,
    )
    chains_3 = discover_chains(
        y=y, base=base, x_matrix=x_matrix,
        cv_folds=3, n_estimators=30, num_leaves=8, random_state=0, top_k=3,
    )
    assert len(chains_1) <= 1, "auto_chain_top_k=1 must return at most one chain"
    assert len(chains_3) >= len(chains_1), "a larger auto_chain_top_k cannot return fewer chains"
