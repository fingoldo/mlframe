"""Regression sensors for two MRMR support-chokepoint hardenings.

1. RAW-FEATURE FLOOR-DROP PROTECTION must NOT resurrect a below-null raw on
   linear-usability alone. ``decoy = x_real**2`` on ``y = sign(x_real)`` has
   MI(decoy; y) ~ 0 (the square loses the sign), so the relevance screen
   correctly drops it -- but an unregularised single-split R^2 increment used
   to re-add it (overfitting on ~n/3 rows). The re-add now also requires the
   candidate to clear the screen's marginal-MI relevance floor.

2. P>=N FP-CONTROL TOTAL CAP: in the p>>n regime the post-screen retention
   passes can admit a few spurious noise raws over the multiple-comparison
   ceiling ``max(20, p//3)``. When p >= n, the total raw support is capped at
   that ceiling by descending relevance.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

_RAW_ONLY = dict(
    fe_max_steps=0,
    fe_univariate_basis_enable=False,
    fe_univariate_fourier_enable=False,
    fe_hinge_enable=False,
    fe_conditional_dispersion_enable=False,
    fe_wavelet_enable=False,
    fe_hybrid_orth_pair_enable=False,
    fe_auto_escalation_enable=False,
    fe_pair_prewarp_enable=False,
    fe_rung_schedule_enable=False,
    fe_stability_vote_enable=False,
    cluster_aggregate_enable=False,
    dcd_enable=False,
    fe_discrete_structural_operators_enable=False,
    fe_hybrid_orth_triplet_enable=False,
    fe_hybrid_orth_quadruplet_enable=False,
)


def test_below_null_squared_decoy_not_readded_by_linear_usability():
    """Below null squared decoy not readded by linear usability."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(7)
    n = 800
    x_real = rng.normal(size=n)
    y = pd.Series((x_real + 0.2 * rng.normal(size=n) > 0).astype(int), name="y")
    df = pd.DataFrame(
        {
            "x_real": x_real,
            "decoy": x_real**2,  # within-null: MI(x^2; sign(x)) ~ 0
            "noise0": rng.normal(size=n),
            "noise1": rng.normal(size=n),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = MRMR(verbose=0, random_seed=42, **_RAW_ONLY).fit(df, y)
    sel = list(fs.get_feature_names_out())
    assert "x_real" in sel, f"genuine signal dropped: {sel}"
    assert "decoy" not in sel, f"below-null squared decoy resurrected by linear-usability re-add (relevance gate missing): {sel}"


def test_p_ge_n_total_support_capped_at_fp_ceiling():
    """P ge n total support capped at fp ceiling."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n, p, p_signal = 60, 300, 4
    X_sig = rng.standard_normal((n, p_signal))
    X_noise = rng.standard_normal((n, p - p_signal))
    X = np.column_stack([X_sig, X_noise])
    score = X_sig.sum(axis=1) + 0.3 * rng.standard_normal(n)
    y = pd.Series((score > np.median(score)).astype(np.int64))
    Xdf = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = MRMR(verbose=0, random_seed=0).fit(Xdf, y)
    n_sel = len(list(fs.get_feature_names_out()))
    ceiling = max(20, p // 3)
    assert n_sel <= ceiling, f"p>=n FP-control breached: {n_sel} > ceiling {ceiling}"
