"""biz_val test for ``BorutaShap``: on a synthetic 10-feature dataset where 2 features are genuinely informative (``y`` derived from ``0.7 * x_inf + 0.3 * x_inf2 + noise``) and the other 8 are pure standard-normal noise, the selector must recover BOTH informative features AND admit at most a handful of noise features.

Locks in the SHAP-driven Boruta core invariant: rejecting noise-only columns while keeping signal-bearing ones. A regression that either rejects the informative pair (overly-strict pvalue / percentile drift) OR accepts all 10 (broken shadow-comparison) would fail this test.

Per CLAUDE.md (biz_value floor = measured-value minus 5-15% headroom):
  - measured dev run (seed=0, n=600, n_trials=30): informative_kept=2/2, noise_kept=0/8
  - asserted floors: informative_kept >= 2, noise_kept <= 5 (well below the 8-noise-everything regression watermark while leaving room for noise correlation luck at non-fixed seeds)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_biz_val_boruta_shap_filters_noise_keeps_informative():
    pytest.importorskip("shap")
    from mlframe.feature_selection.boruta_shap import BorutaShap

    rng = np.random.default_rng(0)
    n = 600
    x_inf = rng.normal(size=n)
    x_inf2 = rng.normal(size=n)
    noise_cols = rng.normal(size=(n, 8))

    # Linear combo with small Gaussian residual; threshold on its median to balance the binary target ~50/50.
    linear = 0.7 * x_inf + 0.3 * x_inf2 + 0.10 * rng.normal(size=n)
    y = (linear > np.median(linear)).astype(np.int64)

    cols = ["inf1", "inf2"] + [f"noise_{i}" for i in range(8)]
    X = pd.DataFrame(np.column_stack([x_inf, x_inf2, noise_cols]), columns=cols)

    sel = BorutaShap(
        importance_measure="gini",
        classification=True,
        n_trials=30,
        random_state=0,
        verbose=False,
        optimistic=True,
    )
    sel.fit(X, pd.Series(y))

    selected = set(sel.selected_features_)
    informative_kept = selected & {"inf1", "inf2"}
    noise_kept = [c for c in selected if c.startswith("noise_")]

    # Recovery: BOTH informative features must survive. Measured 2/2; floor 2/2 (no headroom -- losing one is a real regression we want to detect).
    assert informative_kept == {"inf1", "inf2"}, (
        f"BorutaShap must retain both informative features; got informative_kept={informative_kept}, "
        f"full selected={sorted(selected)}"
    )

    # Discrimination: at most 5 of 8 noise columns admitted. Measured 0/8; floor 5 keeps room for seed-to-seed correlation luck while still catching the "all 10 admitted" failure mode.
    assert len(noise_kept) <= 5, (
        f"BorutaShap admitted too many noise columns ({len(noise_kept)} of 8); noise_kept={noise_kept}"
    )

    # Sanity: support_ shape and dtype match the sklearn-style contract every other selector in the suite exposes.
    assert sel.support_.shape == (10,)
    assert sel.support_.dtype == bool
    # support_ must be consistent with selected_features_.
    support_named = {c for c, m in zip(cols, sel.support_) if m}
    assert support_named == selected, f"support_ disagrees with selected_features_: {support_named} vs {selected}"
