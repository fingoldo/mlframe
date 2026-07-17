"""Regression test: ``compute_target_quantile_attention`` Mode A must refit the RobustScaler PER FOLD on train rows only.

The bug (fixed): the scaler was fit ONCE on the full X_train outside the fold loop, so each fold's held-out rows entered that fold's scaler stats (an X-distribution
leak — the median/IQR a val row is normalised by partly depended on the val row itself).

Sensor: a held-out fold's output features must be invariant to the X-values of OTHER held-out rows IN THE SAME val fold... that is hard to isolate directly, so we
test the equivalent contract: a row's OOF feature must not change when we perturb the X of rows that share its outer fold's VAL partition but are NOT in its training
complement. Pre-fix the single global scaler folded those rows' X into the shared median/IQR; post-fix each fold's scaler sees only its own train complement, so
perturbing rows that never enter that complement leaves the feature unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer import compute_target_quantile_attention


pytestmark = pytest.mark.fast


def test_target_quantile_modea_scaler_is_per_fold_train_only():
    rng = np.random.default_rng(0)
    n, d = 200, 6
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + 0.4 * X[:, 1] + 0.2 * rng.standard_normal(n)).astype(np.float32)
    splitter = KFold(n_splits=5, shuffle=False)
    kw = dict(seed=0, n_quantiles=4, similarity="cosine", standardize=True)

    base = compute_target_quantile_attention(X, y, None, splitter, **kw).to_numpy()

    # Pick a target row and a "perturb" row that fall in the SAME val fold (so the perturb row is never in the target row's train complement).
    splits = list(splitter.split(X))
    _tr0, va0 = splits[0]
    target_row, perturb_row = int(va0[0]), int(va0[1])

    X2 = X.copy()
    X2[perturb_row] += 100.0  # large X-shift on a row that shares the target row's val fold

    perturbed = compute_target_quantile_attention(X2, y, None, splitter, **kw).to_numpy()

    delta = float(np.abs(base[target_row] - perturbed[target_row]).max())
    assert delta < 1e-5, (
        f"Target row {target_row}'s OOF features moved by {delta:.4f} when only a SAME-FOLD val row's X changed. "
        "The scaler must be refit per fold on train rows only; a non-zero delta means a held-out row's X leaked into "
        "the scaler stats (global fit outside the fold loop)."
    )
