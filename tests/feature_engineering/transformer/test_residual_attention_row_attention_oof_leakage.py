"""OOF-leakage regression test for ``compute_residual_attention`` Mode A.

The bug (fixed): the auxiliary residual target was built with a flat ``KFold`` independent of the caller's outer ``splitter``. A val row in outer fold f then attends
to complement rows whose aux residual was produced by an aux model that may have trained on that same val row -> the row's own target leaks into its own attention
feature across the outer split.

The fix nests the aux OOF inside the outer splitter: for outer fold f, the residual bank f's val rows attend to is computed by an aux OOF restricted to f's train
complement only, so f's val rows are fully excluded from the bank they attend to.

Sensor: perturb a single row's target and recompute. In a leak-free Mode A the row is excluded from its own fold's complement bank, so its OWN attention feature is
invariant to its OWN target (exact 0.0 delta). Pre-fix the flat aux residual of the row's neighbours moved with the row's perturbed y, so its own feature changed
(measured delta ~0.35 on this fixture). The filename carries ``row_attention`` so the ANN-backend collection skip in conftest applies.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer import compute_residual_attention


pytestmark = pytest.mark.fast


def test_residual_attention_own_y_does_not_leak_into_own_feature():
    rng = np.random.default_rng(0)
    n, d = 300, 5
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + 0.3 * X[:, 1] + 0.2 * rng.standard_normal(n)).astype(np.float32)
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    kw = dict(
        seed=0,
        n_heads=1,
        head_dim=3,
        k=8,
        aggregate=("y_mean",),
        aux_n_estimators=60,
        aux_max_depth=4,
        projection="random",
    )

    base = compute_residual_attention(X, y, None, splitter, **kw).to_numpy().ravel()

    # Perturb ONLY this row's target by a large amount; every other row's y is identical.
    r = 7
    y_perturbed = y.copy()
    y_perturbed[r] += 50.0
    perturbed = compute_residual_attention(X, y_perturbed, None, splitter, **kw).to_numpy().ravel()

    own_delta = abs(float(base[r] - perturbed[r]))
    assert own_delta < 1e-5, (
        f"Row {r}'s own attention feature moved by {own_delta:.4f} when only its OWN target changed. "
        "A row must be excluded from the residual bank it attends to; a non-zero delta means the aux residual "
        "of the row's neighbours leaked its own target (flat aux KFold independent of the outer splitter)."
    )
