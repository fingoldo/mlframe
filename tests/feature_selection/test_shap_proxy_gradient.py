"""Gradient ("Schrodinger gates") optimizer tests. Skipped unless torch is installed."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")


def test_gradient_returns_valid_candidates_and_beats_random():
    from mlframe.feature_selection._shap_proxy_gradient import gradient_top_n
    from mlframe.feature_selection._shap_proxy_objective import coalition_margin, proxy_loss

    rng = np.random.default_rng(0)
    n, f = 500, 10
    phi = rng.normal(size=(n, f))
    base = np.full(n, 0.2)
    truth = [0, 1, 2, 3]
    y = base + phi[:, truth].sum(axis=1) + 0.02 * rng.normal(size=n)

    top = gradient_top_n(phi, base, y, classification=False, metric="rmse",
                         n_iter=300, random_state=0, top_n=5)
    assert len(top) >= 1
    # ascending, deduplicated, all non-empty
    losses = [l for l, _ in top]
    assert losses == sorted(losses)
    assert all(len(c) >= 1 for _, c in top)
    # beats a random same-size subset
    best = top[0][1]
    rnd = sorted(np.random.default_rng(1).choice(f, size=len(best), replace=False).tolist())
    assert top[0][0] < proxy_loss(coalition_margin(phi, base, rnd), y, "rmse")


def test_gradient_reproducible():
    from mlframe.feature_selection._shap_proxy_gradient import gradient_top_n

    rng = np.random.default_rng(5)
    phi = rng.normal(size=(300, 8))
    base = np.full(300, 0.0)
    y = base + phi[:, [1, 2]].sum(axis=1) + 0.02 * rng.normal(size=300)
    a = gradient_top_n(phi, base, y, classification=False, metric="rmse", n_iter=150, random_state=42, top_n=3)
    b = gradient_top_n(phi, base, y, classification=False, metric="rmse", n_iter=150, random_state=42, top_n=3)
    assert [c for _, c in a] == [c for _, c in b]
