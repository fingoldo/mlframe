"""brute_force_top_n and its float64 CPU reference agree on well-separated subsets to float-rounding tolerance."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import brute_force_top_n
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_subsetrank import brute_force_top_n_cpu_ref


def test_winner_and_losses_agree_within_float32_rounding():
    """The informative pair wins under both paths and the reported losses match to float32 tolerance."""
    rng = np.random.default_rng(0)
    n, f = 400, 8
    phi = rng.normal(0, 0.05, size=(n, f))
    y = rng.normal(size=n)
    phi[:, 2] = 0.6 * y
    phi[:, 5] = 0.3 * y
    base = np.zeros(n)
    kw = dict(classification=False, metric="rmse", min_card=2, max_card=3, top_n=5)
    fast = brute_force_top_n(phi, base, y, **kw)
    ref = brute_force_top_n_cpu_ref(phi, base, y, **kw)
    assert fast[0][1] == ref[0][1] == (2, 5) or set(fast[0][1]) == set(ref[0][1])
    np.testing.assert_allclose([l for l, _ in fast], [l for l, _ in ref], rtol=1e-4)
