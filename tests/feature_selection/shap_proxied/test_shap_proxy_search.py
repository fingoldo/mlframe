"""Unit tests for the numba brute-force subset search.

The incremental-sum kernel with periodic-reset is the one place a subtle accumulation bug could
silently corrupt the ranking, so we assert exact parity against a naive ``itertools.combinations``
recompute, plus serial-vs-parallel agreement.
"""

from __future__ import annotations

import itertools

import numpy as np


def _naive_top(phi, base, y, metric, max_card):
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import coalition_margin, proxy_loss

    best = []
    f = phi.shape[1]
    for r in range(1, max_card + 1):
        for comb in itertools.combinations(range(f), r):
            m = coalition_margin(phi, base, list(comb))
            best.append((proxy_loss(m, y, metric), tuple(comb)))
    best.sort(key=lambda t: t[0])
    return best


def test_brute_force_matches_itertools_regression():
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import brute_force_top_n

    rng = np.random.default_rng(0)
    phi = rng.normal(size=(300, 7))
    base = np.full(300, 0.25)
    y = base + phi[:, [0, 1, 2]].sum(axis=1) + 0.01 * rng.normal(size=300)

    kernel = brute_force_top_n(phi, base, y, classification=False, metric="rmse", max_card=7, top_n=10)
    naive = _naive_top(phi, base, y, "rmse", 7)

    assert set(kernel[0][1]) == set(naive[0][1]) == {0, 1, 2}
    # top-5 sets agree (order-insensitive within ties)
    k_sets = [frozenset(c) for _, c in kernel[:5]]
    n_sets = [frozenset(c) for _, c in naive[:5]]
    assert set(k_sets) == set(n_sets)
    # reported rmse matches the naive rmse for the winner
    np.testing.assert_allclose(kernel[0][0], naive[0][0], rtol=1e-6)


def test_brute_force_classification_brier_matches_itertools():
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import brute_force_top_n

    rng = np.random.default_rng(3)
    phi = rng.normal(size=(400, 6))
    base = np.full(400, 0.0)
    margin = base + phi[:, [1, 3]].sum(axis=1)
    y = (1.0 / (1.0 + np.exp(-margin)) > rng.random(400)).astype(float)

    kernel = brute_force_top_n(phi, base, y, classification=True, metric="brier", max_card=6, top_n=8)
    naive = _naive_top(phi, base, y, "brier", 6)
    assert set(kernel[0][1]) == set(naive[0][1])
    np.testing.assert_allclose(kernel[0][0], naive[0][0], rtol=1e-6)


def test_serial_parallel_agree():
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import brute_force_top_n

    rng = np.random.default_rng(1)
    phi = rng.normal(size=(250, 9))
    base = np.full(250, 0.1)
    y = base + phi[:, [2, 5, 7]].sum(axis=1) + 0.02 * rng.normal(size=250)

    serial = brute_force_top_n(phi, base, y, classification=False, metric="rmse", max_card=9, top_n=10, parallel=False)
    par = brute_force_top_n(phi, base, y, classification=False, metric="rmse", max_card=9, top_n=10,
                            parallel=True, n_chunks=4)
    assert set(serial[0][1]) == set(par[0][1])
    assert {frozenset(c) for _, c in serial[:5]} == {frozenset(c) for _, c in par[:5]}


def test_total_subsets_count():
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import total_subsets

    # sum_{r=1}^{n} C(n, r) == 2^n - 1
    assert total_subsets(10, 1, None) == 2**10 - 1
    assert total_subsets(10, 1, 3) == 10 + 45 + 120
