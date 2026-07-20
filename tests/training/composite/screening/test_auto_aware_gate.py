"""Tests for ``auto_aware_gate`` -- gt_05's auto-selection between ``stacking_aware_gate`` (NNLS) and
``shapley_aware_gate`` (Shapley), gated by the SAME ``max_off_diagonal_correlation`` diagnostic this
package already documents as its "measure-first protocol".
"""

from __future__ import annotations

import numpy as np

from mlframe.training.composite import auto_aware_gate


def _make_duplicate_heavy_pool(n=4000, seed=0):
    """Mirrors gt_05's biz_val fixture: 2 strong + 3 near-identical duplicates-of-strong1 + 2 pure-noise predictors."""
    rng = np.random.default_rng(seed)
    y_margin = rng.standard_normal(n)
    y = (y_margin > 0).astype(np.float64)
    strong1 = 0.6 * y_margin + 0.4 * rng.standard_normal(n)
    strong2 = 0.6 * y_margin + 0.4 * rng.standard_normal(n)
    dups = [strong1 + rng.normal(0, 0.01, n) for _ in range(3)]
    noise = [rng.standard_normal(n) for _ in range(2)]
    names = ["strong1", "strong2", "dup1", "dup2", "dup3", "noise1", "noise2"]
    preds = {n_: arr for n_, arr in zip(names, [strong1, strong2, *dups, *noise])}
    return preds, y


def _make_low_correlation_pool(n=2000, seed=1):
    """Independent predictors -- no duplication, no redundancy to correct for."""
    rng = np.random.default_rng(seed)
    y_margin = rng.standard_normal(n)
    y = (y_margin > 0).astype(np.float64)
    preds = {f"m{i}": 0.5 * y_margin + rng.standard_normal(n) for i in range(4)}
    return preds, y


def test_auto_aware_gate_routes_to_shapley_on_duplicate_heavy_pool():
    """On the duplicate-heavy pool (near-identical predictions, corr ~0.99+), the auto-gate must route
    to Shapley -- the exact regime gt_05's biz_val measured NNLS destabilizing under."""
    preds, y = _make_duplicate_heavy_pool()
    survivors, weights, method = auto_aware_gate(preds, y, min_weight=0.05, engine_kwargs={"n_permutations": 100, "rng": np.random.default_rng(2)})
    assert method == "shapley", f"expected the duplicate-heavy pool to route to shapley, got {method!r}"
    assert survivors, "shapley path returned no survivors"
    assert set(weights.keys()) == set(preds.keys())


def test_auto_aware_gate_routes_to_nnls_on_low_correlation_pool():
    """On a pool with no meaningful redundancy, the auto-gate must route to the cheaper NNLS gate
    (no permutation-Shapley overhead needed when there's no collinearity problem to correct for)."""
    preds, y = _make_low_correlation_pool()
    survivors, weights, method = auto_aware_gate(preds, y, min_weight=0.05)
    assert method == "nnls", f"expected the low-correlation pool to route to nnls, got {method!r}"
    assert survivors
    assert set(weights.keys()) == set(preds.keys())


def test_auto_aware_gate_empty_pool_defaults_to_nnls_method_tag():
    """An empty predictions dict returns empty survivors/weights and the 'nnls' method tag (the
    cheaper path's own empty-input contract), never raises."""
    survivors, weights, method = auto_aware_gate({}, np.array([]))
    assert survivors == []
    assert weights == {}
    assert method == "nnls"


def test_auto_aware_gate_redundancy_threshold_is_respected():
    """A pool whose max off-diagonal correlation sits between two chosen thresholds routes differently
    depending on where ``redundancy_threshold`` is set -- proof the threshold argument actually drives
    the routing decision, not a hardcoded constant."""
    preds, y = _make_duplicate_heavy_pool()
    # Duplicates correlate near 1.0 with strong1; a threshold ABOVE that can never fire.
    _survivors, _weights, method_high_threshold = auto_aware_gate(preds, y, min_weight=0.05, redundancy_threshold=1.5)
    assert method_high_threshold == "nnls", "a redundancy_threshold above any achievable correlation must never route to shapley"
