"""LightGBM routes ``x <= threshold`` with a double threshold; the kernel compares float32 values with ``x < t'``. The converted threshold must give the same side for every float32 value."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import _lightgbm_threshold_to_lt_float32


def _float32_neighbours(t32: np.float32, width: int = 4) -> np.ndarray:
    """The float32 values within ``width`` steps of ``t32`` in both directions."""
    lo = hi = np.float32(t32)
    out = [lo]
    for _ in range(width):
        lo = np.nextafter(lo, np.float32(-np.inf))
        hi = np.nextafter(hi, np.float32(np.inf))
        out += [lo, hi]
    return np.asarray(out, dtype=np.float32)


def test_every_float32_value_goes_the_side_lightgbm_sends_it():
    """Random double thresholds across scales, tested on the float32 values around them: ``x < t'`` equals ``float(x) <= threshold`` (the double comparison LightGBM makes)."""
    rng = np.random.default_rng(0)
    scales = 10.0 ** rng.integers(-6, 6, size=400)
    thresholds = rng.normal(size=400) * scales
    assert thresholds.size > 0
    for threshold in thresholds:
        t_conv = np.float32(_lightgbm_threshold_to_lt_float32(float(threshold)))
        neighbours = list(_float32_neighbours(np.float32(threshold)))
        assert neighbours
        for x in neighbours:
            assert bool(x < t_conv) == bool(float(x) <= float(threshold)), (threshold, x, t_conv)


def test_the_nearest_rounding_of_the_old_conversion_would_misroute():
    """A threshold just under a float32 value, rounded to the nearest float32, lands on that value: the old conversion sent that value left, LightGBM sends it right. The new one agrees with LightGBM."""
    x = np.float32(1.0000001)  # a float32 grid point
    threshold = float(x) - 1e-9  # between the float32 below x and x, nearer to x
    old = float(np.nextafter(np.float32(threshold), np.float32(np.inf)))  # the predecessor's conversion
    assert not (float(x) <= threshold)  # LightGBM: x is above the threshold, it goes right
    assert bool(x < np.float32(old))  # the old conversion: x goes left, wrongly
    assert not bool(x < np.float32(_lightgbm_threshold_to_lt_float32(threshold)))  # the new one: x goes right, as LightGBM does


def test_extractor_keeps_the_additivity_when_two_training_values_sit_on_adjacent_float32_numbers():
    """End to end: a LightGBM model splitting between two adjacent float32 values must give ``base + sum(phi) == raw prediction`` for every row, on the table form and on the path kernel."""
    import pytest

    lgb = pytest.importorskip("lightgbm")
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble, treeshap_phi_base_numba, treeshap_phi_base_path

    rng = np.random.default_rng(3)
    n = 4000
    a = np.float32(1.0)
    b = np.nextafter(a, np.float32(2.0))  # the adjacent float32 number
    x0 = np.where(rng.random(n) < 0.5, a, b).astype(np.float32)
    x1 = rng.normal(size=n).astype(np.float32)
    y = 3.0 * (x0 == b) + 0.5 * x1 + 0.05 * rng.normal(size=n)
    X = np.column_stack([x0, x1]).astype(np.float32)
    model = lgb.LGBMRegressor(n_estimators=30, num_leaves=8, min_child_samples=5, verbose=-1, n_jobs=2).fit(X, y)
    ens = extract_ensemble(model)
    margin = model.predict(X, raw_score=True)
    for fn in (treeshap_phi_base_numba, treeshap_phi_base_path):
        phi, base = fn(ens, X)
        assert float(np.max(np.abs(base + phi.sum(axis=1) - margin))) < 1e-6, fn.__name__
