"""The raw-MI noise floor is bounded on both sides, so it stays meaningful on the narrow frames a stage sees under the step-input contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _mi_classif_batch
from mlframe.feature_selection.filters._unified_fe_gate import _coerce_y_classes, _null_mi_floor, raw_mi_noise_floor


def _hetero(n=4000, seed=0):
    """Four base columns where ``xi`` and ``g1`` carry signal; ``y`` depends on |xi| scaled by an ``xj``-dependent spread."""
    rng = np.random.default_rng(seed)
    xj = rng.random(n)
    sd = np.where(xj > 0.5, 5.0, 1.0)
    xi = rng.standard_normal(n) * sd
    g1 = rng.standard_normal(n)
    y = np.abs(xi) / sd + 0.5 * g1 + rng.standard_normal(n) * 0.1
    return pd.DataFrame({"xi": xi, "xj": xj, "g1": g1, "g2": rng.standard_normal(n)}), y


def _raw_mi(X, y):
    """Marginal MI of every raw column against the coerced target, as the floor itself computes it."""
    return _mi_classif_batch(np.column_stack([X[c].to_numpy(float) for c in X.columns]), _coerce_y_classes(y), nbins=10)


def test_a_narrow_frame_floor_never_exceeds_the_strongest_raw_column():
    """On four raw columns median + 3.5 MAD is 0.59, above the strongest raw column's 0.33; the floor is capped at that column, so a derived column that
    out-scores every raw column (here the heteroscedastic dispersion pair, MI 0.38) can still clear it."""
    X, y = _hetero()
    floor = raw_mi_noise_floor(X, y)
    assert floor <= _raw_mi(X, y).max() + 1e-12, f"floor {floor} is above the strongest raw column"
    assert floor > 0.0


def test_the_floor_never_undercuts_the_independence_null_on_a_wide_pure_noise_frame():
    """A wide pure-noise frame has a tiny median + MAD; the floor is still at least the analytic null quantile."""
    rng = np.random.default_rng(3)
    n = 3000
    X = pd.DataFrame({f"n{i}": rng.standard_normal(n) for i in range(20)})
    y = rng.standard_normal(n)
    floor = raw_mi_noise_floor(X, y)
    y_classes = len(np.unique(pd.qcut(y, 10, labels=False, duplicates="drop")))
    assert floor >= _null_mi_floor(n, 10, y_classes) * 0.999


def test_the_analytic_null_quantile_scales_inversely_with_n_and_grows_with_the_table_size():
    """``2 n MI`` is chi-squared with (nbins - 1)(classes - 1) degrees of freedom: halving n doubles the floor, a larger table raises it."""
    assert abs(_null_mi_floor(2000, 10, 10) / _null_mi_floor(4000, 10, 10) - 2.0) < 1e-12
    assert _null_mi_floor(4000, 10, 10) > _null_mi_floor(4000, 10, 3) > _null_mi_floor(4000, 4, 3) > 0.0
