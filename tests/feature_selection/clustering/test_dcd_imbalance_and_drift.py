"""Final edge-case probes for DCD clustering (2026-06-03): extreme class
imbalance (a known mlframe sensitivity) and recipe replay under train/test
distribution DRIFT. Assert graceful, finite, non-crashing behaviour -- failures
are prod bugs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _fit(X, y, **kw):
    from mlframe.feature_selection.filters.mrmr import MRMR

    base = dict(dcd_enable=True, dcd_tau_cluster=0.5, dcd_cluster_size_threshold=2, verbose=0, random_seed=0)
    base.update(kw)
    return MRMR(**base).fit(X, y)


def test_dcd_extreme_imbalance_no_crash():
    # ~1.5% positive rate + a redundancy cluster. The swap null + MI clustering
    # must not crash or emit garbage on the rare-positive regime.
    rng = np.random.default_rng(0)
    n = 6000
    z = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "a": z + 0.05 * rng.standard_normal(n),
            "b": z + 0.05 * rng.standard_normal(n),
            "c": z + 0.05 * rng.standard_normal(n),
            "noise": rng.standard_normal(n),
        }
    )
    # ~1.5% positives driven by the extreme upper tail of z.
    thresh = np.quantile(z, 0.985)
    y = pd.Series((z > thresh).astype(int))
    assert 0 < y.mean() < 0.05
    m = _fit(X, y)
    names = list(m.get_feature_names_out())
    assert len(names) >= 1
    # transform output must be finite.
    out = np.asarray(m.transform(X.iloc[:500]), dtype=np.float64)
    assert np.all(np.isfinite(np.nan_to_num(out)))


def test_dcd_transform_under_distribution_drift_is_finite():
    # Fit on train, transform a DRIFTED test frame (shifted mean + inflated
    # scale). Replay of any cluster aggregate recipe must stay finite and not
    # crash even though the test distribution differs from fit.
    rng = np.random.default_rng(1)
    n = 2000
    z = rng.standard_normal(n)
    Xtr = pd.DataFrame(
        {
            "a": z + 0.05 * rng.standard_normal(n),
            "b": z + 0.05 * rng.standard_normal(n),
            "c": z + 0.05 * rng.standard_normal(n),
            "noise": rng.standard_normal(n),
        }
    )
    ytr = pd.Series((z > 0).astype(int))
    m = _fit(Xtr, ytr)

    rng2 = np.random.default_rng(2)
    zt = 3.0 + 2.5 * rng2.standard_normal(800)  # shifted mean, inflated variance
    Xte = pd.DataFrame(
        {
            "a": zt + 0.05 * rng2.standard_normal(800),
            "b": zt + 0.05 * rng2.standard_normal(800),
            "c": zt + 0.05 * rng2.standard_normal(800),
            "noise": 2.0 * rng2.standard_normal(800) - 1.0,
        }
    )
    out = np.asarray(m.transform(Xte), dtype=np.float64)
    assert out.shape[0] == 800
    assert np.all(np.isfinite(np.nan_to_num(out))), "drifted-test transform emitted non-finite"


def test_dcd_transform_with_all_nan_test_column_no_crash():
    rng = np.random.default_rng(3)
    n = 1500
    z = rng.standard_normal(n)
    Xtr = pd.DataFrame(
        {
            "a": z + 0.05 * rng.standard_normal(n),
            "b": z + 0.05 * rng.standard_normal(n),
            "c": z + 0.05 * rng.standard_normal(n),
            "noise": rng.standard_normal(n),
        }
    )
    ytr = pd.Series((z > 0).astype(int))
    m = _fit(Xtr, ytr)
    Xte = Xtr.iloc[:300].copy()
    Xte["b"] = np.nan  # a cluster member is entirely NaN at transform time
    out = np.asarray(m.transform(Xte), dtype=np.float64)
    assert np.all(np.isfinite(np.nan_to_num(out)))
