"""biz_val for the pre-flight decision rule (Batch C): it must correctly distinguish the regime where
ShapProxiedFS shines (additive, high-SNR, well-fit) from where it struggles (pure interaction / XOR),
from cheap statistics alone -- so a user can decide whether/how to run it before paying the full cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")


def _additive(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    inf = rng.normal(size=(n, 5))
    noise = rng.normal(size=(n, 5))
    X = pd.DataFrame(np.column_stack([inf, noise]), columns=[f"f{i}" for i in range(10)])
    logit = 1.0 * inf[:, 0] + 0.9 * inf[:, 1] - 0.8 * inf[:, 2] + 0.6 * inf[:, 3] + 0.4 * inf[:, 4]
    y = (logit > 0).astype(int)  # high-SNR additive -> the favourable regime
    return X, y


def _xor(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8))
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(int)  # pure interaction
    return pd.DataFrame(x, columns=[f"f{i}" for i in range(8)]), y


@pytest.mark.slow
def test_biz_val_preflight_favours_additive_flags_interaction():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    add = ShapProxiedFS.preflight(*_additive(), classification=True, random_state=0)
    xor = ShapProxiedFS.preflight(*_xor(), classification=True, random_state=0)

    # Additive high-SNR -> high additive ratio, recommended to run.
    assert add["diagnostics"]["additive_ratio"] > 0.7, add["diagnostics"]
    assert add["recommendation"] == "run", add

    # XOR -> low additive ratio (a depth-1 stump can't model it), flagged interaction-heavy.
    assert xor["diagnostics"]["additive_ratio"] < 0.6, xor["diagnostics"]
    assert xor["recommendation"] in ("caution", "fallback")
    assert any("interaction" in r for r in xor["reasons"]), xor["reasons"]
    assert "enable interaction_aware=True" in xor["suggestions"]


@pytest.mark.slow
def test_biz_val_preflight_flags_redundancy_and_width():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    rng = np.random.default_rng(0)
    n = 2000
    z = rng.normal(size=(n, 3))
    refl = np.hstack([z[:, [k]] + 0.1 * rng.normal(size=(n, 20)) for k in range(3)])  # 60 redundant
    X = pd.DataFrame(refl, columns=[f"f{i}" for i in range(refl.shape[1])])
    y = (z[:, 0] + 0.8 * z[:, 1] - 0.7 * z[:, 2] + 0.3 * rng.normal(size=n) > 0).astype(int)
    rep = ShapProxiedFS.preflight(X, y, classification=True, random_state=0)
    assert rep["diagnostics"]["max_abs_corr"] >= 0.7
    assert "enable cluster_features=True" in rep["suggestions"]
    assert rep["diagnostics"]["n_features"] > 40
