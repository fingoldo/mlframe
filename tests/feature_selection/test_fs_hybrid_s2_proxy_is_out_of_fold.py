"""Audit 2026-11-20 finding 12-F27: the SHAP proxy Brier must be out-of-fold, so it is comparable with the honest Brier.

On pure-noise labels an in-sample model looks much better than chance; an out-of-fold proxy must stay near 0.25.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_FS_HYBRID = Path(__file__).resolve().parents[2] / "src/mlframe/feature_selection/_benchmarks/fs_hybrid"


def test_out_of_fold_proxy_does_not_look_like_the_in_sample_fit():
    """Noise labels, 8 noise columns: the out-of-fold proxy Brier is near chance and clearly above the in-sample one."""
    pytest.importorskip("shap")
    sys.path.insert(0, str(_FS_HYBRID))
    try:
        import lightgbm as lgb
        import shap
        from scipy.special import expit
        from sklearn.metrics import brier_score_loss

        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import _maybe_patch_shap_xgb_base_score

        import s2_interaction_deficit_test as s2
    finally:
        sys.path.remove(str(_FS_HYBRID))
    rng = np.random.default_rng(0)
    Xtr = pd.DataFrame(rng.standard_normal((300, 8)), columns=[f"n{i}" for i in range(8)])
    ytr = rng.integers(0, 2, size=300)
    names = list(Xtr.columns)
    m = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, verbose=-1).fit(Xtr, ytr)
    with _maybe_patch_shap_xgb_base_score():
        e = shap.TreeExplainer(m)
    phi = np.asarray(e.shap_values(Xtr))
    if phi.ndim == 3:
        phi = phi[..., 1] if phi.shape[-1] == 2 else phi[..., 0]
    base = float(np.ravel(e.expected_value)[-1])
    in_sample = brier_score_loss(ytr, np.clip(expit(base + phi.sum(1)), 1e-6, 1 - 1e-6))
    oof = s2.oof_proxy_brier(Xtr, ytr, names, seed=0)
    assert oof > in_sample + 0.03
    assert oof > 0.2
