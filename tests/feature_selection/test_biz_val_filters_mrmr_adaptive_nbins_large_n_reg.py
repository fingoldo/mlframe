"""biz_val + regression tests for the MRMR large-n regression adaptive-quantization gate.

The QUANTITATIVE win (fixed 20-bin quantile beats MDLP on reg n=100k: holdout R2 +0.116 / F1 +0.242, 15/15 seeds) is pinned by the
committed 180-cell campaign at ``feature_selection/_benchmarks/fs_quality/_results/mrmr_largeN_campaign.jsonl`` and confirmed by
``qual24_adaptive_nbins_large_n_reg_confirm.py``. A full n>=50k MRMR fit is ~60s, too slow for the suite, so these tests pin the GATE
LOGIC: it must FIRE (switch to nbins_strategy=None + quantization_nbins=20) exactly on detected-regression AND n>=threshold while the
user left the quantization params at defaults, and must NOT fire on classification, on small-n, or when the user pinned a value.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _reg_xy(n: int, p: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = (1.5 * X[:, 0] - X[:, 1] + 0.8 * X[:, 2] + 0.3 * rng.standard_normal(n)).astype(np.float64)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(p)]), pd.Series(y, name="y")


def _clf_xy(n: int, p: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    logits = 1.5 * X[:, 0] - X[:, 1] + 0.8 * X[:, 2]
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logits))).astype(np.int64)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(p)]), pd.Series(y, name="y")


def _fast_mrmr(**kw):
    from mlframe.feature_selection.filters.mrmr import MRMR

    base = dict(fe_max_steps=0, interactions_max_order=1, full_npermutations=0, baseline_npermutations=0,
                random_seed=0, use_gpu=False, n_jobs=1, verbose=0, cv=2)
    base.update(kw)
    return MRMR(**base)


def test_biz_val_adaptive_nbins_gate_default_on():
    """The gate ships ON by default (corrective mechanism enabled by default)."""
    m = _fast_mrmr()
    assert m.adaptive_nbins_large_n_reg is True
    assert m.adaptive_nbins_large_n_reg_threshold == 50_000
    assert m.adaptive_nbins_large_n_reg_nbins == 20


def test_biz_val_adaptive_nbins_fires_on_large_n_regression():
    """Detected regression at n>=threshold flips the campaign-winner config (nbins_strategy=None, quantization_nbins=20)."""
    X, y = _reg_xy(n=60_000)
    m = _fast_mrmr()
    m.fit(X, y)
    assert getattr(m, "_adaptive_nbins_large_n_reg_fired_", False) is True
    assert m.nbins_strategy is None
    assert m.quantization_nbins == 20


def test_biz_val_adaptive_nbins_no_fire_on_small_n_regression():
    """The fixed-20 path LOSES at reg n=20k (campaign holdout -0.143), so it must stay MDLP below the threshold."""
    X, y = _reg_xy(n=20_000)
    m = _fast_mrmr()
    m.fit(X, y)
    assert getattr(m, "_adaptive_nbins_large_n_reg_fired_", False) is False
    assert m.nbins_strategy == "mdlp"
    assert m.quantization_nbins == 10


def test_biz_val_adaptive_nbins_no_fire_on_large_n_classification():
    """Classification ties (n=100k) or loses (n=20k) in the campaign, so the gate must NOT fire on a detected classifier."""
    X, y = _clf_xy(n=60_000)
    m = _fast_mrmr()
    m.fit(X, y)
    assert getattr(m, "_adaptive_nbins_large_n_reg_fired_", False) is False
    assert m.nbins_strategy == "mdlp"
    assert m.quantization_nbins == 10


def test_biz_val_adaptive_nbins_respects_explicit_user_nbins():
    """An explicit user quantization_nbins (!= default 10) is never overridden -- the gate only resolves defaults."""
    X, y = _reg_xy(n=60_000)
    m = _fast_mrmr(quantization_nbins=12)
    m.fit(X, y)
    assert getattr(m, "_adaptive_nbins_large_n_reg_fired_", False) is False
    assert m.quantization_nbins == 12


def test_biz_val_adaptive_nbins_respects_explicit_user_strategy():
    """An explicit non-mdlp nbins_strategy is never overridden by the gate."""
    X, y = _reg_xy(n=60_000)
    m = _fast_mrmr(nbins_strategy="sturges")
    m.fit(X, y)
    assert getattr(m, "_adaptive_nbins_large_n_reg_fired_", False) is False
    assert m.nbins_strategy == "sturges"


def test_biz_val_adaptive_nbins_disabled_keeps_mdlp_at_large_n():
    """adaptive_nbins_large_n_reg=False is the documented opt-out: MDLP stays on even on large-n regression."""
    X, y = _reg_xy(n=60_000)
    m = _fast_mrmr(adaptive_nbins_large_n_reg=False)
    m.fit(X, y)
    assert getattr(m, "_adaptive_nbins_large_n_reg_fired_", False) is False
    assert m.nbins_strategy == "mdlp"
