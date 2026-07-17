"""biz_value: ShortlistTransformerAdapter.fit_transform must produce Mode-A OUT-OF-FOLD train features, so the
train FE distribution matches the out-of-sample val/test FE that transform (Mode B) produces.

The default sklearn fit().transform() fed the model IN-SAMPLE train features, a train/serving skew that
measurably HURTS honest holdout -- catastrophically for trust_score_oof on binary (measured -0.42 AUC via
bench_curated_fe_holdout_value). fit_transform now routes supervised transformers through Mode A; this pins
that the skew is gone (holdout AUC within a small margin of raw-only) and that the mechanism actually engages
(fit_transform train features differ from the Mode-B fit+transform train features).
"""

import numpy as np
import pandas as pd
import pytest

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from mlframe.feature_engineering.curated_fe import curated_fe_pipelines

pytest.importorskip("lightgbm")
SEED = 0


def _binary(n=4000, seed=SEED):
    """Helper: Binary."""
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, 40, size=n)
    cat_eff = rng.standard_normal(40) * 2.0
    x1, x2, x3 = (rng.standard_normal(n) for _ in range(3))
    noise = rng.standard_normal((n, 4))
    sig = cat_eff[cat] + 1.5 * np.sin(2 * x1) * x2 + 0.7 * x3**2
    p = 1 / (1 + np.exp(-(sig - np.median(sig))))
    y = (rng.uniform(size=n) < p).astype(int)
    cols = ["cat", "x1", "x2", "x3", "n0", "n1", "n2", "n3"]
    return pd.DataFrame(np.column_stack([cat, x1, x2, x3, noise]), columns=cols), pd.Series(y, name="y")


def _auc(Xtr, ytr, Xho, yho):
    """Helper: Auc."""
    import lightgbm as lgb

    m = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=0, verbose=-1, n_jobs=-1)
    m.fit(Xtr, ytr)
    return roc_auc_score(yho, m.predict_proba(Xho)[:, 1])


def test_mode_a_fit_transform_eliminates_trust_score_binary_skew():
    """Mode a fit transform eliminates trust score binary skew."""
    X, y = _binary()
    Xtr, Xho, ytr, yho = train_test_split(X, y, test_size=0.4, random_state=SEED)
    base = _auc(Xtr.values, ytr.values, Xho.values, yho.values)

    pipe = curated_fe_pipelines(task="binary", names=["trust_score_oof"], seed=SEED, passthrough=False)["trust_score_oof"]
    ftr = np.nan_to_num(np.asarray(pipe.fit_transform(Xtr, ytr), dtype=float))  # Mode A (OOF)
    fho = np.nan_to_num(np.asarray(pipe.transform(Xho), dtype=float))  # Mode B (honest)
    auc_fe = _auc(np.hstack([Xtr.values, ftr]), ytr.values, np.hstack([Xho.values, fho]), yho.values)

    # Pre-fix Mode-B fed in-sample train features and cratered holdout AUC by ~0.42. Mode-A must keep it within
    # a small margin of raw-only (the FE may still be mildly unhelpful on binary, but NOT catastrophic skew).
    assert auc_fe >= base - 0.05, f"trust_score Mode-A still skews holdout badly: fe={auc_fe:.4f} base={base:.4f}"


def test_fit_transform_mode_a_differs_from_mode_b_in_sample():
    """The mechanism engages: Mode-A OOF train features are not identical to the Mode-B in-sample train
    features (they would be identical only if fit_transform silently fell back to fit+transform)."""
    X, y = _binary(n=2000)
    pipe_a = curated_fe_pipelines(task="binary", names=["nn_oof_target_mean"], seed=SEED, passthrough=False)["nn_oof_target_mean"]
    ftr_a = np.nan_to_num(np.asarray(pipe_a.fit_transform(X, y), dtype=float))
    pipe_b = curated_fe_pipelines(task="binary", names=["nn_oof_target_mean"], seed=SEED, passthrough=False)["nn_oof_target_mean"]
    pipe_b.fit(X, y)
    ftr_b = np.nan_to_num(np.asarray(pipe_b.transform(X), dtype=float))  # in-sample Mode B
    assert ftr_a.shape == ftr_b.shape
    assert not np.allclose(ftr_a, ftr_b), "fit_transform (Mode A) produced the in-sample Mode-B train features"
