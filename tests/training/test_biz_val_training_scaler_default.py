"""Verdict-pinning biz_value tests for the numeric-scaling default (qual-20).

Lever: ``PreprocessingBackendConfig.scaler_name`` default ``"standard"``.

qual-20 measured StandardScaler vs RobustScaler on downstream honest-holdout
AUC/R2 for scale-sensitive linear models with heavy outliers / heavy tails
present (bench: ``mlframe.training._benchmarks.bench_scaler_default_standard_vs_robust``).
Verdict: KEEP standard -- robust shows no scenario-majority win and only
~1e-3-magnitude deltas, because a single global L2 regulariser absorbs the
std-vs-IQR scale-family difference once every column is individually normalised.

These tests pin BOTH sides of that verdict (per REJECTED != DELETED):
  1. the default really is ``"standard"`` (a silent flip would trip this),
  2. ``"robust"`` is still reachable and produces a valid honest-holdout score,
  3. on the contaminated honest holdout robust does NOT dominate standard --
     it stays within a tiny band (fails if a future change makes robust the
     decisive winner here, which would mean the verdict must be revisited).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytest.importorskip("polars_ds.pipeline")

from sklearn.linear_model import LogisticRegression

from mlframe.training._preprocessing_configs import PreprocessingBackendConfig
from mlframe.training.pipeline import create_polarsds_pipeline


def _holdout_auc(scaler_name: str, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n, p = 3000, 8
    col_scale = rng.uniform(0.3, 8.0, p)
    X = rng.normal(0, 1, size=(n, p)) * col_scale
    beta = rng.normal(0, 1, p) / col_scale
    y = (X @ beta + rng.normal(0, 1, n) > 0).astype(int)
    for j in range(0, p, 2):
        spike = rng.random(n) < 0.05
        X[spike, j] += rng.uniform(20, 80, spike.sum()) * col_scale[j] * rng.choice([-1, 1], spike.sum())
    cut = int(n * 0.7)
    cols = [f"f{j}" for j in range(p)]

    def _frame(rows):
        return pl.DataFrame({c: [float(X[i, j]) for i in rows] for j, c in enumerate(cols)})

    cfg = PreprocessingBackendConfig(
        imputer_strategy="median", scaler_name=scaler_name, categorical_encoding=None, prefer_polarsds=True
    )
    tr, ho = _frame(range(cut)), _frame(range(cut, n))
    pipe = create_polarsds_pipeline(tr, cfg, verbose=0)
    Xtr = np.nan_to_num(pipe.transform(tr).to_numpy())
    Xho = np.nan_to_num(pipe.transform(ho).to_numpy())
    from sklearn.metrics import roc_auc_score

    m = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, y[:cut])
    return roc_auc_score(y[cut:], m.predict_proba(Xho)[:, 1])


def test_biz_val_scaler_default_is_standard():
    """qual-20 verdict: the most-accurate default scaler is ``standard``."""
    assert PreprocessingBackendConfig().scaler_name == "standard"


def test_biz_val_scaler_robust_still_reachable():
    """REJECTED != DELETED: the robust option stays usable end-to-end."""
    cfg = PreprocessingBackendConfig(scaler_name="robust")
    assert cfg.scaler_name == "robust"
    auc = _holdout_auc("robust", seed=23)
    assert 0.5 <= auc <= 1.0


def test_biz_val_scaler_robust_does_not_dominate_standard_on_contaminated_holdout():
    """Pin the KEEP-standard verdict: averaged over seeds on the contaminated
    holdout, robust must NOT beat standard by a material margin. Fails (forcing
    a re-evaluation) only if robust becomes the decisive winner here.
    """
    seeds = [11, 23, 47, 71, 97]
    std = np.mean([_holdout_auc("standard", s) for s in seeds])
    rob = np.mean([_holdout_auc("robust", s) for s in seeds])
    # robust may edge std by measurement noise but must not win by a material margin.
    assert rob - std < 0.01, f"robust unexpectedly dominates standard (std={std:.4f} rob={rob:.4f}); revisit qual-20 verdict"
