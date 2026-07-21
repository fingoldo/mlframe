"""Sign-of-residual baseline: predict sign(y - ŷ) as a classification target.

Iter 88 mechanism. Agent C #5 ranked.

Two-stage OOF:
1. Fit mu_hat baseline.
2. Fit classifier predicting sign(y - mu_hat) ∈ {0, 1} (1 = positive residual).

Per query emit 5 leakage-free features:
- pred_mu
- p_positive_residual (probability of under-prediction at query)
- bias_signal = p_positive_residual - 0.5 (signed bias)
- abs_bias = |bias_signal|
- baseline_score = mu_query + bias_signal * residual_scale (bias-corrected point estimate; residual_scale is the
  train-fold residual std of the mu-model, so a detected directional bias nudges the raw prediction toward the
  side the sign-classifier expects the true residual to fall on)

Captures asymmetric bias the squared-error boosting cannot detect.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_sign_residual_baseline_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    standardize: bool = True,
    column_prefix: str = "signres",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Sign-of-residual baseline features. 5 outputs per row."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("sign_residual_baseline requires lightgbm") from exc

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = 5

    def _process(Xt, Xq, y_t, fold_seed):
        """Fit a baseline mean/probability model, then fit a second classifier to predict the SIGN of the train residual (whether ``y`` under-predicts the baseline); the second model's query-row probability minus 0.5 gives a directional bias signal, plus its absolute value as a magnitude-only feature."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s, Xq_s = Xt, Xq
        if task == "binary":
            m_mu = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                      random_state=int(fold_seed), verbose=-1, n_jobs=-1).fit(Xt_s, y_t.astype(np.int32))
            mu_query = np.asarray(m_mu.predict_proba(Xq_s))[:, 1].astype(np.float32)
        else:
            m_mu = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(fold_seed), verbose=-1, n_jobs=-1).fit(Xt_s, y_t)
            mu_query = np.asarray(m_mu.predict(Xq_s)).astype(np.float32)
        # mu_train drives BOTH sign_target (the second model's classification target) and residual_scale
        # below -- an IN-SAMPLE mu_train is close to y_t almost by construction, making sign_target close
        # to a coin flip on near-zero in-sample residuals (rather than the genuine over/under-prediction
        # direction) and understating residual_scale. Use an inner-KFold(3) OOF mu_train instead -- m_mu
        # itself (fit on the full Xt_s) stays the model used for mu_query, genuinely held-out regardless.
        n_t = Xt_s.shape[0]
        if n_t < 3:
            if task == "binary":
                mu_train = np.asarray(m_mu.predict_proba(Xt_s))[:, 1].astype(np.float32)
            else:
                mu_train = np.asarray(m_mu.predict(Xt_s)).astype(np.float32)
        else:
            from sklearn.model_selection import KFold
            mu_train = np.zeros(n_t, dtype=np.float32)
            inner_splitter = KFold(n_splits=3, shuffle=True, random_state=int(fold_seed) + 11)
            for inner_idx, (in_tr, in_val) in enumerate(inner_splitter.split(Xt_s)):
                if task == "binary":
                    m_inner = lgb.LGBMClassifier(
                        n_estimators=50, max_depth=3, learning_rate=0.1,
                        random_state=int(fold_seed) + 21 + inner_idx, verbose=-1, n_jobs=-1,
                    ).fit(Xt_s[in_tr], y_t[in_tr].astype(np.int32))
                    mu_train[in_val] = np.asarray(m_inner.predict_proba(Xt_s[in_val]))[:, 1].astype(np.float32)
                else:
                    m_inner = lgb.LGBMRegressor(
                        n_estimators=50, max_depth=3, learning_rate=0.1,
                        random_state=int(fold_seed) + 21 + inner_idx, verbose=-1, n_jobs=-1,
                    ).fit(Xt_s[in_tr], y_t[in_tr])
                    mu_train[in_val] = np.asarray(m_inner.predict(Xt_s[in_val])).astype(np.float32)
        # Sign target: 1 if y > mu_hat (under-prediction), else 0.
        sign_target = (y_t > mu_train).astype(np.int32)
        # If degenerate (all same sign), pad fallback
        if sign_target.sum() == 0 or sign_target.sum() == sign_target.shape[0]:
            p_pos = np.full(Xq_s.shape[0], float(sign_target.mean()), dtype=np.float32)
        else:
            m_sign = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                        random_state=int(fold_seed) + 1, verbose=-1, n_jobs=-1).fit(Xt_s, sign_target)
            p_pos = np.asarray(m_sign.predict_proba(Xq_s))[:, 1].astype(np.float32)
        bias_signal = (p_pos - 0.5).astype(np.float32)
        abs_bias = np.abs(bias_signal).astype(np.float32)
        residual_scale = np.float32(np.std(y_t - mu_train))
        baseline_score = (mu_query + bias_signal * residual_scale).astype(np.float32)
        return np.column_stack([mu_query, p_pos, bias_signal, abs_bias, baseline_score])

    def _make_df(feats):
        """Slice the ``_process`` output columns into a name-tagged dict (mu/p_pos_residual/bias_signal/abs_bias/baseline_score), cast to the requested output dtype."""
        cols = {}
        cols[f"{column_prefix}_mu"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_p_pos_residual"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_bias_signal"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_abs_bias"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_baseline_score"] = feats[:, 4].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out: np.ndarray = np.zeros((n_train, n_features_out), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("sign_residual_baseline: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
