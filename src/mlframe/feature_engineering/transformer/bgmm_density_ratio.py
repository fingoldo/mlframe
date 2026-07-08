"""Per-class BGM density ratio features: Bayesian-mixture LDA analog.

Iter 46 mechanism. Hybrid of iter 41 BGM (additive sampling) and iter 28 denrat (density log-ratio).

Fits BayesianGaussianMixture SEPARATELY on positive-class and negative-class rows. The log-density-ratio ``log p(x|y=1) − log p(x|y=0)`` is the Bayes-optimal
decision feature (LDA's nonlinear analog under Gaussian-mixture class-conditionals).

Mechanism (binary):
1. Per fold, fit BGM_pos on positive-class rows.
2. Per fold, fit BGM_neg on negative-class rows.
3. For each query: compute log p_pos(x), log p_neg(x), log_ratio = log_p_pos − log_p_neg.
4. Repeat at multiple component counts {3, 5, 8} to expose multi-scale density modelling.

Features per row: 3 K-scales × 3 quantities (log_p_pos, log_p_neg, log_ratio) = 9 features.

Why this differs from iter 28 denrat:
- iter 28 denrat: kernel-density estimate with Gaussian RBF (non-parametric); single bandwidth per row.
- iter 46 bgmm_density_ratio: parametric Gaussian-mixture density (multi-modal-aware); multiple component counts.

Why this differs from iter 41/45 BGM-virtual:
- iter 41/45: SAMPLES virtuals from BGM_pos posterior, exposes virtual-distance features.
- iter 46: EVALUATES BGM_pos and BGM_neg densities at each query; exposes density-ratio features.

So this is the "density-evaluation" complement to the "density-sampling" of iter 41/45.

Leakage discipline: both BGMs refit per fold from train-fold rows only.

Cost: 2 × BGM fits per K per fold = 6 BGM fits per fold for 3 K-scales. Each is ~1-2 sec; total ~10-15 sec per fold.

Reference: McLachlan 1992 — Gaussian mixture models for classification (mixture-discriminant analysis).
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Literal, Optional, Tuple

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_COMPONENT_COUNTS_DEFAULT = (3, 5, 8)


def _fit_bgmm_and_score(X_fit: np.ndarray, X_score: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    """Fit BayesianGaussianMixture on X_fit, return log-density of X_score under the fitted model."""
    from sklearn.mixture import BayesianGaussianMixture
    n_fit = X_fit.shape[0]
    if n_fit < n_components + 1:
        # Too few rows — fall back to uniform low log-density.
        return np.full(X_score.shape[0], -30.0, dtype=np.float32)
    bgm = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=200,
        random_state=seed,
        reg_covar=1e-4,
        weight_concentration_prior_type="dirichlet_process",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            bgm.fit(X_fit)
            return bgm.score_samples(X_score).astype(np.float32)
        except Exception as exc:
            logger.info("bgmm_density_ratio: BGM fit failed at K=%d (%s); returning low log-density.", n_components, exc)
            return np.full(X_score.shape[0], -30.0, dtype=np.float32)


def compute_bgmm_density_ratio_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    component_counts: Tuple[int, ...] = _COMPONENT_COUNTS_DEFAULT,
    q_high: float = 0.8,
    standardize: bool = True,
    column_prefix: str = "bdr",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Per-class BGM log-density-ratio features at multiple K-scales.

    Output: ``len(component_counts) * 3`` columns per row — log_p_pos, log_p_neg, log_ratio for each K.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _slice(X_sub: np.ndarray, y_sub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if task == "binary":
            pos = y_sub > 0.5
            return X_sub[pos], X_sub[~pos]
        y_hi = np.quantile(y_sub, q_high)
        y_lo = np.quantile(y_sub, 1.0 - q_high)
        return X_sub[y_sub >= y_hi], X_sub[y_sub <= y_lo]

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        Xt_pos, Xt_neg = _slice(Xt_s, y_t)
        n_q = Xq_s.shape[0]
        all_feats = np.zeros((n_q, len(component_counts) * 3), dtype=np.float32)
        if Xt_pos.shape[0] < 2 or Xt_neg.shape[0] < 2:
            return all_feats
        # bench-attempt-rejected (2026-06-08): the 6 BGM fits (len(component_counts) x {pos,neg}) are
        # independent and dominate the FE shortlist wall (~96%: this transformer is ~21s of a ~22s
        # 6k-row FE-transform run; profile attributes 100% to sklearn's EM -- _estimate_log_gaussian_prob
        # 10.4s + _estimate_gaussian_covariances_full 7.5s -- with ~0.006s in this wrapper). Parallelising
        # the 6 fits via ThreadPoolExecutor measured 1.45-1.9x (17.6s -> 9.1-12.1s), BUT is NOT
        # bit-identical: threads>=3 diverged immediately, and threads=2 (identical when idle) diverged
        # 2/4 trials under background CPU load -- OpenBLAS load-dependent thread-partitioning changes the
        # GEMM reduction order, perturbing the EM trajectory. Pinning BLAS to 1 thread (threadpool_limits)
        # is itself non-identical to the current multi-thread output. There is no detectable safe condition
        # (system load is unpredictable), so the parallel path violates the bit-identity gate. The cost is
        # 100% inside sklearn's BayesianGaussianMixture -- the correct library kernel -- so this stays serial.
        for scale_idx, n_comp in enumerate(component_counts):
            k_pos = max(2, min(n_comp, Xt_pos.shape[0] // 3))
            k_neg = max(2, min(n_comp, Xt_neg.shape[0] // 3))
            log_p_pos = _fit_bgmm_and_score(Xt_pos, Xq_s, n_components=k_pos, seed=fold_seed + scale_idx * 7)
            log_p_neg = _fit_bgmm_and_score(Xt_neg, Xq_s, n_components=k_neg, seed=fold_seed + scale_idx * 11)
            log_ratio = log_p_pos - log_p_neg
            base = scale_idx * 3
            all_feats[:, base + 0] = log_p_pos
            all_feats[:, base + 1] = log_p_neg
            all_feats[:, base + 2] = log_ratio
        return all_feats

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        for scale_idx, n_comp in enumerate(component_counts):
            tag = f"K{n_comp}"
            base = scale_idx * 3
            cols[f"{column_prefix}_{tag}_logp_pos"] = feats[:, base + 0].astype(dtype, copy=False)
            cols[f"{column_prefix}_{tag}_logp_neg"] = feats[:, base + 1].astype(dtype, copy=False)
            cols[f"{column_prefix}_{tag}_logratio"] = feats[:, base + 2].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out: np.ndarray = np.zeros((n_train, len(component_counts) * 3), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("bgmm_density_ratio: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
