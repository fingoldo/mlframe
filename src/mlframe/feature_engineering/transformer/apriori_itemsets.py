"""Frequent itemset mining (Apriori) on discretized X with target-lift ranking.

Iter 98 mechanism. Agent A #3 ranked. Mammography rare-positive target.

Quantile-discretize X to 5 bins/feature; mine frequent itemsets via mlxtend FP-growth on train fold;
rank by lift against target; emit top-K itemsets as boolean features per query.
"""
from __future__ import annotations
import logging
from typing import Any, Literal, Optional
import numpy as np
import polars as pl
from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_apriori_itemsets_features(
    X_train, y_train, X_query=None, splitter=None, *, seed, task="regression",
    n_bins=5, top_k=8, min_support=0.05, max_len=3, standardize=True, column_prefix="apri", dtype=np.float32,
):
    # numpy>=2.0 removed in1d; mlxtend still uses it. Monkey-patch before import.
    if not hasattr(np, "in1d"):
        np.in1d = np.isin
    try:
        import mlxtend
        from mlxtend.frequent_patterns import fpgrowth
    except ImportError as exc:
        raise ImportError("apriori_itemsets requires mlxtend") from exc
    import pandas as pd

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = top_k + 2

    def _discretize(X_ref: np.ndarray, X_target: np.ndarray, n_bins: int):
        d = X_ref.shape[1]
        bin_cols = []
        for j in range(d):
            edges = np.quantile(X_ref[:, j], np.linspace(0, 1, n_bins + 1))
            target_bins = np.clip(np.digitize(X_target[:, j], edges[1:-1]), 0, n_bins - 1)
            for b in range(n_bins):
                bin_cols.append(target_bins == b)
        return np.array(bin_cols, dtype=bool).T  # (n_rows, d*n_bins)

    def _process(Xt, Xq, y_t):
        # Discretize
        bin_matrix_train = _discretize(Xt, Xt, n_bins)
        bin_matrix_query = _discretize(Xt, Xq, n_bins)
        col_names = [f"f{j}_b{b}" for j in range(Xt.shape[1]) for b in range(n_bins)]
        # Build pandas DataFrame for mlxtend
        df_train = pd.DataFrame(bin_matrix_train, columns=col_names)
        try:
            frequent = fpgrowth(df_train, min_support=min_support, use_colnames=True, max_len=max_len)
        except Exception:
            frequent = pd.DataFrame()
        if frequent.empty:
            # Fallback
            top_indicators_q = np.zeros((Xq.shape[0], top_k), dtype=np.float32)
            n_matched = np.zeros(Xq.shape[0], dtype=np.float32)
            return np.column_stack([top_indicators_q, n_matched, np.zeros(Xq.shape[0], dtype=np.float32)])
        # Score by lift against target
        if task == "binary":
            target_mean = float(y_t.mean())
        else:
            target_mean = float(np.median(y_t))
        lifts = []
        for _, row in frequent.iterrows():
            itemset = list(row["itemsets"])
            cols_idx = [col_names.index(it) for it in itemset]
            mask_train = np.all(bin_matrix_train[:, cols_idx], axis=1)
            if mask_train.sum() == 0:
                lifts.append(0.0)
                continue
            if task == "binary":
                cond_mean = float(y_t[mask_train].mean())
                lift = (cond_mean + 1e-6) / (target_mean + 1e-6)
            else:
                cond_mean = float(np.mean(y_t[mask_train]))
                lift = abs(cond_mean - target_mean) / (float(np.std(y_t)) + 1e-9)
            lifts.append(lift)
        # Sort by lift, take top_k
        order = np.argsort(lifts)[::-1][:top_k]
        top_indicators_q = np.zeros((Xq.shape[0], top_k), dtype=np.float32)
        for k_i, idx in enumerate(order):
            itemset = list(frequent.iloc[idx]["itemsets"])
            cols_idx = [col_names.index(it) for it in itemset]
            top_indicators_q[:, k_i] = np.all(bin_matrix_query[:, cols_idx], axis=1).astype(np.float32)
        # Pad with zeros if fewer itemsets than top_k
        n_matched = top_indicators_q.sum(axis=1).astype(np.float32)
        return np.column_stack([top_indicators_q, n_matched, np.full(Xq.shape[0], float(len(frequent)), dtype=np.float32)])

    def _make_df(feats):
        cols = {}
        for k in range(top_k):
            cols[f"{column_prefix}_itemset{k}"] = feats[:, k].astype(dtype, copy=False)
        cols[f"{column_prefix}_n_matched"] = feats[:, top_k].astype(dtype, copy=False)
        cols[f"{column_prefix}_n_frequent_total"] = feats[:, top_k + 1].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        return pl.DataFrame(_make_df(_process(X_train_f, Xq, y_train_f)))
    if splitter is None:
        raise ValueError("Mode A requires splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, n_features_out), dtype=dtype)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_f)):
        out[val_idx] = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx]).astype(dtype, copy=False)
        logger.info("apriori_itemsets: fold %d done", fold_idx + 1)
    return pl.DataFrame(_make_df(out))
