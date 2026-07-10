"""``detect_expanding_window_feature_leakage``: catch feature-selection significance inflation from
full-dataset (vs train-only) feature computation in time-ordered CV.

Source: KKBox Music Recommendation Challenge 1st place -- "if we use both train.csv and test.csv to generate
features for validation, we are leaking information from the future to the past... false positive results
when doing feature selection." A common, insidious bug: an aggregate/encoded feature (frequency count,
target encoding, a fitted statistic) is computed ONCE over the full dataset before any CV split, so every
fold's "held-out" validation rows were actually influenced by data that, chronologically, comes AFTER them
-- inflating CV score and making features look more predictive than they really are at serve time.

Mechanism: takes a caller-supplied ``fit_transform_fn(fit_df, transform_df) -> feature array`` (the same
fit/transform contract as any encoder: statistics are learned from ``fit_df``, applied to ``transform_df``).
For each EXPANDING time-ordered fold, computes the SAME feature two ways:

- "leaky": ``fit_transform_fn`` called ONCE on the FULL time-ordered dataset (statistics see every row,
  including rows chronologically after the fold's validation cutoff) -- the bug this function detects.
- "honest": ``fit_transform_fn`` called PER FOLD on ONLY that fold's train-cutoff rows, applied to the whole
  dataset -- the fix (mirrors :func:`mlframe.feature_engineering.as_of_aggregate.leakage_safe_aggregate`'s
  per-row cutoff discipline, generalized to CV-fold granularity for feature-selection significance testing).

Reports both CV score curves; a leaky score inflated well above the honest score at the SAME folds is the
false-positive-feature-selection-significance failure mode the source technique warns about.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_expanding_window_feature_leakage(
    df: pd.DataFrame,
    time_col: str,
    y: np.ndarray,
    fit_transform_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    estimator_factory: Callable[[], Any],
    n_splits: int = 5,
    scoring: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare CV scores between full-dataset ("leaky") and per-fold-train-only ("honest") feature
    computation over expanding time-ordered folds.

    Parameters
    ----------
    df
        Row-ordered-by-nothing-in-particular feature frame; sorted internally by ``time_col``.
    time_col
        Column defining chronological order.
    y
        Target, same row order as ``df`` (BEFORE internal sorting -- reordered alongside ``df``).
    fit_transform_fn
        ``(fit_df, transform_df) -> (len(transform_df),) array``. Learns whatever statistic/encoding from
        ``fit_df`` and applies it to every row of ``transform_df``. E.g. a frequency-count encoder: count
        occurrences of a categorical column WITHIN ``fit_df``, look up each ``transform_df`` row's count.
    estimator_factory
        Zero-arg callable returning a fresh unfitted estimator, used to score each fold with the leaky vs
        honest feature as its sole input column.
    n_splits
        Number of expanding-window folds (the first ``1/(n_splits+1)`` of the sorted data seeds the initial
        train window; each subsequent fold's train window expands to include all previously-validated rows).
    scoring
        sklearn scorer name; None uses the estimator's default ``.score``.

    Returns
    -------
    dict
        ``leaky_scores``, ``honest_scores`` (one per fold), ``leaky_mean``, ``honest_mean``,
        ``inflation`` (``leaky_mean - honest_mean``), ``leak_detected`` (True if ``inflation`` exceeds a
        small tolerance -- a leaky CV score can be equal to or even below honest by chance, but a
        MATERIALLY higher leaky score across folds is the tell).
    """
    from sklearn.model_selection import cross_val_score

    order = np.argsort(df[time_col].to_numpy())
    df_sorted = df.iloc[order].reset_index(drop=True)
    y_sorted = np.asarray(y)[order]
    n = len(df_sorted)

    chunks = np.array_split(np.arange(n), n_splits + 1)
    leaky_feature_full = np.asarray(fit_transform_fn(df_sorted, df_sorted), dtype=np.float64).reshape(-1, 1)

    leaky_scores: List[float] = []
    honest_scores: List[float] = []
    for fold_idx in range(1, n_splits + 1):
        train_idx = np.concatenate(chunks[:fold_idx])
        val_idx = chunks[fold_idx]
        if val_idx.shape[0] == 0:
            continue
        fold_idx_all = np.concatenate([train_idx, val_idx])

        honest_feature = np.asarray(fit_transform_fn(df_sorted.iloc[train_idx], df_sorted.iloc[fold_idx_all]), dtype=np.float64).reshape(-1, 1)
        y_fold = y_sorted[fold_idx_all]
        n_train = train_idx.shape[0]

        cv_2fold = [(np.arange(n_train), np.arange(n_train, n_train + val_idx.shape[0]))]
        leaky_score = float(np.mean(cross_val_score(estimator_factory(), leaky_feature_full[fold_idx_all], y_fold, cv=cv_2fold, scoring=scoring)))
        honest_score = float(np.mean(cross_val_score(estimator_factory(), honest_feature, y_fold, cv=cv_2fold, scoring=scoring)))
        leaky_scores.append(leaky_score)
        honest_scores.append(honest_score)
        logger.info("detect_expanding_window_feature_leakage: fold %d/%d leaky=%.4f honest=%.4f", fold_idx, n_splits, leaky_score, honest_score)

    leaky_mean = float(np.mean(leaky_scores))
    honest_mean = float(np.mean(honest_scores))
    inflation = leaky_mean - honest_mean

    return {
        "leaky_scores": leaky_scores,
        "honest_scores": honest_scores,
        "leaky_mean": leaky_mean,
        "honest_mean": honest_mean,
        "inflation": inflation,
        "leak_detected": bool(inflation > 0.02),
    }


__all__ = ["detect_expanding_window_feature_leakage"]
