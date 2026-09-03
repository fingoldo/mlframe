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


# A leaky CV score can land at or below the honest one by chance; only a materially higher one is the tell.
_LEAK_TOLERANCE = 0.02


def _score_expanding_folds(
    df_sorted: pd.DataFrame,
    y_sorted: np.ndarray,
    chunks: List[np.ndarray],
    n_splits: int,
    leaky_values: np.ndarray,
    fit_transform_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    estimator_factory: Callable[[], Any],
    scoring: Optional[str],
    remediated_sorted: Optional[np.ndarray] = None,
) -> tuple:
    """Score every expanding fold with ``leaky_values`` against a feature refit on that fold's train prefix.

    ``leaky_values`` is whatever candidate series is under suspicion, indexed by TIME-SORTED row: the
    full-dataset computation for detection, or the stitched remediated series when verifying a remediation. The
    honest comparison always refits ``fit_transform_fn`` on the fold's train rows only.

    When ``remediated_sorted`` is supplied it is filled in place with each fold's honest validation values --
    the leakage-safe recomputation boundary.

    Returns ``(leaky_scores, honest_scores, leaking_fold_val_idx)``.
    """
    from sklearn.model_selection import cross_val_score

    leaky_scores: List[float] = []
    honest_scores: List[float] = []
    leaking_folds: List[np.ndarray] = []
    for fold_idx in range(1, n_splits + 1):
        train_idx = np.concatenate(chunks[:fold_idx])
        val_idx = chunks[fold_idx]
        if val_idx.shape[0] == 0:
            continue
        fold_idx_all = np.concatenate([train_idx, val_idx])

        honest_feature_all = np.asarray(fit_transform_fn(df_sorted.iloc[train_idx], df_sorted.iloc[fold_idx_all]), dtype=np.float64).ravel()
        y_fold = y_sorted[fold_idx_all]
        n_train = train_idx.shape[0]

        cv_2fold = [(np.arange(n_train), np.arange(n_train, n_train + val_idx.shape[0]))]
        leaky_score = float(np.mean(cross_val_score(estimator_factory(), leaky_values[fold_idx_all].reshape(-1, 1), y_fold, cv=cv_2fold, scoring=scoring)))
        honest_score = float(np.mean(cross_val_score(estimator_factory(), honest_feature_all.reshape(-1, 1), y_fold, cv=cv_2fold, scoring=scoring)))
        leaky_scores.append(leaky_score)
        honest_scores.append(honest_score)
        logger.info("detect_expanding_window_feature_leakage: fold %d/%d leaky=%.4f honest=%.4f", fold_idx, n_splits, leaky_score, honest_score)

        if remediated_sorted is not None:
            # The honest value for the fold's OWN validation rows (the tail of honest_feature_all) is the
            # leakage-safe recomputation boundary: fit strictly precedes this fold's split.
            remediated_sorted[val_idx] = honest_feature_all[n_train:]
        if (leaky_score - honest_score) > _LEAK_TOLERANCE:
            leaking_folds.append(val_idx)

    return leaky_scores, honest_scores, leaking_folds


def detect_expanding_window_feature_leakage(
    df: pd.DataFrame,
    time_col: str,
    y: np.ndarray,
    fit_transform_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    estimator_factory: Callable[[], Any],
    n_splits: int = 5,
    scoring: Optional[str] = None,
    auto_remediate: bool = False,
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
    auto_remediate
        Opt-in (default False, output bit-identical to the pre-existing detection-only behavior when
        omitted). When True, additionally builds a leakage-safe replacement feature by stitching together
        the per-fold HONEST values already computed for detection (each validation row's value comes from
        ``fit_transform_fn`` fit on ONLY rows strictly before that row's fold train-cutoff -- the same
        recomputation boundary the "honest" score uses), and reports exactly which original row ranges
        were flagged as leaking.

    Returns
    -------
    dict
        ``leaky_scores``, ``honest_scores`` (one per fold), ``leaky_mean``, ``honest_mean``,
        ``inflation`` (``leaky_mean - honest_mean``), ``leak_detected`` (True if ``inflation`` exceeds a
        small tolerance -- a leaky CV score can be equal to or even below honest by chance, but a
        MATERIALLY higher leaky score across folds is the tell).

        When ``auto_remediate=True``, three additional keys are present:

        - ``remediated_feature``: ``(len(df),)`` array, same row order as the ORIGINAL (unsorted) ``df``,
          with each row's value recomputed as of that row's own fold train-cutoff -- safe to feed straight
          back into feature selection / model training in place of the leaky feature.
        - ``leaky_row_positions``: list of ``np.ndarray`` (sorted, into the ORIGINAL ``df`` row order) for
          every fold whose per-fold ``leaky_score - honest_score`` gap exceeds the same tolerance used for
          ``leak_detected`` -- the exact rows whose "held-out" score was inflated by future information.
          Each fold's validation slice is a CONTIGUOUS range in time-sorted order, but ``df`` need not be
          pre-sorted by ``time_col`` (this function sorts internally), so those rows' ORIGINAL positions
          are generally scattered, not contiguous -- hence an exact position array rather than a
          ``(start, end)`` range, which would silently span everything between the scattered positions'
          min and max on unsorted input.
        - ``remediation_verified``: True when re-scoring the folds with ``remediated_feature`` in place of the
          suspect feature, against an honest per-fold refit, shows no residual inflation. This is a measured
          quantity, reported alongside it as ``remediation_inflation``, and it can be False.
        - ``remediation_inflation``: the residual gap that flag is derived from.
    """
    # kind="stable": ties among duplicate ``time_col`` values must break the same way every time, so that
    # ``remediated_sorted``'s row alignment and the ``inverse_order`` round-trip below are reproducible.
    order = np.argsort(df[time_col].to_numpy(), kind="stable")
    df_sorted = df.iloc[order].reset_index(drop=True)
    y_sorted = np.asarray(y)[order]
    n = len(df_sorted)

    chunks = np.array_split(np.arange(n), n_splits + 1)
    leaky_feature_full = np.asarray(fit_transform_fn(df_sorted, df_sorted), dtype=np.float64).ravel()

    # sorted-row-index -> position in the ORIGINAL (unsorted) df, for remediation output only.
    inverse_order = np.empty(n, dtype=np.int64)
    inverse_order[order] = np.arange(n)

    remediated_sorted = np.full(n, np.nan, dtype=np.float64) if auto_remediate else None
    if auto_remediate:
        # Rows in the seed chunk (never a fold's validation slice) get the best available safe value:
        # fit on themselves, since no strictly-earlier data exists to fit on.
        seed_idx = chunks[0]
        assert remediated_sorted is not None
        remediated_sorted[seed_idx] = np.asarray(fit_transform_fn(df_sorted.iloc[seed_idx], df_sorted.iloc[seed_idx]), dtype=np.float64).ravel()

    leaky_scores, honest_scores, leaking_folds = _score_expanding_folds(
        df_sorted, y_sorted, chunks, n_splits, leaky_feature_full, fit_transform_fn, estimator_factory, scoring, remediated_sorted
    )
    # val_idx is contiguous in TIME-SORTED position (an np.array_split chunk), but its ORIGINAL (pre-sort)
    # positions are generally scattered when df wasn't already time-sorted -- collapsing to (min, max+1) would
    # silently claim every row in between as leaked too, on the realistic unsorted-input case spanning nearly the
    # whole dataset. Report the exact scattered positions.
    leaky_row_positions = [np.sort(inverse_order[val_idx]) for val_idx in leaking_folds]

    leaky_mean = float(np.mean(leaky_scores))
    honest_mean = float(np.mean(honest_scores))
    inflation = leaky_mean - honest_mean

    result: Dict[str, Any] = {
        "leaky_scores": leaky_scores,
        "honest_scores": honest_scores,
        "leaky_mean": leaky_mean,
        "honest_mean": honest_mean,
        "inflation": inflation,
        "leak_detected": bool(inflation > _LEAK_TOLERANCE),
    }

    if auto_remediate:
        assert remediated_sorted is not None
        remediated_feature = remediated_sorted[inverse_order]

        # Re-score the folds with the CALLER-VISIBLE remediated array standing in for the suspect feature, against
        # an honest refit of the ORIGINAL feature. Two things have to hold for the remediation to be trustworthy,
        # and both are measured here rather than assumed: the stitched series must carry no residual advantage on
        # a fold's validation rows, and ``remediated_feature[order]`` must round-trip back to the sorted series a
        # consumer would expect -- an index-alignment bug in the inverse permutation shows up as inflation.
        #
        # The previous implementation recursed into this function with a callback that ignored its ``fit_df``
        # argument, so the leaky and honest branches were handed element-for-element identical arrays; the
        # inflation was exactly 0.0 by construction and ``remediation_verified`` was True for every input,
        # including one where the remediation had genuinely failed.
        verif_leaky, verif_honest, _ = _score_expanding_folds(
            df_sorted, y_sorted, chunks, n_splits, remediated_feature[order], fit_transform_fn, estimator_factory, scoring, None
        )
        remediation_inflation = float(np.mean(verif_leaky) - np.mean(verif_honest))

        result["remediated_feature"] = remediated_feature
        result["leaky_row_positions"] = leaky_row_positions
        result["remediation_inflation"] = remediation_inflation
        result["remediation_verified"] = bool(remediation_inflation <= _LEAK_TOLERANCE)

    return result


__all__ = ["detect_expanding_window_feature_leakage"]
