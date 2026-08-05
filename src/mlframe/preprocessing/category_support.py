"""Screen categorical columns for train/test category-support mismatch.

A raw categorical encoding (ordinal/one-hot/target-mean) is only as good as its category overlap between
train and test: a category present only in test has no learned representation, so the model silently falls
back to a constant/unknown bucket and loses whatever signal that category carried. Frequency (count)
encoding degrades gracefully instead — an unseen category still gets a real frequency value computed from
its own split, so the underlying "how common is this category" signal transfers even when the specific
category identities do not (the pattern IEEE-CIS's 2nd place team diagnosed by eyeballing raw-vs-frequency
plots). ``train_test_support_screen`` automates that diagnosis and recommends a remediation per column.

Frequency encoding itself has a blind spot: when every category has roughly the SAME count (near-uniform
cardinality, e.g. hash buckets or an evenly-sharded id), the resulting frequency column is itself near-constant
and carries no signal, even though the categories genuinely differ in target rate and enough of them are shared
between train and test for a per-category estimate to be worth learning. Raw target-mean encoding would work
there but overfits on low-count levels. ``smoothed_target_encode_column`` bridges that gap with a James-Stein-
style shrinkage estimate, and the screener can recommend it automatically (opt-in) when it detects that
frequency encoding would collapse.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def smoothed_target_encode_column(
    train_series: pd.Series,
    test_series: pd.Series,
    y_train: pd.Series,
    smoothing: float = 10.0,
    oof: bool = True,
    n_splits: int = 5,
    random_state: Optional[int] = None,
) -> tuple[pd.Series, pd.Series]:
    """James-Stein-style shrinkage target encoding: per-category mean pulled toward the global mean by
    ``smoothing`` pseudo-observations, so low-count levels don't overfit to a handful of target rows.

    ``enc(cat) = (count(cat) * mean_y(cat) + smoothing * global_mean) / (count(cat) + smoothing)``

    ``test_encoded`` is always computed from stats fit on the FULL ``train_series``/``y_train`` (no leakage:
    test rows never inform their own encoding). ``train_encoded`` is, by default (``oof=True``), computed via
    K-fold out-of-fold encoding: each train row's value comes from stats fit on every OTHER fold, so a row's
    own label never informs its own encoded value. With ``oof=False`` (legacy behavior), ``train_encoded`` is
    computed the same way as ``test_encoded`` but reusing ``train_series``' own stats -- every row's own label
    contributes to its own category's shrunk mean, an in-sample target-encoding leak that inflates
    ``train_encoded``'s apparent correlation with ``y_train`` versus what a model would see on genuinely
    unseen rows. Prefer ``oof=True`` (the default) whenever ``train_encoded`` feeds a downstream model fit on
    the same rows; ``oof=False`` is for pure EDA/screening where in-sample values are being eyeballed
    directly, not fed back into training.

    Test categories unseen in train fall back to the train global mean (same blind spot as plain target-mean
    encoding for genuinely disjoint category sets — this helper is for the *overlapping*, near-uniform-count
    regime, not a fix for zero train/test overlap).

    Parameters
    ----------
    oof
        Default ``True``. See above.
    n_splits, random_state
        K-fold configuration for the OOF path; ignored when ``oof=False``.

    Returns
    -------
    (train_encoded, test_encoded)
        Series aligned to ``train_series``/``test_series`` indices.
    """
    global_mean = float(y_train.mean())
    stats = y_train.groupby(train_series).agg(["mean", "count"])
    shrunk = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
    test_encoded = test_series.map(shrunk).fillna(global_mean)

    if not oof:
        train_encoded = train_series.map(shrunk).fillna(global_mean)
        return train_encoded, test_encoded

    from sklearn.model_selection import KFold

    train_encoded = pd.Series(np.nan, index=train_series.index, dtype=np.float64)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_train_pos, fold_val_pos in kf.split(np.arange(len(train_series))):
        fold_train_idx = train_series.index[fold_train_pos]
        fold_val_idx = train_series.index[fold_val_pos]
        fold_train_series = train_series.loc[fold_train_idx]
        fold_stats = y_train.loc[fold_train_idx].groupby(fold_train_series).agg(["mean", "count"])
        fold_shrunk = (fold_stats["count"] * fold_stats["mean"] + smoothing * global_mean) / (fold_stats["count"] + smoothing)
        train_encoded.loc[fold_val_idx] = train_series.loc[fold_val_idx].map(fold_shrunk).fillna(global_mean)
    return train_encoded, test_encoded


def train_test_support_screen(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_cols: Sequence[str] | None = None,
    unseen_row_threshold: float = 0.05,
    drop_overlap_threshold: float = 0.05,
    near_unique_cardinality_ratio: float = 0.9,
    target_col: str | None = None,
    enable_smoothed_target_encoding_fallback: bool = False,
    freq_collapse_cv_threshold: float = 0.15,
) -> pd.DataFrame:
    """Per-categorical-column train/test support diagnostic with an encoding recommendation.

    Parameters
    ----------
    train_df, test_df
        Frames sharing the columns to screen.
    categorical_cols
        Columns to screen. Defaults to every column present in both frames.
    unseen_row_threshold
        Recommend ``"frequency_encode"`` when the fraction of TEST rows whose category value never appears
        in train exceeds this threshold (fraction of ROWS, not fraction of distinct categories — a column
        can have many test-only rare categories that together touch very few rows, which is harmless).
    drop_overlap_threshold
        Recommend ``"drop"`` instead of ``"frequency_encode"`` when the Jaccard overlap of the train/test
        category SETS is below this AND the column looks like a near-unique-per-row identifier (see
        ``near_unique_cardinality_ratio``) — in that regime even frequency encoding degenerates to an
        uninformative near-constant value, so there's no encoding worth keeping.
    near_unique_cardinality_ratio
        A column is treated as "near-unique-per-row" (id/key-like) when ``n_unique_train / n_train_rows``
        exceeds this ratio.
    target_col
        Opt-in. Name of the target column in ``train_df`` (must not be one of ``categorical_cols``). Required
        for ``enable_smoothed_target_encoding_fallback``; ignored otherwise. Default behavior is bit-identical
        with ``target_col=None``.
    enable_smoothed_target_encoding_fallback
        Opt-in (default ``False``, no change to default output). When ``True`` and a column would be
        recommended ``"frequency_encode"``, also checks whether frequency encoding would itself collapse to a
        near-constant value (near-uniform per-category counts in train, see ``freq_collapse_cv_threshold``).
        If so, recommends ``"smoothed_target_encode"`` instead — use ``smoothed_target_encode_column`` to
        actually produce the encoding. Adds a ``freq_encoding_cv`` column to the output only when enabled.
    freq_collapse_cv_threshold
        Coefficient of variation (``std / mean``) of per-category train row counts below which frequency
        encoding is judged to collapse to a near-constant value. Only used when
        ``enable_smoothed_target_encoding_fallback=True``.

    Returns
    -------
    pd.DataFrame
        One row per screened column: ``{"column", "n_train_categories", "n_test_categories",
        "n_test_only_categories", "jaccard_overlap", "frac_test_rows_unseen", "recommendation"}``, plus
        ``"freq_encoding_cv"`` when ``enable_smoothed_target_encoding_fallback=True``.
        ``recommendation`` is one of ``"keep_raw"``, ``"frequency_encode"``, ``"drop"``, and (opt-in only)
        ``"smoothed_target_encode"``.
    """
    if categorical_cols is None:
        categorical_cols = [c for c in train_df.columns if c in test_df.columns]

    if enable_smoothed_target_encoding_fallback and (target_col is None or target_col not in train_df.columns):
        raise ValueError("enable_smoothed_target_encoding_fallback=True requires a valid target_col present in train_df")

    rows = []
    for col in categorical_cols:
        train_vals = train_df[col]
        test_vals = test_df[col]
        train_cats = set(train_vals.dropna().unique().tolist())
        test_cats = set(test_vals.dropna().unique().tolist())
        union = train_cats | test_cats
        intersect = train_cats & test_cats
        jaccard = (len(intersect) / len(union)) if union else 1.0
        test_only = test_cats - train_cats

        n_test_rows = len(test_vals)
        frac_unseen = float(test_vals.isin(test_only).sum() / n_test_rows) if n_test_rows > 0 else 0.0

        n_train_rows = len(train_vals)
        cardinality_ratio = (len(train_cats) / n_train_rows) if n_train_rows > 0 else 0.0
        near_unique = cardinality_ratio >= near_unique_cardinality_ratio

        if jaccard < drop_overlap_threshold and near_unique:
            recommendation = "drop"
        elif frac_unseen > unseen_row_threshold:
            recommendation = "frequency_encode"
        else:
            recommendation = "keep_raw"

        freq_cv: float | None = None
        if enable_smoothed_target_encoding_fallback:
            if recommendation == "frequency_encode" and train_cats:
                counts = train_vals.value_counts()
                counts_mean = float(counts.mean())
                freq_cv = float(counts.std(ddof=0) / counts_mean) if counts_mean > 0 else 0.0
                if freq_cv < freq_collapse_cv_threshold:
                    recommendation = "smoothed_target_encode"
            else:
                freq_cv = float("nan")

        row = {
            "column": col,
            "n_train_categories": len(train_cats),
            "n_test_categories": len(test_cats),
            "n_test_only_categories": len(test_only),
            "jaccard_overlap": jaccard,
            "frac_test_rows_unseen": frac_unseen,
            "recommendation": recommendation,
        }
        if enable_smoothed_target_encoding_fallback:
            row["freq_encoding_cv"] = freq_cv
        rows.append(row)

    return pd.DataFrame(rows)


__all__ = ["train_test_support_screen", "smoothed_target_encode_column"]
