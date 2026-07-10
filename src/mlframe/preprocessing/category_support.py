"""Screen categorical columns for train/test category-support mismatch.

A raw categorical encoding (ordinal/one-hot/target-mean) is only as good as its category overlap between
train and test: a category present only in test has no learned representation, so the model silently falls
back to a constant/unknown bucket and loses whatever signal that category carried. Frequency (count)
encoding degrades gracefully instead — an unseen category still gets a real frequency value computed from
its own split, so the underlying "how common is this category" signal transfers even when the specific
category identities do not (the pattern IEEE-CIS's 2nd place team diagnosed by eyeballing raw-vs-frequency
plots). ``train_test_support_screen`` automates that diagnosis and recommends a remediation per column.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd


def train_test_support_screen(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_cols: Sequence[str] | None = None,
    unseen_row_threshold: float = 0.05,
    drop_overlap_threshold: float = 0.05,
    near_unique_cardinality_ratio: float = 0.9,
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

    Returns
    -------
    pd.DataFrame
        One row per screened column: ``{"column", "n_train_categories", "n_test_categories",
        "n_test_only_categories", "jaccard_overlap", "frac_test_rows_unseen", "recommendation"}``.
        ``recommendation`` is one of ``"keep_raw"``, ``"frequency_encode"``, ``"drop"``.
    """
    if categorical_cols is None:
        categorical_cols = [c for c in train_df.columns if c in test_df.columns]

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

        rows.append(
            {
                "column": col,
                "n_train_categories": len(train_cats),
                "n_test_categories": len(test_cats),
                "n_test_only_categories": len(test_only),
                "jaccard_overlap": jaccard,
                "frac_test_rows_unseen": frac_unseen,
                "recommendation": recommendation,
            }
        )

    return pd.DataFrame(rows)


__all__ = ["train_test_support_screen"]
