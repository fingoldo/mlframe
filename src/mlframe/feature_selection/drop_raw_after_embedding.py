"""``drop_raw_after_embedding``: drop a raw high-cardinality categorical once its derived encodings exist.

Source: 1st_talkingdata-adtracking-fraud-detection.md - "we removed all raw categorical features except app
since we supposed embedding features cover information... jumped public LB from 0.9821 to 0.9828". Once a raw
high-cardinality categorical has been converted into derived features (target/frequency/count encodings,
entity embeddings, SVD/co-occurrence features), keeping the raw column around mostly adds overfitting
surface (a tree model can memorize per-category splits the encoding already summarized) rather than genuine
signal - this is a small, explicit drop step, not a generic redundancy pruner: it only ever removes the RAW
column, never a derived one, and only once the derived columns it depends on are actually present.

``verify_against`` (opt-in): "embedding columns exist" does not imply "embedding columns retain the raw
column's signal" - the embedding may have been trained for a different task, may be too low-dimensional, or
may be stale versus a since-updated raw column. When supplied, each candidate raw column is only actually
dropped once a cheap signal-strength check confirms its derived columns retain at least a threshold fraction
of the raw column's own univariate signal against a target; otherwise the raw column is kept and reported.
Reuses ``preprocessing.align_feature_direction.batch_univariate_auc`` (the same rank-based AUC machinery as
``drop_near_noise_univariate_auc``) rather than fitting a fresh model per column.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from mlframe.preprocessing.align_feature_direction import batch_univariate_auc


def _univariate_signal(values: np.ndarray, y_arr: np.ndarray) -> float:
    """Signal strength in [0, 1] from univariate AUC against ``y_arr`` (0 = chance, 1 = perfect separation)."""
    auc = float(batch_univariate_auc(values.reshape(-1, 1).astype(np.float64), y_arr)[0])
    return 2.0 * abs(auc - 0.5)


def _raw_column_signal(df: pd.DataFrame, raw_col: str, y_arr: np.ndarray, n_folds: int = 5, random_state: int = 0) -> float:
    """Signal strength of a raw column: numeric columns as-is, categoricals via an OUT-OF-FOLD target-mean encoding.

    Out-of-fold, not in-sample. The in-sample version this replaces gave each row the mean of a group that
    INCLUDES that row, so for a high-cardinality column -- 50k distinct device ids over 200k rows, i.e. ~4 rows
    per group -- the encoding largely reproduces y itself and `raw_signal` approaches its ceiling. The derived
    columns are scored as-is, with no such advantage, so the comparison was not like-for-like and the
    `verify_against` gate almost never passed for EXACTLY the high-cardinality columns this module exists to
    drop. That made a safety check inert by construction: it could not clear the raw column, so it could not
    authorise the drop it was asked about.

    The fold means are computed from the other folds only, which is the same discipline the target-encoding FE
    family uses, and it costs one extra pass over the column.
    """
    col = df[raw_col]
    if pd.api.types.is_numeric_dtype(col):
        values = col.to_numpy(dtype=np.float64)
    else:
        codes = pd.factorize(col, use_na_sentinel=False)[0]
        n = codes.shape[0]
        values = np.full(n, float(np.mean(y_arr)), dtype=np.float64)
        rng = np.random.default_rng(random_state)
        folds = rng.permutation(n) % max(1, int(n_folds))
        for f in range(max(1, int(n_folds))):
            oof_mask = folds == f
            fit_mask = ~oof_mask
            if not fit_mask.any() or not oof_mask.any():
                continue
            fit = pd.DataFrame({"c": codes[fit_mask], "y": np.asarray(y_arr, dtype=np.float64)[fit_mask]})
            means = fit.groupby("c")["y"].mean()
            prior = float(fit["y"].mean())
            values[oof_mask] = pd.Series(codes[oof_mask]).map(means).fillna(prior).to_numpy(dtype=np.float64)
    return _univariate_signal(values, y_arr)


def drop_raw_after_embedding(
    df: pd.DataFrame,
    raw_to_derived: Dict[str, Sequence[str]],
    min_derived_present: int = 1,
    verify_against: Optional[Tuple[np.ndarray, float]] = None,
    safety_report: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Drop each raw column in ``raw_to_derived`` once enough of its derived columns are present in ``df``.

    Parameters
    ----------
    df
        Frame containing both raw categorical columns and their derived features.
    raw_to_derived
        Mapping from a raw column name to the derived column names built from it (e.g. an entity-embedding
        or target-encoding step's output columns).
    min_derived_present
        Minimum number of a raw column's derived columns that must already be present in ``df`` before that
        raw column is dropped - guards against dropping a raw column whose encoding step never ran (e.g. was
        skipped due to low cardinality, or failed upstream), which would silently destroy the only signal
        source for that column.
    verify_against
        Opt-in safety check: ``(y, min_retained_fraction)``. ``y`` is a BINARY (0/1) target array, same row
        order as ``df`` - same contract as ``drop_near_noise_univariate_auc``; binarize a continuous target
        (e.g. a median split or a business-relevant threshold) before calling if needed. For each candidate
        raw column, a raw-column signal strength (univariate AUC-derived, categoricals via
        an in-sample target-mean encoding) is compared against the best signal strength among its present
        derived columns. The raw column is only dropped if
        ``derived_signal >= min_retained_fraction * raw_signal`` (a raw column with ~zero signal of its own is
        always treated as safe to drop, since the embedding can't lose signal that wasn't there). Columns
        failing the check are kept and, if ``safety_report`` is passed, recorded there. Default ``None``
        reproduces the prior unconditional-trust behavior exactly.
    safety_report
        Optional dict to populate in-place with ``{raw_col: reason}`` for every raw column kept for safety
        (i.e. that would have been dropped under ``min_derived_present`` alone, but failed the
        ``verify_against`` check). Ignored when ``verify_against`` is ``None``.

    Returns
    -------
    pd.DataFrame
        ``df`` (shallow copy) with each qualifying raw column removed. Derived columns are always kept as-is.
    """
    candidates = []
    for raw_col, derived_cols in raw_to_derived.items():
        if raw_col not in df.columns:
            continue
        present_derived = [c for c in derived_cols if c in df.columns]
        if len(present_derived) >= min_derived_present:
            candidates.append((raw_col, present_derived))

    if verify_against is None:
        return df.drop(columns=[raw_col for raw_col, _ in candidates])

    y_arr, min_retained_fraction = verify_against
    y_arr = np.asarray(y_arr)

    to_drop = []
    for raw_col, present_derived in candidates:
        raw_signal = _raw_column_signal(df, raw_col, y_arr)
        if raw_signal <= 0.0:
            to_drop.append(raw_col)
            continue
        derived_signal = max(_univariate_signal(df[c].to_numpy(dtype=np.float64), y_arr) for c in present_derived)
        retained_fraction = derived_signal / raw_signal
        if retained_fraction >= min_retained_fraction:
            to_drop.append(raw_col)
        elif safety_report is not None:
            safety_report[raw_col] = (
                f"embedding retains only {retained_fraction:.1%} of raw signal "
                f"(raw={raw_signal:.3f}, derived={derived_signal:.3f}), below threshold {min_retained_fraction:.1%}"
            )

    return df.drop(columns=to_drop)


__all__ = ["drop_raw_after_embedding"]
