"""Uniqueness-count categorical encoding for detecting synthetic/duplicate rows.

*** COMPETITION / EXPLORATORY ONLY - NOT PRODUCTION CODE ***

This module implements a Kaggle-competition-specific trick (Santander Customer
Transaction Prediction 1st place solution: per-feature "has one feat" categorical
encoding built on top of "fake test row" detection). It is generalized here as a
standalone feature-engineering primitive.

Why this is NOT production-safe:
    - The underlying technique is only meaningful for datasets with synthetically
      augmented/duplicated test sets (a Kaggle anti-cheating artifact), which has
      no production analog.
    - The target-conditional grouping ("repeats_only_with_target_1" /
      "repeats_only_with_target_0") is a mild variant of standard count/target
      encoding and can easily leak signal if misused outside a strict CV harness.

This module lives under ``mlframe.competition`` and is NEVER imported by any
production mlframe module, and NEVER re-exported from mlframe's top-level
``__init__.py``. Use it only for exploratory competition work, always inside a
proper train/validation split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNIQUE_GLOBALLY = "unique_globally"
REPEATS_ONLY_TARGET_1 = "repeats_only_with_target_1"
REPEATS_ONLY_TARGET_0 = "repeats_only_with_target_0"
REPEATS_MIXED_TARGET = "repeats_mixed_target"
REPEATS_UNKNOWN_TARGET = "repeats_unknown_target"
MISSING_VALUE = "missing_value"

VALUE_UNIQUENESS_CATEGORIES = (
    UNIQUE_GLOBALLY,
    REPEATS_ONLY_TARGET_1,
    REPEATS_ONLY_TARGET_0,
    REPEATS_MIXED_TARGET,
    REPEATS_UNKNOWN_TARGET,
    MISSING_VALUE,
)


def _build_train_value_to_flag(values: pd.Series, y_train: np.ndarray) -> dict:
    """Causal, train-only value->flag mapping: uses value counts AND target co-occurrence within train.

    Never uses any test-set information (no leakage into the train-side flags). Vectorized via
    groupby sum/size (binary target) instead of a per-group Python callback - the latter measured
    ~17s at n=200k in cProfile (pandas per-group ``unique()`` dominates); this path is groupby-native.
    """
    df = pd.DataFrame({"value": values.to_numpy(), "target": np.asarray(y_train)})

    grouped = df.groupby("value")["target"].agg(["size", "sum"])
    counts = grouped["size"].to_numpy()
    target_sum = grouped["sum"].to_numpy()

    flags = np.full(len(grouped), REPEATS_MIXED_TARGET, dtype=object)
    flags[counts <= 1] = UNIQUE_GLOBALLY
    repeats = counts > 1
    flags[repeats & (target_sum == counts)] = REPEATS_ONLY_TARGET_1
    flags[repeats & (target_sum == 0)] = REPEATS_ONLY_TARGET_0

    return dict(zip(grouped.index.to_numpy(), flags))


def _encode_train_column(values: pd.Series, value_to_flag: dict) -> pd.Series:
    """Map each train value to its precomputed uniqueness/target-co-occurrence flag.

    ``value_to_flag`` (built via a NaN-dropping ``groupby``) has no entry for a NaN train value, so
    ``.map`` would otherwise leave a raw, undocumented ``NaN`` in the output instead of one of the
    documented categories -- routed to the explicit ``MISSING_VALUE`` category instead.
    """
    flags = values.map(value_to_flag).to_numpy(dtype=object)
    missing = values.isna().to_numpy()
    if missing.any():
        flags[missing] = MISSING_VALUE
    return pd.Series(flags, index=values.index, dtype="object")


def _encode_test_column(values: pd.Series, real_mask_col: np.ndarray | None, train_value_to_flag: dict) -> pd.Series:
    """Encode test rows without ever using test-side labels.

    For a value already seen in train, reuses the flag learned purely from train (that flag
    was derived only from train's own ``y_train`` - the fact the same value recurs in test never
    feeds back into it, so this is not leakage).

    For a value never seen in train, falls back to a value-count-based flag computed from test
    rows only (optionally restricted to ``real_mask_col`` to exclude synthetic/duplicate rows,
    replicating the source "fake test row" filtering) - no target information is available or
    used for these novel values.
    """
    raw = values.to_numpy()
    if real_mask_col is not None:
        real_values = raw[real_mask_col]
    else:
        real_values = raw

    novel_counts = pd.Series(real_values).value_counts()

    values_series = pd.Series(raw)
    is_missing = values_series.isna().to_numpy()
    # A NaN test value must never fall through to the novel-value count path below: `.isin`/`.map`
    # both treat NaN as never matching (dict has no NaN key, value_counts() drops NaN by default), so
    # counts_for_novel would silently resolve to 0 -> UNIQUE_GLOBALLY, an undocumented asymmetry with
    # the train-side MISSING_VALUE flag (see _encode_train_column). Route it there explicitly instead.
    seen_in_train = values_series.isin(train_value_to_flag.keys()).to_numpy() & ~is_missing

    flags = np.empty(len(raw), dtype=object)
    if seen_in_train.any():
        flags[seen_in_train] = values_series[seen_in_train].map(train_value_to_flag).to_numpy()
    if is_missing.any():
        flags[is_missing] = MISSING_VALUE

    novel_mask = ~seen_in_train & ~is_missing
    if novel_mask.any():
        counts_for_novel = values_series[novel_mask].map(novel_counts).fillna(0).to_numpy()
        novel_flags = np.where(counts_for_novel <= 1, UNIQUE_GLOBALLY, REPEATS_UNKNOWN_TARGET)
        flags[novel_mask] = novel_flags
    return pd.Series(flags, index=values.index, dtype="object")


def value_uniqueness_encoder(
    train: pd.DataFrame,
    test: pd.DataFrame,
    real_test_mask: np.ndarray | None,
    y_train: np.ndarray,
    columns: list[str],
) -> pd.DataFrame:
    """Produce per-row categorical uniqueness/target-co-occurrence flags for given columns.

    *** COMPETITION / EXPLORATORY ONLY - NOT PRODUCTION CODE. See module docstring. ***

    For each column in ``columns``, computes a categorical flag per row indicating how
    that row's value behaves w.r.t. uniqueness and (for train rows only) target
    co-occurrence:

    - ``"unique_globally"``: the value occurs exactly once (within its own split's
      count basis - train counts within train, test counts within real test rows).
    - ``"repeats_only_with_target_1"``: value repeats in train and is ALWAYS associated
      with ``y_train == 1`` (train rows only).
    - ``"repeats_only_with_target_0"``: value repeats in train and is ALWAYS associated
      with ``y_train == 0`` (train rows only).
    - ``"repeats_mixed_target"``: value repeats in train with both target classes present
      (train rows only).
    - ``"repeats_unknown_target"``: value repeats (count > 1) - assigned to test rows only,
      since test labels are never available/used (no leakage).
    - ``"missing_value"``: the raw value is NaN/None, on either the train or test side -- never
      routed through the count/target logic above (a NaN is not "unique" or "repeating" in any
      meaningful sense).

    Causality / no-leakage guarantees:
        - Train-row flags are computed using ONLY ``train`` values and ``y_train`` -
          test data never influences a train row's flag.
        - Test-row flags are computed using ONLY value counts within ``test`` (optionally
          restricted to ``real_test_mask`` to exclude synthetic/duplicate rows, replicating
          the source "fake test row" filtering) - target information is never used for
          test rows, since ``y_train`` has no test-side labels to leak.

    Parameters
    ----------
    train : pd.DataFrame
        Training frame.
    test : pd.DataFrame
        Test frame (may include synthetic/duplicate rows).
    real_test_mask : np.ndarray | None
        Boolean mask, same length as ``test``, marking rows considered "real" (as opposed
        to synthetically generated/duplicated). If ``None``, all test rows are treated as
        real when computing test-side value counts.
    y_train : np.ndarray
        Binary target aligned with ``train`` rows. Only 0/1 values are supported for the
        target-conditional flags.
    columns : list[str]
        Column names (present in both ``train`` and ``test``) to encode.

    Returns
    -------
    pd.DataFrame
        A frame with index = concatenation of ``train.index`` then ``test.index``, and one
        column per encoded feature named ``f"{col}__value_uniqueness"``, containing one of
        the categorical flags above (as pandas ``category`` dtype).
    """
    if real_test_mask is not None and len(real_test_mask) != len(test):
        raise ValueError("real_test_mask must have the same length as test")
    if len(y_train) != len(train):
        raise ValueError("y_train must have the same length as train")

    y_arr = np.asarray(y_train)
    result = pd.DataFrame(index=pd.Index(list(train.index) + list(test.index)))

    for col in columns:
        if col not in train.columns or col not in test.columns:
            raise KeyError(f"column '{col}' must be present in both train and test")

        value_to_flag = _build_train_value_to_flag(train[col], y_arr)
        train_flags = _encode_train_column(train[col], value_to_flag)
        test_flags = _encode_test_column(test[col], real_test_mask, value_to_flag)

        combined = pd.concat([train_flags, test_flags], ignore_index=False)
        combined.index = result.index
        result[f"{col}__value_uniqueness"] = pd.Categorical(combined, categories=list(VALUE_UNIQUENESS_CATEGORIES))

    return result
