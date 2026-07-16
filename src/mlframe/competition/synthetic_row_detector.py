"""Detection of organizer-injected "fake"/padding rows in a Kaggle public-leaderboard test set.

*** COMPETITION / EXPLORATORY ONLY - NOT PRODUCTION CODE ***

This module implements a Kaggle-competition-specific trick, popularized by the public
"list of fake samples and public/private LB split" kernel for the Santander Customer
Transaction Prediction competition, and independently rediscovered by the 3rd place
solution write-up: some competitions pad the released test set with synthetic rows
(assembled by independently resampling per-column values from the real rows) purely to
prevent competitors from reverse-engineering the private leaderboard split via exact-row
frequency counting. Those synthetic rows must be excluded before computing count/frequency
encodings across train+test, or count-based features get diluted/biased by combinations
that never occurred organically.

Why this is NOT production-safe:
    - Deliberately organizer-injected "fake" padding rows are a Kaggle anti-cheating
      artifact of released *test* sets. Production serving data is never padded this way,
      so there is nothing to "detect" in a real system.
    - The heuristic (rows with no globally-unique feature value across all columns) is
      tuned to how these specific fake rows are synthesized (per-column resampling from
      the empirical marginal, breaking joint value-combination structure) - it is not a
      general anomaly/outlier detector and would misfire on legitimately repetitive
      production data (e.g. low-cardinality categorical columns).

This module lives under ``mlframe.competition`` and is NEVER imported by any production
mlframe module, and NEVER re-exported from mlframe's top-level ``__init__.py``. Use it
only for exploratory competition work.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def detect_synthetic_rows(test_df: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
    """Flag test rows most likely to be organizer-injected synthetic/padding rows.

    *** COMPETITION / EXPLORATORY ONLY - see module docstring. ***

    Heuristic (the public "fake test row" detection pattern): synthetic padding rows are
    typically assembled by independently resampling each column's values from the
    empirical marginal distribution of the real rows, which destroys the *joint*
    value-combination structure a genuinely collected row has. A real, hand-collected row
    tends to have at least one feature value that is globally unique (count == 1) within
    the test set - a coincidence of natural, high-cardinality/continuous measurement.
    A synthetic row, built by recombining already-observed per-column values, tends to have
    every one of its feature values repeated elsewhere (count > 1 for all columns), because
    each individual value was drawn from a pool of values that already occurs more than once
    once the padding rows are added.

    A row is flagged synthetic when it has ZERO globally-unique-within-column values across
    all considered columns.

    Parameters
    ----------
    test_df : pd.DataFrame
        The (potentially contaminated) test set.
    columns : list[str] | None
        Columns to use for the uniqueness-of-combinations heuristic. Defaults to all
        columns of ``test_df``.

    Returns
    -------
    np.ndarray
        Boolean array, same length as ``test_df``, ``True`` where the row is flagged as a
        likely synthetic/padding row (candidate for exclusion before count/frequency
        encoding).
    """
    if len(test_df) == 0:
        return np.zeros(0, dtype=bool)

    cols = list(test_df.columns) if columns is None else columns
    if len(cols) == 0:
        raise ValueError("columns must be non-empty (or test_df must have at least one column)")

    n = len(test_df)
    has_unique_value = np.zeros(n, dtype=bool)

    for col in cols:
        series = test_df[col]
        counts = series.map(series.value_counts())
        has_unique_value |= counts.to_numpy() == 1

    return ~has_unique_value


@dataclass
class CountEncodingShiftReport:
    """Per-column diagnostic of count/frequency-encoding statistics shift.

    *** COMPETITION / EXPLORATORY ONLY - see module docstring of synthetic_row_detector. ***

    Attributes
    ----------
    column_max_relative_shift : dict[str, float]
        Per-column max relative change (over shared values) between value counts computed
        on the full (contaminated) test set vs the detected-real-only subset.
    flagged_columns : list[str]
        Columns whose ``column_max_relative_shift`` exceeds ``threshold``.
    n_synthetic : int
        Number of rows flagged as synthetic by ``detect_synthetic_rows``.
    n_total : int
        Total number of rows in the input test set.
    """

    column_max_relative_shift: dict[str, float] = field(default_factory=dict)
    flagged_columns: list[str] = field(default_factory=list)
    n_synthetic: int = 0
    n_total: int = 0

    @property
    def synthetic_fraction(self) -> float:
        """Return the fraction of rows flagged as synthetic, or 0.0 if there are no rows."""
        return self.n_synthetic / self.n_total if self.n_total else 0.0


def count_encoding_shift_report(
    test_df: pd.DataFrame,
    synthetic_mask: np.ndarray,
    columns: list[str] | None = None,
    threshold: float = 0.10,
    warn: bool = True,
) -> CountEncodingShiftReport:
    """Diagnose whether count/frequency-encoding statistics shift drastically once
    detected-synthetic rows are dropped.

    *** COMPETITION / EXPLORATORY ONLY - see module docstring of synthetic_row_detector. ***

    For each column, computes the value-count table on the full ``test_df`` and on the
    real-only subset (``~synthetic_mask``), then reports the max relative change in count
    for any value present in both (a large shift means synthetic rows are materially
    distorting count-based features, and should be dropped before computing them).

    Parameters
    ----------
    test_df : pd.DataFrame
        The (potentially contaminated) test set.
    synthetic_mask : np.ndarray
        Boolean mask, same length as ``test_df``, as returned by ``detect_synthetic_rows``
        (``True`` = row is a detected synthetic/padding row).
    columns : list[str] | None
        Columns to check. Defaults to all columns of ``test_df``.
    threshold : float
        Relative-shift threshold (fraction) above which a column is flagged and (if
        ``warn``) a warning is raised. Default 0.10 (10%).
    warn : bool
        If True, raise a ``UserWarning`` listing flagged columns.

    Returns
    -------
    CountEncodingShiftReport
    """
    if len(synthetic_mask) != len(test_df):
        raise ValueError("synthetic_mask must have the same length as test_df")

    cols = list(test_df.columns) if columns is None else columns
    real_df = test_df.loc[~np.asarray(synthetic_mask, dtype=bool)]

    report = CountEncodingShiftReport(n_synthetic=int(np.sum(synthetic_mask)), n_total=len(test_df))

    for col in cols:
        full_counts = test_df[col].value_counts()
        real_counts = real_df[col].value_counts()

        shared_values = full_counts.index.intersection(real_counts.index)
        if len(shared_values) == 0:
            continue

        full_shared = full_counts.loc[shared_values].to_numpy(dtype=float)
        real_shared = real_counts.loc[shared_values].to_numpy(dtype=float)
        relative_shift = np.abs(full_shared - real_shared) / full_shared
        max_shift = float(relative_shift.max())

        report.column_max_relative_shift[col] = max_shift
        if max_shift > threshold:
            report.flagged_columns.append(col)

    if warn and report.flagged_columns:
        warnings.warn(
            f"count/frequency-encoding statistics shift by more than {threshold:.0%} on "
            f"{len(report.flagged_columns)} column(s) when computed on the full test set vs "
            f"the detected-real-only subset ({report.n_synthetic}/{report.n_total} rows flagged "
            f"synthetic): {report.flagged_columns}. Consider dropping detected synthetic rows "
            "before computing count-based features.",
            UserWarning,
            stacklevel=2,
        )

    return report
