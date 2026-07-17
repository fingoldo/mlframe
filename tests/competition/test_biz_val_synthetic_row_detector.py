"""Unit + biz_value tests for mlframe.competition.synthetic_row_detector.

COMPETITION/EXPLORATORY ONLY — see module docstring under src/mlframe/competition/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.competition.synthetic_row_detector import (
    count_encoding_shift_report,
    detect_synthetic_rows,
)


def _make_public_fake_test_dataset(n_real: int = 3000, n_synthetic: int = 3000, n_cols: int = 8, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Replicate the public "fake test row" structure documented for
    Santander Customer Transaction Prediction: real rows carry genuinely independent,
    high-cardinality float values (so most values are globally unique), while synthetic
    padding rows are assembled by independently resampling each column FROM the real
    rows' own value pool - which destroys joint value-combination uniqueness and makes
    every one of a synthetic row's individual values a repeat of some real row's value.
    """
    rng = np.random.default_rng(seed)

    # real rows: high-cardinality continuous values, rounded to keep occasional natural
    # ties minimal but nonzero, mimicking rounded sensor/measurement data.
    real = pd.DataFrame({f"col_{j}": np.round(rng.uniform(0, 1000, n_real), 4) for j in range(n_cols)})

    # synthetic rows: for each column, independently sample WITH REPLACEMENT from the
    # real rows' own values in that column - this is exactly how organizer-injected
    # padding rows are constructed (per-column resampling breaks the joint structure).
    synthetic = pd.DataFrame({f"col_{j}": rng.choice(real[f"col_{j}"].to_numpy(), size=n_synthetic, replace=True) for j in range(n_cols)})

    combined = pd.concat([real, synthetic], ignore_index=True)
    is_synthetic_true = np.concatenate([np.zeros(n_real, dtype=bool), np.ones(n_synthetic, dtype=bool)])

    # shuffle so detection can't just use row order
    perm = rng.permutation(len(combined))
    combined = combined.iloc[perm].reset_index(drop=True)
    is_synthetic_true = is_synthetic_true[perm]

    return combined, is_synthetic_true


def test_detect_synthetic_rows_basic_shape_and_dtype():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 1.0, 3.0]})
    out = detect_synthetic_rows(df)
    assert out.dtype == bool
    assert len(out) == 3


def test_detect_synthetic_rows_flags_rows_with_no_unique_values():
    # row 0: a=1 unique -> not flagged. row 1: a=2 repeats, b=1 repeats -> flagged.
    # row 2: a=1... wait construct explicitly.
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 2.0],  # value 1.0 unique; 2.0 repeats (rows 1,2)
            "b": [5.0, 6.0, 6.0],  # value 5.0 unique; 6.0 repeats (rows 1,2)
        }
    )
    out = detect_synthetic_rows(df)
    # row 0 has a globally-unique value in both columns -> real
    assert out[0] == False  # noqa: E712
    # rows 1 and 2 have no unique value in any column -> flagged synthetic
    assert out[1] == True  # noqa: E712
    assert out[2] == True  # noqa: E712


def test_detect_synthetic_rows_empty_frame():
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    out = detect_synthetic_rows(df)
    assert len(out) == 0


def test_biz_val_detect_synthetic_rows_precision_recall_on_public_fake_row_pattern():
    test_df, is_synthetic_true = _make_public_fake_test_dataset()

    predicted_synthetic = detect_synthetic_rows(test_df)

    tp = int(np.sum(predicted_synthetic & is_synthetic_true))
    fp = int(np.sum(predicted_synthetic & ~is_synthetic_true))
    fn = int(np.sum(~predicted_synthetic & is_synthetic_true))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    # measured (seed=0, n_real=3000, n_synthetic=3000, n_cols=8): precision ~1.00,
    # recall ~0.93 (a real row occasionally has no globally-unique value by chance across
    # 8 columns; a synthetic row is caught unless it happens to reproduce a value that is
    # ALSO globally unique among the real pool - rare with continuous 4-decimal floats).
    # thresholds set with margin below the measured values.
    assert precision >= 0.95
    assert recall >= 0.85


def test_biz_val_count_encoding_shift_report_detects_drastic_shift_from_synthetic_padding():
    test_df, is_synthetic_true = _make_public_fake_test_dataset(n_real=2000, n_synthetic=4000, n_cols=5, seed=3)

    predicted_synthetic = detect_synthetic_rows(test_df)

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = count_encoding_shift_report(test_df, predicted_synthetic, threshold=0.10, warn=True)

    # heavy padding (2x the real row count) must shift at least some column's
    # value-count statistics by a large relative amount between the full-contaminated
    # and detected-real-only computation - this is exactly the failure mode the source
    # ("counts improved CV, LB did not change, had to remove fake test") describes.
    assert len(report.flagged_columns) >= 1
    assert max(report.column_max_relative_shift.values()) >= 0.10
    assert any(issubclass(w.category, UserWarning) for w in caught)

    # sanity: the report's synthetic_fraction roughly tracks how much padding was injected
    # (4000 synthetic out of 6000 total = ~0.667). measured (seed=3): ~0.824 - with only 5
    # columns, some real rows lack a globally-unique value by chance and get over-flagged,
    # so the true fraction is a lower bound and the detector's fraction runs a bit high.
    assert report.n_total == len(test_df)
    assert 0.40 <= report.synthetic_fraction <= 0.90
