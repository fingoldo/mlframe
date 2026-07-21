"""``augment_to_match_test_distribution``: pluggable train-to-test degradation augmentation.

Source: PLAsTiCC Astronomical Classification 1st place -- "took every lightcurve in the training set and
degraded it up to 40 times to get something that looks like the less well-sampled lightcurves in the test
set" (brightness/redshift/gaps/photo-z-error simulation). Generalizes to any production setting where
training data is systematically higher-quality/better-sampled than what will actually be served (more
complete sensor coverage, less noise, fewer missing values): rather than only reweighting/upsampling,
programmatically degrade COPIES of train samples toward test's own empirically measured quality profile, so
the model sees the degraded regime during training instead of only at serve time.

Pluggable per-modality: two degradation functions ship built-in (matching the two most common quality-gap
axes) -- ``match_missingness_rate`` (inject additional NaNs, per column, up to test's own missing rate) and
``match_noise_level`` (inject calibrated Gaussian noise, per column, so the augmented std matches test's own
std when test is noisier). ``augment_to_match_test_distribution`` accepts ANY additional caller-supplied
degradation callable with the same ``(X_train, X_test, rng) -> X_degraded`` signature for other modalities
(row-density thinning, measurement dropout patterns, ...).
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np
import pandas as pd

DegradationFn = Callable[[pd.DataFrame, pd.DataFrame, np.random.Generator], pd.DataFrame]


def match_missingness_rate(X_train: pd.DataFrame, X_test: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject additional NaNs into a COPY of ``X_train``, per column, up to ``X_test``'s own missing rate.

    Only ADDS missingness (never removes existing NaNs); a column already at or above test's missing rate
    is left unchanged. Operates on the underlying numpy array (one column pass, no per-row/per-column
    pandas ``.loc`` indexing) -- a per-column ``.loc[idx, col] =`` loop was measured as the dominant
    cProfile hotspot (pandas indexer resolution overhead, not real work).
    """
    # Numeric columns only (matches match_noise_level's own dispatch): a non-numeric
    # (categorical/object/string) column can't round-trip through a float64 numpy array, so
    # X_train.to_numpy(dtype=np.float64) on the WHOLE frame crashed with "could not convert
    # string to float" on any realistic mixed-dtype frame. Non-numeric columns pass through unchanged.
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    out = X_train.copy(deep=False)
    values = out[numeric_cols].to_numpy(dtype=np.float64, copy=True)
    test_values = X_test[numeric_cols].to_numpy(dtype=np.float64, copy=False)
    n_rows = values.shape[0]
    for j in range(values.shape[1]):
        col_vals = values[:, j]
        test_missing_rate = float(np.isnan(test_values[:, j]).mean())
        train_missing_rate = float(np.isnan(col_vals).mean())
        additional_rate = test_missing_rate - train_missing_rate
        if additional_rate <= 0:
            continue
        non_null_idx = np.flatnonzero(~np.isnan(col_vals))
        n_to_null = round(additional_rate * n_rows)
        n_to_null = min(n_to_null, len(non_null_idx))
        if n_to_null > 0:
            drop_idx = rng.choice(non_null_idx, size=n_to_null, replace=False)
            col_vals[drop_idx] = np.nan
    out[numeric_cols] = values
    return out


def match_noise_level(X_train: pd.DataFrame, X_test: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject calibrated Gaussian noise into a COPY of ``X_train``, per NUMERIC column, so the augmented
    column's std matches ``X_test``'s own std when test is noisier (``additional_std = sqrt(max(0,
    test_std**2 - train_std**2))`` -- the noise addition that brings variance up to test's level assuming
    independent additive noise). Operates on the underlying numpy array, same rationale as
    ``match_missingness_rate``.
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    out = X_train.copy(deep=False)
    values = out[numeric_cols].to_numpy(dtype=np.float64, copy=True)
    test_values = X_test[numeric_cols].to_numpy(dtype=np.float64, copy=False)
    for j in range(values.shape[1]):
        col_vals = values[:, j]
        train_std = float(np.nanstd(col_vals))
        test_std = float(np.nanstd(test_values[:, j]))
        if not np.isfinite(train_std) or not np.isfinite(test_std) or test_std <= train_std:
            continue
        additional_std = float(np.sqrt(max(0.0, test_std**2 - train_std**2)))
        if additional_std <= 0:
            continue
        mask = ~np.isnan(col_vals)
        col_vals[mask] += rng.normal(scale=additional_std, size=int(mask.sum()))
    out[numeric_cols] = values
    return out


def augment_to_match_test_distribution(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    degradation_fns: Sequence[DegradationFn] = (match_missingness_rate, match_noise_level),
    n_augments: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Apply ALL ``degradation_fns`` (composed in order) to ``n_augments`` independently-sampled degraded
    copies of ``X_train``, concatenate with the original clean ``X_train`` (broadcasting ``y_train``).

    Parameters
    ----------
    X_train, y_train
        Original (higher-quality) training data.
    X_test
        Test frame the degraded copies are calibrated toward -- FEATURES ONLY, never touches any label.
    degradation_fns
        Callables applied in sequence to each augmented copy; each sees the RUNNING degraded frame (so a
        later function's calibration reflects an earlier function's already-injected missingness/noise).
    n_augments
        Number of independently-degraded copies of ``X_train`` to generate (matching the source's "degraded
        it up to 40 times" -- default kept modest; raise for a closer-to-source augmentation ratio).
    random_state
        Seed; each augmented copy gets an independent sub-generator so copies aren't identical.

    Returns
    -------
    tuple
        ``(X_augmented, y_augmented)`` -- the original clean rows followed by ``n_augments`` degraded
        copies, ``len(X_train) * (1 + n_augments)`` rows total.
    """
    rng = np.random.default_rng(random_state)
    y_arr = np.asarray(y_train)

    frames: List[pd.DataFrame] = [X_train.reset_index(drop=True)]
    labels: List[np.ndarray] = [y_arr]
    for _ in range(n_augments):
        degraded = X_train.reset_index(drop=True)
        copy_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        for fn in degradation_fns:
            degraded = fn(degraded, X_test, copy_rng)
        frames.append(degraded)
        labels.append(y_arr)

    X_augmented = pd.concat(frames, axis=0, ignore_index=True)
    y_augmented = np.concatenate(labels)
    return X_augmented, y_augmented


__all__ = ["augment_to_match_test_distribution", "match_missingness_rate", "match_noise_level"]
