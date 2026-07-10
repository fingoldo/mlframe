"""Control-difference synthetic data augmentation: inject realistic batch/plate noise into treated samples.

Real experimental panel data (biological assays, A/B tests with repeated control measurements) has structured
"batch noise" -- systematic run-to-run variation shared by all samples processed together. A 3rd-place
Mechanisms-of-Action team's single decisive trick: synthesize new augmented rows as
``treated_sample + control_sample_1 - control_sample_2`` -- the difference between two randomly-drawn control
samples IS a realistic draw from that batch-noise distribution (both are known-null, so their difference is
pure noise), and adding it to a real treated sample produces a new, still-valid training row with the SAME
treatment label but different realistic noise, effectively multiplying the usable training set.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def control_difference_augment(
    treated_df: pd.DataFrame,
    control_df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    n_augmented_per_treated: int = 1,
    random_state: int = 0,
) -> pd.DataFrame:
    """Synthesize ``treated + control_1 - control_2`` augmented rows for every treated sample.

    Parameters
    ----------
    treated_df
        Real treated-group rows to augment; non-``feature_cols`` columns (e.g. the label) are copied through
        unchanged onto every augmented row derived from that treated sample.
    control_df
        Control-group (known-null) rows to draw the noise difference from; must share ``feature_cols`` with
        ``treated_df``.
    feature_cols
        Numeric columns to augment; defaults to every numeric column shared by both frames.
    n_augmented_per_treated
        How many independent augmented rows to synthesize per real treated row (each draws its own random
        pair of control samples).
    random_state
        Seed for the control-sample draws.

    Returns
    -------
    pd.DataFrame
        ``n_augmented_per_treated * len(treated_df)`` synthesized rows, same columns as ``treated_df``.
    """
    if feature_cols is None:
        feature_cols = [
            c for c in treated_df.select_dtypes(include=[np.number]).columns if c in control_df.columns
        ]
    feature_cols = list(feature_cols)
    if len(control_df) < 2:
        raise ValueError("control_difference_augment: control_df needs at least 2 rows to draw a difference pair")

    rng = np.random.default_rng(random_state)
    n_treated = len(treated_df)
    n_control = len(control_df)
    control_values = control_df[feature_cols].to_numpy(dtype=np.float64)
    treated_values = treated_df[feature_cols].to_numpy(dtype=np.float64)
    other_cols = [c for c in treated_df.columns if c not in feature_cols]
    other_block = treated_df[other_cols].reset_index(drop=True) if other_cols else None

    # build the feature block as a fresh DataFrame + one axis=1 concat with the (label/other) columns per
    # augmented copy, instead of `augmented[feature_cols] = augmented_values` on a full-frame .copy() -- the
    # latter triggers a per-COLUMN __setitem__ pandas internally (profiled: 1000 calls at n_features=100,
    # n_aug=10), while this is one bulk block assignment per copy. ~2.4x faster at n=10k/100 features.
    augmented_frames = []
    for _ in range(n_augmented_per_treated):
        idx_1 = rng.integers(0, n_control, n_treated)
        idx_2 = rng.integers(0, n_control, n_treated)
        noise_diff = control_values[idx_1] - control_values[idx_2]
        augmented_values = treated_values + noise_diff

        feature_block = pd.DataFrame(augmented_values, columns=feature_cols)
        augmented = pd.concat([feature_block, other_block], axis=1)[list(treated_df.columns)] if other_block is not None else feature_block
        augmented_frames.append(augmented)

    return pd.concat(augmented_frames, axis=0, ignore_index=True)


__all__ = ["control_difference_augment"]
