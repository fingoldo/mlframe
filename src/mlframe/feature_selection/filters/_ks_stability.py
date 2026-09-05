"""Per-feature train/test KS-test stability filter.

A feature can pass the model-based adversarial-AUC diagnostic (the CLASSIFIER can't separate train from test
using it, or its contribution to the combined signal is small) yet still have a distribution that shifted
meaningfully between splits when examined ALONE. A 1st-place LANL-earthquake-prediction team's rule: only
keep a feature if its train-vs-test Kolmogorov-Smirnov test p-value clears a threshold (``<= 0.05`` rejects
the feature) - a per-feature, distribution-based check that complements (not duplicates) adversarial
validation's model-based, joint-feature-set check.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def ks_stability_filter(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    p_value_threshold: float = 0.05,
    n_splits: int = 1,
    split_frac: float = 0.5,
    random_state: int = 0,
) -> pd.DataFrame:
    """Per-feature two-sample KS test between ``train_df`` and ``test_df``; flag unstable features.

    Parameters
    ----------
    train_df, test_df
        Frames sharing the columns to screen; only finite values are compared per column.
    feature_cols
        Numeric columns to screen; defaults to every numeric column present in both frames.
    p_value_threshold
        A feature is flagged unstable when its KS p-value is AT OR BELOW this threshold (the source
        convention: ``p <= 0.05`` means the train/test distributions are significantly different, i.e. the
        feature is unstable and a candidate to drop).
    n_splits
        Number of KS checks to run per feature. ``1`` (the default) reproduces the original single-pair
        behavior bit-identically: one KS test on the full ``train_df``/``test_df`` values. When ``> 1``, a
        majority-vote mode kicks in: each split draws an independent random subsample (without replacement,
        size ``split_frac`` of each side) from ``train_df``/``test_df`` and runs its own KS test; a feature
        is flagged unstable only when a STRICT MAJORITY of the ``n_splits`` checks flag it, which damps false
        flags caused by a single noisy split. The reported ``p_value``/``ks_statistic`` are the medians
        across splits (used only for the report's sort order, not for the stability verdict itself).
    split_frac
        Fraction of each side's finite values drawn per split when ``n_splits > 1`` (default ``0.5``: small
        enough that splits are not near-duplicates of each other, which is what lets the majority vote
        actually decorrelate the false-flag noise instead of just repeating the same verdict ``n_splits``
        times). Ignored when ``n_splits == 1``.
    random_state
        Seed for the per-split subsampling when ``n_splits > 1``. Ignored when ``n_splits == 1``.

    Returns
    -------
    pd.DataFrame
        One row per screened column: ``{"column", "ks_statistic", "p_value", "stable", "measured"}`` plus, when
        ``n_splits > 1``, ``{"n_splits", "n_unstable_splits"}``. Sorted by ``p_value`` ascending (most
        unstable first). ``stable=False`` marks features to consider dropping. ``measured=False`` marks a
        column with no finite values on one side or the other: ``stable`` is left True there so nothing is
        dropped on evidence nobody has, but it means "not judged" rather than "checked and fine", which the
        report previously could not express.
    """
    from scipy.stats import ks_2samp

    if feature_cols is None:
        # A drift guard that inspected nothing must not report a clean bill. This derivation is a DOUBLE
        # filter -- shared name AND pandas-numeric dtype -- so it empties on a rename, a prefixed test frame,
        # a pipeline that emits engineered names on one side only, or columns arriving as object/string or a
        # non-pandas-numeric extension dtype. The loop below would then run zero times and every caller would
        # read 'no unstable features', making a fully drifted feature set indistinguishable from a clean one.
        # An explicitly-passed empty list is the caller's own decision and is left alone.
        derived = [c for c in train_df.columns if c in test_df.columns and pd.api.types.is_numeric_dtype(train_df[c])]
        if not derived:
            shared = [c for c in train_df.columns if c in test_df.columns]
            raise ValueError(
                "ks_stability_filter: no shared numeric columns to screen, so no drift could be measured. "
                f"train has {len(train_df.columns)} column(s), test {len(test_df.columns)}, {len(shared)} shared; "
                f"of the shared ones, dtypes are {sorted({str(train_df[c].dtype) for c in shared})[:6]}. "
                "Pass feature_cols explicitly if this is intended."
            )
        feature_cols = derived
    feature_cols = list(feature_cols)
    multi_split = n_splits > 1
    if multi_split and not (0.0 < split_frac <= 1.0):
        # a split_frac > 1.0 (or <= 0) used to crash
        # deep inside rng.choice(..., replace=False) with an opaque "a cannot be greater than population
        # size" ValueError - validate up front and name the bad argument clearly.
        raise ValueError(f"ks_stability_filter: split_frac must be in (0.0, 1.0]; got {split_frac!r}")
    rng = np.random.default_rng(random_state) if multi_split else None

    rows = []
    for col in feature_cols:
        train_vals = train_df[col].to_numpy(dtype=np.float64)
        test_vals = test_df[col].to_numpy(dtype=np.float64)
        train_vals = train_vals[np.isfinite(train_vals)]
        test_vals = test_vals[np.isfinite(test_vals)]
        if train_vals.size == 0 or test_vals.size == 0:
            # `stable` stays True so the column is not proposed for dropping on evidence nobody has -- callers
            # such as drop_noninformative_vs_reference use it directly as a boolean mask. But True here means
            # 'not judged', not 'checked and fine', so `measured` says which of the two it is: an all-NaN
            # column used to be indistinguishable in this report from one whose distributions genuinely match.
            row = {"column": col, "ks_statistic": np.nan, "p_value": np.nan, "stable": True, "measured": False}
            if multi_split:
                row["n_splits"] = n_splits
                row["n_unstable_splits"] = 0
            rows.append(row)
            continue

        if not multi_split:
            result = ks_2samp(train_vals, test_vals)
            p_value = float(result.pvalue)
            rows.append({"column": col, "ks_statistic": float(result.statistic), "p_value": p_value, "stable": p_value > p_value_threshold, "measured": True})
            continue

        assert rng is not None
        train_size = max(1, round(train_vals.size * split_frac))
        test_size = max(1, round(test_vals.size * split_frac))
        statistics = []
        p_values = []
        n_unstable = 0
        for _ in range(n_splits):
            train_sample = rng.choice(train_vals, size=train_size, replace=False)
            test_sample = rng.choice(test_vals, size=test_size, replace=False)
            split_result = ks_2samp(train_sample, test_sample)
            split_p_value = float(split_result.pvalue)
            statistics.append(float(split_result.statistic))
            p_values.append(split_p_value)
            if split_p_value <= p_value_threshold:
                n_unstable += 1

        rows.append(
            {
                "column": col,
                "ks_statistic": float(np.median(statistics)),
                "p_value": float(np.median(p_values)),
                "stable": n_unstable <= n_splits // 2,
                "measured": True,
                "n_splits": n_splits,
                "n_unstable_splits": n_unstable,
            }
        )

    columns = ["column", "ks_statistic", "p_value", "stable", "measured"] + (["n_splits", "n_unstable_splits"] if multi_split else [])
    if not rows:
        # An explicitly empty `feature_cols` is a legitimate no-op, but `sort_values("p_value")` on a frame
        # built from no rows raises KeyError -- the column does not exist. Return the documented shape empty.
        return pd.DataFrame(columns=columns)
    report = pd.DataFrame(rows)
    return report.sort_values("p_value", ascending=True, na_position="last").reset_index(drop=True)


__all__ = ["ks_stability_filter"]
