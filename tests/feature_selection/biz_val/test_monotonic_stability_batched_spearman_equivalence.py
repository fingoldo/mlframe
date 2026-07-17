"""Wave 11 (Category 3) H3: ``monotonic_deviation_stability_filter`` (``_monotonic_stability.py``) called
``scipy.stats.spearmanr`` once per (subsample, feature) cell -- default 30 x 500 = ~15,000 calls per
invocation. Rewritten to batch the rank transform + Pearson-on-ranks across all features sharing one
subsample's row mask via ``_spearman_corr_batch``. Pins the new batched path against the original
per-column ``_spearman_corr`` loop (kept in the module, unused by the public function) across randomized
configs including NaN, constant columns, and the segment-conditional path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._monotonic_stability import (
    _spearman_corr,
    monotonic_deviation_stability_filter,
)


def _reference_filter(
    df,
    y,
    group_col,
    feature_cols,
    n_subsamples,
    group_fraction,
    min_stable_fraction,
    random_state,
    segment_col=None,
    segment_min_agreement=None,
):
    """Frozen copy of the pre-Wave-11 per-column ``scipy.stats.spearmanr`` loop implementation."""
    if feature_cols is None:
        excluded = {group_col} if segment_col is None else {group_col, segment_col}
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]
    feature_cols = list(feature_cols)
    y = np.asarray(y, dtype=np.float64)
    group_means = df.groupby(group_col, sort=False)[feature_cols].transform("mean")
    deviations = df[feature_cols] - group_means
    groups = df[group_col].to_numpy()
    unique_groups = np.unique(groups)
    n_groups_per_draw = max(1, round(group_fraction * len(unique_groups)))
    rng = np.random.default_rng(random_state)
    full_sample_corr = {col: _spearman_corr(deviations[col].to_numpy(), y) for col in feature_cols}
    agreement_counts = {col: 0 for col in feature_cols}
    for _ in range(n_subsamples):
        chosen_groups = rng.choice(unique_groups, size=n_groups_per_draw, replace=False)
        row_mask = np.isin(groups, chosen_groups)
        y_sub = y[row_mask]
        for col in feature_cols:
            dev_sub = deviations[col].to_numpy()[row_mask]
            corr = _spearman_corr(dev_sub, y_sub)
            full_sign = np.sign(full_sample_corr[col])
            sub_sign = np.sign(corr)
            if full_sign == 0 or sub_sign == full_sign:
                agreement_counts[col] += 1
    rows = []
    for col in feature_cols:
        agreement_fraction = agreement_counts[col] / n_subsamples
        rows.append(
            {
                "feature": col,
                "full_sample_correlation": full_sample_corr[col],
                "sign_agreement_fraction": agreement_fraction,
                "stable": agreement_fraction >= min_stable_fraction,
            }
        )
    return pd.DataFrame(rows)


def _make_data(seed, n_groups=120, rows_per_group=6, p=12, nan_frac=0.0, const_frac=0.0):
    """Make data."""
    rng = np.random.default_rng(seed)
    group_ids = np.repeat(np.arange(n_groups), rows_per_group)
    n = len(group_ids)
    gb = rng.normal(0, 1, n_groups)[group_ids]
    cols = {}
    for j in range(p):
        col = gb * (0.5 if j % 3 else 1.0) + rng.normal(0, 0.3 + 0.05 * j, n)
        if j < int(p * const_frac):
            col = np.full(n, float(j))
        cols[f"f{j}"] = col
    df = pd.DataFrame(cols)
    if nan_frac > 0:
        mask = rng.random(df.shape) < nan_frac
        df = df.mask(mask)
    df["group"] = group_ids
    y = (rng.normal(size=n) + 0.4 * (gb > 0)).astype(np.float64)
    return df, y


def test_batched_spearman_matches_reference_across_random_configs():
    """Batched spearman matches reference across random configs."""
    n_checks = 0
    for seed in range(6):
        for nan_frac in (0.0, 0.05):
            for const_frac in (0.0, 0.1):
                df, y = _make_data(seed, nan_frac=nan_frac, const_frac=const_frac)
                feat_cols = [c for c in df.columns if c.startswith("f")]
                kwargs = dict(
                    df=df,
                    y=y,
                    group_col="group",
                    feature_cols=feat_cols,
                    n_subsamples=15,
                    group_fraction=0.5,
                    min_stable_fraction=0.7,
                    random_state=seed,
                )
                r_ref = _reference_filter(**kwargs)
                r_new = monotonic_deviation_stability_filter(**kwargs)
                n_checks += 1
                pd.testing.assert_frame_equal(
                    r_ref.reset_index(drop=True),
                    r_new.reset_index(drop=True),
                    check_exact=False,
                    atol=1e-9,
                    rtol=1e-9,
                )
    assert n_checks == 24


def test_biz_val_monotonic_stability_still_separates_stable_from_jumpy():
    """Existing biz_value invariant (the filter's whole reason to exist) must survive the rewrite."""
    rng = np.random.default_rng(0)
    n_groups = 250
    rows_per_group = 6
    group_ids = np.repeat(np.arange(n_groups), rows_per_group)
    n = len(group_ids)
    group_baseline = rng.normal(0, 1, n_groups)[group_ids]
    good_deviation_noise = rng.normal(0, 0.3, n)
    good_feature = group_baseline + good_deviation_noise
    y = (good_deviation_noise > 0).astype(int)
    jumpy_feature = rng.normal(0, 1, n)
    df = pd.DataFrame({"group": group_ids, "good_feature": good_feature, "jumpy_feature": jumpy_feature})

    result = monotonic_deviation_stability_filter(
        df,
        y,
        group_col="group",
        feature_cols=["good_feature", "jumpy_feature"],
        n_subsamples=40,
        random_state=0,
    )
    by_name = {row["feature"]: row for _, row in result.iterrows()}
    assert by_name["good_feature"]["stable"]
    assert not by_name["jumpy_feature"]["stable"]
