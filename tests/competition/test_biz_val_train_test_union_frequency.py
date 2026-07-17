"""Unit + biz_value tests for mlframe.competition.train_test_union_frequency.

COMPETITION/EXPLORATORY ONLY — see module docstring under src/mlframe/competition/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mlframe.competition.train_test_union_frequency import (
    train_test_union_frequency_encode,
    train_test_union_frequency_encode_hierarchical_components,
)


def _make_drifting_version_dataset(n_rows: int = 20000, n_versions: int = 60, seed: int = 0):
    """Simulate software-version rows where each version has a fixed windowed lifespan on a time axis.

    Each version ``v`` has a true underlying population weight ``true_weight[v]`` (drawn
    independently of any windowing effect) and a lifespan window on ``t in [0, 1]`` — versions
    are introduced and retired over time, mimicking real version-drift data. A row's version is
    sampled proportional to ``true_weight[v] * window_kernel(t; center_v, width)``.

    ``train`` = rows with ``t < 0.6`` (older window), ``test`` = rows with ``t >= 0.6`` (newer
    window). Many higher-index versions barely appear in train (their window is mostly/entirely
    in the test region), so train-only counts systematically underestimate their true
    prevalence — the drift/time-dependence the trick is meant to remove. ``true_weight`` is the
    ground truth the encoded features are compared against.
    """
    rng = np.random.default_rng(seed)

    majors = rng.integers(1, 6, size=n_versions)
    minors = rng.integers(0, 10, size=n_versions)
    patches = rng.integers(0, 10, size=n_versions)
    version_strings = np.array([f"{ma}.{mi}.{pa}" for ma, mi, pa in zip(majors, minors, patches)])

    true_weight = rng.lognormal(mean=0.0, sigma=0.8, size=n_versions)
    # windows spread across [0, 1], ordered so later-index versions skew to later windows
    centers = np.sort(rng.uniform(0.0, 1.0, size=n_versions))
    width = 0.12

    t = rng.uniform(0.0, 1.0, size=n_rows)
    # kernel(t) per version per row: gaussian window around each version's center
    diffs = (t[:, None] - centers[None, :]) / width
    kernel = np.exp(-0.5 * diffs**2)
    row_weights = kernel * true_weight[None, :]
    row_probs = row_weights / row_weights.sum(axis=1, keepdims=True)

    version_idx = np.array([rng.choice(n_versions, p=row_probs[i]) for i in range(n_rows)])

    df = pd.DataFrame(
        {
            "t": t,
            "version": version_strings[version_idx],
            "true_weight": true_weight[version_idx],
            "version_idx": version_idx,
        }
    )

    train_df = df[df["t"] < 0.6].reset_index(drop=True)
    test_df = df[df["t"] >= 0.6].reset_index(drop=True)
    return train_df, test_df, version_strings, true_weight


def test_train_test_union_frequency_encode_basic_shapes_and_values():
    train_series = pd.Series(["a", "a", "b", "c"], index=[10, 11, 12, 13])
    test_series = pd.Series(["a", "b", "b"], index=[20, 21, 22])

    train_encoded, test_encoded = train_test_union_frequency_encode(train_series, test_series)

    assert list(train_encoded.index) == [10, 11, 12, 13]
    assert list(test_encoded.index) == [20, 21, 22]
    # "a": 2 train + 1 test = 3; "b": 1 train + 2 test = 3; "c": 1 train + 0 test = 1
    assert train_encoded.tolist() == [3.0, 3.0, 3.0, 1.0]
    assert test_encoded.tolist() == [3.0, 3.0, 3.0]


def test_train_test_union_frequency_encode_hierarchical_components_levels():
    train_series = pd.Series(["1.0.0", "1.0.1", "2.0.0"])
    test_series = pd.Series(["1.0.0", "1.1.0"])

    components = train_test_union_frequency_encode_hierarchical_components(train_series, test_series, ".")

    assert set(components.keys()) == {"major", "major_minor", "major_minor_patch"}
    major_train, major_test = components["major"]
    # major "1" appears 2x train ("1.0.0","1.0.1") + 2x test ("1.0.0","1.1.0") = 4
    assert major_train.tolist() == [4.0, 4.0, 1.0]
    assert major_test.tolist() == [4.0, 4.0]

    full_train, full_test = components["major_minor_patch"]
    # "1.0.0" appears 1x train + 1x test = 2
    assert full_train.tolist() == [2.0, 1.0, 1.0]
    assert full_test.tolist() == [2.0, 1.0]


def test_biz_val_train_test_union_frequency_beats_train_only_under_drift():
    """Union-pooled frequency correlates better with the true (drift-free) prevalence than train-only counts.

    On the version-lifespan-drift dataset, train-only counts are stale for versions whose
    window sits mostly in the test-time region, so their rank-correlation with the true
    underlying population weight degrades relative to counting over the train+test union.
    """
    train_df, test_df, _, _ = _make_drifting_version_dataset(n_rows=20000, n_versions=60, seed=0)

    train_series = train_df["version"]
    test_series = test_df["version"]

    # train-only frequency baseline: counts computed from train alone, applied to test rows
    train_only_counts = train_series.value_counts()
    train_only_feature_on_test = test_series.map(train_only_counts).fillna(0.0).to_numpy()

    union_train_encoded, union_test_encoded = train_test_union_frequency_encode(train_series, test_series)

    y_true_test = test_df["true_weight"].to_numpy()

    corr_train_only = spearmanr(train_only_feature_on_test, y_true_test).statistic
    corr_union = spearmanr(union_test_encoded.to_numpy(), y_true_test).statistic

    assert corr_train_only < 0.55, f"expected degraded train-only correlation, got {corr_train_only:.3f}"
    assert corr_union >= 0.70, f"expected union correlation >= 0.70, got {corr_union:.3f}"
    assert corr_union > corr_train_only + 0.15, f"expected union to beat train-only by a real margin: union={corr_union:.3f} train_only={corr_train_only:.3f}"

    # sanity: train-side encoding also aligns with the same union counts used for test
    assert len(union_train_encoded) == len(train_series)


def test_biz_val_hierarchical_split_helps_novel_full_version_strings():
    """For a full version string never seen in train, coarser hierarchical levels still carry signal.

    Builds a case where test contains patch-level version strings that never occur in train
    (full-string train-side frequency is exactly 0 for all of them), but the parent
    major.minor family already appeared in train. The hierarchical geometric-mean feature
    should correlate materially better with true prevalence than the flat full-string
    union frequency for exactly these novel rows.
    """
    rng = np.random.default_rng(3)

    n_families = 150
    # unique (major, minor) pairs per family, so major_minor-level counts never mix two
    # different families' weights together
    majors = rng.integers(1, 4, size=n_families)
    minors = np.arange(n_families)
    family_weight = rng.lognormal(mean=0.0, sigma=0.6, size=n_families)

    train_rows: list[str] = []
    test_rows: list[str] = []
    test_true_weight: list[float] = []

    for fam_idx in range(n_families):
        major, minor = majors[fam_idx], minors[fam_idx]
        weight = family_weight[fam_idx]
        # train has abundant, low-noise history for this family (precise weight signal);
        # test only has a handful of brand-new-patch rows (noisy small-sample signal) - the
        # flat full-string feature can only ever see the noisy test-only count for a novel
        # string, while the hierarchical major.minor level also inherits train's precise count
        n_train_reps = max(5, int(round(weight * 800 * rng.uniform(0.97, 1.03))))
        # train only ever sees patch "0" for this family
        train_rows += [f"{major}.{minor}.0"] * n_train_reps

        # test sees a brand-new patch number never present in train, same family - the
        # small test-only sample size for a first-seen string carries essentially no
        # weight signal on its own (independent of the family's true weight)
        n_test_reps = int(rng.integers(1, 6))
        novel_patch = 99
        test_rows += [f"{major}.{minor}.{novel_patch}"] * n_test_reps
        test_true_weight += [weight] * n_test_reps

    train_series = pd.Series(train_rows)
    test_series = pd.Series(test_rows)
    y_true_test = np.array(test_true_weight)

    # flat full-string union frequency: all test values are novel-to-train, so it collapses
    # to (near-)constant test-side counts and cannot rank-order families by true weight
    flat_train_encoded, flat_test_encoded = train_test_union_frequency_encode(train_series, test_series)
    corr_flat = spearmanr(flat_test_encoded.to_numpy(), y_true_test).statistic

    # the combined (geometric-mean-of-all-levels) feature still folds in the noisy
    # full-patch level, so check it against a weaker bar - the strong bar belongs to the
    # raw major_minor component, which is exactly where the precise train history lives
    hier_train_encoded, hier_test_encoded = train_test_union_frequency_encode(train_series, test_series, hierarchical_split_sep=".")
    corr_hier_combined = spearmanr(hier_test_encoded.to_numpy(), y_true_test).statistic

    components = train_test_union_frequency_encode_hierarchical_components(train_series, test_series, ".")
    _, major_minor_test = components["major_minor"]
    corr_major_minor = spearmanr(major_minor_test.to_numpy(), y_true_test).statistic

    assert abs(corr_flat) < 0.35, f"expected near-uninformative flat correlation, got {corr_flat:.3f}"
    assert corr_major_minor >= 0.70, f"expected major_minor correlation >= 0.70, got {corr_major_minor:.3f}"
    assert corr_major_minor > abs(corr_flat) + 0.30, (
        f"expected major_minor to beat flat by a real margin: major_minor={corr_major_minor:.3f} flat={corr_flat:.3f}"
    )
    assert corr_hier_combined > abs(corr_flat), (
        f"expected combined hierarchical feature to still beat flat: combined={corr_hier_combined:.3f} flat={corr_flat:.3f}"
    )

    assert len(hier_train_encoded) == len(train_series)
