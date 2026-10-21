"""A continuous target is quantile-binned before bootstrap-MI scoring, and the honest null re-bins only the columns that read the target.

``coerce_y_classif`` gave every distinct value of a continuous target its own class. The plug-in MI of any binned feature with such a label equals the feature's entropy (about ln 10), so the
uplift of every engineered column over its source was about 1 and the modular family found nothing on a regression target. The honest null re-binned every column on each permuted target;
only the columns whose supervised codes depend on the target need it.
"""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_shared import coerce_y_classif
from mlframe.feature_selection.filters._periodic_fe import hybrid_modular_fe_with_recipes


def test_a_continuous_target_becomes_a_few_quantile_classes():
    """A float target with many distinct values gets at most 10 classes of similar size, not one class per row."""
    y = np.random.default_rng(0).standard_normal(2000)
    classes = coerce_y_classif(y)
    counts = np.bincount(classes)
    assert 2 <= counts.size <= 10
    assert counts.min() > 100


def test_integer_and_few_valued_targets_keep_one_class_per_value():
    """Integer labels pass through and a float target with few distinct values stays categorical (dense ranks)."""
    ints = np.array([3, 3, 7, 7, 9])
    assert np.array_equal(coerce_y_classif(ints), ints)
    few = np.array([0.5, 1.5, 0.5, 2.5, 1.5])
    assert np.array_equal(coerce_y_classif(few), np.array([0, 1, 0, 2, 1]))


def test_modular_family_finds_the_period_of_a_counter_on_a_regression_target():
    """On ``y = cos(2 pi t / 7) + noise`` the modular family appends a period-7 column of the counter: its uplift over the raw counter is large, not about 1."""
    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame({"t": rng.integers(0, 200, n), "x1": rng.standard_normal(n)})
    y = np.cos(2 * np.pi * X["t"].to_numpy() / 7.0) + 0.2 * rng.standard_normal(n)
    _, appended, _, scores = hybrid_modular_fe_with_recipes(X, y, cols=None, periods=(7, 12, 24, 30, 365), top_k=6, seed=0)
    assert any("_7" in name for name in appended), f"no period-7 column appended: {appended}"
    best = scores.iloc[0]
    assert best["uplift_mean"] > 5.0


def test_columns_that_do_not_read_the_target_keep_their_codes_under_a_permuted_target():
    """Categorical and low-cardinality columns are binned without the target, so their codes under a permuted target equal the observed ones; only continuous columns change."""
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._binning import _target_dependent_columns
    from mlframe.feature_selection.filters.discretization import categorize_dataset

    rng = np.random.default_rng(1)
    n = 800
    cont = rng.standard_normal(n)
    cont[::9] = np.nan
    X = pd.DataFrame({"cont": cont, "cont2": rng.standard_normal(n), "low": rng.integers(0, 5, n), "cat": pd.Categorical(rng.choice(list("abc"), n))})
    y = (X["cont2"].to_numpy() + rng.standard_normal(n) > 0).astype(np.int64)
    kwargs = dict(method="quantile", n_bins=10, dtype=np.int16, max_categorical_cardinality=None, missing_strategy="separate_bin", nbins_strategy="mdlp", nbins_strategy_kwargs=None, cache_dir=None)
    observed, cols, nbins = categorize_dataset(df=X, y_for_strategy=y, **kwargs)
    dependent = _target_dependent_columns(X, list(cols))
    assert dependent == {"cont", "cont2"}
    permuted, cols_p, nbins_p = categorize_dataset(df=X, y_for_strategy=y[rng.permutation(n)], **kwargs)
    position = {str(c): j for j, c in enumerate(cols_p)}
    for j, c in enumerate(cols):
        if str(c) in dependent:
            continue
        k = position[str(c)]
        assert np.array_equal(observed[:, j], permuted[:, k]) and nbins[j] == nbins_p[k], f"column {c!r} changed under a permuted target"
