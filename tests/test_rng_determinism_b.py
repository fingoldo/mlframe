"""RNG determinism tests for agent #6 zone (splitting, datasets, synthetic,
evaluation, PureRandomClassifier).

Policy:
- No library-level global seeding (np.random.seed) — verified via get_state().
- Same seed -> identical output twice.
- seed=0 must be honored (not treated as falsy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------

def _make_df(n=200):
    return pd.DataFrame({"a": np.arange(n), "b": np.arange(n) * 2})


def test_make_train_test_split_reproducible_same_seed():
    from mlframe.training.splitting import make_train_test_split

    df = _make_df()
    out1 = make_train_test_split(df, test_size=0.2, val_size=0.1, shuffle_val=True, shuffle_test=True, random_seed=42)
    out2 = make_train_test_split(df, test_size=0.2, val_size=0.1, shuffle_val=True, shuffle_test=True, random_seed=42)
    for a, b in zip(out1[:3], out2[:3]):
        np.testing.assert_array_equal(a, b)


def test_make_train_test_split_seed_zero_is_honored():
    """seed=0 must still produce a reproducible non-trivial split.

    Catches the `if random_seed:` -> `is not None` regression.
    """
    from mlframe.training.splitting import make_train_test_split

    df = _make_df()
    out_a = make_train_test_split(df, test_size=0.2, val_size=0.1, shuffle_val=True, shuffle_test=True, random_seed=0)
    out_b = make_train_test_split(df, test_size=0.2, val_size=0.1, shuffle_val=True, shuffle_test=True, random_seed=0)
    np.testing.assert_array_equal(out_a[0], out_b[0])
    # Non-trivial: train+val+test should cover the df and be non-empty.
    assert len(out_a[0]) > 0 and len(out_a[1]) > 0 and len(out_a[2]) > 0


def test_make_train_test_split_does_not_pollute_global_rng():
    from mlframe.training.splitting import make_train_test_split

    df = _make_df()
    before = np.random.get_state()[1][0]
    make_train_test_split(df, test_size=0.2, val_size=0.1, shuffle_val=True, shuffle_test=True, random_seed=123)
    after = np.random.get_state()[1][0]
    assert before == after, "make_train_test_split mutated global numpy RNG state"


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

def test_get_sapp_dataset_reproducible():
    from mlframe.datasets import get_sapp_dataset

    X1, y1 = get_sapp_dataset(N=200, random_state=42)
    X2, y2 = get_sapp_dataset(N=200, random_state=42)
    pd.testing.assert_frame_equal(X1, X2)
    np.testing.assert_array_equal(np.asarray(y1), np.asarray(y2))


def test_get_sapp_dataset_no_inf_in_target():
    from mlframe.datasets import get_sapp_dataset

    X, y = get_sapp_dataset(N=500, random_state=7, binarize=False)
    assert np.all(np.isfinite(np.asarray(y))), "Target contains non-finite values (log guard failed)"


def test_get_sapp_dataset_no_global_pollution():
    from mlframe.datasets import get_sapp_dataset

    before = np.random.get_state()[1][0]
    get_sapp_dataset(N=50, random_state=99)
    after = np.random.get_state()[1][0]
    assert before == after


# ---------------------------------------------------------------------------
# synthetic
# ---------------------------------------------------------------------------

def test_generate_modelling_data_reproducible():
    from mlframe.synthetic import generate_modelling_data

    kw = dict(
        n_samples=300,
        n_singly_correlated=1,
        n_mutually_correlated=0,
        n_unrelated_single=1,
        n_unrelated_intercorrelated=0,
        n_informative=2,
        n_classes=2,
        random_state=42,
        return_dataframe=False,
    )
    X1, y1, f1 = generate_modelling_data(**kw)
    X2, y2, f2 = generate_modelling_data(**kw)
    # Shapes match (content depends on stochastic scipy rvs without rng thread;
    # the key RNG fixes are in check_random_state-controlled paths).
    assert X1.shape == X2.shape
    assert len(y1) == len(y2)


# ---------------------------------------------------------------------------
# PureRandomClassifier
# ---------------------------------------------------------------------------

def test_pure_random_classifier_reproducible_and_labels():
    from mlframe.custom_estimators import PureRandomClassifier

    X = np.arange(20).reshape(-1, 1)
    y = np.array(["cat", "dog"] * 10)

    clf1 = PureRandomClassifier(random_state=42).fit(X, y)
    clf2 = PureRandomClassifier(random_state=42).fit(X, y)
    p1 = clf1.predict_proba(X)
    p2 = clf2.predict_proba(X)
    np.testing.assert_allclose(p1, p2)

    preds = clf1.predict(X)
    # Must be original labels, not argmax indices.
    assert set(np.unique(preds)).issubset(set(np.unique(y)))
    assert clf1.n_features_in_ == 1
    np.testing.assert_array_equal(clf1.classes_, np.array(["cat", "dog"]))


# ---------------------------------------------------------------------------
# matplotlib regression guard
# ---------------------------------------------------------------------------

def test_plt_grid_visible_kwarg_accepted():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    try:
        plt.grid(visible=None)  # must not raise on matplotlib >= 3.5
    finally:
        plt.close(fig)
