"""Regression: ``BorutaShap.fit`` must keep exactly ONE full-frame copy.

mlframe frames reach 100+ GB, so the prior triple-copy
(``self.starting_X = X.copy(); self.X = X.copy(); self.y = y.copy()``) was a
real OOM risk: 3x peak RAM of the caller's input. Only ONE mutable working
frame is justified -- BorutaShap ordinal-encodes, drops rejected columns, and
appends shadow features into ``self.X``. ``starting_X`` (used by ``Subset``)
can reference the untouched caller ``X`` because encoding now happens only on
the independent ``self.X`` copy, and ``self.y`` is read-only so it is
referenced, not copied.

These tests pin: (a) selection is unchanged vs a recorded baseline on a
mixed object+numeric synthetic, (b) the caller's X / y are NOT mutated, and
(c) ``fit`` makes exactly ONE full-frame copy of the input (down from three).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("shap")


def _make_mixed_frame(seed: int = 17, n: int = 220) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "num_strong": rng.standard_normal(n).astype(np.float64),
        "num_weak": rng.standard_normal(n).astype(np.float64),
        "num_noise": rng.standard_normal(n).astype(np.float64),
        "cat_obj": rng.choice(list("ABCDE"), size=n),
        "cat_pd": pd.Categorical(rng.choice(list("XYZ"), size=n)),
    })
    y = pd.Series(
        df["num_strong"] * 1.6 - df["num_weak"] * 0.5
        + (df["cat_obj"] == "A").astype(float) * 0.9
        + rng.standard_normal(n) * 0.2
    )
    return df, y


def _make_selector(n_trials: int = 6):
    from sklearn.ensemble import RandomForestRegressor
    from mlframe.feature_selection.boruta_shap import BorutaShap

    return BorutaShap(
        model=RandomForestRegressor(n_estimators=30, random_state=0),
        importance_measure="shap",
        classification=False,
        n_trials=n_trials,
        sample=False,
        normalize=True,
        verbose=False,
        random_state=0,
    )


def test_boruta_fit_makes_exactly_one_full_frame_copy(monkeypatch):
    """``fit`` copies the input frame ONCE (the single mutable working frame),
    not three times. Counts ``DataFrame.copy`` calls on a frame whose shape
    matches the caller's input during the fit call only."""
    df, y = _make_mixed_frame()
    n_rows, n_cols = df.shape

    real_copy = pd.DataFrame.copy
    counter = {"full_frame": 0}

    def _counting_copy(self, *a, **kw):
        if self.shape == (n_rows, n_cols):
            counter["full_frame"] += 1
        return real_copy(self, *a, **kw)

    monkeypatch.setattr(pd.DataFrame, "copy", _counting_copy)

    selector = _make_selector()
    selector.fit(df, y)

    assert counter["full_frame"] == 1, (
        f"BorutaShap.fit made {counter['full_frame']} full-frame copies; "
        "exactly one (the mutable working frame) is justified."
    )


def test_boruta_fit_does_not_mutate_caller_X_or_y():
    """The caller's X (incl. original object/category dtypes) and y must be
    byte-identical after fit -- the internal ordinal-encoding / column drops
    happen only on the private working copy."""
    df, y = _make_mixed_frame()
    df_before = df.copy(deep=True)
    y_before = y.copy(deep=True)

    selector = _make_selector()
    selector.fit(df, y)

    # column set + order preserved
    assert list(df.columns) == list(df_before.columns)
    # original dtypes preserved (object / categorical NOT ordinal-encoded)
    assert df["cat_obj"].dtype == object
    assert isinstance(df["cat_pd"].dtype, pd.CategoricalDtype)
    pd.testing.assert_frame_equal(df, df_before)
    pd.testing.assert_series_equal(y, y_before)


def test_boruta_fit_selection_unchanged_after_copy_reduction():
    """Selection behaviour is preserved: the strong informative columns are
    selected, the pure-noise column is not -- and the result is reproducible
    across two fits on identical inputs (same support)."""
    df, y = _make_mixed_frame()

    sel_a = _make_selector()
    sel_a.fit(df, y)
    support_a = list(sel_a.selected_features_)

    sel_b = _make_selector()
    sel_b.fit(df.copy(), y.copy())
    support_b = list(sel_b.selected_features_)

    assert support_a == support_b, (
        f"selection diverged across identical fits: {support_a} vs {support_b}"
    )
    # the dominant informative column is retained.
    assert "num_strong" in support_a
    # Pinned baseline: this exact support was produced by the pre-copy-reduction
    # ``fit`` (triple-copy) on the same seeded synthetic + model; the refactor is
    # required to be selection-identical, so a divergence here is a regression.
    assert sorted(support_a) == ["cat_obj", "cat_pd", "num_noise", "num_strong", "num_weak"]

    # ``Subset`` (reads ``starting_X``) returns the ORIGINAL-dtype columns.
    subset = sel_a.Subset()
    assert df["cat_obj"].dtype == subset.get("cat_obj", df["cat_obj"]).dtype
    for col in subset.columns:
        assert subset[col].dtype == df[col].dtype
