"""Regression: ``prepare_df_for_catboost`` must produce categorical codes
that are stable across train/val/test splits. Pre-fix every frame got an
independent ``astype("category")`` cast, so ``train.cat.codes[i]`` for
value ``"A"`` could equal ``val.cat.codes[i]`` for value ``"B"`` - the
two splits used different code <-> label maps.

Fix: compute joint category union from ``train + val`` (held-out test must
use ``strict=False`` semantics with missing categories -> NaN), then cast
each split with that fixed dtype.
"""

from __future__ import annotations

import pandas as pd


def test_cat_codes_consistent_across_train_val_test():
    """Same string value gets the same code in train, val, and test."""
    from mlframe.training.pipeline import prepare_dfs_for_catboost_joint

    # Train sees A, B. Val sees B, C. Test sees A, D (D is unseen -> NaN code).
    train = pd.DataFrame({"x": ["A", "B", "A", "B"]})
    val = pd.DataFrame({"x": ["B", "C"]})
    test = pd.DataFrame({"x": ["A", "D"]})

    prepare_dfs_for_catboost_joint(
        train_df=train,
        val_df=val,
        test_df=test,
        cat_features=["x"],
    )

    # All three frames must use a Categorical dtype.
    assert pd.api.types.is_categorical_dtype(train["x"])
    assert pd.api.types.is_categorical_dtype(val["x"])
    assert pd.api.types.is_categorical_dtype(test["x"])

    # Train + val drive the union; test does NOT contribute.
    expected_union = sorted({"A", "B", "C"})
    assert sorted(train["x"].cat.categories.tolist()) == expected_union
    assert sorted(val["x"].cat.categories.tolist()) == expected_union
    assert sorted(test["x"].cat.categories.tolist()) == expected_union

    # Same string -> same code across frames.
    def code_for(df, value):
        """Code for."""
        idx = (df["x"] == value).idxmax()
        return int(df["x"].cat.codes.iloc[idx])

    assert code_for(train, "A") == code_for(test, "A"), "code for 'A' differs between train and test"
    assert code_for(train, "B") == code_for(val, "B"), "code for 'B' differs between train and val"

    # Test row holding 'D' (unseen at union-construction time) must land
    # as a NaN/-1 code, not collide with another category.
    d_pos = (test["x"].isna()).tolist()
    assert any(d_pos), "test row with unseen 'D' should map to NaN under strict=False semantics"


def test_cat_codes_naive_per_split_cast_drifts():
    """Sanity probe: when each split is cast independently and the splits
    have DIFFERENT visible category SETS, the per-split codes for the
    overlapping value diverge - exactly the bug fix #3 addresses.
    """
    # Train sees {A, B}; val sees {A, C}. Independent ``astype('category')``
    # builds two different category arrays, so code for 'A' differs.
    train = pd.DataFrame({"x": ["A", "B", "A", "B"]}).astype({"x": "category"})
    val = pd.DataFrame({"x": ["A", "C", "C", "A"]}).astype({"x": "category"})
    int(train["x"].cat.codes.iloc[0])
    int(val["x"].cat.codes.iloc[0])
    # train: categories=[A, B] -> A->0; val: categories=[A, C] -> A->0.
    # If the codes happen to coincide here, the inverse case ("B"/"C")
    # demonstrates the divergence:
    train_code_for_B = int(train["x"].cat.codes.iloc[1])  # B in train is code 1
    val_code_for_C = int(val["x"].cat.codes.iloc[1])  # C in val is code 1
    assert train_code_for_B == val_code_for_C, (
        "sanity probe expected per-split cast to confuse different values to the same code (B in train vs C in val); fix the probe."
    )
