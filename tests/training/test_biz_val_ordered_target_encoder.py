"""biz_value test for ``training.feature_handling.ordered_target_encode``.

The win: on a synthetic where the category has ZERO true relationship to the target (pure noise, many rare
categories), a naive full-data mean encoding (using each row's OWN label as part of its own encoding) shows a
spurious, artificially high AUC purely from self-leakage. The ordered (row-level expanding, prior-rows-only)
encoding correctly avoids this and stays near chance -- the honest result, matching CatBoost's motivation for
ordered target statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mlframe.training.feature_handling.ordered_target_encoder import ordered_target_encode


def test_biz_val_ordered_target_encode_avoids_self_leakage_naive_encoding_falls_for():
    rng = np.random.default_rng(0)
    n = 3000
    n_cats = 1000  # many small/rare categories
    cats = rng.integers(0, n_cats, n)
    y = rng.integers(0, 2, n)  # category has NO true relationship to y

    df = pd.DataFrame({"cat": cats, "y": y})
    naive_encoding = df.groupby("cat")["y"].transform("mean").to_numpy()
    naive_auc = roc_auc_score(y, naive_encoding)

    ordered_encoding = ordered_target_encode(cats, y, smoothing=1.0)
    ordered_auc = roc_auc_score(y, ordered_encoding)

    assert naive_auc > 0.7, f"naive full-data mean encoding should show a suspiciously inflated AUC from self-leakage: {naive_auc:.4f}"
    assert abs(ordered_auc - 0.5) < 0.05, f"ordered encoding should stay near chance (no true signal, no leak): {ordered_auc:.4f}"


def test_ordered_target_encode_first_occurrence_gets_prior():
    cats = np.array(["a", "a", "b"])
    y = np.array([1.0, 0.0, 1.0])
    encoded = ordered_target_encode(cats, y, smoothing=1.0, prior=0.5)
    assert encoded[0] == 0.5  # "a"'s first occurrence: zero running count -> pure prior
    assert encoded[2] == 0.5  # "b"'s first (only) occurrence


def test_ordered_target_encode_matches_manual_expanding_computation():
    cats = np.array(["a", "a", "a"])
    y = np.array([1.0, 1.0, 0.0])
    encoded = ordered_target_encode(cats, y, smoothing=1.0, prior=0.5)
    # row 0: running_sum=0, running_count=0 -> (0 + 1*0.5) / (0 + 1) = 0.5
    assert np.isclose(encoded[0], 0.5)
    # row 1: running_sum=1 (row 0's y), running_count=1 -> (1 + 0.5) / (1 + 1) = 0.75
    assert np.isclose(encoded[1], 0.75)
    # row 2: running_sum=2 (rows 0,1), running_count=2 -> (2 + 0.5) / (2 + 1) = 0.8333
    assert np.isclose(encoded[2], 2.5 / 3.0)


def test_ordered_target_encode_respects_custom_order():
    cats = np.array(["a", "a"])
    y = np.array([10.0, 0.0])
    order = np.array([1, 0])  # row 1 (y=0) comes FIRST in causal order, row 0 (y=10) second
    encoded = ordered_target_encode(cats, y, order=order, smoothing=1.0, prior=5.0)
    # in causal order, row index 1 is first (gets pure prior), row index 0 sees row 1's y=0.0 as prior history.
    assert np.isclose(encoded[1], 5.0)
    assert np.isclose(encoded[0], (0.0 + 1.0 * 5.0) / (1 + 1.0))
