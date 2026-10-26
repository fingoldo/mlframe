"""Continuous columns handed to the cat-triple search do not change what it finds, and cost nothing."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._cat_triple_fe import hybrid_cat_triple_fe


def _frame(n: int = 6000, with_noise: bool = True):
    """Four small categoricals whose parity carries the target, plus continuous distractors."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"i{k}": rng.integers(0, 6, n) for k in range(4)})
    y = ((X.i0 + X.i1 + X.i2) % 2 + rng.normal(0, 0.3, n) > 0.5).astype(int).to_numpy()
    if with_noise:
        for k in range(6):
            X[f"x{k}"] = rng.normal(size=n)
    return X, y


def test_distractor_columns_leave_the_result_unchanged():
    """Same appended triples with and without continuous columns in ``cat_cols``."""
    X, y = _frame()
    with_noise = hybrid_cat_triple_fe(X, y, cat_cols=list(X.columns), random_state=0)[1]
    Xc, yc = _frame(with_noise=False)
    without = hybrid_cat_triple_fe(Xc, yc, cat_cols=list(Xc.columns), random_state=0)[1]
    assert with_noise == without
    assert with_noise


def test_a_frame_of_only_continuous_columns_returns_nothing():
    """Every column is above the half-of-n cardinality bar: no category left to cross."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(3000, 5)), columns=list("abcde"))
    _out_x, appended, recipes, scores = hybrid_cat_triple_fe(X, rng.integers(0, 2, 3000), cat_cols=list(X.columns))
    assert appended == [] and recipes == [] and scores.empty
