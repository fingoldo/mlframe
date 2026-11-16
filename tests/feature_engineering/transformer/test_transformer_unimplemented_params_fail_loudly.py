"""Transformer parameters that are not implemented raise instead of being silently ignored."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_engineering.transformer.apriori_itemsets import compute_apriori_itemsets_features
from mlframe.feature_engineering.transformer.ib_baseline_codes import compute_ib_baseline_codes_features
from mlframe.feature_engineering.transformer.mdl_binning_pairwise import compute_mdl_binning_pairwise_features


def _frame(n=60, seed=0):
    """A small numeric frame with a binary target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = (X["a"] > 0).astype(int).to_numpy()
    return X, y


def test_ib_baseline_rejects_a_non_default_n_bits():
    """n_bits other than the supported default raises NotImplementedError rather than being ignored."""
    X, y = _frame()
    with pytest.raises(NotImplementedError):
        compute_ib_baseline_codes_features(X, y, X, seed=0, n_bits=5)


def test_apriori_rejects_standardize_false():
    """standardize=False is not implemented and must not be silently treated as True."""
    X, y = _frame()
    with pytest.raises(NotImplementedError):
        compute_apriori_itemsets_features(X, y, X, seed=0, standardize=False)


def test_mdl_binning_rejects_a_non_default_top_k_pairs():
    """top_k_pairs other than the supported default raises NotImplementedError."""
    X, y = _frame()
    with pytest.raises(NotImplementedError):
        compute_mdl_binning_pairwise_features(X, y, X, seed=0, top_k_pairs=3)
