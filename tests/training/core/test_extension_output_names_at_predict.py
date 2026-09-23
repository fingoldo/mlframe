"""Predict serves the extension pipeline's fit-time column names and refuses a different output width."""

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from mlframe.training.core._predict_pre_pipeline import _extension_output_names


def _fitted():
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 4)), columns=list("abcd"))
    return Pipeline([("dim_reducer", PCA(n_components=2))]).fit(X)


def test_stamped_names_are_served_verbatim():
    pipe = _fitted()
    pipe._mlframe_output_columns_ = ["ext_dim_reducer_0", "ext_dim_reducer_1"]
    assert _extension_output_names(pipe, 2) == ["ext_dim_reducer_0", "ext_dim_reducer_1"]


def test_a_width_change_raises_instead_of_renaming_positionally():
    pipe = _fitted()
    pipe._mlframe_output_columns_ = ["t__tfidf_0", "t__tfidf_1"]
    with pytest.raises(RuntimeError, match="2 columns at fit time but 3"):
        _extension_output_names(pipe, 3)


def test_an_unstamped_pipeline_falls_back_to_the_train_side_names():
    pipe = _fitted()
    assert _extension_output_names(pipe, 2) == ["pca0", "pca1"]
    assert _extension_output_names(pipe, 5) == [f"ext_dim_reducer_{i}" for i in range(5)], "must mirror the train-side fallback, not ext_<i>"
