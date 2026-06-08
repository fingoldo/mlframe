"""Wiring test: NeuralNetStrategy.build_pipeline inserts the embedding/text -> numeric step and the resulting
pipeline turns an embedding-``List`` frame into a pure-numeric matrix (the MLP-consumable form).

Pipeline-level (no full suite, no HF model -- embedding-only), so it pins the strategy wiring fast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from mlframe.training.strategies.neural import NeuralNetStrategy


def test_build_pipeline_no_emb_text_has_no_extra_step():
    pipe = NeuralNetStrategy().build_pipeline(
        None, cat_features=[], imputer=SimpleImputer(), scaler=StandardScaler(),
        embedding_features=None, text_features=None,
    )
    assert not any(name == "neural_emb_text" for name, _ in pipe.steps)


def test_build_pipeline_inserts_emb_text_step_and_outputs_numeric():
    s = NeuralNetStrategy()
    pipe = s.build_pipeline(
        None, cat_features=[], category_encoder=None,
        imputer=SimpleImputer(), scaler=StandardScaler(),
        embedding_features=["emb_0"], text_features=[],
    )
    assert any(name == "neural_emb_text" for name, _ in pipe.steps), "neural_emb_text step missing from pipeline"

    n, d = 24, 4
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "num_0": rng.normal(size=n).astype(np.float32),
        "emb_0": [rng.normal(size=d).astype(np.float32) for _ in range(n)],
    })
    out = pipe.fit_transform(X)
    arr = np.asarray(out)
    assert arr.dtype.kind in ("f", "i", "u", "b"), f"pipeline output not numeric: {arr.dtype}"
    # emb_0 expanded into its components -> more than the single original numeric column
    assert arr.shape[1] >= 1 + d
