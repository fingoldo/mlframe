"""TF-IDF entity embedding keeps NaN entities and matches OOV tokens in the vectorizer's case (audit 2026-11-20, findings 24-F18, 24-F19)."""

import numpy as np
import pandas as pd

from mlframe.feature_engineering.tfidf_svd_entity_embedding import _build_documents


def test_nan_entity_is_kept_as_its_own_document():
    """A NaN entity key used to drop out of groupby and then reindex to a NaN document."""
    df = pd.DataFrame({"e": ["a", None, "a"], "tok": ["x", "y", "z"]})
    entities, documents = _build_documents(df, "e", "tok")
    assert len(entities) == len(documents) == 2
    assert all(isinstance(doc, str) for doc in documents)
    assert "y" in documents[list(pd.isna(entities)).index(True)]


def test_oov_check_matches_lowercased_vocabulary():
    """Uppercase tokens that the vectorizer lowercases into its vocabulary must not count as out-of-vocabulary."""
    from mlframe.feature_engineering.tfidf_svd_entity_embedding import tfidf_svd_entity_embedding

    fit_df = pd.DataFrame({"e": ["a", "a", "b", "b"], "tok": ["red", "blue", "red", "green"]})
    _, fitted = tfidf_svd_entity_embedding(fit_df, entity_col="e", token_col="tok", n_components=1, random_state=0, return_fitted=True)
    new_df = pd.DataFrame({"e": ["z", "z"], "tok": ["RED", "Blue"]})
    out = fitted.transform_new_entities(new_df, entity_col="e", token_col="tok")
    np.testing.assert_allclose(out["tfidf_svd_oov_fraction"].to_numpy(), [0.0])
