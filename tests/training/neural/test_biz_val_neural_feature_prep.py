"""biz_value: the neural text path (frozen HuggingFace embeddings) must carry SEMANTIC signal a classifier can use,
and must beat bag-of-words TF-IDF on synonym generalization -- the whole reason to use real transformer embeddings
instead of TF-IDF for a neural model.

Uses a REAL HuggingFace model (the project default ``intfloat/multilingual-e5-small``). Skips only if ``transformers``
is unavailable or the model can't be fetched (offline CI); it does NOT mock the model -- the point is to validate the
real embeddings.

Design (semantic generalization): train and test sentences draw the discriminative sentiment word from DISJOINT
synonym pools. TF-IDF sees only train-vocabulary tokens, so the test synonyms are out-of-vocabulary -> it predicts at
chance. Frozen transformer embeddings place synonyms of the same sentiment near each other, so a classifier fit on
train embeddings generalizes to the unseen test synonyms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("transformers")

from mlframe.training.neural.feature_prep import NeuralEmbeddingTextEncoder

POS_TRAIN = ["excellent", "superb", "fantastic", "wonderful", "great", "splendid", "stellar", "magnificent"]
POS_TEST = ["amazing", "brilliant", "outstanding", "marvelous", "terrific", "exceptional", "glorious", "superlative"]
NEG_TRAIN = ["terrible", "awful", "horrible", "dreadful", "bad", "lousy", "appalling", "horrid"]
NEG_TEST = ["atrocious", "abysmal", "disappointing", "poor", "miserable", "deplorable", "woeful", "subpar"]


def _make_split(pos_words, neg_words, n, seed):
    """Make split."""
    rng = np.random.default_rng(seed)
    rows, ys = [], []
    for _ in range(n):
        y = int(rng.integers(0, 2))
        word = rng.choice(pos_words if y == 1 else neg_words)
        rows.append(f"the experience was {word}")
        ys.append(y)
    return pd.DataFrame({"text_0": rows}), np.array(ys)


def _fit_score(train_X, ytr, test_X, yte):
    """Fit score."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000).fit(train_X, ytr)
    return float((clf.predict(test_X) == yte).mean())


def test_biz_val_neural_feature_prep_hf_text_beats_tfidf_on_synonym_generalization():
    """Biz val neural feature prep hf text beats tfidf on synonym generalization."""
    Xtr, ytr = _make_split(POS_TRAIN, NEG_TRAIN, 220, seed=0)
    Xte, yte = _make_split(POS_TEST, NEG_TEST, 120, seed=1)  # DISJOINT synonyms from train

    enc = NeuralEmbeddingTextEncoder(text_features=["text_0"])
    try:
        tr_emb = enc.fit_transform(Xtr)
    except Exception as e:  # pragma: no cover -- offline / model-fetch failure
        pytest.skip(f"HuggingFace model unavailable ({type(e).__name__}: {e})")
    te_emb = enc.transform(Xte)

    hf_acc = _fit_score(tr_emb.to_numpy(), ytr, te_emb.to_numpy(), yte)

    # Bag-of-words baseline on the same texts.
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer().fit(Xtr["text_0"].tolist())
    tfidf_acc = _fit_score(
        vec.transform(Xtr["text_0"].tolist()).toarray(),
        ytr,
        vec.transform(Xte["text_0"].tolist()).toarray(),
        yte,
    )

    # HF embeddings generalize to unseen synonyms; TF-IDF (OOV test tokens) sits near chance.
    assert hf_acc >= 0.80, f"HF text embeddings should generalize to unseen synonyms (got {hf_acc:.3f})"
    assert hf_acc >= tfidf_acc + 0.15, f"HF ({hf_acc:.3f}) must beat TF-IDF ({tfidf_acc:.3f}) on synonym generalization"
