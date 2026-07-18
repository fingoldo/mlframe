"""End-to-end: a real recurrent (LSTM/GRU) model in HYBRID mode fits + predicts with NATIVE learnable categorical embeddings on the TABULAR
(auxiliary) block, for every target type. Mirrors the flat-MLP ``test_estimator_categorical_embeddings_boundary`` but for the recurrent stack.

Each HYBRID test passes a string categorical tabular column + numerics + a per-sample sequence and ``cat_features=[...]`` at the wrapper ``fit``
boundary; the wrapper factorizes the tabular cats to int codes BEFORE the scaler/dataset (no ``dtype('O')`` error), the model prepends a
``CategoricalEmbedding`` on the aux block (BEFORE the shared output head), and predictions are finite + correctly shaped. SEQUENCE_ONLY has no
tabular block so the cat path no-ops cleanly. Also covers a fit->pickle->predict bit-identical round-trip, unseen-category-at-predict, the knob-off
fallback, and a biz_value check that learnable embeddings beat an ordinal cat-code on a non-monotone category->target signal.
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from mlframe.training.neural.recurrent import (
    RecurrentRegressorWrapper,
    RecurrentClassifierWrapper,
)
from mlframe.training.neural._recurrent_config import RecurrentConfig, InputMode, RNNType
from mlframe.training.neural._categorical_embeddings import CategoricalEmbedding

# n=256 / batch=32 / 3 epochs keeps total stepping batches comfortably > 1 (OneCycleLR divides by (end-start) steps; a handful of total steps
# trips a ZeroDivisionError inside torch's scheduler -- unrelated to cat embeddings, just degenerate-tiny-data scheduler math).
_N = 256
_COLORS = ["red", "green", "blue", "yellow"]


def _cfg(input_mode=InputMode.HYBRID, num_classes=2, rnn_type=RNNType.GRU):
    """Cfg."""
    return RecurrentConfig(
        input_mode=input_mode,
        rnn_type=rnn_type,
        hidden_size=8,
        num_layers=1,
        bidirectional=False,
        use_attention=False,
        mlp_hidden_sizes=(8,),
        dropout=0.0,
        max_epochs=3,
        batch_size=32,
        accelerator="cpu",
        precision="32-true",
        num_workers=0,
        num_classes=num_classes,
        use_stratified_sampler=False,
    )


def _make_data(n=_N, seed=0):
    """Make data."""
    rng = np.random.default_rng(seed)
    cats = rng.choice(_COLORS, size=n)
    feats = pd.DataFrame(
        {
            "num_a": rng.normal(size=n).astype(np.float32),
            "color": cats,  # string categorical (object dtype)
            "num_b": rng.normal(size=n).astype(np.float32),
        }
    )
    seqs = [rng.normal(size=(int(rng.integers(3, 8)), 2)).astype(np.float32) for _ in range(n)]
    return feats, seqs, cats, rng


def test_regression_hybrid_native_cat_embeddings():
    """Regression hybrid native cat embeddings."""
    feats, seqs, cats, rng = _make_data()
    lut = {"red": 1.0, "green": -2.0, "blue": 5.0, "yellow": 0.0}
    y = np.array([lut[c] for c in cats], dtype=np.float32) + rng.normal(scale=0.05, size=len(cats)).astype(np.float32)
    reg = RecurrentRegressorWrapper(config=_cfg(), random_state=42)
    reg.fit(features=feats, labels=y, sequences=seqs, cat_features=["color"])
    assert reg._cat_cardinalities_ == [4]
    assert reg._n_cat_features_ == 1
    assert reg._cat_cols_ == ["color"]
    # The embedding is built on the aux block, BEFORE the shared MLPHead.
    assert isinstance(reg.model.aux_cat_embedding, CategoricalEmbedding)
    assert reg.model.mlp_head.input_size == reg.model.aux_cat_embedding.out_features + reg.model.config.hidden_size
    preds = np.asarray(reg.predict(features=feats, sequences=seqs))
    assert preds.shape == (len(cats),)
    assert np.all(np.isfinite(preds))


def test_binary_hybrid_native_cat_embeddings():
    """Binary hybrid native cat embeddings."""
    feats, seqs, cats, _rng = _make_data(seed=1)
    y = np.isin(cats, ["red", "blue"]).astype(np.int64)
    clf = RecurrentClassifierWrapper(config=_cfg(num_classes=2), random_state=42)
    clf.fit(features=feats, labels=y, sequences=seqs, cat_features=["color"])
    assert clf._cat_cardinalities_ == [4]
    assert isinstance(clf.model.aux_cat_embedding, CategoricalEmbedding)
    proba = np.asarray(clf.predict_proba(features=feats, sequences=seqs))
    preds = np.asarray(clf.predict(features=feats, sequences=seqs))
    assert proba.shape == (len(cats), 2)
    assert preds.shape == (len(cats),)
    assert np.all(np.isfinite(proba))


def test_multiclass_hybrid_native_cat_embeddings():
    """Multiclass hybrid native cat embeddings."""
    feats, seqs, cats, _rng = _make_data(seed=2)
    mapping = {"red": 0, "green": 1, "blue": 2, "yellow": 0}
    y = np.array([mapping[c] for c in cats], dtype=np.int64)
    clf = RecurrentClassifierWrapper(config=_cfg(num_classes=3), random_state=42)
    clf.fit(features=feats, labels=y, sequences=seqs, cat_features=["color"])
    assert clf._cat_cardinalities_ == [4]
    proba = np.asarray(clf.predict_proba(features=feats, sequences=seqs))
    assert proba.shape == (len(cats), 3)
    assert np.all(np.isfinite(proba))


def test_sequence_only_no_tabular_cats_trains_identically():
    # SEQUENCE_ONLY has NO tabular block; the cat factorizer must no-op (cardinalities None) and the model builds no aux embedding. Training +
    # prediction proceed exactly as without the feature.
    """Sequence only no tabular cats trains identically."""
    _feats, seqs, _cats, _rng = _make_data(seed=3)
    y = np.array([float(s[:, 0].mean()) for s in seqs], dtype=np.float32)
    reg = RecurrentRegressorWrapper(config=_cfg(input_mode=InputMode.SEQUENCE_ONLY), random_state=42)
    reg.fit(features=None, labels=y, sequences=seqs)  # no features, no cat_features
    assert reg._cat_cardinalities_ is None
    assert reg._n_cat_features_ == 0
    assert reg.model.aux_cat_embedding is None
    preds = np.asarray(reg.predict(sequences=seqs))
    assert preds.shape == (len(seqs),)
    assert np.all(np.isfinite(preds))


def test_sequence_only_ignores_cat_features_passed_by_caller():
    # Even if the orchestrator threads cat_features for a SEQUENCE_ONLY member (features=None), the wrapper must no-op cleanly -- no aux block.
    """Sequence only ignores cat features passed by caller."""
    _feats, seqs, _cats, _rng = _make_data(seed=4)
    y = np.array([float(s[:, 0].mean()) for s in seqs], dtype=np.float32)
    reg = RecurrentRegressorWrapper(config=_cfg(input_mode=InputMode.SEQUENCE_ONLY), random_state=42)
    reg.fit(features=None, labels=y, sequences=seqs, cat_features=["color"])
    assert reg._cat_cardinalities_ is None
    preds = np.asarray(reg.predict(sequences=seqs))
    assert np.all(np.isfinite(preds))


def test_fit_pickle_predict_round_trip_bit_identical():
    """Fit pickle predict round trip bit identical."""
    feats, seqs, cats, _rng = _make_data(seed=7)
    y = (feats["num_a"].to_numpy() + np.isin(cats, ["red"]).astype(np.float32) * 3.0).astype(np.float32)
    reg = RecurrentRegressorWrapper(config=_cfg(), random_state=42)
    reg.fit(features=feats, labels=y, sequences=seqs, cat_features=["color"])
    p1 = np.asarray(reg.predict(features=feats, sequences=seqs))
    restored = pickle.loads(pickle.dumps(reg))  # nosec B301 -- round-trip of a locally-created, trusted object
    p2 = np.asarray(restored.predict(features=feats, sequences=seqs))
    assert np.allclose(p1, p2, atol=1e-5)
    assert isinstance(restored._cat_code_maps_, dict)
    assert isinstance(restored._cat_cardinalities_, list)
    assert restored._cat_cardinalities_ == [4]


def test_unseen_category_at_predict_maps_to_reserved_code():
    """Unseen category at predict maps to reserved code."""
    feats, seqs, cats, _rng = _make_data(seed=11)
    y = feats["num_a"].to_numpy().astype(np.float32)
    reg = RecurrentRegressorWrapper(config=_cfg(), random_state=42)
    reg.fit(features=feats, labels=y, sequences=seqs, cat_features=["color"])
    feats_new = feats.copy()
    feats_new.loc[feats_new.index[0], "color"] = "MAGENTA_never_seen"
    preds = np.asarray(reg.predict(features=feats_new, sequences=seqs))
    assert preds.shape == (len(cats),)
    assert np.all(np.isfinite(preds))


def test_knob_off_disables_factorization():
    # use_learnable_cat_embeddings=False -> the wrapper does NOT factorize (cardinalities None). The caller is then responsible for upstream
    # encoding; we pass an all-numeric frame so the fit completes.
    """Knob off disables factorization."""
    rng = np.random.default_rng(5)
    n = _N
    feats = pd.DataFrame(
        {
            "cat_code": rng.integers(0, 3, size=n).astype(np.float32),
            "num_0": rng.normal(size=n).astype(np.float32),
        }
    )
    seqs = [rng.normal(size=(int(rng.integers(3, 8)), 2)).astype(np.float32) for _ in range(n)]
    y = feats["num_0"].to_numpy().astype(np.float32)
    reg = RecurrentRegressorWrapper(config=_cfg(), random_state=42, use_learnable_cat_embeddings=False)
    reg.fit(features=feats, labels=y, sequences=seqs, cat_features=["cat_code"])
    assert reg._cat_cardinalities_ is None
    assert reg.model.aux_cat_embedding is None
    preds = np.asarray(reg.predict(features=feats, sequences=seqs))
    assert preds.shape == (n,)


def test_biz_value_learnable_embeddings_beat_ordinal_on_nonmonotone_signal():
    """Learnable per-cat embeddings recover a NON-MONOTONE category->target mapping that a single ordinal cat-code (treated as a numeric feature)
    cannot express linearly. We fit two HYBRID regressors on identical data: one with learnable cat embeddings (cat_features=["color"]) and one
    where the same cat is fed as a raw ordinal float code (no cat_features). The embedding fit must achieve a materially lower train RMSE on the
    pure-identity target. Floor is loose (1.10x) so seed noise doesn't trip it but a regression that disables the embedding (-> ordinal-equivalent)
    is caught.
    """
    rng = np.random.default_rng(123)
    n = _N
    cats = rng.choice(_COLORS, size=n)
    # Strongly non-monotone in the factorize order: code 0->red->+4, 1->green->-4, 2->blue->+4, 3->yellow->-4 (zig-zag, not linearly separable by
    # the ordinal code). The sequence + numerics carry no target signal here, so all discriminative power is in the cat identity.
    lut = {"red": 4.0, "green": -4.0, "blue": 4.0, "yellow": -4.0}
    y = np.array([lut[c] for c in cats], dtype=np.float32)
    seqs = [rng.normal(size=(5, 2)).astype(np.float32) for _ in range(n)]

    feats_emb = pd.DataFrame(
        {
            "num_a": rng.normal(scale=0.01, size=n).astype(np.float32),
            "color": cats,
        }
    )
    # Ordinal baseline: the SAME factorize codes, fed as a plain numeric column (no cat_features -> no embedding).
    codes, _uniques = pd.factorize(pd.Series(cats), sort=False)
    feats_ord = pd.DataFrame(
        {
            "num_a": feats_emb["num_a"].to_numpy(),
            "color_code": codes.astype(np.float32),
        }
    )

    cfg_emb = _cfg()
    cfg_emb.max_epochs = 40
    reg_emb = RecurrentRegressorWrapper(config=cfg_emb, random_state=42)
    reg_emb.fit(features=feats_emb, labels=y, sequences=seqs, cat_features=["color"])
    pred_emb = np.asarray(reg_emb.predict(features=feats_emb, sequences=seqs))
    rmse_emb = float(np.sqrt(np.mean((pred_emb - y) ** 2)))

    cfg_ord = _cfg()
    cfg_ord.max_epochs = 40
    reg_ord = RecurrentRegressorWrapper(config=cfg_ord, random_state=42)
    reg_ord.fit(features=feats_ord, labels=y, sequences=seqs)  # ordinal code, no embedding
    pred_ord = np.asarray(reg_ord.predict(features=feats_ord, sequences=seqs))
    rmse_ord = float(np.sqrt(np.mean((pred_ord - y) ** 2)))

    assert np.all(np.isfinite(pred_emb)) and np.all(np.isfinite(pred_ord))
    assert reg_emb._cat_cardinalities_ == [4]
    assert reg_ord._cat_cardinalities_ is None
    assert (
        rmse_emb < rmse_ord / 1.10
    ), f"learnable embeddings (RMSE {rmse_emb:.4f}) should beat ordinal cat-code (RMSE {rmse_ord:.4f}) by >=1.10x on the non-monotone signal"
