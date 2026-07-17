"""End-to-end: a real PyTorch-Lightning MLP fits + predicts with NATIVE learnable categorical embeddings via the estimator fit/predict boundary
(``_factorize_cats_fit`` / ``_apply_cat_codes`` + the ``CategoricalEmbedding`` hook in ``generate_mlp``), for EVERY target type.

Each test passes a string categorical column + numerics and ``cat_features=[...]`` via fit_params; the estimator must factorize the cats to int
codes BEFORE its NaN/inf validator (no ``dtype('O')`` error) and produce finite predictions of the right shape. Also covers the no-cat regression
(the hook is a no-op when no ``cat_features`` are named) and a fit->pickle->unpickle->predict round-trip.
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from mlframe.training.neural import (
    MLPTorchModel,
    PytorchLightningRegressor,
    PytorchLightningClassifier,
    TorchDataModule,
)


def _common(labels_dtype, loss_fn):
    return dict(
        model_class=MLPTorchModel,
        model_params={"loss_fn": loss_fn, "learning_rate": 1e-3},
        network_params={"nlayers": 2, "first_layer_num_neurons": 16, "dropout_prob": 0.0, "activation_function": torch.nn.ReLU},
        datamodule_class=TorchDataModule,
        datamodule_params={
            "read_fcn": None,
            "data_placement_device": None,
            "features_dtype": torch.float32,
            "labels_dtype": labels_dtype,
            "dataloader_params": {"batch_size": 16, "num_workers": 0},
        },
        trainer_params={
            "max_epochs": 1,
            "enable_model_summary": False,
            "default_root_dir": None,
            "log_every_n_steps": 1,
            "devices": 1,
            "logger": False,
            "accelerator": "cpu",
        },
    )


def _make_frame(n=64, seed=0):
    rng = np.random.default_rng(seed)
    cats = rng.choice(["red", "green", "blue", "yellow"], size=n)
    X = pd.DataFrame(
        {
            "color": cats,  # string categorical
            "num_0": rng.normal(size=n).astype(np.float32),
            "num_1": rng.normal(size=n).astype(np.float32),
        }
    )
    return X, cats, rng


def test_regression_native_cat_embeddings():
    X, cats, rng = _make_frame()
    lut = {"red": 1.0, "green": -2.0, "blue": 5.0, "yellow": 0.0}
    y = np.array([lut[c] for c in cats], dtype=np.float32) + rng.normal(scale=0.1, size=len(cats)).astype(np.float32)
    reg = PytorchLightningRegressor(**_common(torch.float32, torch.nn.MSELoss()))
    reg.fit(X, y, cat_features=["color"])
    assert reg._cat_cardinalities_ == [4]
    assert reg._n_cat_features_ == 1
    preds = np.asarray(reg.predict(X))
    assert preds.shape == (len(cats),)
    assert np.all(np.isfinite(preds))


def test_binary_classification_native_cat_embeddings():
    X, cats, _rng = _make_frame()
    y = (np.isin(cats, ["red", "blue"])).astype(np.int64)
    clf = PytorchLightningClassifier(**_common(torch.int64, torch.nn.CrossEntropyLoss()))
    clf.fit(X, y, cat_features=["color"])
    assert clf._cat_cardinalities_ == [4]
    preds = np.asarray(clf.predict(X))
    proba = np.asarray(clf.predict_proba(X))
    assert preds.shape == (len(cats),)
    assert proba.shape == (len(cats), 2)
    assert np.all(np.isfinite(proba))


def test_multiclass_classification_native_cat_embeddings():
    X, cats, _rng = _make_frame()
    mapping = {"red": 0, "green": 1, "blue": 2, "yellow": 0}
    y = np.array([mapping[c] for c in cats], dtype=np.int64)
    clf = PytorchLightningClassifier(**_common(torch.int64, torch.nn.CrossEntropyLoss()))
    clf.fit(X, y, cat_features=["color"])
    proba = np.asarray(clf.predict_proba(X))
    assert proba.shape == (len(cats), 3)
    assert np.all(np.isfinite(proba))


def test_multilabel_classification_native_cat_embeddings():
    X, cats, _rng = _make_frame()
    y = np.column_stack(
        [
            np.isin(cats, ["red", "blue"]).astype(np.int64),
            np.isin(cats, ["green", "blue"]).astype(np.int64),
            (X["num_0"].to_numpy() > 0).astype(np.int64),
        ]
    )
    clf = PytorchLightningClassifier(**_common(torch.float32, torch.nn.functional.binary_cross_entropy_with_logits))
    clf.fit(X, y, cat_features=["color"])
    assert clf._cat_cardinalities_ == [4]
    proba = np.asarray(clf.predict_proba(X))
    assert proba.shape == (len(cats), 3)
    assert np.all(np.isfinite(proba))
    preds = np.asarray(clf.predict(X))
    assert preds.shape == (len(cats), 3)


def test_no_cat_features_trains_identically_hook_noop():
    # No cat_features named -> the factorizer is a no-op (cardinalities stays None), the generate_mlp hook is skipped, and the all-numeric
    # frame trains + predicts exactly as before. (object-dtype-free frame so the NaN/inf validator is happy.)
    rng = np.random.default_rng(3)
    n = 64
    X = pd.DataFrame({"num_0": rng.normal(size=n).astype(np.float32), "num_1": rng.normal(size=n).astype(np.float32)})
    y = (X["num_0"] * 2.0).to_numpy().astype(np.float32)
    reg = PytorchLightningRegressor(**_common(torch.float32, torch.nn.MSELoss()))
    reg.fit(X, y)  # no cat_features
    assert reg._cat_cardinalities_ is None
    assert reg._n_cat_features_ == 0
    preds = np.asarray(reg.predict(X))
    assert preds.shape == (n,)
    assert np.all(np.isfinite(preds))


def test_mlp_auto_factorizes_raw_object_cat_without_explicit_cat_features():
    """Regression (bug-hunt): learnable cat embeddings on -> strategy skips the CatBoostEncoder, so a raw object categorical reaches the MLP.
    When the caller does NOT thread cat_features (the suite path), the fit boundary must AUTO-DETECT + factorize the non-numeric column itself;
    pre-fix it fell through to _validate_no_nan_inf and raised "X has dtype('O'); PytorchLightningEstimator requires numeric dtype" -- the
    single dominant fuzz failure class (44 combos) at 1k rows."""
    X, cats, rng = _make_frame()  # 'color' object col + 2 numeric columns; NO cat_features threaded
    lut = {"red": 1.0, "green": -2.0, "blue": 5.0, "yellow": 0.0}
    y = np.array([lut[c] for c in cats], dtype=np.float32) + rng.normal(scale=0.1, size=len(cats)).astype(np.float32)
    reg = PytorchLightningRegressor(**_common(torch.float32, torch.nn.MSELoss()))
    reg.fit(X, y)  # no cat_features -> the raw object 'color' column must be auto-factorized, not crash
    assert reg._n_cat_features_ == 1  # auto-detected the object column
    assert reg._cat_cardinalities_ == [4]
    preds = np.asarray(reg.predict(X))
    assert preds.shape == (len(cats),) and np.all(np.isfinite(preds))


def test_fit_pickle_unpickle_predict_round_trip():
    X, cats, _rng = _make_frame(seed=7)
    y = (X["num_0"] + np.isin(cats, ["red"]).astype(np.float32) * 3.0).to_numpy().astype(np.float32)
    reg = PytorchLightningRegressor(**_common(torch.float32, torch.nn.MSELoss()), random_state=42)
    reg.fit(X, y, cat_features=["color"])
    p1 = np.asarray(reg.predict(X))
    restored = pickle.loads(pickle.dumps(reg))  # nosec B301 -- round-trip of a locally-created, trusted object
    p2 = np.asarray(restored.predict(X))
    assert np.allclose(p1, p2, atol=1e-5)
    # Stored maps/cardinalities are plain dict/list (picklable).
    assert isinstance(restored._cat_code_maps_, dict)
    assert isinstance(restored._cat_cardinalities_, list)


def test_unseen_category_at_predict_maps_to_reserved_code():
    X, cats, _rng = _make_frame(seed=11)
    y = (X["num_0"]).to_numpy().astype(np.float32)
    reg = PytorchLightningRegressor(**_common(torch.float32, torch.nn.MSELoss()))
    reg.fit(X, y, cat_features=["color"])
    X_new = X.copy()
    X_new.loc[X_new.index[0], "color"] = "MAGENTA_never_seen"
    preds = np.asarray(reg.predict(X_new))
    assert preds.shape == (len(cats),)
    assert np.all(np.isfinite(preds))


def test_knob_off_disables_factorization():
    # use_learnable_cat_embeddings=False -> the estimator does NOT factorize; with a raw string cat column present the NaN/inf validator (which
    # rejects object dtype on X) is bypassed by allow_object only for y, so the string cat would reach the network. We assert the factorizer is
    # a no-op (cardinalities None); the user is then responsible for upstream encoding. We pass an all-numeric frame so the fit still completes.
    rng = np.random.default_rng(5)
    n = 48
    X = pd.DataFrame({"cat_code": rng.integers(0, 3, size=n).astype(np.float32), "num_0": rng.normal(size=n).astype(np.float32)})
    y = X["num_0"].to_numpy().astype(np.float32)
    reg = PytorchLightningRegressor(**_common(torch.float32, torch.nn.MSELoss()), use_learnable_cat_embeddings=False)
    reg.fit(X, y, cat_features=["cat_code"])
    assert reg._cat_cardinalities_ is None  # knob off -> no factorization
    preds = np.asarray(reg.predict(X))
    assert preds.shape == (n,)


def test_categorical_dtype_cat_predict_routes_unseen_without_setitem_error():
    """Regression (bug-hunt c0005, mlp): a pandas CATEGORICAL-dtype cat column (not object/string -- the dtype the fuzz frame builder and real
    callers use). The apply path ``_apply_cat_codes`` (predict, and the eval_set at fit) did ``X[col].map(mapping)``, which on a Categorical
    RETURNS a Categorical, so ``.fillna(reserved_code)`` then raised "Cannot setitem on a Categorical with a new category (card)" -- a cardinality-7
    Categorical made the reserved code 7.0 a new category. The original boundary tests only used object/string cats, so they missed this. Fit +
    predict on an int Categorical-dtype column, including an all-UNSEEN level at predict that must route to the reserved code, must not raise."""
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(
        {
            "c": pd.Categorical(rng.integers(0, 7, size=n)),  # int Categorical DTYPE, cardinality up to 7 (reserved code == 7.0)
            "num_0": rng.normal(size=n).astype(np.float32),
        }
    )
    y = rng.normal(size=n).astype(np.float32)
    reg = PytorchLightningRegressor(**_common(torch.float32, torch.nn.MSELoss()))
    reg.fit(X, y, cat_features=["c"])
    preds = np.asarray(reg.predict(X))  # predict -> _apply_cat_codes on the Categorical column
    assert preds.shape == (n,) and np.all(np.isfinite(preds))
    X_new = X.copy()
    X_new["c"] = pd.Categorical(np.full(n, 999))  # an all-UNSEEN level -> every cell routes to the reserved unknown code
    preds2 = np.asarray(reg.predict(X_new))
    assert preds2.shape == (n,) and np.all(np.isfinite(preds2))
