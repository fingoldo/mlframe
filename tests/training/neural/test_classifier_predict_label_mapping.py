"""Regression test for F-01 (mlp audit 2026-05-30).

``PytorchLightningClassifier.predict`` must return ENTRIES of ``self.classes_``,
not argmax INDICES. This is the sklearn convention every other classifier in
the repo follows (``mlframe.estimators.custom.py:259/281/325`` do
``self.classes_[np.argmax(self.predict_proba(X), axis=1)]``).

Existing tests in this directory only ever exercise dense ``{0, 1, ..., K-1}``
labels (via ``sklearn.datasets.make_classification`` which emits ``y in
{0, 1, 2}``) -- in that special case argmax indices coincide with class
labels and the bug is invisible. This test uses ``y in {10, 20}`` to
DECOUPLE indices from labels: argmax returns ``0`` / ``1`` while sklearn
convention requires ``10`` / ``20``.

Pre-fix: ``set(preds) == {0, 1}`` -> assertion fails.
Post-fix (``return self.classes_[np.argmax(proba, axis=1)]``): ``set(preds)
== {10, 20}`` -> assertion passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Project root on path so tests can be invoked directly without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.neural import (
    MLPTorchModel,
    PytorchLightningClassifier,
    TorchDataModule,
)


@pytest.fixture
def binary_data_nondense_labels():
    """Binary classification with labels in ``{10, 20}`` -- argmax indices
    ``{0, 1}`` cannot equal the labels, so any code path that confuses the
    two surfaces immediately."""
    X, y01 = make_classification(
        n_samples=160,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    # Map {0, 1} -> {10, 20}: argmax(softmax) can only emit 0/1, never 10/20.
    y = np.where(y01 == 0, 10, 20).astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    return {
        "X_train": X_train.astype(np.float32),
        "y_train": y_train,
        "X_test": X_test.astype(np.float32),
        "y_test": y_test,
    }


@pytest.fixture
def tiny_classifier_params():
    """Cheapest viable PytorchLightningClassifier config: 1 epoch, no logger,
    no progress bar, single device."""
    network_params = {
        "nlayers": 1,
        "first_layer_num_neurons": 16,
        "dropout_prob": 0.0,
        "inputs_dropout_prob": 0.0,
        "use_layernorm": False,
        "activation_function": torch.nn.ReLU,
    }
    model_params = {
        "loss_fn": torch.nn.CrossEntropyLoss(),
        "learning_rate": 1e-2,
    }
    datamodule_params = {
        "features_dtype": torch.float32,
        "labels_dtype": torch.int64,
        "dataloader_params": {"batch_size": 32, "num_workers": 0},
    }
    trainer_params = {
        "max_epochs": 1,
        "enable_model_summary": False,
        "enable_progress_bar": False,
        "log_every_n_steps": 1,
        "devices": 1,
        "logger": False,
        "accelerator": "cpu",
    }
    return {
        "model_class": MLPTorchModel,
        "model_params": model_params,
        "network_params": network_params,
        "datamodule_class": TorchDataModule,
        "datamodule_params": datamodule_params,
        "trainer_params": trainer_params,
    }


def test_predict_returns_class_labels_not_argmax_indices(
    binary_data_nondense_labels,
    tiny_classifier_params,
):
    """F-01 regression: ``predict`` output must be a subset of ``classes_``.

    Pre-fix base.py:990 returned argmax indices ``{0, 1}``; with labels
    ``{10, 20}`` those are NOT in ``classes_`` so the assertion fails.
    """
    clf = PytorchLightningClassifier(**tiny_classifier_params)
    clf.fit(binary_data_nondense_labels["X_train"], binary_data_nondense_labels["y_train"])

    # classes_ is the sklearn-canonical sorted-unique-of-y_train ndarray.
    assert hasattr(clf, "classes_")
    assert set(clf.classes_.tolist()) == {10, 20}

    preds = clf.predict(binary_data_nondense_labels["X_test"])
    pred_set = set(np.asarray(preds).tolist())

    assert pred_set.issubset(set(clf.classes_.tolist())), (
        f"PytorchLightningClassifier.predict must return entries of "
        f"classes_ ({clf.classes_.tolist()}), but got values: {sorted(pred_set)}. "
        "Pre-fix base.py:990 returned argmax indices instead of "
        "classes_[argmax]; this is F-01 in the 2026-05-30 mlp audit."
    )


def test_predict_consistent_with_predict_proba_argmax(
    binary_data_nondense_labels,
    tiny_classifier_params,
):
    """``predict`` and ``classes_[predict_proba.argmax]`` must agree
    sample-by-sample. Holds independently of model quality (works even on a
    barely-trained 1-epoch MLP) because it only checks INTERNAL CONSISTENCY
    between the two public methods of the same estimator instance.
    """
    clf = PytorchLightningClassifier(**tiny_classifier_params)
    clf.fit(binary_data_nondense_labels["X_train"], binary_data_nondense_labels["y_train"])

    proba = clf.predict_proba(binary_data_nondense_labels["X_test"])
    expected = clf.classes_[np.argmax(proba, axis=1)]
    actual = clf.predict(binary_data_nondense_labels["X_test"])

    np.testing.assert_array_equal(actual, expected)


def test_accuracy_score_against_predict_matches_manual_remap(
    binary_data_nondense_labels,
    tiny_classifier_params,
):
    """``sklearn.metrics.accuracy_score(y_test, clf.predict(X_test))``
    must equal the same metric computed on the manually re-mapped argmax.
    Pre-fix this diverges silently: ``accuracy_score(y={10,20},
    preds={0,1})`` returns 0.0 regardless of how good the classifier is.
    """
    clf = PytorchLightningClassifier(**tiny_classifier_params)
    clf.fit(binary_data_nondense_labels["X_train"], binary_data_nondense_labels["y_train"])

    proba = clf.predict_proba(binary_data_nondense_labels["X_test"])
    manual = clf.classes_[np.argmax(proba, axis=1)]

    acc_predict = accuracy_score(binary_data_nondense_labels["y_test"], clf.predict(binary_data_nondense_labels["X_test"]))
    acc_manual = accuracy_score(binary_data_nondense_labels["y_test"], manual)

    assert acc_predict == acc_manual, (
        f"accuracy via predict={acc_predict} diverges from manual "
        f"classes_[argmax]={acc_manual}; F-01 means predict returns "
        f"indices {{0, 1}} while y_test is {{10, 20}}, so the sklearn "
        "metric collapses to 0."
    )


# =============================================================================
# Hand-picked edge-case fixtures + tests.
# Each fixture builds a dataset whose label set DECOUPLES from argmax indices,
# so any leftover argmax-as-label code path surfaces.
# =============================================================================


def _build_binary(y0_label, y1_label, dtype=None, n=120):
    """Build binary classification data with custom label values."""
    X, y01 = make_classification(
        n_samples=n,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=7,
    )
    y_dtype = dtype if dtype is not None else object
    y = np.empty(y01.shape, dtype=y_dtype)
    y[y01 == 0] = y0_label
    y[y01 == 1] = y1_label
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y01)
    return {
        "X_train": Xtr.astype(np.float32),
        "y_train": ytr,
        "X_test": Xte.astype(np.float32),
        "y_test": yte,
    }


def test_fit_and_predict_with_string_labels(tiny_classifier_params):
    """Sklearn-canonical: y in {"low", "high"} -> classes_ sorted to
    ["high","low"]; predict returns the strings, not 0/1."""
    data = _build_binary("low", "high")
    clf = PytorchLightningClassifier(**tiny_classifier_params)
    clf.fit(data["X_train"], data["y_train"])
    assert set(clf.classes_.tolist()) == {"low", "high"}
    preds = clf.predict(data["X_test"])
    assert set(np.asarray(preds).tolist()).issubset({"low", "high"})


def test_fit_and_predict_with_boolean_labels(tiny_classifier_params):
    """Bool labels {False, True}. Bug here is invisible to accuracy_score
    (Python True==1) but still wrong: predict should emit np.bool_, not
    integer indices."""
    data = _build_binary(False, True, dtype=bool)
    clf = PytorchLightningClassifier(**tiny_classifier_params)
    clf.fit(data["X_train"], data["y_train"])
    assert set(clf.classes_.tolist()) == {False, True}
    preds = np.asarray(clf.predict(data["X_test"]))
    assert preds.dtype == np.dtype(bool), f"expected bool dtype, got {preds.dtype}"
    assert set(preds.tolist()).issubset({False, True})


def test_fit_and_predict_with_negative_int_labels(tiny_classifier_params):
    """Classical binary convention: y in {-1, +1}. Argmax indices {0,1}
    cannot equal these labels."""
    data = _build_binary(-1, 1, dtype=np.int64)
    clf = PytorchLightningClassifier(**tiny_classifier_params)
    clf.fit(data["X_train"], data["y_train"])
    assert set(clf.classes_.tolist()) == {-1, 1}
    preds = np.asarray(clf.predict(data["X_test"]))
    assert set(preds.tolist()).issubset({-1, 1})


def test_fit_and_predict_with_multiclass_non_dense_labels(tiny_classifier_params):
    """Multiclass K=3 with labels {100, 200, 300}. Verifies the fix
    generalises past binary -- the same code path handles K classes."""
    X, y012 = make_classification(
        n_samples=180,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=11,
    )
    y = np.where(y012 == 0, 100, np.where(y012 == 1, 200, 300)).astype(np.int64)
    Xtr, Xte, ytr, _yte = train_test_split(X, y, test_size=0.3, random_state=11, stratify=y012)
    clf = PytorchLightningClassifier(**tiny_classifier_params)
    clf.fit(Xtr.astype(np.float32), ytr)
    assert set(clf.classes_.tolist()) == {100, 200, 300}
    preds = np.asarray(clf.predict(Xte.astype(np.float32)))
    assert set(preds.tolist()).issubset({100, 200, 300})
    # predict_proba columns must align with sorted classes_
    proba = clf.predict_proba(Xte.astype(np.float32))
    assert proba.shape == (Xte.shape[0], 3)
    np.testing.assert_array_equal(clf.classes_, np.array([100, 200, 300]))


def test_fit_with_pandas_series_y_train(tiny_classifier_params):
    """y delivered as pd.Series should produce the same classes_ / predict
    semantics as ndarray-y."""
    import pandas as pd

    data = _build_binary(10, 20, dtype=np.int64)
    y_series = pd.Series(data["y_train"], name="target")
    clf = PytorchLightningClassifier(**tiny_classifier_params)
    clf.fit(data["X_train"], y_series)
    assert set(clf.classes_.tolist()) == {10, 20}
    preds = clf.predict(data["X_test"])
    assert set(np.asarray(preds).tolist()).issubset({10, 20})


def test_fit_with_pandas_dataframe_single_col_y_train(tiny_classifier_params):
    """(N, 1) pd.DataFrame y must be ravel'd, not treated as multilabel."""
    import pandas as pd

    data = _build_binary(10, 20, dtype=np.int64)
    y_df = pd.DataFrame(data["y_train"], columns=["target"])
    clf = PytorchLightningClassifier(**tiny_classifier_params)
    clf.fit(data["X_train"], y_df)
    assert set(clf.classes_.tolist()) == {10, 20}
    preds = clf.predict(data["X_test"])
    assert set(np.asarray(preds).tolist()).issubset({10, 20})


def test_classes_attribute_is_sorted(tiny_classifier_params):
    """sklearn invariant: classes_ is sorted unique. LabelEncoder.fit
    produces a sorted classes_; the partial_fit branch that previously
    did ``np.asarray(classes)`` (NOT sorted) is now also routed through
    the encoder for consistency."""
    data = _build_binary(20, 10, dtype=np.int64)
    clf = PytorchLightningClassifier(**tiny_classifier_params)
    clf.fit(data["X_train"], data["y_train"])
    np.testing.assert_array_equal(clf.classes_, np.array([10, 20]))


def test_eval_set_with_nondense_labels_is_encoded(tiny_classifier_params):
    """Validation labels must use the SAME encoder. Pre-fix the val side
    also crashed with ``IndexError: Target N is out of bounds``."""
    data = _build_binary(10, 20, dtype=np.int64)
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        data["X_train"],
        data["y_train"],
        test_size=0.3,
        random_state=3,
        stratify=data["y_train"],
    )
    clf = PytorchLightningClassifier(**tiny_classifier_params)
    clf.fit(X_tr2, y_tr2, eval_set=(X_val, y_val))
    assert set(clf.classes_.tolist()) == {10, 20}
    preds = clf.predict(data["X_test"])
    assert set(np.asarray(preds).tolist()).issubset({10, 20})


def test_partial_fit_with_explicit_classes_universe(tiny_classifier_params):
    """partial_fit(classes=[10, 20]) must use the caller's full universe
    even if this batch only sees a subset. Otherwise a single-label batch
    would set num_classes=1 and the network output shape would be wrong."""
    data = _build_binary(10, 20, dtype=np.int64)
    mask = data["y_train"] == 10
    X_first = data["X_train"][mask]
    y_first = data["y_train"][mask]
    clf = PytorchLightningClassifier(**tiny_classifier_params)
    clf.partial_fit(X_first, y_first, classes=np.array([10, 20]))
    assert set(clf.classes_.tolist()) == {10, 20}
    preds = clf.predict(data["X_test"])
    assert set(np.asarray(preds).tolist()).issubset({10, 20})


# =============================================================================
# Hypothesis property tests.
# Hand-picked tests above pin specific scenarios for readability; these
# property tests sweep broader corners of label-space the human author
# wouldn't enumerate. Each example fits a 1-epoch MLP (~3 s); cap
# ``max_examples`` low to keep total runtime bounded.
# =============================================================================

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_HYP_SETTINGS = settings(
    max_examples=6,
    deadline=None,  # fitting a tiny MLP exceeds the default 200 ms deadline
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,  # tiny_classifier_params is function-scoped
        HealthCheck.too_slow,
    ],
)


@_HYP_SETTINGS
@given(
    label_pair=st.lists(
        st.integers(min_value=-1000, max_value=1000),
        min_size=2,
        max_size=2,
        unique=True,
    ),
)
def test_property_binary_int_labels_round_trip(tiny_classifier_params, label_pair):
    """For any 2-element int label set, predict outputs must lie in that
    set and classes_ must equal sorted(label_set)."""
    a, b = label_pair
    data = _build_binary(a, b, dtype=np.int64, n=80)
    clf = PytorchLightningClassifier(**tiny_classifier_params)
    clf.fit(data["X_train"], data["y_train"])
    np.testing.assert_array_equal(clf.classes_, np.sort(np.array([a, b])))
    preds = set(np.asarray(clf.predict(data["X_test"])).tolist())
    assert preds.issubset({a, b}), f"preds={sorted(preds)} not subset of {{{a},{b}}}; F-01 regression"


@_HYP_SETTINGS
@given(
    label_pair=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
            min_size=1,
            max_size=8,
        ),
        min_size=2,
        max_size=2,
        unique=True,
    ),
)
def test_property_binary_string_labels_round_trip(tiny_classifier_params, label_pair):
    """Any 2-element string label set: predict outputs lie in the set,
    classes_ is the sorted set."""
    a, b = label_pair
    data = _build_binary(a, b, n=80)
    clf = PytorchLightningClassifier(**tiny_classifier_params)
    clf.fit(data["X_train"], data["y_train"])
    assert set(clf.classes_.tolist()) == {a, b}
    preds = set(np.asarray(clf.predict(data["X_test"])).tolist())
    assert preds.issubset({a, b})


@_HYP_SETTINGS
@given(
    labels=st.lists(
        st.integers(min_value=-10_000, max_value=10_000),
        min_size=2,
        max_size=4,
        unique=True,
    ),
)
def test_property_multiclass_int_labels_round_trip(tiny_classifier_params, labels):
    """K-class (K in {2,3,4}) int label set: predict outputs lie in the
    set, classes_ is sorted, predict_proba has K columns."""
    K = len(labels)
    X, y_dense = make_classification(
        n_samples=60 * K,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=K,
        n_clusters_per_class=1,
        random_state=23,
    )
    label_arr = np.asarray(labels, dtype=np.int64)
    y = label_arr[y_dense]
    Xtr, Xte, ytr, _yte = train_test_split(X, y, test_size=0.3, random_state=23, stratify=y_dense)
    clf = PytorchLightningClassifier(**tiny_classifier_params)
    clf.fit(Xtr.astype(np.float32), ytr)
    np.testing.assert_array_equal(clf.classes_, np.sort(label_arr))
    preds = set(np.asarray(clf.predict(Xte.astype(np.float32))).tolist())
    assert preds.issubset(set(labels))
    proba = clf.predict_proba(Xte.astype(np.float32))
    assert proba.shape == (Xte.shape[0], K)


@_HYP_SETTINGS
@given(
    label_pair=st.lists(
        st.integers(min_value=-1000, max_value=1000),
        min_size=2,
        max_size=2,
        unique=True,
    ),
)
def test_property_predict_consistent_with_proba_argmax(tiny_classifier_params, label_pair):
    """Internal consistency: predict == classes_[argmax(predict_proba)]
    sample-by-sample, across the label-pair space."""
    a, b = label_pair
    data = _build_binary(a, b, dtype=np.int64, n=80)
    clf = PytorchLightningClassifier(**tiny_classifier_params)
    clf.fit(data["X_train"], data["y_train"])
    proba = clf.predict_proba(data["X_test"])
    expected = clf.classes_[np.argmax(proba, axis=1)]
    actual = clf.predict(data["X_test"])
    np.testing.assert_array_equal(actual, expected)
