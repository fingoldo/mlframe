"""Target-type combination coverage against the training entrypoint.

Parametrized tests over (task_type, target_dtype) combos that ``train_mlframe_models_suite``
must accept without raising, plus a behavioural check per combo: regression float target -> a
regressor head under the REGRESSION key; integer / string multiclass target -> a classifier head
under MULTICLASS_CLASSIFICATION whose fitted estimator inferred K classes; binary -> a binary
classifier head. Behavioural only -- no source inspection.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.training import TargetTypes, train_mlframe_models_suite
from tests.training.shared import SimpleFeaturesAndTargetsExtractor

pytest.importorskip("lightgbm")


def _make_xy(y: np.ndarray, *, n_features: int = 4, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame with a signal-bearing feature block + the supplied target column."""
    n = len(y)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y
    return df


def _make_target(task_label: str, n: int) -> np.ndarray:
    """Class-balanced / signal-free targets sized to ``n`` for each task label."""
    rng = np.random.default_rng(123)
    if task_label == "binary":
        return (np.arange(n) % 2).astype(np.int64)
    if task_label == "binary_float":
        return (np.arange(n) % 2).astype(np.float64)
    if task_label == "multiclass":
        return (np.arange(n) % 3).astype(np.int64)
    if task_label == "multiclass_str":
        return np.array(["a", "b", "c"], dtype=object)[np.arange(n) % 3]
    if task_label == "regression":
        return (rng.standard_normal(n) * 2.0 + 5.0).astype(np.float64)
    raise KeyError(task_label)


# (task_label, target_type override or None for default regression/binary handling).
# ``multiclass_str`` feeds a raw string-labelled multiclass column through the extractor: the suite
# factorizes it to integer codes (leakage-safe) so it behaves identically to the int-labelled path,
# and the numeric-only regression-refit collapse guard no longer runs np.isfinite over strings.
TASK_TARGET_COMBOS = [
    ("binary", None),
    ("binary_float", None),
    ("multiclass", TargetTypes.MULTICLASS_CLASSIFICATION),
    ("multiclass_str", TargetTypes.MULTICLASS_CLASSIFICATION),
    ("regression", None),
]


def _train_combo(task_label: str, target_type=None, n: int = 600):
    y = _make_target(task_label, n)
    df = _make_xy(y)
    regression = task_label == "regression"
    fte = SimpleFeaturesAndTargetsExtractor(
        target_column="target",
        regression=regression,
        target_type=target_type,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name=f"ttc_{task_label}",
            features_and_targets_extractor=fte,
            mlframe_models=["lgb"],
            use_mlframe_ensembles=False,
            verbose=0,
            hyperparams_config={"iterations": 10, "lgb_kwargs": {"device_type": "cpu"}},
        )
    return models, metadata


def _iter_fitted_models(models, target_type):
    """Yield the inner fitted estimators stored under ``target_type``."""
    target_models = models.get(target_type, {})
    for _name, ns_list in target_models.items():
        for ns in ns_list:
            inner = getattr(ns, "model", None)
            if inner is not None:
                yield inner


@pytest.mark.parametrize(
    "task_label,target_type", TASK_TARGET_COMBOS, ids=[c[0] for c in TASK_TARGET_COMBOS]
)
def test_target_type_combo_trains_without_raising(task_label, target_type):
    """Each combo must return a (models, metadata) pair and fit at least one head for its
    inferred target type -- not merely construct a synthetic frame."""
    models, metadata = _train_combo(task_label, target_type)
    assert isinstance(models, dict) and models, f"{task_label}: empty models dict"
    assert isinstance(metadata, dict), f"{task_label}: metadata is not a dict"

    if task_label == "regression":
        expected_key = TargetTypes.REGRESSION
    elif task_label in ("multiclass", "multiclass_str"):
        expected_key = TargetTypes.MULTICLASS_CLASSIFICATION
    else:
        expected_key = TargetTypes.BINARY_CLASSIFICATION
    assert expected_key in models, (
        f"{task_label}: expected a head under {expected_key!r}; got keys {list(models)}"
    )
    assert any(True for _ in _iter_fitted_models(models, expected_key)), (
        f"{task_label}: no fitted estimator under {expected_key!r}"
    )


def _assert_classifier_inferred_k(models, target_type, expected_k: int, label: str) -> None:
    fitted = list(_iter_fitted_models(models, target_type))
    assert fitted, f"{label}: no fitted classifier head"
    saw_classes = False
    for est in fitted:
        classes = getattr(est, "classes_", None)
        if classes is not None:
            saw_classes = True
            assert len(classes) == expected_k, (
                f"{label}: classifier inferred {len(classes)} classes, expected {expected_k}"
            )
    assert saw_classes, f"{label}: fitted head exposed no classes_ (n_classes not inferred)"


def test_multiclass_int_infers_three_classes():
    models, _ = _train_combo("multiclass", TargetTypes.MULTICLASS_CLASSIFICATION)
    _assert_classifier_inferred_k(models, TargetTypes.MULTICLASS_CLASSIFICATION, 3, "multiclass")


def test_multiclass_str_trains_and_infers_three_classes():
    """A raw string-labelled multiclass target ("a"/"b"/"c") must train (no np.isfinite-on-string
    crash in the regression-refit guard) and infer K=3, with the string->code mapping stamped for
    predict-time inverse."""
    models, metadata = _train_combo("multiclass_str", TargetTypes.MULTICLASS_CLASSIFICATION)
    _assert_classifier_inferred_k(models, TargetTypes.MULTICLASS_CLASSIFICATION, 3, "multiclass_str")
    label_classes = metadata.get("target_label_classes", {}).get("target")
    assert label_classes is not None, "string multiclass target did not stamp target_label_classes for inverse mapping"
    assert list(label_classes) == ["a", "b", "c"], f"expected sorted string classes; got {label_classes}"


def test_regression_head_is_not_a_classifier():
    """The regression head exposes a regressor surface (predict, no classes_)."""
    models, _ = _train_combo("regression")
    fitted = list(_iter_fitted_models(models, TargetTypes.REGRESSION))
    assert fitted, "regression: no fitted estimator"
    for est in fitted:
        assert hasattr(est, "predict"), "regression head lacks predict()"
        assert getattr(est, "classes_", None) is None, "regression head unexpectedly has classes_"
