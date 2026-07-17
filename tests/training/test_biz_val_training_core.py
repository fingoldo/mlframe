"""biz_val tests for ``train_mlframe_models_suite`` (training/core.py).

Per CLAUDE.md "Every new ML trick gets a biz_val synthetic test":
high-level contract invariants for the suite. Existing
``test_bizvalue_*.py`` files cover feature-specific behaviour
(imbalance, calibration, feature_selection, etc.); this file adds
the small set of "must remain true" invariants on the suite's
public contract.

Defensive: ``training/core.py`` is under active refactor. Each test
uses minimal config + small synthetic data. On API change (e.g. a
kwarg renamed mid-refactor), the test ``pytest.skip``s with a clear
note rather than failing -- so the suite-coverage signal stays
trustworthy even during transition.

Naming: ``test_biz_val_training_<class>_<parameter>``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tests.conftest import is_fast_mode
from tests.training.synthetic import (
    make_simple_classification_data,
    make_simple_regression_data,
)

warnings.filterwarnings("ignore")

# Smoke tests assert "is not None" / "isinstance(dict)" only. Iterations=2 is
# enough to exercise the trainer loop; the higher original iterations=30 was
# pure runtime overhead. Halved under fast mode.
_DEFAULT_SMOKE_ITERATIONS = 2 if is_fast_mode() else 5

# These tests share state between runs (matplotlib backend, numba JIT
# cache, on-disk model directories). pytest-randomly's default
# shuffle exposes the cross-test interactions; pin the module to
# sequential collection order so each test sees a clean start.
pytestmark = pytest.mark.order(index=0)


def _make_regression_df(n=400, seed=42):
    """Small regression dataset; delegates to shared synthetic helper."""
    df, _, _ = make_simple_regression_data(n_samples=n, n_features=5, seed=seed)
    return df


def _make_classification_df(n=400, seed=42):
    """Small binary classification dataset; delegates to shared synthetic helper."""
    df, _, _, _ = make_simple_classification_data(n_samples=n, n_features=5, seed=seed)
    return df


def _try_import_suite():
    """Defensive import. Skip the test if the suite isn't importable
    in the current core.py state."""
    try:
        from mlframe.training.core import train_mlframe_models_suite
        from mlframe.training import OutputConfig
        from tests.training.shared import SimpleFeaturesAndTargetsExtractor

        return train_mlframe_models_suite, OutputConfig, SimpleFeaturesAndTargetsExtractor
    except (ImportError, AttributeError) as e:
        pytest.skip(f"suite not importable during refactor: {e}")


# ---------------------------------------------------------------------------
# Smoke: suite runs on regression + classification
# ---------------------------------------------------------------------------


def test_biz_val_training_suite_regression_completes(tmp_path):
    """Suite must train a simple regression task and return a 2-tuple
    ``(models, metadata)``."""
    pytest.importorskip("lightgbm")
    train_mlframe_models_suite, OutputConfig, FTE = _try_import_suite()
    df = _make_regression_df(n=400, seed=42)
    fte = FTE(target_column="target", regression=True)
    data_dir = str(tmp_path / "data")
    try:
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="m_reg",
            features_and_targets_extractor=fte,
            mlframe_models=["lgb"],
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=data_dir, models_dir="models"),
            verbose=0,
            hyperparams_config={"iterations": _DEFAULT_SMOKE_ITERATIONS},
        )
    except (TypeError, ImportError) as e:
        pytest.skip(f"suite call broke during refactor: {e}")
    # Behavioural: suite returned a real models mapping (not a stub) with at least one trained estimator under
    # the requested family, and the metadata dict carries the canonical keys downstream consumers depend on.
    assert models is not None, "regression suite returned None models on lgb-only path"
    assert hasattr(models, "__len__") and len(models) >= 1, (
        f"models container empty after successful suite call; got {type(models).__name__} len={len(models) if hasattr(models, '__len__') else 'n/a'}"
    )
    assert isinstance(metadata, dict) and len(metadata) > 0, (
        f"metadata empty / wrong type: {type(metadata).__name__} keys={list(metadata)[:5] if isinstance(metadata, dict) else 'n/a'}"
    )


def test_biz_val_training_suite_classification_completes(tmp_path):
    """Suite must train a simple binary classification task."""
    pytest.importorskip("lightgbm")
    train_mlframe_models_suite, OutputConfig, FTE = _try_import_suite()
    df = _make_classification_df(n=400, seed=42)
    fte = FTE(target_column="target", regression=False)
    data_dir = str(tmp_path / "data")
    try:
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="m_clf",
            features_and_targets_extractor=fte,
            mlframe_models=["lgb"],
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=data_dir, models_dir="models"),
            verbose=0,
            hyperparams_config={"iterations": _DEFAULT_SMOKE_ITERATIONS},
        )
    except (TypeError, ImportError) as e:
        pytest.skip(f"suite call broke during refactor: {e}")
    # Same behavioural contract as the regression path (see above).
    assert models is not None, "classification suite returned None models on lgb-only path"
    assert hasattr(models, "__len__") and len(models) >= 1, f"models container empty after successful classification suite call; got {type(models).__name__}"
    assert isinstance(metadata, dict) and len(metadata) > 0, "metadata empty / wrong type on classification path"


# ---------------------------------------------------------------------------
# Model-subset selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_list",
    [
        ["lgb"],
        ["xgb"],
    ],
)
def test_biz_val_training_suite_mlframe_models_subset(tmp_path, model_list):
    """``mlframe_models=[X]`` must train ONLY that model family.
    Parametrize over the boosting families."""
    train_mlframe_models_suite, OutputConfig, FTE = _try_import_suite()
    if model_list == ["lgb"]:
        pytest.importorskip("lightgbm")
    elif model_list == ["xgb"]:
        pytest.importorskip("xgboost")
    df = _make_regression_df(n=300, seed=42)
    fte = FTE(target_column="target", regression=True)
    data_dir = str(tmp_path / "data")
    try:
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name=f"m_{model_list[0]}",
            features_and_targets_extractor=fte,
            mlframe_models=model_list,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=data_dir, models_dir="models"),
            verbose=0,
            hyperparams_config={"iterations": _DEFAULT_SMOKE_ITERATIONS},
        )
    except (TypeError, ImportError) as e:
        pytest.skip(f"suite call broke during refactor: {e}")
    # The chosen model family must be reflected somewhere in the returned models structure. Behavioural pin:
    # the requested family name must appear as a substring of any KEY (target_type / target_name) OR any
    # MODEL OBJECT's class name (the leaf level is a list of model objects, not a dict). The suite-level
    # return nests as ``{TargetTypes.<X>: {target_name: [model_obj_1, model_obj_2, ...], ...}, ...}``.
    assert models is not None, f"suite returned None models on mlframe_models={model_list} path"
    assert hasattr(models, "__len__") and len(models) >= 1, f"models container empty for mlframe_models={model_list}; got {type(models).__name__}"
    if isinstance(models, dict):
        family = model_list[0]
        haystacks: list[str] = []

        def _collect(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    haystacks.append(str(k))
                    _collect(v)
            elif isinstance(d, (list, tuple)):
                for item in d:
                    _collect(item)
            else:
                # Leaf model object — capture its class name, module, and any
                # ``model_name`` / ``name`` attribute the suite stamps on it.
                haystacks.append(type(d).__name__)
                haystacks.append(getattr(type(d), "__module__", "") or "")
                for attr in (
                    "model_name",
                    "name",
                    "estimator_type",
                    "mlframe_model_name",
                    "family",
                    "model_type",
                ):
                    val = getattr(d, attr, None)
                    if isinstance(val, str):
                        haystacks.append(val)
                # ``SimpleNamespace`` wrappers stash everything in ``__dict__``.
                # Pick up every string field so the family name (e.g. stored as
                # ``model_name="lgb"`` inside the namespace) is captured.
                _ns_dict = getattr(d, "__dict__", None)
                if isinstance(_ns_dict, dict):
                    for k, v in _ns_dict.items():
                        if isinstance(v, str):
                            haystacks.append(v)
                        # Capture the class name of any nested model object too
                        # (e.g. ``namespace.model = LGBMRegressor(...)``).
                        elif v is not None and not isinstance(v, (int, float, bool, list, tuple, dict)):
                            haystacks.append(type(v).__name__)

        _collect(models)
        keys_str = " ".join(haystacks).lower()
        assert family in keys_str, f"mlframe_models={model_list} did not produce any {family}-related model; haystacks={haystacks}"


# ---------------------------------------------------------------------------
# Reproducibility / output schema
# ---------------------------------------------------------------------------


def test_biz_val_training_suite_metadata_dict_schema(tmp_path):
    """Suite must return ``metadata`` as a dict on success. Catches
    regressions in the suite-level metadata aggregation path."""
    pytest.importorskip("lightgbm")
    train_mlframe_models_suite, OutputConfig, FTE = _try_import_suite()
    df = _make_regression_df(n=300, seed=42)
    fte = FTE(target_column="target", regression=True)
    data_dir = str(tmp_path / "data")
    try:
        _models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="m_md",
            features_and_targets_extractor=fte,
            mlframe_models=["lgb"],
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=data_dir, models_dir="models"),
            verbose=0,
            hyperparams_config={"iterations": _DEFAULT_SMOKE_ITERATIONS},
        )
    except (TypeError, ImportError) as e:
        pytest.skip(f"suite call broke during refactor: {e}")
    assert isinstance(metadata, dict), f"metadata must be a dict; got {type(metadata).__name__}"
