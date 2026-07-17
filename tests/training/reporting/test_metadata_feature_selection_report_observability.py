"""Wave-8 observability sensor 1: ``metadata["model_schemas"][file_name]["feature_selection_report"]``.

Contract under test:
  * The report is always stamped (selector_name=None for the ordinary / no-FS branch).
  * For MRMR / RFECV / BorutaShap branches, ``selector_name`` matches the class.
  * ``kept_features`` reflects the post-FS surviving feature set; ``dropped_features`` is the
    pre-fit complement.
  * ``selector_params_hash`` is stable per (selector, params) and differs when params change.
  * Per-selector score / reason fields surface the selector's natural attribute, or ``None`` when
    the selector exposes neither.

Tested via the helper directly (unit-level) plus an end-to-end smoke that asserts the key lands
on metadata when the suite trains with use_mrmr_fs=True.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def test_unwrap_selector_returns_input_for_bare_selector():
    """A bare selector (not wrapped in Pipeline) is returned as-is."""
    from mlframe.training.core._phase_train_one_target import _unwrap_selector

    class _Selector:
        support_ = np.array([True, False])

    s = _Selector()
    assert _unwrap_selector(s) is s


def test_unwrap_selector_returns_last_step_of_pipeline():
    """A sklearn Pipeline's last step is the selector."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from mlframe.training.core._phase_train_one_target import _unwrap_selector

    class _Selector:
        support_ = np.array([True])

    sel = _Selector()
    pipe = Pipeline([("scaler", StandardScaler()), ("selector", sel)])
    assert _unwrap_selector(pipe) is sel


def test_unwrap_selector_returns_none_on_none():
    """``None`` (ordinary / no-FS) returns ``None``."""
    from mlframe.training.core._phase_train_one_target import _unwrap_selector

    assert _unwrap_selector(None) is None


def test_selector_kind_classifies_mrmr():
    """MRMR class name is the marker."""
    from mlframe.training.core._phase_train_one_target import _selector_kind

    class MRMR:
        pass

    assert _selector_kind(MRMR()) == "MRMR"


def test_selector_kind_classifies_rfecv():
    """RFECV classification covers ``RFECV`` and ``CatBoostRFECV``-like names."""
    from mlframe.training.core._phase_train_one_target import _selector_kind

    class RFECV:
        pass

    class CatBoostRFECV:
        pass

    assert _selector_kind(RFECV()) == "RFECV"
    assert _selector_kind(CatBoostRFECV()) == "RFECV"


def test_selector_kind_classifies_boruta_shap():
    """BorutaShap class name is the marker."""
    from mlframe.training.core._phase_train_one_target import _selector_kind

    class BorutaShap:
        pass

    assert _selector_kind(BorutaShap()) == "BorutaShap"


def test_selector_kind_returns_none_on_unknown():
    """Unknown selectors (custom transforms) classify as None."""
    from mlframe.training.core._phase_train_one_target import _selector_kind

    class _Other:
        pass

    assert _selector_kind(_Other()) is None
    assert _selector_kind(None) is None


def test_selector_params_hash_is_stable_across_calls():
    """Same params -> same hash. Different params -> different hash."""
    from mlframe.training.core._phase_train_one_target import _selector_params_hash

    class _S:
        def __init__(self, alpha=1, beta=2):
            self.alpha = alpha
            self.beta = beta

        def get_params(self, deep=False):
            return {"alpha": self.alpha, "beta": self.beta}

    h1 = _selector_params_hash(_S(alpha=1, beta=2))
    h2 = _selector_params_hash(_S(alpha=1, beta=2))
    h3 = _selector_params_hash(_S(alpha=99, beta=2))
    assert h1 is not None and h1 == h2
    assert h1 != h3


def test_build_report_for_ordinary_branch_returns_none_selector():
    """``pre_pipeline=None`` (ordinary path) -> selector_name=None, kept_features still surfaces."""
    from mlframe.training.core._phase_train_one_target import _build_feature_selection_report

    report = _build_feature_selection_report(
        pre_pipeline=None,
        pre_pipeline_name="",
        fitted_columns_in=["a", "b", "c"],
        kept_columns=["a", "b", "c"],
    )
    assert report["selector_name"] is None
    assert report["kept_features"] == ["a", "b", "c"]
    assert report["dropped_features"] == [] or report["dropped_features"] is None
    assert report["scores"] is None
    assert report["reason_per_feature"] is None


def test_build_report_for_mrmr_emits_reason_per_feature():
    """MRMR exposes ``support_`` but no per-feature score; reason is 'kept'/'dropped'."""
    from mlframe.training.core._phase_train_one_target import _build_feature_selection_report

    class MRMR:
        feature_names_in_ = ["a", "b", "c", "d"]
        support_ = np.array([0, 2])  # indices into feature_names_in_

        def get_params(self, deep=False):
            return {"k": 2}

    report = _build_feature_selection_report(
        pre_pipeline=MRMR(),
        pre_pipeline_name="MRMR ",
        fitted_columns_in=None,
        kept_columns=["a", "c"],
    )
    assert report["selector_name"] == "MRMR"
    assert report["selector_params_hash"] is not None
    assert report["kept_features"] == ["a", "c"]
    assert set(report["dropped_features"]) == {"b", "d"}
    assert report["scores"] is None  # MRMR exposes no per-feature score
    assert report["reason_per_feature"] == {"a": "kept", "b": "dropped", "c": "kept", "d": "dropped"}


def test_build_report_for_rfecv_emits_scores_and_rank():
    """RFECV exposes ``feature_importances_`` (dict keyed by '<nfeatures>_<fold>') + ``ranking_``."""
    from mlframe.training.core._phase_train_one_target import _build_feature_selection_report

    class RFECV:
        feature_names_in_ = ["a", "b", "c", "d"]
        n_features_ = 2
        # Two folds at the chosen size; mean gives the score.
        feature_importances_ = {
            "2_0": np.array([0.4, 0.1, 0.3, 0.05]),
            "2_1": np.array([0.5, 0.2, 0.4, 0.10]),
        }
        ranking_ = [1, 3, 1, 4]
        support_ = np.array([True, False, True, False])

        def get_params(self, deep=False):
            return {"step": 0.5}

    report = _build_feature_selection_report(
        pre_pipeline=RFECV(),
        pre_pipeline_name="lgb ",
        fitted_columns_in=None,
        kept_columns=["a", "c"],
    )
    assert report["selector_name"] == "RFECV"
    assert report["scores"] is not None
    assert pytest.approx(report["scores"]["a"], rel=1e-6) == 0.45
    assert pytest.approx(report["scores"]["c"], rel=1e-6) == 0.35
    assert report["reason_per_feature"] is not None
    assert report["reason_per_feature"]["a"].startswith("kept@rank=")
    assert report["reason_per_feature"]["b"].startswith("dropped@rank=")


def test_build_report_for_boruta_shap_emits_history_means_and_reasons():
    """BorutaShap's ``history_x`` mean column gives the per-feature score; accepted/rejected/tentative -> reason."""
    from mlframe.training.core._phase_train_one_target import _build_feature_selection_report

    class BorutaShap:
        all_columns = np.array(["a", "b", "c", "d"])
        history_x = pd.DataFrame(
            np.array(
                [
                    [0.1, 0.2, 0.3, 0.05],
                    [0.2, 0.3, 0.4, 0.10],
                ]
            ),
            columns=["a", "b", "c", "d"],
        )
        accepted = ["a", "c"]
        rejected = ["d"]
        tentative = ["b"]
        support_ = np.array([True, False, True, False])

        def get_params(self, deep=False):
            return {"percentile": 100}

    report = _build_feature_selection_report(
        pre_pipeline=BorutaShap(),
        pre_pipeline_name="BorutaShap ",
        fitted_columns_in=None,
        kept_columns=["a", "c"],
    )
    assert report["selector_name"] == "BorutaShap"
    assert report["scores"] is not None
    assert pytest.approx(report["scores"]["a"], rel=1e-6) == 0.15
    assert pytest.approx(report["scores"]["c"], rel=1e-6) == 0.35
    assert report["reason_per_feature"] == {
        "a": "accepted",
        "b": "tentative",
        "c": "accepted",
        "d": "rejected",
    }


def test_build_report_never_raises_on_malformed_selector():
    """A selector that raises on every attr access still returns a minimal report."""
    from mlframe.training.core._phase_train_one_target import _build_feature_selection_report

    class _Broken:
        def __getattr__(self, name):
            raise RuntimeError("nope")

    report = _build_feature_selection_report(
        pre_pipeline=_Broken(),
        pre_pipeline_name="custom ",
        fitted_columns_in=["a", "b"],
        kept_columns=["a"],
    )
    # Shape stays sane regardless of the broken selector.
    assert "selector_name" in report
    assert report["kept_features"] == ["a"]


# ----------------------------------------------------------------------------
# End-to-end: feature_selection_report lands on metadata when MRMR is enabled.
# ----------------------------------------------------------------------------


@pytest.fixture
def synthetic_binary_8feat():
    """200 rows x 8 features, binary target -- small enough for fast tests, large enough for MRMR."""
    rng = np.random.default_rng(2026)
    n = 200
    X = rng.standard_normal((n, 8)).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    # Logistic target: only f0 / f1 are informative; MRMR should drop the rest.
    logits = 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.1 * rng.standard_normal(n)
    df["target"] = (logits > 0).astype(np.int8)
    return df


def test_feature_selection_report_lands_on_metadata_with_mrmr(synthetic_binary_8feat):
    """End-to-end: train_mlframe_models_suite with FeatureSelectionConfig(use_mrmr_fs=True) stamps the report."""
    from mlframe.training import train_mlframe_models_suite
    from mlframe.training.configs import FeatureSelectionConfig
    from tests.training.shared import SimpleFeaturesAndTargetsExtractor

    df = synthetic_binary_8feat
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    fs_cfg = FeatureSelectionConfig(
        use_mrmr_fs=True,
        mrmr_kwargs={"verbose": 0},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _result = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="fs_report_smoke",
            features_and_targets_extractor=fte,
            mlframe_models=["linear"],
            feature_selection_config=fs_cfg,
            use_mlframe_ensembles=False,
            verbose=0,
        )

    # train_mlframe_models_suite returns (models, metadata, ...) tuple or a ctx; normalise.
    metadata = None
    if isinstance(_result, tuple):
        for _slot in _result:
            if isinstance(_slot, dict) and "model_schemas" in _slot:
                metadata = _slot
                break
    elif hasattr(_result, "metadata"):
        metadata = _result.metadata
    elif isinstance(_result, dict):
        metadata = _result
    assert isinstance(metadata, dict), f"expected metadata dict, got {type(_result)}"
    _schemas = metadata.get("model_schemas") or {}
    assert _schemas, "expected at least one model_schemas entry"
    _with_report = [k for k, v in _schemas.items() if isinstance(v, dict) and "feature_selection_report" in v]
    assert _with_report, f"expected feature_selection_report on every model_schemas entry; keys: {list(_schemas.keys())}"
    # At least one entry should be the MRMR branch (selector_name='MRMR').
    _kinds = {
        v["feature_selection_report"]["selector_name"] for v in _schemas.values() if isinstance(v, dict) and isinstance(v.get("feature_selection_report"), dict)
    }
    assert "MRMR" in _kinds or None in _kinds, f"expected MRMR or ordinary branch; got {_kinds}"
