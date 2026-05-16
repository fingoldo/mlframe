"""``_build_pre_pipelines`` wiring contract for the new ``use_boruta_shap`` knob:

  * default (False) -> no BorutaShap instance appears in the returned list, legacy ordering preserved byte-for-byte
  * opt-in (True)   -> exactly one BorutaShap instance appended AFTER the cheaper MRMR / RFECV selectors

Why default-off matters: BorutaShap is the most expensive selector in the suite (TreeExplainer per trial on doubled feature matrix); any caller relying on the historical ``pre_pipelines`` shape must not see a new entry without explicit opt-in.
"""

from __future__ import annotations

import pytest


def test_build_pre_pipelines_no_boruta_shap_by_default():
    from mlframe.training.core._setup_helpers import _build_pre_pipelines

    pipelines, names = _build_pre_pipelines(
        use_ordinary_models=True,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=False,
        mrmr_kwargs={},
    )
    # No BorutaShap (it would import shap; instead check by class name to avoid the import).
    for p in pipelines:
        cls_name = type(p).__name__
        assert cls_name != "BorutaShap", f"BorutaShap leaked into default pipelines: {names!r}"
    # Just the ordinary (None) branch.
    assert pipelines == [None]
    assert names == [""]


def test_build_pre_pipelines_appends_boruta_shap_when_enabled():
    pytest.importorskip("shap")
    from mlframe.feature_selection.boruta_shap import BorutaShap
    from mlframe.training.core._setup_helpers import _build_pre_pipelines

    pipelines, names = _build_pre_pipelines(
        use_ordinary_models=True,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=False,
        mrmr_kwargs={},
        use_boruta_shap=True,
        boruta_shap_kwargs={"n_trials": 3, "verbose": False},
    )

    # Exactly one BorutaShap; appended (not prepended).
    boruta_indices = [i for i, p in enumerate(pipelines) if isinstance(p, BorutaShap)]
    assert len(boruta_indices) == 1, f"expected exactly one BorutaShap entry; got pipelines={pipelines!r}"
    assert boruta_indices[0] == len(pipelines) - 1, "BorutaShap must be appended after the cheaper selectors"
    assert "BorutaShap " in names

    # Kwargs forwarded to the constructor.
    boruta_instance = pipelines[boruta_indices[0]]
    assert boruta_instance.n_trials == 3
    assert boruta_instance.verbose is False


def test_build_pre_pipelines_boruta_shap_after_mrmr():
    """When both MRMR and BorutaShap are enabled, ordering must be: ordinary -> MRMR -> BorutaShap."""
    pytest.importorskip("shap")
    from mlframe.feature_selection.boruta_shap import BorutaShap
    from mlframe.feature_selection.filters import MRMR
    from mlframe.training.core._setup_helpers import _build_pre_pipelines

    pipelines, _ = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=True,
        mrmr_kwargs={"verbose": 0},
        use_boruta_shap=True,
        boruta_shap_kwargs={"verbose": False},
    )

    assert len(pipelines) == 2, f"expected 2 entries (MRMR + BorutaShap); got {pipelines!r}"
    assert isinstance(pipelines[0], MRMR), f"MRMR must come first; got {type(pipelines[0]).__name__}"
    assert isinstance(pipelines[1], BorutaShap), f"BorutaShap must follow MRMR; got {type(pipelines[1]).__name__}"
