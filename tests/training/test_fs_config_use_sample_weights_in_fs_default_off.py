"""``FeatureSelectionConfig.use_sample_weights_in_fs`` wiring contract:

  * Default is False (preserves FS cache reuse across weight schemas).
  * When False, MRMR / RFECV instances produced by ``_build_pre_pipelines`` carry the marker
    ``_mlframe_use_sample_weights_in_fs_ = False`` so downstream fit drivers do NOT thread sample_weight.
  * When True, the marker flips to True so suite code forwards sample_weight via fit_params.

Why default-off matters: any caller relying on the FS cache (keyed on params + content hashes, NOT weights)
must not see a weight-aware refit silently invalidate every cached selection.
"""

from __future__ import annotations

import pytest


def _rfecv_selectors(pipelines):
    """The suite's RFECV is wrapped in GroupAwareMRMR (cluster-medoid pre-reduction, default-ON) before it
    enters pre_pipelines, with the suite markers stamped on the OUTER wrapper. Identify the RFECV-kind
    selectors by the dedicated dispatch marker rather than ``isinstance(p, RFECV)`` (which the wrapper isn't)."""
    return [p for p in pipelines if getattr(p, "_mlframe_selector_kind_", None) == "RFECV"]


def test_fs_config_use_sample_weights_in_fs_default_is_false():
    from mlframe.training.configs import FeatureSelectionConfig

    cfg = FeatureSelectionConfig()
    assert cfg.use_sample_weights_in_fs is False, "default must be False to preserve FS cache reuse"


def test_build_pre_pipelines_marker_default_off_on_mrmr():
    """When use_sample_weights_in_fs is unset / False, MRMR's marker attribute must be False."""
    from mlframe.training.core._setup_helpers import _build_pre_pipelines
    from mlframe.feature_selection.filters import MRMR

    pipelines, _ = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=True,
        mrmr_kwargs={"verbose": 0},
    )
    mrmrs = [p for p in pipelines if isinstance(p, MRMR)]
    assert len(mrmrs) == 1
    assert getattr(mrmrs[0], "_mlframe_use_sample_weights_in_fs_", None) is False


def test_build_pre_pipelines_marker_on_when_flag_flipped():
    """When use_sample_weights_in_fs=True, both MRMR and RFECV instances get marker=True."""
    from mlframe.training.core._setup_helpers import _build_pre_pipelines
    from mlframe.feature_selection.filters import MRMR
    from mlframe.feature_selection.wrappers import RFECV
    from sklearn.linear_model import LogisticRegression

    # RFECV instance must come from rfecv_models_params (locked construction path in production); for unit
    # purposes we construct one directly and pass it through the params dict.
    rfecv_instance = RFECV(estimator=LogisticRegression(max_iter=100), cv=2, verbose=0, max_runtime_mins=0.01)
    pipelines, names = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=["lr"],
        rfecv_models_params={"lr": rfecv_instance},
        use_mrmr_fs=True,
        mrmr_kwargs={"verbose": 0},
        use_sample_weights_in_fs=True,
    )
    mrmrs = [p for p in pipelines if isinstance(p, MRMR)]
    rfecvs = _rfecv_selectors(pipelines)
    assert len(mrmrs) == 1
    assert len(rfecvs) == 1
    assert getattr(mrmrs[0], "_mlframe_use_sample_weights_in_fs_", None) is True
    assert getattr(rfecvs[0], "_mlframe_use_sample_weights_in_fs_", None) is True


def test_build_pre_pipelines_marker_default_off_on_rfecv():
    from mlframe.training.core._setup_helpers import _build_pre_pipelines
    from mlframe.feature_selection.wrappers import RFECV
    from sklearn.linear_model import LogisticRegression

    rfecv_instance = RFECV(estimator=LogisticRegression(max_iter=100), cv=2, verbose=0, max_runtime_mins=0.01)
    pipelines, _ = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=["lr"],
        rfecv_models_params={"lr": rfecv_instance},
        use_mrmr_fs=False,
        mrmr_kwargs={},
    )
    rfecvs = _rfecv_selectors(pipelines)
    assert len(rfecvs) == 1
    assert getattr(rfecvs[0], "_mlframe_use_sample_weights_in_fs_", None) is False
