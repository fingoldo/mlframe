"""Sensor for the training/core/_setup_helpers.py monolith split (wave w6b).

Verifies:
- Every previously-importable public name still resolves via the parent facade.
- Identity is preserved (parent.X is sibling.X) for moved symbols.
- Parent facade LOC stays under the 800-line budget.
- Smoke calls into moved bodies (import-only sensors miss runtime NameErrors).
"""

from __future__ import annotations

from pathlib import Path

import pytest


PARENT = "mlframe.training.core._setup_helpers"
FACADE_LOC_BUDGET = 800


def test_setup_helpers_facade_loc_budget():
    import mlframe.training.core._setup_helpers as parent

    n = len(Path(parent.__file__).read_text(encoding="utf-8").splitlines())
    assert n <= FACADE_LOC_BUDGET, (
        f"{PARENT} grew back over the budget ({n} > {FACADE_LOC_BUDGET}); carve another sibling rather than letting the facade bloat."
    )


def test_setup_helpers_re_exports_resolve():
    """All carved symbols importable from the parent (historical public API surface)."""
    from mlframe.training.core._setup_helpers import (  # noqa: F401
        _pipeline_disk_cache_path,
        _pipeline_disk_cache_version_tag,
        _load_pipeline_disk_cache_into_memory,
        _persist_pipeline_disk_cache,
        _PolarsDsPipelineJsonProxy,
        _polars_ds_pipeline_from_json,
        _PIPELINE_JSON_ROUNDTRIP_CACHE,
        _apply_outlier_detection_global,
        _build_pre_pipelines,
        _create_initial_metadata,
        _initialize_training_defaults,
        _finalize_and_save_metadata,
    )


def test_setup_helpers_identity_pipeline_cache():
    import mlframe.training.core._setup_helpers as parent
    from mlframe.training.core import _setup_helpers_pipeline_cache as sib

    assert parent._pipeline_disk_cache_path is sib._pipeline_disk_cache_path
    assert parent._pipeline_disk_cache_version_tag is sib._pipeline_disk_cache_version_tag
    assert parent._load_pipeline_disk_cache_into_memory is sib._load_pipeline_disk_cache_into_memory
    assert parent._persist_pipeline_disk_cache is sib._persist_pipeline_disk_cache
    assert parent._PolarsDsPipelineJsonProxy is sib._PolarsDsPipelineJsonProxy
    assert parent._polars_ds_pipeline_from_json is sib._polars_ds_pipeline_from_json


def test_setup_helpers_identity_outliers():
    import mlframe.training.core._setup_helpers as parent
    from mlframe.training.core import _setup_helpers_outliers as sib

    assert parent._apply_outlier_detection_global is sib._apply_outlier_detection_global


def test_setup_helpers_identity_pre_pipelines():
    import mlframe.training.core._setup_helpers as parent
    from mlframe.training.core import _setup_helpers_pre_pipelines as sib

    assert parent._build_pre_pipelines is sib._build_pre_pipelines


def test_setup_helpers_identity_metadata():
    import mlframe.training.core._setup_helpers as parent
    from mlframe.training.core import _setup_helpers_metadata as sib

    assert parent._create_initial_metadata is sib._create_initial_metadata
    assert parent._initialize_training_defaults is sib._initialize_training_defaults
    assert parent._finalize_and_save_metadata is sib._finalize_and_save_metadata


def test_setup_helpers_smoke_pipeline_disk_cache_path():
    """Exercise the moved body so a runtime NameError surfaces."""
    from mlframe.training.core._setup_helpers import _pipeline_disk_cache_path

    out = _pipeline_disk_cache_path()
    assert isinstance(out, str)
    assert out.endswith(".json")


def test_setup_helpers_smoke_pipeline_disk_cache_version_tag():
    from mlframe.training.core._setup_helpers import _pipeline_disk_cache_version_tag

    tag = _pipeline_disk_cache_version_tag()
    assert isinstance(tag, str) and "polars" in tag


def test_setup_helpers_smoke_initialize_training_defaults():
    from mlframe.training.core._setup_helpers import _initialize_training_defaults

    common, rfecv, mrmr = _initialize_training_defaults(None, None, None, suite_verbose=0)
    assert common == {}
    assert rfecv == []
    assert "n_workers" in mrmr
    assert mrmr["max_runtime_mins"] == 300


def test_setup_helpers_smoke_build_pre_pipelines_empty():
    """Empty pre-pipeline list path doesn't trigger MRMR / BorutaShap heavy imports."""
    from mlframe.training.core._setup_helpers import _build_pre_pipelines

    pipelines, names = _build_pre_pipelines(
        use_ordinary_models=True,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=False,
        mrmr_kwargs={},
    )
    assert pipelines == [None]
    assert names == [""]


def test_setup_helpers_shared_pipeline_cache_state():
    """The in-memory cache dict is shared (same object) between parent and sibling -
    metadata sibling writes must be visible to anyone reading via the parent facade.
    """
    import mlframe.training.core._setup_helpers as parent
    from mlframe.training.core import _setup_helpers_pipeline_cache as pc

    assert parent._PIPELINE_JSON_ROUNDTRIP_CACHE is pc._PIPELINE_JSON_ROUNDTRIP_CACHE
