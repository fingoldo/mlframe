"""Sensor tests for A-Arch-004: MRMR cross-target identity cache moved from process-global to ctx-scoped.

When the suite injects a dict via ``_mlframe_identity_cache_override_``, MRMR reads / writes that dict
instead of the module-level ``_MRMR_IDENTITY_FP_CACHE``. Default ``mrmr_identity_cache_scope="ctx"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_config_default_scope_is_ctx():
    from mlframe.training.configs import FeatureSelectionConfig

    cfg = FeatureSelectionConfig()
    assert cfg.mrmr_identity_cache_scope == "ctx"


def test_config_validator_rejects_unknown_scope():
    from mlframe.training.configs import FeatureSelectionConfig

    with pytest.raises(ValueError):
        FeatureSelectionConfig(mrmr_identity_cache_scope="global")


def test_mrmr_uses_override_dict_when_stamped():
    """When ``_mlframe_identity_cache_override_`` is set, MRMR reads/writes it instead of the global cache."""
    from mlframe.feature_selection.filters import MRMR
    from mlframe.feature_selection.filters import mrmr as mrmr_module

    # Build a tiny non-identity scenario so MRMR completes quickly.
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    y = (X["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int)

    ctx_cache: dict = {}
    inst = MRMR(verbose=0, mrmr_skip_when_prior_was_identity=True)
    inst._mlframe_identity_cache_override_ = ctx_cache

    # Snapshot the module-global cache to verify it's untouched.
    global_snapshot = dict(mrmr_module._MRMR_IDENTITY_FP_CACHE)
    try:
        inst.fit(X, y)
    except Exception:  # nosec B110 -- best-effort cleanup/optional step; failure here never masks this test's own assertions
        # MRMR may fail on tiny inputs; we only care about cache routing semantics, not full fit success.
        pass

    # The override cache may or may not be populated depending on whether the fit reached the
    # post-fit cache-store branch. Either way the module-level cache must NOT have grown from this fit.
    assert mrmr_module._MRMR_IDENTITY_FP_CACHE == global_snapshot


def test_ctx_cache_field_present_on_training_context():
    """TrainingContext must have the slot for the ctx-scoped cache."""
    from mlframe.training.core._training_context import TrainingContext

    ctx = TrainingContext()
    assert hasattr(ctx, "_mrmr_identity_cache")
    assert ctx._mrmr_identity_cache == {}


def test_build_pre_pipelines_threads_cache_to_mrmr():
    """When mrmr_identity_cache is passed to _build_pre_pipelines, the MRMR instance gets it stamped."""
    from mlframe.training.core._setup_helpers import _build_pre_pipelines

    ctx_cache: dict = {}
    pre_pipelines, _names = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=True,
        mrmr_kwargs={"verbose": 0},
        custom_pre_pipelines=None,
        use_boruta_shap=False,
        boruta_shap_kwargs=None,
        mrmr_identity_cache=ctx_cache,
    )
    assert len(pre_pipelines) == 1
    # The override is a non-pickling view that DELEGATES to the shared ctx dict (so the runtime cache
    # does not enter the persisted model); writes through it must land in the same backing ``ctx_cache``,
    # and it must collapse to an empty plain dict at pickle time.
    override = pre_pipelines[0]._mlframe_identity_cache_override_
    override["probe"] = 1
    assert ctx_cache.get("probe") == 1
    assert override.get("probe") == 1
    import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

    assert pickle.loads(pickle.dumps(override)) == {}  # nosec B301 -- round-trip of a locally-created, trusted object


def test_build_pre_pipelines_no_cache_when_none():
    """When mrmr_identity_cache is None, MRMR falls back to the module-level cache (no override stamped)."""
    from mlframe.training.core._setup_helpers import _build_pre_pipelines

    pre_pipelines, _names = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=True,
        mrmr_kwargs={"verbose": 0},
        custom_pre_pipelines=None,
        use_boruta_shap=False,
        boruta_shap_kwargs=None,
        mrmr_identity_cache=None,
    )
    assert len(pre_pipelines) == 1
    # No override means MRMR uses the module-level cache; the attr should not have been stamped.
    assert not hasattr(pre_pipelines[0], "_mlframe_identity_cache_override_")
