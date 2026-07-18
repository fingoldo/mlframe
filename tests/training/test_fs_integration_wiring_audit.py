"""Regression sensors for the feature-selector wiring/integration layer.

Covers two integration-layer bugs found in the 2026-06-22 FS suite-integration audit:

  1. The MRMR cross-target identity cache (``_mlframe_identity_cache_override_``) is a
     ctx-scoped runtime dict stamped onto every per-target MRMR. Stamped as a plain
     attribute it gets pickled into the saved model bundle (size bloat + stale replay on
     reload). The wiring now stamps a non-pickling view that collapses to ``{}`` on pickle
     while still delegating in-process reads/writes to the shared backing dict.

  2. ``FeatureSelectionConfig.rfecv_kwargs`` used to whitelist the cluster-medoid keys
     (cluster_reduce, ...). The suite builds RFECV directly (NOT via the registry wrap that
     consumes those keys), so they were forwarded verbatim to ``RFECV(**kwargs)`` and crashed
     at construction. The validator must now reject them at config time.
"""

from __future__ import annotations

import os
import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from mlframe.training import FeatureSelectionConfig
from mlframe.training.core._setup_helpers_pre_pipelines import _NonPicklingCacheView


def test_non_pickling_cache_view_delegates_and_prunes_on_pickle():
    """The view shares the backing dict in-process but pickles to an empty plain dict."""
    backing: dict = {}
    view = _NonPicklingCacheView(backing)

    # Writes/reads delegate to the shared backing dict (cross-target reuse contract).
    view["fp123"] = (True, None)
    assert backing["fp123"] == (True, None)
    assert view.get("fp123") == (True, None)
    assert "fp123" in view
    backing["fp456"] = 7
    assert view["fp456"] == 7  # a write to the backing dict by a sibling stamp is visible

    # Pickle round-trip collapses to an empty plain dict -- no suite-runtime cache persisted.
    restored = pickle.loads(pickle.dumps(view))  # nosec B301 -- round-trip of a locally-created, trusted object
    assert type(restored) is dict
    assert restored == {}


def test_fitted_mrmr_does_not_pickle_identity_cache_override():
    """A fitted MRMR carrying the stamped override must not drag the cache into its pickle."""
    from mlframe.training.core._setup_helpers_pre_pipelines import _build_pre_pipelines

    shared_cache: dict = {"prior_target_fp": (True, None)}
    pre_pipelines, names = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=True,
        mrmr_kwargs={"use_simple_mode": True, "verbose": 0},
        mrmr_identity_cache=shared_cache,
    )
    mrmr = pre_pipelines[names.index("MRMR ")]
    assert isinstance(getattr(mrmr, "_mlframe_identity_cache_override_"), _NonPicklingCacheView)

    blob = pickle.dumps(mrmr)
    # The unique sentinel fingerprint from the shared cache must not appear in the bytes.
    assert b"prior_target_fp" not in blob
    restored = pickle.loads(blob)  # nosec B301 -- round-trip of a locally-created, trusted object
    assert restored._mlframe_identity_cache_override_ == {}


def test_fs_runtime_markers_stripped_from_saved_bundle_and_restored(tmp_path):
    """F5: the FS suite-runtime ``_mlframe_*`` markers must not enter the saved model bundle.

    ``_build_pre_pipelines`` stamps selectors with private dispatch/weight/cache markers used only by
    the in-process training loop. ``save_mlframe_model`` strips them before pickling (temp-del + restore)
    so the persisted bundle carries no FS-internal suite state, while the in-memory object keeps them.
    """
    from types import SimpleNamespace

    from mlframe.training.io import load_mlframe_model, save_mlframe_model

    class _Selector:
        """Groups tests covering selector."""
        def __init__(self):
            self._mlframe_selector_kind_ = "MRMR"
            self._mlframe_use_sample_weights_in_fs_ = True
            self._mlframe_identity_cache_override_ = {"some_fp": (True, None)}
            self.real_attr = "kept"

    sel = _Selector()
    payload = SimpleNamespace(pre_pipeline=sel, model=None)
    out = tmp_path / "m.dump"
    assert save_mlframe_model(payload, str(out), verbose=0) is True

    # In-memory object keeps its markers (restored in the finally block).
    assert sel._mlframe_selector_kind_ == "MRMR"
    assert sel._mlframe_use_sample_weights_in_fs_ is True
    assert sel._mlframe_identity_cache_override_ == {"some_fp": (True, None)}

    # The reloaded bundle carries none of the FS-internal markers, but keeps the real attr.
    loaded = load_mlframe_model(str(out), safe=False)
    lsel = loaded.pre_pipeline
    assert lsel.real_attr == "kept"
    assert not hasattr(lsel, "_mlframe_selector_kind_")
    assert not hasattr(lsel, "_mlframe_use_sample_weights_in_fs_")
    assert not hasattr(lsel, "_mlframe_identity_cache_override_")


def test_rfecv_kwargs_rejects_cluster_reduce_keys():
    """cluster_reduce in rfecv_kwargs is a config-time-green / fit-time-crash trap; reject it."""
    for bad in ("cluster_reduce", "cluster_corr_threshold", "cluster_min_reduction", "cluster_corr_method"):
        with pytest.raises(ValueError, match="unknown key"):
            FeatureSelectionConfig(rfecv_models=["lgb"], rfecv_kwargs={bad: True})


def test_boruta_shap_kwargs_still_allows_cluster_reduce_keys():
    """BorutaShap DOES route through the registry cluster wrap, so its validator keeps the keys."""
    cfg = FeatureSelectionConfig(use_boruta_shap=True, boruta_shap_kwargs={"cluster_reduce": False})
    assert cfg.boruta_shap_kwargs["cluster_reduce"] is False
