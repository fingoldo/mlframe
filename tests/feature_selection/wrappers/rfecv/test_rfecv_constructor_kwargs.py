"""Sensor tests for A-Arch-003: RFECV leakage / mbh thresholds applied via set_params, not raw setattr.

The suite previously called ``setattr(_rfecv_instance, "leakage_corr_threshold", ...)`` directly,
bypassing any property-setter side effects. The fix routes through sklearn's ``set_params`` which
validates the parameter name against ``get_params`` and fires any setter side effects.
"""

from __future__ import annotations


from mlframe.training.core._setup_helpers import _build_pre_pipelines


class _FakeRFECVWithSetParams:
    """Mimics RFECV.set_params behavior; records what was applied."""

    def __init__(self):
        self.leakage_corr_threshold = None
        self.mbh_adaptive_threshold = None
        self._set_params_calls: list[dict] = []

    def set_params(self, **kwargs):
        """Set params."""
        self._set_params_calls.append(dict(kwargs))
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self


class _FakeNoSetParams:
    """Selector without sklearn set_params; the fallback path uses setattr."""

    pass


def test_set_params_path_invoked_for_sklearn_style_instances():
    """Set params path invoked for sklearn style instances."""
    inst = _FakeRFECVWithSetParams()
    _pre_pipelines, _names = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=["fake"],
        rfecv_models_params={"fake": inst},
        use_mrmr_fs=False,
        mrmr_kwargs={},
        custom_pre_pipelines=None,
        rfecv_leakage_corr_threshold=0.77,
        rfecv_mbh_adaptive_threshold=42,
        use_boruta_shap=False,
        boruta_shap_kwargs=None,
    )
    assert len(inst._set_params_calls) == 1
    applied = inst._set_params_calls[0]
    assert applied["leakage_corr_threshold"] == 0.77
    assert applied["mbh_adaptive_threshold"] == 42
    assert inst.leakage_corr_threshold == 0.77
    assert inst.mbh_adaptive_threshold == 42


def test_setattr_fallback_for_non_sklearn_instances():
    """Setattr fallback for non sklearn instances."""
    inst = _FakeNoSetParams()
    _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=["fake"],
        rfecv_models_params={"fake": inst},
        use_mrmr_fs=False,
        mrmr_kwargs={},
        custom_pre_pipelines=None,
        rfecv_leakage_corr_threshold=0.5,
        rfecv_mbh_adaptive_threshold=10,
        use_boruta_shap=False,
        boruta_shap_kwargs=None,
    )
    assert inst.leakage_corr_threshold == 0.5
    assert inst.mbh_adaptive_threshold == 10


def test_sample_weight_marker_still_stamped_via_setattr():
    """Per user constraint: the suite-internal ``_mlframe_use_sample_weights_in_fs_`` marker stays as setattr
    (never routed through set_params). It is stamped on the OUTER object that enters ``pre_pipelines`` -- with
    cluster-reduce default-ON that is the GroupAwareMRMR wrapper, which is exactly what the downstream
    ``_selector_kind`` / weight-aware fit driver reads it off. The pre-cluster-wrap shape (marker on the bare
    inner instance) was the stale proxy."""
    inst = _FakeRFECVWithSetParams()
    pre_pipelines, _ = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=["fake"],
        rfecv_models_params={"fake": inst},
        use_mrmr_fs=False,
        mrmr_kwargs={},
        custom_pre_pipelines=None,
        rfecv_leakage_corr_threshold=0.95,
        rfecv_mbh_adaptive_threshold=30,
        use_boruta_shap=False,
        boruta_shap_kwargs=None,
        use_sample_weights_in_fs=True,
    )
    outer = pre_pipelines[0]
    assert getattr(outer, "_mlframe_use_sample_weights_in_fs_") is True
    # The marker is NOT routed through set_params (kept suite-internal) on the inner instance.
    for call in inst._set_params_calls:
        assert "_mlframe_use_sample_weights_in_fs_" not in call
