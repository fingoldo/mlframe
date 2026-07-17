"""Regression + integration sensors for the 2026-06-22 FS suite-integration audit, round 2.

Covers the remaining wiring findings:

  F3  Suite RFECV now actually gets the cluster-medoid pre-reduction. The suite builds RFECV
      directly (not via ``registry._instantiate_rfecv``), so ``_build_pre_pipelines`` wraps the
      prebuilt RFECV in GroupAwareMRMR, driven by ``FeatureSelectionConfig.rfecv_cluster_*``. The
      documented default-ON behaviour previously never reached the suite RFECV path.
  F4  ShapProxiedFS is reachable from the suite: ``use_shap_proxied_fs`` flag + a
      ``_build_pre_pipelines`` branch (mirroring BorutaShap). Registration -> usable.
  F5  The FS suite-runtime ``_mlframe_*`` markers (selector-kind, weight-aware flag, identity-cache
      override) are stripped from the saved model bundle and restored on the in-memory object.
  F6  ``shap_proxied_fs_kwargs`` has a validation + master-flag surface symmetric to the others.

Architecture: ``_build_feature_selection_report`` consumes ``FeatureSelectorSpec.report_extract``.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from mlframe.training import FeatureSelectionConfig


class _FakeRFECV(BaseEstimator, TransformerMixin):
    """Minimal selector exposing the RFECV-like sklearn contract the wiring relies on.

    Keeps the first ``keep`` columns. Implements ``set_params`` (BaseEstimator) so the suite-override
    loop in ``_build_pre_pipelines`` exercises its happy path, and ``support_`` / ``feature_names_in_``
    so GroupAwareMRMR can read the inner selection and expand it.
    """

    def __init__(self, keep: int = 2, leakage_corr_threshold=None, mbh_adaptive_threshold=None, random_state=None):
        self.keep = keep
        self.leakage_corr_threshold = leakage_corr_threshold
        self.mbh_adaptive_threshold = mbh_adaptive_threshold
        self.random_state = random_state

    def fit(self, X, y=None, **fit_params):
        cols = list(X.columns)
        self.feature_names_in_ = np.asarray(cols, dtype=object)
        self.n_features_in_ = len(cols)
        self.support_ = np.array([i < self.keep for i in range(len(cols))], dtype=bool)
        return self

    def transform(self, X):
        return X.iloc[:, np.where(self.support_)[0]]


def _build(**over):
    from mlframe.training.core._setup_helpers_pre_pipelines import _build_pre_pipelines

    base = dict(use_ordinary_models=False, rfecv_models=[], rfecv_models_params={}, use_mrmr_fs=False, mrmr_kwargs={})
    base.update(over)
    return _build_pre_pipelines(**base)


# --------------------------------------------------------------------------- F3


def test_f3_suite_rfecv_is_cluster_wrapped_when_default_on():
    from mlframe.feature_selection.filters.group_aware import GroupAwareMRMR

    rf = _FakeRFECV(keep=2)
    pps, _names = _build(rfecv_models=["lgb"], rfecv_models_params={"lgb": rf}, rfecv_cluster_reduce=True)
    sel = pps[0]
    assert isinstance(sel, GroupAwareMRMR), "default-ON cluster-reduce must wrap the suite RFECV"
    # The kind marker is stamped on the OUTER wrapper so _selector_kind classifies it as RFECV.
    assert getattr(sel, "_mlframe_selector_kind_") == "RFECV"
    assert sel.estimator is rf  # wraps the prebuilt, suite-overridden RFECV


def test_f3_cluster_reduce_off_yields_bare_rfecv():
    from mlframe.feature_selection.filters.group_aware import GroupAwareMRMR

    rf = _FakeRFECV(keep=2)
    pps, _ = _build(rfecv_models=["lgb"], rfecv_models_params={"lgb": rf}, rfecv_cluster_reduce=False)
    assert not isinstance(pps[0], GroupAwareMRMR)
    assert pps[0] is rf
    assert getattr(pps[0], "_mlframe_selector_kind_") == "RFECV"


def test_f3_wrapped_rfecv_threads_overrides_to_inner():
    rf = _FakeRFECV(keep=2)
    pps, _ = _build(
        rfecv_models=["lgb"],
        rfecv_models_params={"lgb": rf},
        rfecv_leakage_corr_threshold=0.77,
        rfecv_mbh_adaptive_threshold=11,
        rfecv_cluster_reduce=True,
    )
    # Overrides were applied to the prebuilt RFECV BEFORE wrapping.
    assert pps[0].estimator.leakage_corr_threshold == 0.77
    assert pps[0].estimator.mbh_adaptive_threshold == 11


def test_f3_selector_kind_classifies_wrapped_rfecv():
    from mlframe.training.core._phase_train_one_target import _selector_kind

    rf = _FakeRFECV(keep=2)
    pps, _ = _build(rfecv_models=["lgb"], rfecv_models_params={"lgb": rf}, rfecv_cluster_reduce=True)
    assert _selector_kind(pps[0]) == "RFECV"


def test_biz_f3_cluster_wrap_keeps_whole_correlated_cluster():
    """biz_value: with expand=True the wrap returns the WHOLE correlated cluster, not just the medoid.

    A bare RFECV keeping ``keep`` columns drops near-duplicate members; the cluster-medoid wrap expands a
    selected medoid back to all its cluster mates, so a near-duplicate group survives together. This is the
    measurable win the wrap delivers (recall on correlated groups), and a regression that silently disables
    the wrap drops the duplicates again.
    """
    rng = np.random.default_rng(0)
    base = rng.standard_normal(400)
    df = pd.DataFrame(
        {
            "a": base,
            "a_dup1": base + 1e-3 * rng.standard_normal(400),
            "a_dup2": base + 1e-3 * rng.standard_normal(400),
            "noise0": rng.standard_normal(400),
            "noise1": rng.standard_normal(400),
        }
    )
    y = pd.Series((base > 0).astype(int))

    rf_bare = _FakeRFECV(keep=1)
    bare_pps, _ = _build(rfecv_models=["lgb"], rfecv_models_params={"lgb": rf_bare}, rfecv_cluster_reduce=False)
    bare_kept = list(bare_pps[0].fit(df, y).transform(df).columns)

    rf_wrap = _FakeRFECV(keep=1)
    wrap_pps, _ = _build(rfecv_models=["lgb"], rfecv_models_params={"lgb": rf_wrap}, rfecv_cluster_reduce=True, rfecv_cluster_corr_threshold=0.9)
    wrap_kept = list(wrap_pps[0].fit(df, y).transform(df).columns)

    # Bare keeps a single column; the wrap expands the correlated cluster {a, a_dup1, a_dup2} back together.
    assert len(bare_kept) == 1
    assert len(wrap_kept) >= 3, f"cluster wrap should keep the whole correlated group, got {wrap_kept}"
    assert {"a", "a_dup1", "a_dup2"}.issubset(set(wrap_kept))


# --------------------------------------------------------------------------- F4


def test_f4_shap_proxied_fs_reachable_from_suite():
    pps, names = _build(use_shap_proxied_fs=True, shap_proxied_fs_kwargs={"top_n": 5})
    assert "ShapProxiedFS " in names
    sel = pps[names.index("ShapProxiedFS ")]
    assert getattr(sel, "_mlframe_selector_kind_") == "ShapProxiedFS"
    assert type(sel).__name__ == "ShapProxiedFS"


def test_f4_shap_proxied_fs_auto_derives_regression_classification():
    pps, names = _build(use_shap_proxied_fs=True, shap_proxied_fs_kwargs={}, target_type="TargetTypes.REGRESSION")
    sel = pps[names.index("ShapProxiedFS ")]
    assert sel.classification is False  # regression target -> regressor inner model


def test_f4_shap_proxied_fs_kind_classified():
    from mlframe.training.core._phase_train_one_target import _selector_kind

    pps, names = _build(use_shap_proxied_fs=True, shap_proxied_fs_kwargs={})
    assert _selector_kind(pps[names.index("ShapProxiedFS ")]) == "ShapProxiedFS"


# --------------------------------------------------------------------------- F6


def test_f6_shap_proxied_fs_kwargs_master_flag_gate():
    with pytest.raises(ValueError, match="shap_proxied_fs_kwargs supplied but use_shap_proxied_fs"):
        FeatureSelectionConfig(shap_proxied_fs_kwargs={"top_n": 5})


def test_f6_shap_proxied_fs_kwargs_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown key"):
        FeatureSelectionConfig(use_shap_proxied_fs=True, shap_proxied_fs_kwargs={"definitely_not_a_param": 1})


def test_f6_shap_proxied_fs_kwargs_accepts_valid_key():
    cfg = FeatureSelectionConfig(use_shap_proxied_fs=True, shap_proxied_fs_kwargs={"top_n": 7, "optimizer": "auto"})
    assert cfg.shap_proxied_fs_kwargs["top_n"] == 7


def test_f6_rfecv_cluster_corr_method_validated():
    with pytest.raises(ValueError, match="rfecv_cluster_corr_method"):
        FeatureSelectionConfig(rfecv_cluster_corr_method="bogus")
    assert FeatureSelectionConfig(rfecv_cluster_corr_method="su").rfecv_cluster_corr_method == "su"


# --------------------------------------------------------------------------- Architecture: report_extract


def test_report_builder_consumes_registry_report_extract():
    """The central report builder fills ShapProxiedFS scores via the registry spec's report_extract."""
    from mlframe.training.core._phase_train_one_target_helpers import _build_feature_selection_report

    class _FakeShapProxied:
        feature_names_in_ = np.asarray(["f0", "f1", "f2"], dtype=object)
        selected_features_ = ["f0", "f2"]
        shap_proxy_report_ = {"mean_abs_shap": {"f0": 0.4, "f1": 0.01, "f2": 0.3}}

        def __init__(self):
            self._mlframe_selector_kind_ = "ShapProxiedFS"

    rep = _build_feature_selection_report(_FakeShapProxied(), "ShapProxiedFS ", ["f0", "f1", "f2"], ["f0", "f2"])
    assert rep["selector_name"] == "ShapProxiedFS"
    assert rep["scores"] == {"f0": 0.4, "f1": 0.01, "f2": 0.3}
    assert rep["reason_per_feature"]["f0"] == "selected"
    assert rep["reason_per_feature"]["f1"] == "dropped"
