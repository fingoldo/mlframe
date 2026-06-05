"""Pre-pipeline (FS selectors) builder for ``_setup_helpers``.

Carved from ``_setup_helpers.py``. Re-exported from the parent.
Preserves the deferred MRMR import pattern verbatim - MRMR transitively
pulls in the entire mlframe.feature_selection package (numba kernels +
filter wrappers + sklearn estimators), which adds ~10-25s to first-call
import time even when the suite doesn't opt into MRMR.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlframe.feature_selection.filters import MRMR  # noqa: F401

logger = logging.getLogger(__name__)


def _build_pre_pipelines(
    use_ordinary_models: bool,
    rfecv_models: list[str],
    rfecv_models_params: dict[str, Any],
    use_mrmr_fs: bool,
    mrmr_kwargs: dict[str, Any],
    custom_pre_pipelines: dict[str, Any] | None = None,
    rfecv_leakage_corr_threshold: float | None = 0.95,
    rfecv_mbh_adaptive_threshold: int = 30,
    use_boruta_shap: bool = False,
    boruta_shap_kwargs: dict[str, Any] | None = None,
    use_sample_weights_in_fs: bool = False,
    mrmr_identity_cache: dict | None = None,
    target_type: Any = None,
    fs_random_seed: int | None = None,
    fs_use_groups: bool = False,
) -> tuple[list[Any], list[str]]:
    """Build lists of pre-pipelines and their names for feature selection.

    Both ``rfecv_leakage_corr_threshold`` and ``rfecv_mbh_adaptive_threshold`` are applied to every RFECV instance fetched from ``rfecv_models_params`` via ``setattr``; ``configure_training_params`` constructs those instances before the suite-level config is in scope, so this is the canonical place to override the suite-controllable knobs without rebuilding the RFECV objects.

    ``use_boruta_shap`` appends a BorutaShap selector AFTER MRMR / RFECV: SHAP-driven Boruta is a comparatively-expensive wrapper (per-trial TreeExplainer on a doubled feature matrix) so it makes sense to evaluate it as an alternative branch rather than chained behind the cheaper selectors. Default OFF preserves the legacy pre_pipelines ordering byte-for-byte.

    ``use_sample_weights_in_fs`` (``FeatureSelectionConfig.use_sample_weights_in_fs``): when True, stamps the
    marker attribute ``_mlframe_use_sample_weights_in_fs_`` on every MRMR / RFECV instance so the suite-level
    fit driver knows to forward the active ``sample_weight`` via fit_params (weight-aware FS, FS cache misses
    per weight schema). When False (default), the marker is False and the suite skips weight forwarding so the
    FS cache stays valid across weight iterations and selected features reflect the uniform-weight assumption.
    """
    pre_pipelines = []
    pre_pipeline_names = []

    if use_ordinary_models:
        pre_pipelines.append(None)
        pre_pipeline_names.append("")

    if not rfecv_models:
        rfecv_models = []
    if not rfecv_models_params:
        rfecv_models_params = {}
    unknown_rfecv_models = [m for m in rfecv_models if m not in rfecv_models_params]
    if unknown_rfecv_models:
        raise ValueError(f"Unknown RFECV model(s): {unknown_rfecv_models}. " f"Available: {list(rfecv_models_params.keys())}")
    for rfecv_model_name in rfecv_models:
        _rfecv_instance = rfecv_models_params[rfecv_model_name]
        # Suite-level overrides win over the RFECV defaults. Use sklearn's ``set_params`` instead of raw
        # ``setattr`` so any future property-setter side effects (e.g. recomputing a derived bound) fire as
        # the constructor would; ``set_params`` is the documented sklearn API for post-construction kwarg
        # overrides and validates the parameter names against ``get_params``. Falls back to ``setattr`` for
        # non-BaseEstimator instances used in tests / custom wrappers that don't implement set_params.
        if _rfecv_instance is not None:
            _rfecv_overrides = {
                "leakage_corr_threshold": rfecv_leakage_corr_threshold,
                "mbh_adaptive_threshold": rfecv_mbh_adaptive_threshold,
            }
            _set_params = getattr(_rfecv_instance, "set_params", None)
            if callable(_set_params):
                try:
                    _set_params(**_rfecv_overrides)
                except (ValueError, TypeError):
                    for _k, _v in _rfecv_overrides.items():
                        setattr(_rfecv_instance, _k, _v)
            else:
                for _k, _v in _rfecv_overrides.items():
                    setattr(_rfecv_instance, _k, _v)
            # Reproducibility: when the operator did not pin an RFECV random_state, default it from the
            # split seed so the whole pipeline (split + FS + model) is reproducible from one seed. An
            # explicitly-set random_state is left untouched.
            if fs_random_seed is not None and getattr(_rfecv_instance, "random_state", None) is None:
                _seed_set = getattr(_rfecv_instance, "set_params", None)
                if callable(_seed_set):
                    try:
                        _seed_set(random_state=int(fs_random_seed))
                    except (ValueError, TypeError):
                        setattr(_rfecv_instance, "random_state", int(fs_random_seed))
                else:
                    setattr(_rfecv_instance, "random_state", int(fs_random_seed))
            # Suite-internal marker (user constraint: keep as setattr; not a public RFECV constructor kwarg).
            setattr(_rfecv_instance, "_mlframe_use_sample_weights_in_fs_", bool(use_sample_weights_in_fs))
            # Dedicated dispatch marker so downstream report-build / cache code can identify the selector
            # kind without class-name string matching or abusing the weight-marker as a type tag.
            setattr(_rfecv_instance, "_mlframe_selector_kind_", "RFECV")
        pre_pipelines.append(_rfecv_instance)
        pre_pipeline_names.append(f"{rfecv_model_name} ")

    if use_mrmr_fs:
        # MRMR handles NaN natively via ``nan_strategy`` (default "separate_bin" routes NaN rows to a
        # dedicated discretization bin instead of imputing them; see MRMR._validate_inputs). Wrapping in
        # SimpleImputer would discard that signal and silently degrade downstream NaN-aware backends
        # (catboost / lgb / xgb). Registry-driven dispatch (A-Arch-002): adding a fourth selector
        # registers a spec instead of touching this function.
        from mlframe.feature_selection.registry import get as _get_selector_spec
        _mrmr_spec = _get_selector_spec("MRMR")
        # Reproducibility: default MRMR's seed from the split seed when the operator didn't pass one
        # (random_seed=None makes MRMR non-deterministic via pid^id derivation). Explicit seeds win.
        mrmr_kwargs = dict(mrmr_kwargs or {})
        if fs_random_seed is not None and mrmr_kwargs.get("random_seed") is None and mrmr_kwargs.get("random_state") is None:
            mrmr_kwargs["random_seed"] = int(fs_random_seed)
        # MRMR's MI estimator is group-naive: under a group-aware split, silently ignoring groups risks
        # cross-group leakage in the MI estimate on panel / session data. Default strict_groups=True so a
        # group-threaded fit raises loudly rather than computing group-naive MI. Operator override wins.
        if fs_use_groups and "strict_groups" not in mrmr_kwargs:
            mrmr_kwargs["strict_groups"] = True
        _mrmr = _mrmr_spec.instantiate(**mrmr_kwargs)
        setattr(_mrmr, "_mlframe_use_sample_weights_in_fs_", bool(use_sample_weights_in_fs))
        setattr(_mrmr, "_mlframe_selector_kind_", "MRMR")
        # When the suite caller passes a ctx-scoped cache dict (default per FeatureSelectionConfig.mrmr_identity_cache_scope="ctx"),
        # stamp it on the MRMR instance so fit-time identity-cache reads/writes route to the suite-bounded dict instead of the
        # process-global module-level cache. None falls back to the module-level cache (mrmr_identity_cache_scope="process").
        if mrmr_identity_cache is not None:
            setattr(_mrmr, "_mlframe_identity_cache_override_", mrmr_identity_cache)
        pre_pipelines.append(_mrmr)
        pre_pipeline_names.append("MRMR ")

    if use_boruta_shap:
        # Registry-driven dispatch (A-Arch-002). The BorutaShap spec hides the lazy-import behind
        # ``instantiate`` so shap / matplotlib / seaborn (~2s cold cost) only load when this branch fires.
        from mlframe.feature_selection.registry import get as _get_selector_spec
        _bs_spec = _get_selector_spec("BorutaShap")
        # BorutaShap's ``classification`` defaults to True, so the inner
        # default model is RandomForestClassifier --
        # which raises ValueError("Unknown label type: 'continuous'") inside
        # sklearn.multiclass on regression targets. When the caller hasn't
        # set ``classification`` explicitly in boruta_shap_kwargs AND
        # target_type is known, derive it from target_type so the inner
        # RandomForestRegressor is picked on regression targets.
        _bs_kwargs = dict(boruta_shap_kwargs or {})
        if "classification" not in _bs_kwargs and target_type is not None:
            _tt_str = str(target_type).lower()
            # TargetTypes enum stringifies to e.g. "targettypes.regression"; substring match handles
            # both the enum and a plain string variant.
            _is_regression = "regression" in _tt_str
            _bs_kwargs["classification"] = not _is_regression
        _bs = _bs_spec.instantiate(**_bs_kwargs)
        setattr(_bs, "_mlframe_selector_kind_", "BorutaShap")
        pre_pipelines.append(_bs)
        pre_pipeline_names.append("BorutaShap ")

    if custom_pre_pipelines:
        # Clone every user-supplied pre-pipeline before insertion so fit-time
        # state from one model never leaks across the others in this suite.
        # sklearn.base.clone is the canonical path; non-BaseEstimator objects
        # fall back to copy.deepcopy so callers can pass custom transformers
        # that don't implement the sklearn estimator protocol.
        import copy as _copy
        try:
            from sklearn.base import clone as _sk_clone
        except Exception:
            _sk_clone = None
        for pipeline_name, pipeline_obj in custom_pre_pipelines.items():
            try:
                _cloned = _sk_clone(pipeline_obj) if _sk_clone is not None else _copy.deepcopy(pipeline_obj)
            except Exception:
                _cloned = _copy.deepcopy(pipeline_obj)
            pre_pipelines.append(_cloned)
            pre_pipeline_names.append(f"{pipeline_name} ")

    return pre_pipelines, pre_pipeline_names
