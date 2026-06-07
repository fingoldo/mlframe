"""Prediction + load entry points for mlframe trained suites."""
from __future__ import annotations

import glob
import logging
import os
import pickle as _pickle
from collections import defaultdict
from copy import deepcopy
from os.path import exists, join

from scipy import stats
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import polars as pl
from pyutilz.strings import slugify

from ..configs import TargetTypes
from ..extractors import FeaturesAndTargetsExtractor
from ..io import load_mlframe_model
from ..pipeline import prepare_df_for_catboost
from ..cb import _predict_with_fallback
from ..utils import drop_columns_from_dataframe, get_pandas_view_of_polars_df
from .utils import (
    DEFAULT_PROBABILITY_THRESHOLD,
    _drop_cols_df,
    _setup_model_directories,
    _validate_input_columns_against_metadata,
    _validate_trusted_path,
)

logger = logging.getLogger(__name__)


_SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})
_CURRENT_SCHEMA_VERSION = 2


def _validate_metadata_version_envelope(metadata: dict, models_path: str) -> None:
    """Wave 19 P0 #2: validate the version-envelope fields the WRITE side
    has been populating but the READ side previously ignored.

    Pre-fix (before 2026-05-20) the load path never checked
    ``metadata["schema_version"]`` or
    ``metadata["composite_target_env_signature"]``, so a bundle written
    by code path A could be silently consumed by code path B that
    interpreted the same field set differently. Worst case: composite
    targets specs (schema_version=2 contract) interpreted by code that
    still assumed the v1 layout, then silent wrong predictions.

    Severity policy:
    - Missing ``schema_version`` AND no composite_target_specs ->
      treat as legacy v1; INFO-log once, accept (back-compat).
    - Missing ``schema_version`` AND composite_target_specs present ->
      raise: the bundle can't be safely loaded without the version
      stamp because the composite-spec contract changed at v2.
    - Unsupported ``schema_version`` (not in
      ``_SUPPORTED_SCHEMA_VERSIONS``) -> raise.
    - ``schema_version`` < ``_CURRENT_SCHEMA_VERSION`` -> WARN, continue.
    - ``composite_target_env_signature`` recorded but the live
      ``env_signature()`` differs in major/minor lib versions -> WARN
      with both signatures; predict proceeds (booster libraries are
      typically forward-compatible for minor versions, but operators
      should see the skew before chasing weird metric drift).
    """
    if not isinstance(metadata, dict):
        # Some legacy bundles used a SimpleNamespace; nothing to validate.
        return
    schema_version = metadata.get("schema_version")
    has_composite = bool(metadata.get("composite_target_specs"))
    if schema_version is None:
        if has_composite:
            raise ValueError(
                f"mlframe predict: bundle at {models_path!r} contains "
                f"composite_target_specs but lacks schema_version. The "
                f"composite-spec contract requires schema_version >= 2. "
                f"Refusing to load: this bundle was written by code "
                f"older than 2026-02 and the spec semantics it expects "
                f"are unknowable from the artifact alone. Retrain on "
                f"current mlframe."
            )
        logger.info(
            "mlframe predict: bundle at %s has no schema_version "
            "(legacy v1 artifact, no composite specs) -- loading with "
            "v1 semantics.", models_path,
        )
        return
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"mlframe predict: bundle at {models_path!r} has unsupported "
            f"schema_version={schema_version!r}. This mlframe build "
            f"supports {sorted(_SUPPORTED_SCHEMA_VERSIONS)}. Either "
            f"retrain on current mlframe OR pin to the mlframe version "
            f"that wrote this schema."
        )
    if schema_version < _CURRENT_SCHEMA_VERSION:
        logger.warning(
            "mlframe predict: bundle at %s was written with "
            "schema_version=%d, current is %d. Loading with backward-"
            "compat semantics; some newer attributes may be missing.",
            models_path, schema_version, _CURRENT_SCHEMA_VERSION,
        )
    # Composite-target env_signature drift check (booster lib versions).
    saved_sig = metadata.get("composite_target_env_signature")
    if saved_sig is not None:
        try:
            from ..composite import env_signature as _env_sig
            live_sig = _env_sig()
        except Exception as _e_env:
            logger.debug(
                "mlframe predict: env_signature() unavailable (%s); "
                "skipping composite-env skew check.", _e_env,
            )
            return
        if live_sig != saved_sig:
            logger.warning(
                "mlframe predict: composite-target env signature drift "
                "between train and predict. Saved: %r. Live: %r. Booster "
                "libraries are typically forward-compatible for minor "
                "versions; if you see unexplained metric drift, retrain "
                "on the live environment.", saved_sig, live_sig,
            )


def _polars_native_class_names() -> tuple[str, ...]:
    """Bare class-name allowlist for the polars fastpath: CatBoost and XGBoost sklearn-API estimators (and the
    DMatrix-reuse shim subclasses).

    Returns bare names (not class objects) so we don't pay the import cost when the user has neither library installed and so a class-name match still works across the entire MRO -- the XGB shims (``XGBRegressorWithDMatrixReuse``, ``XGBClassifierWithDMatrixReuse``) inherit from XGBRegressor / XGBClassifier so the MRO check below catches them, but listing the shim names too keeps the heuristic explicit."""
    return (
        "CatBoostClassifier", "CatBoostRegressor", "CatBoostRanker",
        "XGBClassifier", "XGBRegressor", "XGBRanker",
        "XGBClassifierWithDMatrixReuse", "XGBRegressorWithDMatrixReuse",
    )


def _is_polars_native_model(model_obj: Any) -> bool:
    """Return True if ``model_obj.model`` is a CB or XGB sklearn-API class that accepts a polars DataFrame directly in ``predict`` / ``predict_proba``. MRO-aware so DMatrix-reuse shims inheriting from XGBRegressor / XGBClassifier are recognised even though their bare class names differ."""
    if model_obj is None:
        return False
    _m = getattr(model_obj, "model", model_obj)
    if _m is None:
        return False
    _allowed = _polars_native_class_names()
    return any(cls.__name__ in _allowed for cls in type(_m).__mro__)


def _ensure_pandas_view(df: Any, view_cache: dict) -> Any:
    """Return a pandas view of ``df``. If ``df`` is already pandas, returned as-is. If polars, the converted view is cached in ``view_cache`` keyed by ``id(df)`` so repeated calls on the same source frame within one predict call pay one conversion."""
    if not isinstance(df, pl.DataFrame):
        return df
    _src_id = id(df)
    _view = view_cache.get(_src_id)
    if _view is None:
        _view = get_pandas_view_of_polars_df(df)
        view_cache[_src_id] = _view
    return _view




def _load_ct_ensemble_entries(models_path: str, slug_to_original_target_type: dict, slug_to_original_target_name: dict, trusted_root: str | None = None) -> dict:
    """Scan ``models_path`` for cross-target ensemble dumps stored under ``<target_type_slug>/_CT_ENSEMBLE__<original_target>/CT_ENSEMBLE.dump`` and return a nested dict keyed ``{target_type: {_CT_ENSEMBLE__<orig>: [entry]}}`` so callers can merge it into the loaded ``models`` structure."""
    out: dict = defaultdict(lambda: defaultdict(list))
    _ct_files = glob.glob(join(models_path, "**", "_CT_ENSEMBLE__*", "*.dump"), recursive=True)
    for _f in _ct_files:
        _validate_trusted_path(_f, trusted_root or os.path.abspath(models_path))
        _rel = os.path.relpath(_f, models_path)
        _parts = _rel.split(os.sep)
        if len(_parts) < 3:
            continue
        _tt_slug = _parts[0]
        _tname_slug = _parts[1]
        if not _tname_slug.startswith("_CT_ENSEMBLE__"):
            continue
        _tt = slug_to_original_target_type.get(_tt_slug, _tt_slug)
        # The _CT_ENSEMBLE__<orig> directory name is the in-memory dict key, so reuse it verbatim (no slugify round-trip).
        _ct_key = _tname_slug
        _entry = load_mlframe_model(_f)
        if _entry is not None:
            out[_tt][_ct_key].append(_entry)
    return out


def _combine_probs(
    all_probs: list,
    flavour: str | None,
    *,
    quantile_alphas: Sequence[float] | None = None,
    quantile_mode: str = "sort",
    rrf_k: int = 60,
    ensure_prob_limits: bool = True,
    is_calibrated_per_model: Sequence[bool] | None = None,
    metadata: dict | None = None,
    target_label: str | None = None,
) -> np.ndarray:
    """Combine per-model prediction probabilities using the training-time-selected ``flavour``.

    Delegates to the canonical ``combine_probs`` helper in ``mlframe.models.ensembling`` to guarantee
    byte-identical (within fp64 tolerance) replay against the train-stamped output. ``rrf_k`` defaults
    to 60 but the caller (predict pipeline) should plumb the persisted ``metadata["rrf_k"]`` /
    ``metadata["ensembles_chosen_params"]`` so a non-default-k train run replays correctly.

    When ``quantile_alphas`` is provided the aggregated output is treated as a quantile-regression matrix
    ``(N, K=len(alphas))`` and ``fix_quantile_crossing`` is applied after the ensemble step -- arithmetic /
    harmonic / quadratic aggregations can break per-row monotonicity even when each member's own quantiles
    are monotone. Without this the SKEW-QUANTILE-ENS bug surfaced.

    Arch-4: ``is_calibrated_per_model`` carries one bool per member of ``all_probs`` indicating whether
    that model's probabilities have been post-hoc calibrated. A mix (some True, some False) produces a
    ``logger.warning`` because blending calibrated with uncalibrated probs degrades calibration of the
    ensemble; the function does NOT refuse (user policy: WARN, NOT REFUSE). When ``metadata`` is
    supplied, ``metadata["ensembles_calibrated"]`` is stamped as a bool reflecting "all members
    calibrated" (True only when every flag is True; False otherwise).
    """
    from mlframe.models.ensembling import combine_probs as _shared_combine_probs

    if is_calibrated_per_model is not None:
        _flags = [bool(f) for f in is_calibrated_per_model]
        if _flags and any(_flags) and not all(_flags):
            _n_cal = sum(_flags)
            logger.warning(
                "[_combine_probs] mixing calibrated and uncalibrated probabilities for ensemble"
                "%s: %d/%d members are calibrated. Blending the two degrades the ensemble's "
                "calibration; the predict path still proceeds (Arch-4 WARN, not REFUSE) but the "
                "deployed reliability diagram will diverge from each member's individual one.",
                f" target={target_label!r}" if target_label else "",
                _n_cal, len(_flags),
            )
        if isinstance(metadata, dict) and _flags:
            metadata["ensembles_calibrated"] = bool(all(_flags))

    stacked = np.stack(all_probs)
    # When the caller passes ``quantile_alphas`` (even an empty list) they are
    # signalling "this is quantile-regression mode, outputs are unbounded".
    # ``ensure_prob_limits=True`` would clip those raw quantile predictions to
    # [0, 1] and destroy them (observed in
    # test_combine_probs_quantile_alphas_zero_alpha_count_passes_through:
    # member preds in the 1.0-5.0 range were all clipped to 1.0 and the
    # "mean" flavour returned [[1, 1, 1]] instead of the actual mean
    # [[2, 3.5, 4.5]]). Disable the clip in quantile mode; the
    # ``ensure_prob_limits`` flag stays the source of truth in non-quantile
    # (classification) paths.
    _effective_ensure_prob_limits = ensure_prob_limits and (quantile_alphas is None)
    combined = _shared_combine_probs(
        stacked, flavour or "arithm",
        rrf_k=int(rrf_k),
        ensure_prob_limits=_effective_ensure_prob_limits,
    )

    if quantile_alphas is not None and combined.ndim == 2:
        # P2-6: when the aggregated matrix's K dim does not match the alpha tuple, log + skip
        # rather than silently letting an aligned-by-coincidence call apply the wrong fix.
        if combined.shape[1] == len(quantile_alphas):
            try:
                from ..quantile_postproc import fix_quantile_crossing
                combined = fix_quantile_crossing(combined, quantile_alphas, mode=quantile_mode)
            except Exception as _qe:
                logger.warning("[_combine_probs] fix_quantile_crossing(post-aggregation) failed: %s", _qe)
        else:
            logger.warning(
                "[_combine_probs] quantile_alphas has %d entries but aggregated matrix has %d cols; "
                "skipping fix_quantile_crossing to avoid mis-aligned column rewrites.",
                len(quantile_alphas), combined.shape[1],
            )
    return combined




def _coerce_cat_dtype_for_lgb_xgb(input_for_model, *, model, cat_features, enum_domains=None):
    """Cast cat_features to pandas ``category`` (or pl.Enum / pl.Categorical for polars XGB).

    Wave 89 (2026-05-21): extracted from the predict.py:1372 mega-try body.
    Combines two adjacent ~40-line blocks (LGB + XGB) that share the same
    "detect-model-family-by-module + iterate cat_features + cast non-category
    to category" structure. Returns the possibly-mutated input_for_model.

    Why this dance:
      - LGB auto-detects categorical_feature from input dtype at predict and
        compares against the fit-time spec; object / float64 cat_low triggers
        "categorical_feature do not match". Cast to ``category`` only for LGB
        so sklearn HGB / linear models that reject categorical dtype are not
        broken.
      - XGB's QuantileDMatrix builder rejects object / pl.String / pl.LargeString
        cat cols with "Invalid columns: cat_low: object" (pandas path) or
        "KeyError: DataType(large_string)" (polars path). Cast to pandas
        ``category`` for the pandas path, pl.Categorical for the polars path.
    """
    if not cat_features or not hasattr(input_for_model, "columns"):
        return input_for_model
    _model_module = type(model).__module__ or ""
    _model_cls_name = type(model).__name__
    _is_lgb = (
        _model_module.startswith("lightgbm")
        or _model_module.endswith("lgb_shim")
        or "LGBM" in _model_cls_name
    )
    _is_xgb = (
        _model_module.startswith("xgboost")
        or _model_module.endswith("xgb_shim")
        or "XGB" in _model_cls_name
    )
    if not (_is_lgb or _is_xgb):
        return input_for_model

    # Pandas path (LGB + XGB both use the same assign(**dict) pattern --
    # the prior implementation's input_for_model.copy() allocated a fresh
    # copy of every column even when only 1-2 cat cols needed casting; the
    # assign(**) keeps BlockManager-level reuse for un-cast columns on
    # pandas >= 2.0, the biggest single allocation saving on wide frames).
    if not isinstance(input_for_model, pl.DataFrame):
        _to_cast: dict[str, Any] = {}
        for _cf in cat_features:
            if _cf not in input_for_model.columns:
                continue
            try:
                if input_for_model[_cf].dtype.name != "category":
                    _to_cast[_cf] = input_for_model[_cf].astype("category")
            except Exception as _exc:
                logger.debug(
                    "predict_from_models: %s cat-cast for %r failed (%s); leaving as-is",
                    "LGB" if _is_lgb else "XGB", _cf, type(_exc).__name__,
                )
        if _to_cast:
            input_for_model = input_for_model.assign(**_to_cast)
        return input_for_model

    # Polars path: only the XGB branch needs this -- LGB on polars reads
    # categorical dtype natively without an explicit cast.
    if _is_xgb:
        _pl_cast_exprs = []
        _enum_domains = enum_domains or {}
        for _cf in cat_features:
            if _cf not in input_for_model.columns:
                continue
            _dt = input_for_model.schema.get(_cf)
            if _dt in (pl.String, pl.Utf8) or str(_dt).startswith("LargeString"):
                _dom = _enum_domains.get(_cf)
                if _dom:
                    # Persisted Enum domain from training: out-of-domain values cast to null via strict=False (matches training semantics for "truly unseen" test categories) and avoids widening the polars global string cache.
                    _pl_cast_exprs.append(pl.col(_cf).cast(pl.Enum(list(_dom)), strict=False))
                else:
                    # Legacy bundle without persisted enum_domains: fall back to pl.Categorical with a one-time WARN. Categorical participates in the process-wide string cache that grows monotonically; subsequent inference calls accumulate stale categories. Re-train + re-save the bundle to populate enum_domains.
                    logger.warning(
                        "predict_from_models: XGB polars cat-cast for %r falling back to pl.Categorical (no enum_domains in bundle). Widens global string cache; re-train+save to persist enum domain.",
                        _cf,
                    )
                    _pl_cast_exprs.append(pl.col(_cf).cast(pl.Categorical))
        if _pl_cast_exprs:
            try:
                input_for_model = input_for_model.with_columns(_pl_cast_exprs)
            except Exception as _exc:
                logger.debug(
                    "predict_from_models: XGB polars cat-cast failed (%s); leaving as-is",
                    type(_exc).__name__,
                )
    return input_for_model




def _is_post_hoc_calibrated_model(model_obj: Any) -> bool:
    """Detect whether ``model_obj`` is a post-hoc calibrated wrapper.

    Class-name matching keeps the heavy training-module imports out of the predict path. Recognises
    the two wrappers stamped by mlframe.training: ``_PostHocCalibratedModel`` (single-output) and
    ``_PostHocMultiCalibratedModel`` (multi-output). Anything else is treated as uncalibrated for
    Arch-4 mix detection.
    """
    if model_obj is None:
        return False
    try:
        _name = type(model_obj).__name__
    except Exception:
        return False
    return _name in ("_PostHocCalibratedModel", "_PostHocMultiCalibratedModel")


def _resolve_quantile_alphas(metadata: dict, target_type: Any, target_name: Any, model_obj: Any = None) -> Sequence[float] | None:
    """Resolve quantile alphas for a (target_type, target_name) pair.

    Tries metadata-stored keys first; then falls back to model-object introspection
    (CatBoost MultiQuantile / XGB ``quantile_alpha`` / sklearn ``QuantileRegressor.alphas_``).
    Returns ``None`` when the target type isn't quantile or alphas can't be recovered.
    """
    if target_type not in ("quantile_regression", "regression_quantile") and str(target_type) != "quantile_regression":
        return None
    # Metadata-stored alphas: support a few shapes that have been used historically.
    if isinstance(metadata, dict):
        _qa = metadata.get("quantile_alphas")
        if isinstance(_qa, dict):
            _by_tt = _qa.get(target_type) or _qa.get(str(target_type))
            if isinstance(_by_tt, dict):
                _alphas = _by_tt.get(target_name) or _by_tt.get(str(target_name))
                if _alphas:
                    return list(_alphas)
            elif isinstance(_by_tt, (list, tuple)):
                return list(_by_tt)
        elif isinstance(_qa, (list, tuple)):
            return list(_qa)
    # Model-object introspection: a single member usually carries the alpha list (CB MultiQuantile, XGB quantile_alpha).
    if model_obj is not None:
        _m = getattr(model_obj, "model", model_obj)
        for _attr in ("quantile_alpha", "alphas_", "alphas", "_quantile_alphas"):
            _val = getattr(_m, _attr, None)
            if _val is not None and hasattr(_val, "__iter__"):
                try:
                    _alphas = [float(a) for a in _val]
                    if _alphas:
                        return _alphas
                except (TypeError, ValueError):
                    continue
    return None


def _resolve_chosen_ensemble_params(metadata: dict, target_type: Any = None, target_name: Any = None) -> dict:
    """Look up persisted ensemble replay params (rrf_k, etc.) for ``(target_type, target_name)``.

    C-P1-11: predict-side blend math must use the SAME rrf_k the train side recorded; pre-fix the
    predict path hard-coded k=60 which silently drifted when a user changed it at train time.
    ``metadata["ensembles_chosen_params"][tt][tname] = {"rrf_k": ...}`` is written by the per-target
    stamper in ``_phase_train_one_target``; this helper tolerates the absence (legacy models) and
    returns ``{}`` -- callers should treat that as "use defaults".
    """
    _ep = metadata.get("ensembles_chosen_params") if isinstance(metadata, dict) else None
    if not isinstance(_ep, dict):
        return {}
    if target_type is not None:
        _by_tt = _ep.get(target_type) or _ep.get(str(target_type))
        if isinstance(_by_tt, dict):
            if target_name is not None:
                _params = _by_tt.get(target_name) or _by_tt.get(str(target_name))
                if isinstance(_params, dict):
                    return _params
            # single-entry fallback
            if len(_by_tt) == 1:
                _only = next(iter(_by_tt.values()))
                if isinstance(_only, dict):
                    return _only
    # suite-wide fallback when only one (tt, tname) entry exists across the map
    _all_params: list[dict] = []
    for _v in _ep.values():
        if isinstance(_v, dict):
            for _vv in _v.values():
                if isinstance(_vv, dict):
                    _all_params.append(_vv)
    if len(_all_params) == 1:
        return _all_params[0]
    return {}


def _resolve_chosen_flavour(metadata: dict, target_type: Any = None, target_name: Any = None) -> str | None:
    """Look up the persisted chosen ensemble flavour for ``(target_type, target_name)``.

    Arch-3: ``metadata['ensembles_chosen']`` is sub-keyed per ensemble family --
    ``{"simple": {tt: {tname: flavour}}, "cross_target": {tt: {tname: flavour}}}``.
    ``_CT_ENSEMBLE__*``-prefixed target names dispatch to the ``cross_target`` bucket; everything
    else reads the ``simple`` bucket.
    """
    _ec = metadata.get("ensembles_chosen") if isinstance(metadata, dict) else None
    if _ec is None:
        return None
    if isinstance(_ec, str):
        return _ec
    if not isinstance(_ec, dict):
        return None
    # Pick bucket by target-name prefix. CT_ENSEMBLE keys live under "cross_target"; everything
    # else under "simple". Falls back to whichever bucket has the entry when only one is populated.
    _is_ct = isinstance(target_name, str) and target_name.startswith("_CT_ENSEMBLE__")
    _bucket = _ec.get("cross_target" if _is_ct else "simple")
    if not isinstance(_bucket, dict):
        # Legacy / flat-schema fallback: when the metadata was written before
        # the Arch-3 ``simple`` / ``cross_target`` sub-keying landed,
        # ``ensembles_chosen`` looked like ``{target_type: {target_name:
        # flavour}}`` directly. Detect that shape by checking whether any
        # known TargetType-style key sits at the top level; if so, use _ec
        # itself as the bucket so old persisted metadata still resolves.
        _looks_flat = isinstance(_ec, dict) and any(
            (isinstance(_v, dict) and _v and all(isinstance(_kk, str) and isinstance(_vv, str) for _kk, _vv in _v.items()))
            for _v in _ec.values()
        )
        _bucket = _ec if _looks_flat else None
    if _bucket is not None and target_type is not None:
        _by_tt = _bucket.get(target_type) or _bucket.get(str(target_type))
        if isinstance(_by_tt, dict):
            if target_name is not None:
                _fl = _by_tt.get(target_name) or _by_tt.get(str(target_name))
                if _fl:
                    return _fl
            # Single-target fallback -- if there's only one entry, use it.
            if len(_by_tt) == 1:
                return next(iter(_by_tt.values()))
        elif isinstance(_by_tt, str):
            return _by_tt
    # No (tt, tname) match in the dispatched bucket -- if the whole map has a single flavour
    # everywhere across BOTH buckets, use it.
    _all = []
    for _fam in _ec.values():
        if isinstance(_fam, dict):
            for _v in _fam.values():
                if isinstance(_v, dict):
                    _all.extend(_v.values())
                elif isinstance(_v, str):
                    _all.append(_v)
        elif isinstance(_fam, str):
            _all.append(_fam)
    _all_set = {f for f in _all if f}
    if len(_all_set) == 1:
        # Set with cardinality 1 is deterministic, but sort-then-pick documents the intent and survives the
        # accidental len==N>1 case if a future caller widens the guard.
        return sorted(_all_set)[0]
    return None


def _replay_suite_datetime_decomposition(df, metadata, verbose: int = 0):
    """Replay the training-side ``create_date_features`` calls that the SUITE owns and drop any FTE-owned datetime source columns (FTE-handled cols are re-derived by the FTE's own ``transform`` on predict input, but FTE leaves the raw source via ``delete_original_cols=False`` -- the suite-side decomposition phase dropped those at training time, so predict must too or the raw datetime64 col will reach LGB / XGB / sklearn and fail dtype promotion).

    The suite persists ``metadata["datetime_methods"] = {src_col: {accessor: dtype_name}}``; this helper re-applies the same expansion to the predict-time frame so derived columns are byte-identical to training. When a source col is missing (already FTE-derived or absent), the corresponding entry is skipped.
    """
    from mlframe.feature_engineering.basic import create_date_features
    import numpy as _np

    _fte_emitted_map = metadata.get("ftextractor_emitted_columns") or {}
    if _fte_emitted_map and hasattr(df, "columns"):
        _drop = [c for c in _fte_emitted_map.keys() if c in df.columns]
        if _drop:
            if isinstance(df, pl.DataFrame):
                df = df.drop(_drop)
            else:
                df = df.drop(columns=_drop)

    _methods_map = metadata.get("datetime_methods") or {}
    if not _methods_map:
        return df

    _dtype_resolvers = {"int8": _np.int8, "int16": _np.int16, "int32": _np.int32, "int64": _np.int64,
                        "uint8": _np.uint8, "uint16": _np.uint16, "uint32": _np.uint32, "uint64": _np.uint64,
                        "float32": _np.float32, "float64": _np.float64}
    _present_sources = [c for c in _methods_map.keys() if c in df.columns]
    if not _present_sources:
        return df
    # Group sources by their methods dict so we batch-decompose cols sharing the same expansion (typical case: every suite-owned datetime col uses the same configured methods).
    _by_methods: Dict[Tuple[Tuple[str, str], ...], List[str]] = {}
    for _src in _present_sources:
        _m = _methods_map[_src]
        _key = tuple(sorted((str(k), str(v)) for k, v in _m.items()))
        _by_methods.setdefault(_key, []).append(_src)
    for _methods_key, _srcs in _by_methods.items():
        _resolved_methods = {}
        for _accessor, _dtype_name in _methods_key:
            _resolved_methods[_accessor] = _dtype_resolvers.get(_dtype_name, _np.int8)
        if verbose:
            logger.info("Replaying datetime decomposition (%s) on %d source col(s): %s",
                        "/".join(sorted(_resolved_methods.keys())), len(_srcs), _srcs)
        df = create_date_features(df, cols=_srcs, delete_original_cols=True, methods=_resolved_methods)
    return df


def _slice_frame(df: Any, start: int, length: int) -> Any:
    """Polars/pandas-agnostic slice. Returns rows [start, start+length). Out-of-range tail truncates safely."""
    if isinstance(df, pl.DataFrame):
        return df.slice(start, length)
    if isinstance(df, pd.DataFrame):
        return df.iloc[start: start + length]
    raise TypeError(f"_slice_frame: unsupported type {type(df).__name__}")


def _concat_probs_dicts(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate per-batch probability dicts along axis 0 by model_name. Missing keys in any part are skipped
    (a model that crashed on one batch shouldn't poison the others; the caller's warn-and-continue handler
    already logged the failure)."""
    if not parts:
        return {}
    out: dict[str, np.ndarray] = {}
    keys: set[str] = set()
    for part in parts:
        keys.update(part.keys())
    for key in keys:
        chunks = [part[key] for part in parts if key in part and part[key] is not None]
        if chunks:
            out[key] = np.concatenate(chunks, axis=0)
    return out


def _run_batched(
    entry_fn: Callable,
    df: Any,
    predict_batch_rows: int,
    *args, **kwargs,
) -> dict[str, Any]:
    """Iterate ``entry_fn(_slice_frame(df, start, predict_batch_rows), *args, **kwargs)`` over batches and
    concatenate per-key arrays. Used by predict_mlframe_models_suite + predict_from_models when the caller sets
    ``predict_batch_rows`` to bound peak RSS on multi-million-row inference. Per-model probabilities /
    predictions / ensemble outputs are all concatenated row-wise; metadata + models_used + per_target dicts come
    from the FIRST batch (they're row-count-invariant)."""
    n = len(df)
    if n == 0:
        return entry_fn(df, *args, **kwargs)
    batch_outs: list[dict[str, Any]] = []
    _start = 0
    while _start < n:
        _length = min(predict_batch_rows, n - _start)
        _slice = _slice_frame(df, _start, _length)
        _out = entry_fn(_slice, *args, **kwargs)
        batch_outs.append(_out)
        _start += _length

    if len(batch_outs) == 1:
        return batch_outs[0]

    merged: dict[str, Any] = dict(batch_outs[0])  # carry metadata / models_used etc. from batch-0
    for _key in ("predictions", "probabilities", "per_target_probabilities", "per_target_predictions"):
        if _key in merged and isinstance(merged[_key], dict):
            merged[_key] = _concat_probs_dicts([b.get(_key, {}) or {} for b in batch_outs])
    for _key in ("ensemble_predictions", "ensemble_probabilities"):
        _parts = [b.get(_key) for b in batch_outs if b.get(_key) is not None]
        if _parts:
            try:
                merged[_key] = np.concatenate(_parts, axis=0)
            except ValueError:
                merged[_key] = _parts[0]
    # input_df concatenated row-wise so consumers reading the post-extensions frame from the result still see all rows.
    _input_parts = [b.get("input_df") for b in batch_outs if b.get("input_df") is not None]
    if _input_parts:
        try:
            if isinstance(_input_parts[0], pl.DataFrame):
                merged["input_df"] = pl.concat(_input_parts, how="vertical")
            else:
                merged["input_df"] = pd.concat(_input_parts, axis=0)
        except (TypeError, ValueError):
            merged["input_df"] = _input_parts[0]
    return merged






def load_mlframe_suite(models_path: str, trusted_root: str | None = None) -> tuple[dict, dict]:
    """
    Load a trained mlframe models suite from disk.

    Args:
        models_path: Path to the models directory (e.g., "data/models/target_name/model_name")

    Returns:
        Tuple of (models dict, metadata dict) in the same format as train_mlframe_models_suite:
        - models: Dict[target_type][target_name] = [model_obj, ...]
        - metadata: Dict with training configuration and artifacts
    """
    if not isinstance(models_path, str):
        raise TypeError(f"models_path must be string, got {type(models_path).__name__}")
    if not os.path.isdir(models_path):
        raise ValueError(f"models_path must be a valid directory, got: {models_path}")

    # Prefer pickle-proto5 + zstd; fall back to legacy metadata.joblib.
    metadata_file_new = join(models_path, "metadata.pkl.zst")
    metadata_file_uncompressed = join(models_path, "metadata.pkl")
    metadata_file_legacy = join(models_path, "metadata.joblib")
    if exists(metadata_file_new):
        metadata_file = metadata_file_new
        _kind = "pkl.zst"
    elif exists(metadata_file_uncompressed):
        metadata_file = metadata_file_uncompressed
        _kind = "pkl"
    elif exists(metadata_file_legacy):
        metadata_file = metadata_file_legacy
        _kind = "joblib"
    else:
        raise FileNotFoundError(
            f"Metadata file not found in {models_path}; expected one of "
            f"metadata.pkl.zst, metadata.pkl, metadata.joblib"
        )

    _root = trusted_root if trusted_root is not None else os.path.abspath(models_path)
    _validate_trusted_path(metadata_file, _root)
    if _kind == "pkl.zst":
        # ``pickle.loads`` of a zstd-decompressed in-memory buffer; the file path went through
        # ``_validate_trusted_path`` above and the version envelope is checked post-load below.
        # Route through safe_pickle.verify_sidecar so a tampered .pkl.zst is rejected by digest
        # mismatch before the loads(); legacy bundles without a sidecar still load (the trusted-
        # path validation + version envelope are the existing gates).
        from mlframe.utils.safe_pickle import verify_sidecar as _vsidecar
        import pickle as _pickle
        import zstandard as _zstd
        if not _vsidecar(metadata_file, allow_unverified=True):
            raise RuntimeError(
                f"predict_from_models: sha256 sidecar mismatch on {metadata_file!r}; refusing to load."
            )
        _dctx = _zstd.ZstdDecompressor()
        with open(metadata_file, "rb") as _f:
            metadata = _pickle.loads(_dctx.decompress(_f.read()))  # BARE_PICKLE_OK: in-memory buffer, sidecar already verified above
    elif _kind == "pkl":
        from mlframe.utils.safe_pickle import safe_load as _sload
        metadata = _sload(metadata_file, allow_unverified=True)
    else:
        metadata = joblib.load(metadata_file)
    # Wave 19 P0 #2: validate version envelope here too (the second predict
    # entry point at predict_from_models had the same dead-stamp blind spot).
    _validate_metadata_version_envelope(metadata, models_path)

    slug_to_original_target_type = metadata.get("slug_to_original_target_type", {})
    slug_to_original_target_name = metadata.get("slug_to_original_target_name", {})

    # Structure: models[target_type][target_name] = [model_obj, ...]
    # On-disk layout from _setup_model_directories: models_path/target_type/target_name/model.dump
    models = defaultdict(lambda: defaultdict(list))
    model_files = glob.glob(join(models_path, "**", "*.dump"), recursive=True)

    for model_file in model_files:
        rel_path = os.path.relpath(model_file, models_path)
        path_parts = rel_path.split(os.sep)

        if len(path_parts) >= 3:
            slugified_target_type = path_parts[0]
            slugified_target_name = path_parts[1]
            target_type = slug_to_original_target_type.get(slugified_target_type, slugified_target_type)
            target_name = slug_to_original_target_name.get(slugified_target_name, slugified_target_name)
        else:
            target_type = "unknown"
            target_name = "unknown"

        model_obj = load_mlframe_model(model_file)
        if model_obj is not None:
            models[target_type][target_name].append(model_obj)

    return dict(models), metadata


__all__ = [
    "predict_mlframe_models_suite",
    "predict_from_models",
    "load_mlframe_suite",
]


# ----------------------------------------------------------------------
# Sibling-module re-exports. Big predict entrypoints + pre_pipeline
# helpers live in ``_predict_main.py`` / ``_predict_pre_pipeline.py`` so
# this file stays below the 1k-LOC monolith threshold.
# ----------------------------------------------------------------------
from ._predict_main import (  # noqa: E402,F401
    predict_from_models, predict_mlframe_models_suite,
)
from ._predict_pre_pipeline import (  # noqa: E402,F401
    _apply_extensions_pipeline, _apply_pre_pipeline_with_passthrough, _try_predict_with_pp_fallback,
)
