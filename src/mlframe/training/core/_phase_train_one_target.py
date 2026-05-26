"""_train_one_target - per-target training entry point."""
from __future__ import annotations

import hashlib
import inspect
import logging
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import psutil as _ps_module  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _ps_module = None  # type: ignore[assignment]

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

# Cache for ``inspect.signature(cls.__init__)`` results -- the NGBoost / sklearn TypeError fallback
# branch calls ``inspect.signature`` on every TypeError, paying ~0.5-1ms per hit. The signature is
# purely a property of the class, so we memoize per-class and reuse across iterations.
#
# Keyed via WeakKeyDictionary on cls itself, NOT id(cls). Pre-fix used id() which Python recycles
# across GC events; a dynamically-built shim class (XGBClassifierWithDatasetReuse vs
# XGBRegressorWithDatasetReuse rebuilt between suite calls) could id()-collide with a freed
# predecessor and the cache returned the WRONG kwarg set, silently dropping a hyperparameter or
# pass-through extra-kwarg to a method that doesn't accept it. Weakref-keyed: when the class is
# GC'd, the cache entry vanishes too -- no recycle hazard.
import weakref as _weakref
_INIT_SIG_CACHE: "_weakref.WeakKeyDictionary[type, set[str]]" = _weakref.WeakKeyDictionary()


def _cached_init_params(cls) -> set[str]:
    """Return the set of accepted ``__init__`` kwargs for ``cls`` (excl. ``self``), memoised per-class."""
    cached = _INIT_SIG_CACHE.get(cls)
    if cached is None:
        cached = set(inspect.signature(cls.__init__).parameters) - {"self"}
        try:
            _INIT_SIG_CACHE[cls] = cached
        except TypeError:
            # WeakKeyDictionary rejects classes that can't be weakref'd (rare:
            # some C-extension types). Fall back to no caching for these -- the
            # signature lookup costs ~0.5-1ms which is fine on the slow path.
            pass
    return cached

from sklearn.base import clone

from functools import lru_cache

from pyutilz.strings import slugify as _slugify_raw
from pyutilz.system import tqdmu_lazy_start


@lru_cache(maxsize=512)
def _cached_slugify(name: str) -> str:
    """SLUGIFY-PER-TGT: memoize the pure slugify() helper so the per-target call is O(1) after the
    first hit. Bounded LRU prevents accidental unbounded growth in long-lived processes that
    re-train the suite on rotating target names."""
    return _slugify_raw(name)


# ----------------------------------------------------------------------
# Pack H wiring: auto-pick MAE / Huber loss for heavy-tail regression
# residuals.
# ----------------------------------------------------------------------


def _is_regression_target_type(target_type: Any) -> bool:
    """True when target_type designates a regression target.

    Imported lazily so the module load order stays unchanged on configs
    refactor; the configs module pulls in sklearn / pydantic which we
    don't want at import-time for this hot file.
    """
    try:
        from ..configs import TargetTypes as _TT
    except Exception:
        return False
    return target_type == _TT.REGRESSION


def _apply_loss_recommendation_in_place(
    *,
    models_params: dict,
    target_values: Any,
    composite_name: str,
    logger_: Any,
    verbose: bool,
) -> None:
    """Mutate ``models_params`` in place so CB / LGB / XGB inner estimators use a robust loss when ``target_values`` is heavy-tailed.

    Hooks the ``recommend_boosting_regression_loss`` helper (Pack H) onto the per-target loop: after ``select_target`` returns the per-backend templates, override their ``loss_function`` / ``objective`` based on excess-kurtosis of the target. Production motivator: a composite ``y-linres-lag1`` had excess_kurt=+2.40 (Laplace-like) but CB / LGB / XGB defaulted to RMSE objective and early-stopped at iter=4-10 -- the RMSE gradient collapses near zero on Laplace residuals.

    Backends:
    - CatBoost: ``set_params(loss_function='MAE' | 'Huber:delta=1.345')``.
    - LightGBM: ``set_params(objective='regression_l1' | 'huber')``.
    - XGBoost: ``set_params(objective='reg:absoluteerror' | 'reg:pseudohubererror')``.

    Non-applicable cases (Gaussian, degenerate input) are no-ops; the helper logs the rationale at INFO once per call so an operator can confirm the auto-switch fired (or did not) when reading suite output.

    Mutation is in-place to keep parity with the surrounding code (which already mutates ``models_params`` for cache restoration).
    """
    try:
        from ..loss_recommendation import recommend_boosting_regression_loss
    except Exception as _imp_err:
        # Best-effort: never crash training because the auto-loss helper failed to import.
        if verbose:
            logger_.warning(
                "[auto-loss] could not import loss recommendation helper: %s. "
                "Inner regression models keep their default loss.", _imp_err,
            )
        return

    try:
        rec = recommend_boosting_regression_loss(target_values)
    except Exception as _rec_err:
        if verbose:
            logger_.warning(
                "[auto-loss] recommend_boosting_regression_loss failed on target='%s': %s. "
                "Inner regression models keep their default loss.",
                composite_name, _rec_err,
            )
        return

    # Backend -> (param name on the estimator, recommended value).
    _backend_param = {
        "cb": ("loss_function", rec.get("cb")),
        "lgb": ("objective", rec.get("lgb")),
        "xgb": ("objective", rec.get("xgb")),
    }
    # Align eval_metric to the chosen objective so early-stopping
    # tracks the surface the optimiser is actually descending. Observed in prod:
    # raw target (low-kurt -> RMSE objective) was early-stopping at
    # iter=147 (CB) / 76 (LGB) on a 5000-iter cap because the SUITE DEFAULT
    # ``def_regr_metric="MAE"`` left eval_metric pinned to MAE while objective
    # was RMSE -- MAE-eval surface plateaued much earlier than RMSE objective
    # could keep improving. Mismatch caused systematic under-convergence,
    # boosters' TEST RMSE was WORSE than the trivial lag_predict baseline.
    def _eval_metric_for(_backend: str, _value: str) -> tuple[str, Any] | None:
        if _backend == "cb":
            if _value == "RMSE":
                return ("eval_metric", "RMSE")
            if _value == "MAE":
                return ("eval_metric", "MAE")
            if _value.startswith("Huber"):
                # Huber:delta=X IS a valid CB eval_metric (CatBoost 1.0+);
                # match the loss exactly so ES tracks the same surface
                # the optimiser descends. Prior code returned MAE which
                # has a constant-magnitude gradient and stops ES at iter=1
                # on small-residual composite targets (observed in prod:
                # CB pred [-25,+5] for T in [-45,+45], R2=-0.41).
                return ("eval_metric", _value)
        elif _backend == "lgb":
            if _value == "regression":
                return ("metric", "l2")  # l2 == rmse-squared, LGB stops on it
            if _value == "regression_l1":
                return ("metric", "l1")
            if _value == "huber":
                return ("metric", "huber")
        elif _backend == "xgb":
            if _value == "reg:squarederror":
                return ("eval_metric", "rmse")
            if _value == "reg:absoluteerror":
                return ("eval_metric", "mae")
            if _value == "reg:pseudohubererror":
                return ("eval_metric", "mphe")
        return None

    _applied: list[str] = []
    _skipped: list[str] = []
    for _backend, (_param_name, _value) in _backend_param.items():
        if not _value:
            _skipped.append(f"{_backend}:no_recommendation")
            continue
        _entry = models_params.get(_backend)
        if not isinstance(_entry, dict):
            _skipped.append(f"{_backend}:no_entry_in_models_params")
            continue
        _model = _entry.get("model")
        if _model is None or not hasattr(_model, "set_params"):
            _skipped.append(f"{_backend}:model_none_or_no_set_params")
            continue
        _set_kwargs: dict = {_param_name: _value}
        _em = _eval_metric_for(_backend, _value)
        if _em is not None:
            _set_kwargs[_em[0]] = _em[1]
        # Backend-specific extra params from the recommendation
        # (e.g. xgb_extra_params={"huber_slope": MAD*1.345} when XGB
        # gets reg:pseudohubererror; without this the slope defaults to
        # 1.0 and on T-scale composite targets with std~13 the loss
        # is effectively MSE on tails -> pred range blows out by 30x).
        _extra_params = rec.get(f"{_backend}_extra_params") or {}
        if isinstance(_extra_params, dict):
            _set_kwargs.update(_extra_params)
        try:
            _model.set_params(**_set_kwargs)
            _applied.append(
                f"{_backend}:{_param_name}={_value}"
                + (f",{_em[0]}={_em[1]}" if _em is not None else "")
                + (f",{'+'.join(_extra_params)}" if _extra_params else "")
            )
        except (ValueError, TypeError) as _set_err:
            # Backend may reject the combined value. Try objective-only as fallback.
            try:
                _model.set_params(**{_param_name: _value})
                _applied.append(f"{_backend}:{_param_name}={_value}")
            except (ValueError, TypeError) as _set_err2:
                if verbose:
                    logger_.debug(
                        "[auto-loss] %s.set_params(%s=%r) rejected: %s / %s. Keeping default.",
                        _backend, _param_name, _value, _set_err, _set_err2,
                    )

    if verbose and _applied:
        logger_.info(
            "[auto-loss] target='%s' excess_kurt=%.2f (n_finite=%d) -- %s. Applied: %s.",
            composite_name, float(rec.get("excess_kurt", float("nan"))),
            int(rec.get("n_finite", 0)),
            rec.get("rationale", ""), ", ".join(_applied),
        )
    # Surface skipped backends too: an operator who expected a custom objective on
    # ALL three backends gets no signal about which ones were dropped pre-fix.
    # Log at INFO (operator-relevant) only when at least one backend was applied,
    # so the surrounding noise is gated on the auto-loss path having actually fired
    # for some backend; otherwise this is silent (no skip-only spam on backends
    # that simply have no recommendation engine wired).
    if verbose and _applied and _skipped:
        logger_.info(
            "[auto-loss] target='%s' skipped backends: %s",
            composite_name, ", ".join(_skipped),
        )


# Back-compat alias so existing call sites that read ``slugify`` continue to work.
slugify = _cached_slugify

from mlframe.models.ensembling import score_ensemble
from ..configs import TargetTypes as _TargetTypes  # TARGETTYPES-IMPORT-LOOP: hoist out of inner write loop
from .._ram_helpers import estimate_df_size_mb, get_process_rss_mb, maybe_clean_ram_and_gpu
from ..phases import phase
from ..models import is_neural_model
from ..strategies import get_strategy
from ..train_eval import process_model, select_target
from ..utils import compute_model_input_fingerprint, filter_existing, get_pandas_view_of_polars_df, log_ram_usage
from ._misc_helpers import _build_tier_dfs, _compute_neural_max_time, _elapsed_str, _filter_polars_cat_features_by_dtype, _maybe_clear_shim_cache, _prep_polars_df, _split_preds_probs
from ._phase_diagnostics import run_per_target_diagnostics
from ._phase_dummy_baselines import run_dummy_baselines
from ._phase_temporal_audit import _format_temporal_audit_report, _plot_target_over_time
from ._setup_helpers import _build_common_params_for_target, _build_pre_pipelines, _build_process_model_kwargs, _setup_model_directories, _should_skip_catboost_metamodel
from ..strategies import PipelineCache

logger = logging.getLogger(__name__)


from ._ensemble_chooser import (
    _ENSEMBLE_RANK_METRIC_CANDIDATES,
    _choose_ensemble_flavour,
    _read_ensemble_metric,
)


def _unwrap_selector(pre_pipeline) -> Any:
    """Return the inner MRMR / RFECV / BorutaShap instance, unwrapping a sklearn Pipeline if needed.

    ``_build_pre_pipelines`` returns selectors directly (MRMR / a CatBoostRFECV instance / BorutaShap)
    OR a custom sklearn Pipeline whose last step may be a selector. Probe ``.steps[-1][1]`` lazily; if
    the input isn't a Pipeline or doesn't end in a selector, fall back to the input itself so callers
    can still introspect attributes on a plain selector.

    ``getattr`` is wrapped: pathological transformers that override ``__getattr__`` to raise on every
    miss (rare but seen in third-party FS libs) would break the report-build otherwise.
    """
    if pre_pipeline is None:
        return None
    try:
        _steps = getattr(pre_pipeline, "steps", None)
    except Exception:
        return pre_pipeline
    if isinstance(_steps, list) and _steps:
        _last = _steps[-1]
        if isinstance(_last, tuple) and len(_last) == 2:
            return _last[1]
    return pre_pipeline


def _selector_kind(selector) -> str | None:
    """Classify the fitted selector as 'MRMR' / 'RFECV' / 'BorutaShap' / None.

    Uses class-name suffix rather than ``isinstance`` so the import of MRMR / RFECV / BorutaShap stays
    confined to ``_build_pre_pipelines`` (BorutaShap pulls shap + matplotlib + seaborn). ``None`` for
    the ordinary (no-FS) branch and for unrecognised custom pipelines.
    """
    if selector is None:
        return None
    # Dedicated dispatch marker stamped by ``_build_pre_pipelines`` -- preferred over class-name
    # matching because it survives subclassing and doesn't conflate with the weight-aware marker.
    try:
        _kind_marker = getattr(selector, "_mlframe_selector_kind_", None)
    except Exception:
        _kind_marker = None
    if isinstance(_kind_marker, str) and _kind_marker in ("MRMR", "RFECV", "BorutaShap"):
        return _kind_marker
    try:
        _cls_name = type(selector).__name__
    except Exception:
        return None
    if _cls_name == "MRMR":
        return "MRMR"
    if "RFECV" in _cls_name:
        return "RFECV"
    if _cls_name == "BorutaShap":
        return "BorutaShap"
    # Defence: the suite stamps ``_mlframe_use_sample_weights_in_fs_`` on MRMR / RFECV but not
    # BorutaShap; if the marker exists, classify via attribute shape (support_ -> selector-like).
    try:
        _has_marker = hasattr(selector, "_mlframe_use_sample_weights_in_fs_") and hasattr(selector, "support_")
    except Exception:
        return None
    if _has_marker:
        if hasattr(selector, "cv_results_"):
            return "RFECV"
        return "MRMR"
    return None


def _selector_params_hash(selector) -> str | None:
    """Hash the selector's ``get_params()`` (sklearn) or ``__dict__`` (BorutaShap) so the report key
    invalidates when the operator changes selector hyperparameters between runs.

    Returns ``None`` on any introspection / serialisation failure so the report never blocks training.
    Uses ``repr`` over JSON because selector params can carry sklearn estimators / callables that JSON
    can't serialise; ``repr`` is stable per-process and good enough for change detection.
    """
    if selector is None:
        return None
    try:
        if hasattr(selector, "get_params"):
            _params = selector.get_params(deep=False)
        else:
            _params = {k: v for k, v in vars(selector).items() if not k.startswith("_") and not k.endswith("_")}
        _digest_src = repr(sorted(_params.items(), key=lambda kv: kv[0]))
        # digest_size=16 (128-bit) instead of 8 (64-bit): birthday-bound collision floor moves from
        # ~2^32 to ~2^64, harmless cost on a per-suite hash; protects against accidental dedup at
        # suite scale (targets x weights x models can produce thousands of param dicts).
        return hashlib.blake2b(_digest_src.encode("utf-8", errors="replace"), digest_size=16).hexdigest()
    except Exception:
        return None




def _compute_pipeline_cache_key(
    strategy_cache_key: str,
    pre_pipeline_name: str | None,
    feature_tier,
    supports_polars: bool,
    cat_features,
    text_features,
    embedding_features,
    train_df=None,
) -> str:
    """Build the PipelineCache lookup key for a (strategy, pre_pipeline, tier, kind, features) combo.

    The features digest folds (cat, text, embedding) lists through blake2b so cache HIT invalidates
    when the user reshapes those lists between sessions yet stays stable across list ordering;
    without it, a tier frame prepared for one (cat/text/embedding) split could be served to a later
    session that toggled a column's role. Sorting before serialization gives a deterministic byte
    stream: frozenset.__repr__ iterates in hash-seeded order (PYTHONHASHSEED) and would change the
    digest across processes for the same membership.

    CACHE-KEY-CONTENT-OMITTED-POLARS-SCHEMA: when ``train_df`` is supplied AND it's a polars frame,
    fold a tuple of (col_name, dtype_str) into the hash. Pre-fix Int64 vs Int32 frames with
    otherwise-equal cat/text/embedding splits collided on the same cache key, so a re-typed second
    target would be served target 1's preprocessed frame and downstream consumers would fail on
    dtype-aware checks. Pandas frames are unaffected because pandas dtype changes already perturb
    the upstream split_features pipeline.
    """
    _tier_suffix = f"_tier{feature_tier}"
    _kind_suffix = f"_kind{'pl' if supports_polars else 'pd'}"
    _feats_repr = repr((
        tuple(sorted(cat_features or ())),
        tuple(sorted(text_features or ())),
        tuple(sorted(embedding_features or ())),
    ))
    _feats_suffix = f"_feats{hashlib.blake2b(_feats_repr.encode(), digest_size=8).hexdigest()}"
    # Canonical dtype suffix that's POLARS / PANDAS AGNOSTIC.
    # Pre-fix: only polars frames got the ``_dt`` suffix, pandas frames
    # got nothing. Observed in prod: same logical (CB, tier, feats)
    # cached under TWO different keys -- one with _dt suffix (polars
    # call) and one without (pandas call after polars->pandas
    # conversion). Result: cache MISS on the second call even though
    # the work was identical. The new normalized form canonicalises
    # both polars ``Int32`` / pandas ``int32`` to ``i32``, etc., so
    # identical column dtypes produce identical suffixes regardless
    # of which DataFrame backend the call site happened to use.
    _dtype_suffix = ""
    if train_df is not None:
        try:
            _canon_pairs = _canonical_dtype_pairs(train_df)
            if _canon_pairs:
                _dtype_suffix = f"_dt{hashlib.blake2b(repr(_canon_pairs).encode(), digest_size=6).hexdigest()}"
        except Exception:
            _dtype_suffix = ""
    if pre_pipeline_name:
        return f"{strategy_cache_key}_{pre_pipeline_name}{_tier_suffix}{_kind_suffix}{_feats_suffix}{_dtype_suffix}"
    return f"{strategy_cache_key}{_tier_suffix}{_kind_suffix}{_feats_suffix}{_dtype_suffix}"


def _canonical_dtype_pairs(train_df) -> tuple:
    """Polars / pandas-agnostic ``((col, canonical_dtype), ...)`` for cache-key hashing.

    Canonical mapping (the ones we hit in practice):
    - ``Int8/16/32/64`` / ``int8/16/32/64`` -> ``i8/i16/i32/i64``
    - ``UInt8/16/32/64`` / ``uint8/16/32/64`` -> ``u8/u16/u32/u64``
    - ``Float32/64`` / ``float32/64`` -> ``f32/f64``
    - ``Boolean`` / ``bool`` -> ``b``
    - ``Utf8`` / ``String`` / ``object`` (str-ish) -> ``s``
    - ``Categorical`` / ``category`` -> ``c``
    - Other / unknown -> str(dtype).lower()
    Same on-disk dtype yields the same canonical form across polars / pandas; cache reuse works irrespective of which backend the call site uses.
    """
    if pl is not None and isinstance(train_df, pl.DataFrame):
        items = [(c, str(train_df.schema[c])) for c in train_df.columns]
    elif hasattr(train_df, "dtypes") and hasattr(train_df, "columns"):
        # pandas
        items = [(c, str(train_df.dtypes[c])) for c in train_df.columns]
    else:
        return ()
    return tuple((c, _canonicalise_dtype(dt)) for c, dt in items)


def _canonicalise_dtype(dt: str) -> str:
    """Map polars / pandas dtype strings to a single canonical form (see ``_canonical_dtype_pairs`` docstring for table)."""
    s = str(dt).strip().lower()
    if s.startswith("int"):
        return "i" + s[len("int"):]
    if s.startswith("uint"):
        return "u" + s[len("uint"):]
    if s.startswith("float"):
        return "f" + s[len("float"):]
    if s in ("boolean", "bool"):
        return "b"
    if s in ("utf8", "string", "object", "str"):
        return "s"
    if s in ("categorical", "category"):
        return "c"
    return s


from ._phase_train_one_target_dataset_cache import (  # noqa: E402,F401
    _DATASET_REUSE_CACHE_ATTRS,
    _DATASET_REUSE_CACHE_KEY,
    _FEATURE_SIDE_CACHE_KEY,
    _POLARS_RELEASE_MIN_RECLAIM_FRACTION,
    _capture_dataset_reuse_cache,
    _dataset_reuse_cache_key,
    _ensure_ctx_artifacts,
    _ensure_dataset_reuse_cache,
    _ensure_feature_side_cache,
    _forward_dataset_reuse_cache,
    _invalidate_polars_feature_side_cache,
    _purge_fh_cache_by_df_tokens,
    _release_ctx_polars_frames,
    _restore_dataset_reuse_cache,
)


# ----------------------------------------------------------------------
# Sibling-module re-exports. Carved out for monolith-split; loaded AFTER
# the parent finishes binding helpers so the lazy imports inside the
# sibling function bodies resolve.
# ----------------------------------------------------------------------
from ._phase_train_one_target_helpers import (  # noqa: E402,F401
    _build_feature_selection_report,
    _maybe_run_feature_handling_apply,
)
from ._phase_train_one_target_body import _train_one_target  # noqa: E402,F401
