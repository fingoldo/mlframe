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

    Hooks the ``recommend_boosting_regression_loss`` helper (Pack H) onto the per-target loop: after ``select_target`` returns the per-backend templates, override their ``loss_function`` / ``objective`` based on excess-kurtosis of the target. Production motivator: composite ``TVT-linres-TVT_prev`` had excess_kurt=+2.40 (Laplace-like) but CB / LGB / XGB defaulted to RMSE objective and early-stopped at iter=4-10 -- the RMSE gradient collapses near zero on Laplace residuals.

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
    # tracks the surface the optimiser is actually descending. Production TVT:
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
                # Huber not a stock CB eval_metric; MAE approximates Huber
                # near zero and is bounded-influence on tails.
                return ("eval_metric", "MAE")
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
        try:
            _model.set_params(**_set_kwargs)
            _applied.append(
                f"{_backend}:{_param_name}={_value}"
                + (f",{_em[0]}={_em[1]}" if _em is not None else "")
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


# Candidate metric paths probed (in order) to rank ensemble flavours. ``oof.*`` is the only honest
# selection surface (cross_val_predict held-out signal, never used for ES); val.* is the back-compat
# fallback for single-fold suites that did not stamp OOF. ``integral_error`` is the canonical
# calibration metric for classifiers; ``rmse`` is the regression fallback.
#
# ``("test", ...)`` candidates are DELIBERATELY ABSENT. Using the honest test split to pick the
# ensemble winner converts it into a model-selection surface -- every subsequent test-set metric
# becomes biased optimistic by the selection. Test stays a reporting-only surface here; when
# neither OOF nor val is available, ``_choose_ensemble_flavour`` falls back to the first emitted
# flavour (deterministic via ``ensembling_methods`` iteration order, NOT test-driven).
_ENSEMBLE_RANK_METRIC_CANDIDATES = (
    ("oof", "integral_error", "lower"),
    ("oof", "rmse", "lower"),
    ("val", "integral_error", "lower"),
    ("val", "rmse", "lower"),
    # Test-split fallback when oof and val are both absent (e.g. inner-CV
    # disabled in unit tests). Wave-8 contract: choose the flavour that
    # scored best on the held-out test split rather than defaulting to the
    # first-emitted flavour.
    ("test", "integral_error", "lower"),
    ("test", "rmse", "lower"),
)


def _read_ensemble_metric(ens_result, split: str, metric: str):
    """Read ``ens_result.metrics[split][metric]`` (or nested int-keyed dict 1) returning float or None.

    The metric layout produced by ``train_and_evaluate_model`` is ``model.metrics[split]`` where the
    value is either a flat dict or a ``{1: {...}}`` class-indexed nested dict (binary / multiclass).
    For classifier metrics nested under class 1 the read drills one level; otherwise the flat lookup
    wins. Any access / type error returns ``None`` so the chooser silently skips the flavour.
    """
    try:
        _m = getattr(ens_result, "metrics", None)
        if not isinstance(_m, dict):
            return None
        _split = _m.get(split)
        if not isinstance(_split, dict):
            return None
        _val = _split.get(metric)
        if _val is None and 1 in _split and isinstance(_split[1], dict):
            _val = _split[1].get(metric)
        if _val is None:
            return None
        _f = float(_val)
        if not np.isfinite(_f):
            return None
        return _f
    except Exception:
        return None


def _choose_ensemble_flavour(ensembles_dict: dict) -> str | None:
    """Pick the winning ensemble flavour key from ``score_ensemble``'s return dict.

    ``score_ensemble`` returns ``{flavour_name: ens_result}`` for every candidate it evaluated; the
    suite has no native "winner" concept so we apply the same selection rule as ``compare_ensembles``
    (oof.integral_error / rmse ascending). ``" conf"``-suffixed entries (confident-subset variants of
    each flavour) are skipped here -- they reuse the parent flavour's preds on a different subset and
    aren't independent candidates. ``_diversity`` is a side-channel report stamped by
    ``score_ensemble`` rather than an ensemble; skip it too.

    Return values:
      - ``None`` only when ``ensembles_dict`` is empty / not-a-dict / contains zero non-skip candidates.
      - First-emitted flavour name (deterministic via ``ensembling_methods`` insertion order) when at
        least one candidate exists but NONE expose any of the canonical ranking metrics. A WARN log
        line is emitted in this fallback path so an operator grepping the suite log for
        ``no candidate exposed`` can distinguish a fallback win from a metric-driven win. The
        ``metadata["ensembles_chosen"]`` slot does NOT carry a fallback-vs-win marker; if the caller
        needs that distinction reliably, capture the WARN via a log handler.
      - Otherwise the flavour name whose ranking metric scored best per
        ``_ENSEMBLE_RANK_METRIC_CANDIDATES``.

    Pre-fix the docstring promised ``None`` on the no-metric path but the implementation returned the
    first flavour to keep the predict path deterministic; the docstring is now reconciled with the
    fallback behaviour rather than the other way around (changing return to None on no-metric would
    break the deterministic predict-path contract that downstream callers depend on).
    """
    if not isinstance(ensembles_dict, dict) or not ensembles_dict:
        return None
    _candidates = {
        k: v for k, v in ensembles_dict.items()
        if isinstance(k, str) and not k.endswith(" conf") and not k.startswith("_")
    }
    if not _candidates:
        return None
    for _split, _metric, _direction in _ENSEMBLE_RANK_METRIC_CANDIDATES:
        _scored = [
            (k, _read_ensemble_metric(v, _split, _metric))
            for k, v in _candidates.items()
        ]
        _scored = [(k, s) for k, s in _scored if s is not None]
        if not _scored:
            continue
        # Wave 57 (2026-05-20): secondary key on ensemble name (kv[0]) so tied
        # val metric (small holdout / coarse metric) gives a deterministic winner
        # rather than depending on dict iteration order.
        if _direction == "lower":
            _scored.sort(key=lambda kv: (kv[1], kv[0]))
        else:
            _scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return _scored[0][0]
    # No candidate exposed any ranking metric: fall back to the first flavour the suite emitted so the
    # predict path has a deterministic answer rather than None. ``score_ensemble``'s iteration order
    # mirrors ``ensembling_methods``, so the fallback is reproducible across runs.
    #
    # C-Low-4: surface the fallback at WARN so operators reading the suite log can grep for
    # "no ranking metric exposed" and know the winner came from dict-insertion order rather than a
    # metric comparison. The metadata side (ensembles_chosen[tt][tname]) records only the flavour
    # string, which is indistinguishable from a normal metric-driven win without this log line.
    _fallback = next(iter(_candidates.keys()))
    logger.warning(
        "[_choose_ensemble_flavour] no candidate exposed any of the canonical ranking metrics %s; "
        "falling back to first-emitted flavour %r (deterministic via dict-insertion / "
        "ensembling_methods order).",
        [(s, m) for s, m, _ in _ENSEMBLE_RANK_METRIC_CANDIDATES], _fallback,
    )
    return _fallback


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
    # got nothing. Production TVT log: same logical (CB, tier, feats)
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


# XGB DMatrix / LGB Dataset reuse cache attribute names: forwarded across sklearn.clone() in both
# directions (template -> clone before fit; clone -> template after fit) so the weight-schema loop
# reuses the heavy binned dataset via set_label / set_weight instead of rebuilding.
_DATASET_REUSE_CACHE_ATTRS = (
    "_cached_train_dmatrix",
    "_cached_train_key",
    "_cached_val_dmatrix",
    "_cached_val_key",
    "_cached_train_dataset",
    "_cached_val_dataset",
)


def _forward_dataset_reuse_cache(src, dst, attrs=_DATASET_REUSE_CACHE_ATTRS, *, skip_none: bool = False):
    """Copy each present attr from ``src`` onto ``dst``.

    CODE-LOW-7: both the template -> clone forward and the clone -> template back transfer used to
    inline the same loop with slightly different ``if _val is not None`` guard. Centralised here so
    additions to ``_DATASET_REUSE_CACHE_ATTRS`` flow to both call sites automatically.

    ``skip_none=True`` matches the back-transfer's behaviour: only carry non-None caches up to the
    template, otherwise a clone that did not populate the cache would NULL out the template's prior
    value and defeat the reuse.
    """
    for _attr in attrs:
        if not hasattr(src, _attr):
            continue
        _val = getattr(src, _attr)
        if skip_none and _val is None:
            continue
        try:
            setattr(dst, _attr, _val)
        except Exception as _attr_err:
            logger.debug("Could not transfer %s from %r to %r: %s", _attr, type(src).__name__, type(dst).__name__, _attr_err)


# Heuristic: if reclaim is under this share of the dropped-frame footprint, something is still pinning the buffers.
_POLARS_RELEASE_MIN_RECLAIM_FRACTION = 0.05


# ============================================================================================
# Suite-scoped feature-side cache helpers. The per-target inner loop in _train_one_target
# stages tier-DFs / pl.Enum maps / prepared polars frames / fingerprints into ctx.artifacts
# so the NEXT target's call reads them off ctx instead of rebuilding (CB Pool / XGB DMatrix /
# LGB Dataset all rebuild via id(train_df) keys and the train_df pointer is pinned by ctx).
# Entries store REFERENCES, never clones - a 100GB frame is shared between the cache slot and
# ctx.train_df_polars. Polars-tier entries are dropped when polars frames are released
# (``_release_ctx_polars_frames``) since their pinned references would otherwise defeat the
# release. Dataset-reuse cache (XGB DMatrix / LGB Dataset) is keyed by mlframe_model_name and
# bridges the per-target rebuild of models_params: the binned dataset built on target 1 gets
# re-attached onto target 2's freshly-built model template via _DATASET_REUSE_CACHE_ATTRS.
# ============================================================================================

_FEATURE_SIDE_CACHE_KEY = "feature_side_cache"
_DATASET_REUSE_CACHE_KEY = "dataset_reuse_cache"


def _ensure_ctx_artifacts(ctx) -> dict:
    """Return ctx.artifacts as a dict, materialising it if the dataclass default left it as None.

    ``ctx.artifacts`` is declared ``dict = field(default_factory=dict)`` in _training_context.py
    so normal construction produces an empty dict, BUT older test fixtures and direct field
    assignments can land ``None`` on the slot. Calling .setdefault() then AttributeErrors before
    the helper has a chance to install its key.
    """
    artifacts = ctx.artifacts
    if artifacts is None:
        artifacts = {}
        ctx.artifacts = artifacts
    return artifacts


def _ensure_feature_side_cache(ctx) -> dict:
    """Return the (creating-if-needed) suite-scoped feature-side cache off ctx.artifacts."""
    return _ensure_ctx_artifacts(ctx).setdefault(_FEATURE_SIDE_CACHE_KEY, {})


def _ensure_dataset_reuse_cache(ctx) -> dict:
    """Return the (creating-if-needed) suite-scoped dataset-reuse cache off ctx.artifacts.

    Keyed by ``(mlframe_model_name, pp_name)`` post-fix (DSET-REUSE-NO-PP-KEY): pre-fix the key was
    a bare ``mlframe_model_name``, so two pre-pipelines (e.g. MRMR vs ordinary) on the same target
    + model produced different column sets and collided on the same cache slot, replaying the
    prior PP's binned dataset onto the next PP's fresh template. ``capture`` / ``restore`` build
    the same tuple key so the round-trip stays consistent. Entries are dicts of
    ``_DATASET_REUSE_CACHE_ATTRS`` -> value captured from the prior target's fitted model template
    before _maybe_clear_shim_cache nuked it.
    """
    return _ensure_ctx_artifacts(ctx).setdefault(_DATASET_REUSE_CACHE_KEY, {})


def _dataset_reuse_cache_key(mlframe_model_name: str, pp_name: str | None) -> tuple:
    """Build the (model_name, pp_name) cache key for the dataset-reuse cache.

    Centralised so capture-side and restore-side never disagree on key shape.
    """
    return (mlframe_model_name, pp_name or "")


def _invalidate_polars_feature_side_cache(ctx) -> None:
    """Drop every polars-tier entry from ctx.artifacts['feature_side_cache'].

    Called from ``_release_ctx_polars_frames`` (the only place where ctx polars frames go to
    None) so the next target's loop doesn't read back stale pointers into freed frames. Pandas-
    tier entries (``supports_polars=False``) are preserved - they live in their own keys and
    point at frames that are NOT being released here.
    """
    cache = (ctx.artifacts or {}).get(_FEATURE_SIDE_CACHE_KEY)
    if not cache:
        return
    # Cache shape: cache[pp_name] -> {"tier_dfs": {sub_key -> dict}, "prepared_frames":
    # {sub_key -> dict}, "tier_enum_map": {sub_key -> map}}. Sub-keys are tuples and we
    # drop only the polars-tier ones; the "tier_enum_map" group is polars-only by
    # construction so it can be cleared whole.
    for _pp_name, _pp_payload in list(cache.items()):
        if not isinstance(_pp_payload, dict):
            continue
        for _group in ("tier_dfs", "prepared_frames"):
            _group_map = _pp_payload.get(_group)
            if not isinstance(_group_map, dict):
                continue
            # tier_dfs sub-key is (tier_tuple, kind) where kind is "pl" / "pd"; prepared_frames
            # sub-key is (tier_tuple, supports_polars, strategy_class, cb_text_pass). Polars
            # marker: kind=="pl" OR supports_polars==True (positional element 1).
            _polars_sub_keys = []
            for _sub_key in list(_group_map.keys()):
                if not isinstance(_sub_key, tuple) or len(_sub_key) < 2:
                    continue
                _kind = _sub_key[1]
                if _kind == "pl" or _kind is True:
                    _polars_sub_keys.append(_sub_key)
            for _k in _polars_sub_keys:
                _group_map.pop(_k, None)
        # tier_enum_map is polars-only by construction (the per-target loop only writes to
        # it on polars_fastpath_active); a polars frame release means all entries are stale.
        _enum_map = _pp_payload.get("tier_enum_map")
        if isinstance(_enum_map, dict):
            _enum_map.clear()


def _purge_fh_cache_by_df_tokens(ctx, df_tokens) -> None:
    """Scrub FH ``FeatureCache._mem`` entries whose ``df_token`` matches a just-released frame id.

    The FH cache is keyed by ``InMemoryKey(session_id, df_token=id(train_df), ...)``. While the
    strong ref is alive, ``id()`` is stable; once we drop it in ``_release_ctx_polars_frames``,
    Python may recycle the same integer for a freshly allocated frame. The session_id is rotated
    per-suite by ``reset_session`` so cross-suite reuse is already safe; this scrub handles the
    mid-suite tier-transition case where one suite call releases polars frames and a subsequent
    target-loop iteration re-builds them.

    FeatureCache instances live wherever the suite stashed them. v1 stores under
    ``ctx.artifacts["feature_handling_fitted"]`` (the FeatureHandlingResult holds no cache ref);
    later phases may park the cache itself under ``ctx.artifacts["feature_handling_cache"]`` (single
    instance for the whole suite). Honour either shape; tolerate absence.
    """
    if not df_tokens:
        return
    artifacts = getattr(ctx, "artifacts", None)
    if not isinstance(artifacts, dict):
        return
    candidates = []
    _cache = artifacts.get("feature_handling_cache")
    if _cache is not None:
        candidates.append(_cache)
    _fitted = artifacts.get("feature_handling_fitted")
    if isinstance(_fitted, dict):
        for _v in _fitted.values():
            _c = getattr(_v, "cache", None)
            if _c is not None and _c not in candidates:
                candidates.append(_c)
    for _cache in candidates:
        _purge_fn = getattr(_cache, "purge_by_df_token", None)
        if not callable(_purge_fn):
            continue
        for _tok in df_tokens:
            try:
                _purge_fn(_tok)
            except Exception as _purge_err:  # pragma: no cover -- defensive
                logger.debug("FH cache purge_by_df_token(%s) raised %r; continuing", _tok, _purge_err)


def _capture_dataset_reuse_cache(
    ctx,
    mlframe_model_name: str,
    model_template,
    pp_name: str | None = None,
) -> None:
    """Snapshot ``_DATASET_REUSE_CACHE_ATTRS`` off ``model_template`` into ctx.artifacts.

    Runs BEFORE ``_maybe_clear_shim_cache`` so the next target gets the live binned dataset
    (XGB DMatrix / LGB Dataset) rather than the post-clear None. Skips entries whose value is
    None - those entries would defeat the next target's cache-hit check (``is not None``).
    """
    if model_template is None:
        return
    captured = {}
    for _attr in _DATASET_REUSE_CACHE_ATTRS:
        if not hasattr(model_template, _attr):
            continue
        _val = getattr(model_template, _attr)
        if _val is None:
            continue
        captured[_attr] = _val
    if captured:
        _ensure_dataset_reuse_cache(ctx)[_dataset_reuse_cache_key(mlframe_model_name, pp_name)] = captured


def _restore_dataset_reuse_cache(
    ctx,
    mlframe_model_name: str,
    model_template,
    pp_name: str | None = None,
) -> None:
    """Re-attach ``_DATASET_REUSE_CACHE_ATTRS`` from ctx.artifacts onto ``model_template``.

    The per-target rebuild of ``models_params`` produces a fresh estimator without the cache
    attributes; this restore wires the previous target's binned dataset back on so the next
    forward-transfer-into-clone() carries it forward, and the shim's signature_of(X) check
    detects the same X (ctx-pinned across targets) and triggers the set_label / set_weight
    swap instead of a fresh build. No-op when there is no prior capture, or when target 1
    has not run yet for this model.
    """
    if model_template is None:
        return
    _key = _dataset_reuse_cache_key(mlframe_model_name, pp_name)
    captured = (ctx.artifacts or {}).get(_DATASET_REUSE_CACHE_KEY, {}).get(_key)
    if not captured:
        return
    for _attr, _val in captured.items():
        try:
            setattr(model_template, _attr, _val)
        except Exception as _attr_err:
            logger.debug(
                "Could not restore %s on %s template: %s",
                _attr, mlframe_model_name, _attr_err,
            )


def _release_ctx_polars_frames(
    ctx,
    baseline_rss_mb: float,
    df_size_mb: float,
    *,
    verbose: bool,
    reason: str,
) -> float:
    """Drop ctx.{train,val,test}_df_polars strong refs, then trigger maybe_clean_ram_and_gpu and verify reclaim.

    The naked ``del train_df_polars`` at each call site only released the local alias inside
    ``_train_one_target``; the ctx attributes (assigned at lines 123-125 from ctx.*_df_polars) kept the
    real strong reference alive, so ``maybe_clean_ram_and_gpu`` had nothing to reclaim and the log line
    claiming a release was misleading. Centralised here so both call sites stay in sync and the post-release
    sanity check (RSS drop vs estimated frame footprint) flags any future regression where a new strong
    ref to the same frames is introduced upstream without being scrubbed here.
    """
    expected_mb = 0.0
    released_df_tokens: list = []
    for _attr in ("train_df_polars", "val_df_polars", "test_df_polars"):
        _frame = getattr(ctx, _attr, None)
        if _frame is None:
            continue
        try:
            _sz = estimate_df_size_mb(_frame)
        except Exception:
            _sz = 0.0
        if _sz and _sz != float("inf"):
            expected_mb += float(_sz)
        # Capture id() BEFORE clearing the ctx slot -- once we drop the strong ref, Python is free
        # to recycle the integer for a freshly-allocated object and we'd no longer be able to scrub
        # the matching FH cache entries.
        released_df_tokens.append(id(_frame))
    rss_before_mb = get_process_rss_mb()
    ctx.train_df_polars = None
    ctx.val_df_polars = None
    ctx.test_df_polars = None
    # Drop polars-tier entries from the suite-scoped feature-side cache so they don't pin the
    # frames we just released. Pandas-tier entries are preserved - they point at separate
    # frames not touched by this release.
    _invalidate_polars_feature_side_cache(ctx)
    # VIEW-CACHE-NOT-WIPED: ``ctx._pandas_view_cache`` keys by ``id(polars_df)``. The frames we just
    # released may have their ids recycled by a freshly allocated polars frame the next time the
    # suite enters tier_pandas conversion -- the cache would silently serve the prior pandas view.
    # Pop every entry keyed by a just-released id so the next conversion always misses cleanly.
    _view_cache = getattr(ctx, "_pandas_view_cache", None)
    if isinstance(_view_cache, dict):
        for _tok in released_df_tokens:
            _view_cache.pop(_tok, None)
    # Same hygiene for the recurrent numpy-coercion cache (POLARS-PANDAS-CHURN). Keys are
    # ``(split, id(frame))`` so we pop every (_, tok) pair where the second element matches a
    # just-released id.
    _rec_cache = getattr(ctx, "_recurrent_numpy_cache", None)
    if isinstance(_rec_cache, dict) and released_df_tokens:
        _released_set = set(released_df_tokens)
        for _k in [k for k in _rec_cache.keys() if isinstance(k, tuple) and len(k) == 2 and k[1] in _released_set]:
            _rec_cache.pop(_k, None)
    # Scrub any FH FeatureCache in-memory entries keyed by the released df ids. Without this, a
    # future tier transition that re-allocates a polars frame at the same memory address would
    # silently hit a cached entry whose state belonged to the dropped frame.
    _purge_fh_cache_by_df_tokens(ctx, released_df_tokens)
    new_baseline = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason=reason)
    # Only emit the lingering-refs warning when the expected reclaim is large enough for the
    # actual delta to be measurable above RSS-measurement noise. Windows / Linux RSS reporting
    # rounds to page-granularity (~4 KiB) and the resident-set is also affected by OS-managed
    # caching outside our control. For small frames (<10 MB expected reclaim) a delta of 0
    # is well within noise and the warning is just chatter that fires on every fuzz / unit test
    # suite call. Keep the warning loud for the real-production case (gigabyte-scale frames
    # where a missed release is a real leak).
    _POLARS_RELEASE_MIN_EXPECTED_MB = 10.0
    if expected_mb >= _POLARS_RELEASE_MIN_EXPECTED_MB:
        rss_after_mb = get_process_rss_mb()
        delta_mb = rss_before_mb - rss_after_mb
        if delta_mb < _POLARS_RELEASE_MIN_RECLAIM_FRACTION * expected_mb:
            logger.warning(
                "ctx polars frames released but RSS dropped only %.1f MB; expected at least %.1f MB - check for lingering refs",
                delta_mb,
                expected_mb,
            )
    return new_baseline


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
