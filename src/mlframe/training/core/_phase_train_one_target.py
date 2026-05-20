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
# purely a property of the class, so we memoize by ``id(cls)`` and reuse across iterations.
_INIT_SIG_CACHE: dict[int, set[str]] = {}


def _cached_init_params(cls) -> set[str]:
    """Return the set of accepted ``__init__`` kwargs for ``cls`` (excl. ``self``), cached by id."""
    key = id(cls)
    cached = _INIT_SIG_CACHE.get(key)
    if cached is None:
        cached = set(inspect.signature(cls.__init__).parameters) - {"self"}
        _INIT_SIG_CACHE[key] = cached
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
    _applied: list[str] = []
    for _backend, (_param_name, _value) in _backend_param.items():
        if not _value:
            continue
        _entry = models_params.get(_backend)
        if not isinstance(_entry, dict):
            continue
        _model = _entry.get("model")
        if _model is None or not hasattr(_model, "set_params"):
            continue
        try:
            _model.set_params(**{_param_name: _value})
            _applied.append(f"{_backend}:{_param_name}={_value}")
        except (ValueError, TypeError) as _set_err:
            # Backend may reject the value (e.g. CB rejecting 'Huber:delta=1.345' on a specific board). Log and move on; default stays.
            if verbose:
                logger_.debug(
                    "[auto-loss] %s.set_params(%s=%r) rejected: %s. Keeping default.",
                    _backend, _param_name, _value, _set_err,
                )

    if verbose and _applied:
        logger_.info(
            "[auto-loss] target='%s' excess_kurt=%.2f (n_finite=%d) -- %s. Applied: %s.",
            composite_name, float(rec.get("excess_kurt", float("nan"))),
            int(rec.get("n_finite", 0)),
            rec.get("rationale", ""), ", ".join(_applied),
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

    Returns ``None`` when no candidate exposes any of the canonical ranking metrics.
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
        if _direction == "lower":
            _scored.sort(key=lambda kv: kv[1])
        else:
            _scored.sort(key=lambda kv: -kv[1])
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


def _build_feature_selection_report(
    pre_pipeline,
    pre_pipeline_name: str | None,
    fitted_columns_in: list[str] | None,
    kept_columns: list[str] | None,
) -> dict:
    """Build the per-model FS report stamped onto ``metadata["model_schemas"][file_name]``.

    Layout (per task spec):
        selector_name : "MRMR" | "RFECV" | "BorutaShap" | None
        selector_params_hash : 8-byte blake2b hex of selector.get_params(), or None
        kept_features : list[str] -- post-FS surviving feature names
        dropped_features : list[str] -- pre-FS names NOT in kept_features
        scores : {feature: float} -- per-feature score from the selector's natural attribute
                                     (None when the selector exposes no per-feature score)
        reason_per_feature : {feature: str} -- per-feature decision label
                                                (None when the selector lacks a reason)

    Reads selector-specific attributes:
      * MRMR: ``support_`` (integer index list into ``feature_names_in_``); no per-feature score is
        exposed, so ``scores=None``. Reason: "kept" / "dropped".
      * RFECV: ``feature_importances_`` is a ``{nfeatures_nfold: ndarray}`` dict keyed by
        ``"<best_top_n>_<n_folds>"``; aggregate over folds at ``best_top_n=self.n_features_`` by mean
        across folds. ``ranking_`` (when present) gives the eliminated-order; we surface it as a
        per-feature reason ("kept@rank=N" / "dropped@rank=N").
      * BorutaShap: ``history_x`` is a DataFrame of per-iteration shap importances (one row per
        iteration, one column per feature); mean across rows is the canonical "average importance"
        score. Reason: "accepted" / "rejected" / "tentative" via ``self.accepted`` / ``self.rejected``
        / ``self.tentative`` (set when ``calculate_rejected_accepted_tentative`` ran).

    Falls back to a minimal report (selector_name + kept/dropped only) if any attribute access fails;
    a failed FS report must never abort the training run.
    """
    selector = _unwrap_selector(pre_pipeline)
    _kind = _selector_kind(selector)
    _report: dict = {
        "selector_name": _kind,
        "selector_params_hash": _selector_params_hash(selector),
        "kept_features": list(kept_columns) if kept_columns is not None else None,
        "dropped_features": None,
        "scores": None,
        "reason_per_feature": None,
    }
    # Compute dropped = feature_names_in_ \ kept; selectors stamp ``feature_names_in_`` post-fit.
    _all_in = None
    if selector is not None:
        try:
            _all_in = getattr(selector, "feature_names_in_", None)
        except Exception:
            _all_in = None
    if _all_in is None:
        _all_in = fitted_columns_in
    if _all_in is not None and kept_columns is not None:
        try:
            _kept_set = set(kept_columns)
            _report["dropped_features"] = [c for c in _all_in if c not in _kept_set]
        except Exception:
            _report["dropped_features"] = None

    if _kind == "MRMR":
        # MRMR exposes ``support_`` as integer indices into ``feature_names_in_``; no per-feature score.
        _report["scores"] = None
        if _all_in is not None and kept_columns is not None:
            try:
                _kept_set = set(kept_columns)
                _report["reason_per_feature"] = {
                    str(c): ("kept" if c in _kept_set else "dropped") for c in _all_in
                }
            except Exception:
                pass
    elif _kind == "RFECV":
        # ``feature_importances_`` is dict keyed by "<nfeatures>_<fold>"; pick the rows matching
        # ``n_features_`` (the chosen size) and mean across folds. ``ranking_`` exposes the per-
        # feature elimination order when the suite went through the full RFECV loop.
        try:
            _fi_dict = getattr(selector, "feature_importances_", None)
            _n_feat = getattr(selector, "n_features_", None)
            if isinstance(_fi_dict, dict) and _n_feat and _all_in is not None:
                _stride = str(int(_n_feat))
                _rows = [v for k, v in _fi_dict.items() if str(k).startswith(_stride + "_")]
                if _rows:
                    _arr = np.asarray(_rows, dtype=np.float64)
                    _mean = np.nanmean(_arr, axis=0)
                    if _mean.shape[0] == len(_all_in):
                        _report["scores"] = {str(c): float(_mean[i]) for i, c in enumerate(_all_in)}
        except Exception:
            _report["scores"] = None
        # Ranking-based reason
        try:
            _ranking = getattr(selector, "ranking_", None)
            if _ranking is not None and _all_in is not None and kept_columns is not None:
                _kept_set = set(kept_columns)
                _rank_arr = list(_ranking)
                if len(_rank_arr) == len(_all_in):
                    _report["reason_per_feature"] = {
                        str(c): (
                            f"kept@rank={_rank_arr[i]}" if c in _kept_set
                            else f"dropped@rank={_rank_arr[i]}"
                        )
                        for i, c in enumerate(_all_in)
                    }
        except Exception:
            pass
    elif _kind == "BorutaShap":
        # ``history_x`` columns map 1:1 to ``all_columns``; the per-feature score is the mean
        # historical SHAP importance across all Boruta iterations. ``accepted`` / ``rejected`` /
        # ``tentative`` carry the final per-feature verdicts.
        try:
            _history = getattr(selector, "history_x", None)
            if _history is not None and hasattr(_history, "mean"):
                _means = _history.mean(axis=0)
                if hasattr(_means, "to_dict"):
                    _report["scores"] = {str(k): float(v) for k, v in _means.to_dict().items()}
        except Exception:
            _report["scores"] = None
        try:
            _accepted = set(getattr(selector, "accepted", None) or [])
            _rejected = set(getattr(selector, "rejected", None) or [])
            _tentative = set(getattr(selector, "tentative", None) or [])
            _all_cols_attr = getattr(selector, "all_columns", None)
            _iter = list(_all_cols_attr) if _all_cols_attr is not None else (_all_in or [])
            if _iter:
                _reasons = {}
                for c in _iter:
                    _cs = str(c)
                    if c in _accepted:
                        _reasons[_cs] = "accepted"
                    elif c in _rejected:
                        _reasons[_cs] = "rejected"
                    elif c in _tentative:
                        _reasons[_cs] = "tentative"
                    else:
                        _reasons[_cs] = "unknown"
                _report["reason_per_feature"] = _reasons
        except Exception:
            pass
    return _report


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


def _get_feature_side_cache(ctx) -> dict:
    """Return the (creating-if-needed) suite-scoped feature-side cache off ctx.artifacts."""
    return _ensure_ctx_artifacts(ctx).setdefault(_FEATURE_SIDE_CACHE_KEY, {})


def _get_dataset_reuse_cache(ctx) -> dict:
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
        _get_dataset_reuse_cache(ctx)[_dataset_reuse_cache_key(mlframe_model_name, pp_name)] = captured


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


def _maybe_run_feature_handling_apply(
    ctx,
    *,
    cur_target_name: str,
    train_df,
    val_df,
    test_df,
    current_train_target,
    sample_weight=None,
):
    """Run feature_handling_apply once per target when ctx carries a FeatureHandlingConfig; else no-op.

    Returns the FeatureHandlingResult on success or None when disabled / failed. Fitted state is also
    stashed under ctx.artifacts["feature_handling_fitted"][cur_target_name] so a future predict-side
    wave can replay handlers without re-fitting. ctx.artifacts is the only ctx slot we may write to
    here -- TrainingContext uses slots=True so adding a new attribute would AttributeError, and the
    SCOPE constraint forbids touching _training_context.py in this wave.

    sample_weight is accepted for forward compatibility: feature_handling_apply does not yet take it
    (validated against the current apply.py). The keyword is plumbed through so a later apply.py
    extension picks it up without a second wire-in change here. NOTE: the underlying handlers do
    consume sample_weight via LeakageSafeEncoder -- once apply.py grows the kwarg, drop the silent
    discard below.

    model_kind comes from ctx.sorted_mlframe_models[0] -- the first concrete kind drives FHC
    validation; the resulting fitted state is model-agnostic for the handlers wired in v1 (TF-IDF,
    target-encoder, custom), so one call seeds the FeatureCache for every model that follows.
    """
    fhc = getattr(ctx, "feature_handling_config", None)
    if fhc is None:
        # `_phase_config_setup.setup_configuration` stores the validated config under `ctx.artifacts`
        # because TrainingContext uses slots=True and exposes no dedicated slot. Honour that storage path.
        fhc = ctx.artifacts.get("feature_handling_config") if isinstance(getattr(ctx, "artifacts", None), dict) else None
    if fhc is None:
        return None
    try:
        from mlframe.training.feature_handling import feature_handling_apply  # local: avoid suite-import cost when FHC is off
    except ImportError:  # pragma: no cover
        return None

    sorted_models = getattr(ctx, "sorted_mlframe_models", None) or getattr(ctx, "mlframe_models", None) or []
    if not sorted_models:
        return None
    model_kind = sorted_models[0]

    try:
        result = feature_handling_apply(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_target=current_train_target,
            fhc=fhc,
            model_kind=model_kind,
        )
    except ValueError as fhc_err:
        # Surface configuration errors with the kwarg name so users grep the right place; chain so the
        # original validation traceback is preserved.
        raise ValueError(
            f"feature_handling_config rejected for model_kind={model_kind!r} on target "
            f"{cur_target_name!r}: {fhc_err}"
        ) from fhc_err
    except Exception as fhc_err:
        logger.warning(
            "feature_handling_apply failed for target %r (model_kind=%s): %s; continuing without FHC enrichment for this target.",
            cur_target_name, model_kind, fhc_err,
        )
        return None

    # ctx.artifacts is a plain dict on the dataclass, so we can nest a sub-dict here without slots issues.
    fitted_store = ctx.artifacts.setdefault("feature_handling_fitted", {})
    fitted_store[cur_target_name] = result

    # TODO(wave-N): downstream consumption -- the assembled matrices in `result.train/val/test` are
    # currently fit-and-stash-only. Phase F (CB embedding_features) / phase G (TabularInputEncoder)
    # will route these into the model.fit() path. Until then the call exists to seed the per-suite
    # FeatureCache and exercise the validate_against_models guard so misconfig is caught at fit time.
    return result


def _train_one_target(ctx, target_type, targets, cur_target_name, cur_target_values):
    """Train all models for one (target_type, target_name) pair."""
    # Suite-once unsupervised pre-screen (A-Arch-001): drop variance==0 and nulls>99% columns from the
    # TRAIN split ONLY, then reapply to val / test. Conservative defaults; train-only fit by contract so
    # no held-out distribution leaks into the drop decision. Latched on ctx so multi-target suites pay
    # the cost once. Opt out via FeatureSelectionConfig.pre_screen_unsupervised=False.
    _fs_cfg = ctx.feature_selection_config
    if _fs_cfg is not None and getattr(_fs_cfg, "pre_screen_unsupervised", False) and not ctx._pre_screen_done:
        try:
            # Canonical home is ``mlframe.feature_selection.pre_screen`` (not under ``.filters``).
            # The shorter path avoids triggering ``filters/__init__.py``'s ``from ._legacy import *``,
            # which cascades into ``_numba_utils`` and pays ~0.8s of @njit decorator init on
            # cold-start (measured 2026-05-20). Saves that wall on every suite call that doesn't
            # also use MRMR (which is the majority of fuzz iters / non-FS production combos).
            from mlframe.feature_selection.pre_screen import compute_unsupervised_drops, apply_drops
            _protected = set()
            if isinstance(targets, dict):
                _protected.update(str(k) for k in targets.keys())
            if ctx.cat_features:
                pass
            _train_for_screen = ctx.filtered_train_df if ctx.filtered_train_df is not None else (ctx.train_df_polars or ctx.train_df_pd)
            _drops = compute_unsupervised_drops(
                _train_for_screen,
                variance_threshold=getattr(_fs_cfg, "pre_screen_variance_threshold", 0.0),
                null_fraction_threshold=getattr(_fs_cfg, "pre_screen_null_fraction_threshold", 0.99),
                protected_columns=_protected,
            )
            ctx._pre_screen_dropped_cols = list(_drops)
            ctx._pre_screen_done = True
            if _drops:
                for _frame_attr in (
                    "filtered_train_df", "filtered_val_df",
                    "train_df_pd", "val_df_pd", "test_df_pd",
                    "train_df_polars", "val_df_polars", "test_df_polars",
                ):
                    _f = getattr(ctx, _frame_attr, None)
                    if _f is not None:
                        try:
                            setattr(ctx, _frame_attr, apply_drops(_f, _drops))
                        except Exception:
                            pass
                if ctx.verbose:
                    logger.info(
                        "[pre-screen] dropped %d column(s) suite-wide (variance=%s, null_fraction>%s): %s",
                        len(_drops),
                        getattr(_fs_cfg, "pre_screen_variance_threshold", 0.0),
                        getattr(_fs_cfg, "pre_screen_null_fraction_threshold", 0.99),
                        _drops[:20] + (["..."] if len(_drops) > 20 else []),
                    )
        except Exception as _e:
            # Pre-screen is a perf optimization; never block training on its failure.
            ctx._pre_screen_done = True
            ctx._pre_screen_dropped_cols = []
            logger.warning("[pre-screen] skipped due to error: %s", _e)
    model_name = ctx.model_name
    target_name = ctx.target_name
    split_config = ctx.split_config
    hyperparams_config = ctx.hyperparams_config
    behavior_config = ctx.behavior_config
    reporting_config = ctx.reporting_config
    feature_selection_config = ctx.feature_selection_config
    baseline_diagnostics_config = ctx.baseline_diagnostics_config
    dummy_baselines_config = ctx.dummy_baselines_config
    quantile_regression_config = ctx.quantile_regression_config
    verbose = ctx.verbose
    linear_model_config = ctx.linear_model_config
    data_dir = ctx.data_dir
    models_dir = ctx.models_dir
    save_charts = ctx.save_charts
    outlier_detector = ctx.outlier_detector
    use_mrmr_fs = ctx.use_mrmr_fs
    use_ordinary_models = ctx.use_ordinary_models
    use_mlframe_ensembles = ctx.use_mlframe_ensembles
    mrmr_kwargs = ctx.mrmr_kwargs
    rfecv_models = ctx.rfecv_models
    multilabel_dispatch_config = ctx.multilabel_dispatch_config
    custom_pre_pipelines = ctx.custom_pre_pipelines
    common_params_dict = ctx.common_params_dict
    mlframe_models = ctx.mlframe_models
    metadata = ctx.metadata
    target_by_type = ctx.target_by_type
    group_ids = ctx.group_ids
    timestamps = ctx.timestamps
    sample_weights = ctx.sample_weights
    baseline_rss_mb = ctx.baseline_rss_mb
    df_size_mb = ctx.df_size_mb
    train_idx = ctx.train_idx
    test_idx = ctx.test_idx
    train_details = ctx.train_details
    val_details = ctx.val_details
    test_details = ctx.test_details
    fairness_subgroups = ctx.fairness_subgroups
    pipeline = ctx.pipeline
    polars_pipeline_applied = ctx.polars_pipeline_applied
    cat_features = ctx.cat_features
    text_features = ctx.text_features
    embedding_features = ctx.embedding_features
    _dropped_high_card_data = ctx._dropped_high_card_data
    train_df_pd = ctx.train_df_pd
    val_df_pd = ctx.val_df_pd
    test_df_pd = ctx.test_df_pd
    train_df_polars = ctx.train_df_polars
    val_df_polars = ctx.val_df_polars
    test_df_polars = ctx.test_df_polars
    filtered_train_df = ctx.filtered_train_df
    filtered_val_df = ctx.filtered_val_df
    filtered_train_idx = ctx.filtered_train_idx
    filtered_val_idx = ctx.filtered_val_idx
    train_od_idx = ctx.train_od_idx
    val_od_idx = ctx.val_od_idx
    category_encoder = ctx.category_encoder
    imputer = ctx.imputer
    scaler = ctx.scaler
    trainset_features_stats = ctx.trainset_features_stats
    defer_pandas_conv = ctx.defer_pandas_conv
    train_df_size_bytes_cached = ctx.train_df_size_bytes_cached
    val_df_size_bytes_cached = ctx.val_df_size_bytes_cached
    _all_target_audits = ctx._all_target_audits
    _non_neural_train_times = ctx._non_neural_train_times
    models = ctx.models
    slug_to_original_target_type = ctx.slug_to_original_target_type
    slug_to_original_target_name = ctx.slug_to_original_target_name
    # Initialised pre-conditional so a later reference doesn't NameError when mlframe_models is empty.
    rfecv_models_params = {}
    if mlframe_models:
        # Identity assignment is intentional: keep the slug key registered even when it equals the original name,
        # so downstream lookups via slug never KeyError on round-trip identity targets.
        # Registered ONLY when at least one model is trained -- otherwise the predict-time loader would resolve
        # this slug to a target name that has no corresponding model on disk.
        slug_to_original_target_name[slugify(cur_target_name)] = cur_target_name
        plot_file, model_file = _setup_model_directories(
            target_name=target_name,
            model_name=model_name,
            target_type=target_type,
            cur_target_name=cur_target_name,
            data_dir=data_dir,
            models_dir=models_dir,
            save_charts=save_charts,
        )

        _train_idx = filtered_train_idx if filtered_train_idx is not None else train_idx
        current_train_target = (
            cur_target_values[_train_idx]
            if isinstance(cur_target_values, (np.ndarray, pl.Series))
            else cur_target_values.iloc[_train_idx]
        )
        current_val_target = None
        if filtered_val_idx is not None:
            current_val_target = (
                cur_target_values[filtered_val_idx]
                if isinstance(cur_target_values, (np.ndarray, pl.Series))
                else cur_target_values.iloc[filtered_val_idx]
            )
        # test_idx is intentionally raw (not OD-filtered) - test must never be filtered by outlier detector.
        current_test_target = None
        if test_idx is not None:
            current_test_target = (
                cur_target_values[test_idx]
                if isinstance(cur_target_values, (np.ndarray, pl.Series))
                else cur_target_values.iloc[test_idx]
            )

        # Feature-handling wire-in: opt-in via ctx.feature_handling_config. Sits after the per-target
        # OD-filtered frames + targets are bound (this is the "post-FS / pre-final-pipeline" seam for
        # the inner pre_pipelines x models loops below) and before per-target diagnostics so any
        # FHC-detected text columns surface in the same log block. No-op when fhc is None, so the
        # default code path is unchanged. polars-fastpath frames are preferred when present; the
        # underlying handlers detect polars vs pandas via _extract_column_values. A blanket
        # polars->pandas conversion here would defeat the suite's polars fastpath -- left to apply.py
        # to keep frame container as-given.
        _fhc_train_df = train_df_polars if train_df_polars is not None else filtered_train_df
        _fhc_val_df = val_df_polars if val_df_polars is not None else filtered_val_df
        _fhc_test_df = test_df_polars if test_df_polars is not None else test_df_pd
        _maybe_run_feature_handling_apply(
            ctx,
            cur_target_name=cur_target_name,
            train_df=_fhc_train_df,
            val_df=_fhc_val_df,
            test_df=_fhc_test_df,
            current_train_target=current_train_target,
            sample_weight=sample_weights,
        )

        metadata = run_per_target_diagnostics(
            target_type=target_type,
            cur_target_name=cur_target_name,
            current_train_target=current_train_target,
            current_val_target=current_val_target,
            current_test_target=current_test_target,
            filtered_train_df=filtered_train_df,
            baseline_diagnostics_config=baseline_diagnostics_config,
            cat_features=cat_features,
            metadata=metadata,
        )

        metadata = run_dummy_baselines(
            target_type=target_type,
            cur_target_name=cur_target_name,
            target_name=target_name,
            model_name=model_name,
            current_train_target=current_train_target,
            current_val_target=current_val_target,
            current_test_target=current_test_target,
            filtered_train_df=filtered_train_df,
            filtered_val_df=filtered_val_df,
            test_df_pd=test_df_pd,
            filtered_train_idx=filtered_train_idx,
            filtered_val_idx=filtered_val_idx,
            test_idx=test_idx,
            timestamps=timestamps,
            cat_features=cat_features,
            dummy_baselines_config=dummy_baselines_config,
            quantile_regression_config=quantile_regression_config,
            reporting_config=reporting_config,
            _dropped_high_card_data=_dropped_high_card_data,
            train_od_idx=train_od_idx,
            val_od_idx=val_od_idx,
            plot_file=plot_file,
            metadata=metadata,
            target_by_type=target_by_type,
            _split_preds_probs=_split_preds_probs,
        )

        # Audits are precomputed once for all targets via the batch API; this lookup is the per-target render.
        _audit = _all_target_audits.get(target_type, {}).get(cur_target_name)
        if _audit is not None:
            try:
                logger.info(_format_temporal_audit_report(_audit))
                if (getattr(behavior_config, "target_temporal_audit_save_plot", True)
                        and plot_file):
                    _plot_path = f"{plot_file}_target_temporal_audit.png"
                    _plot_target_over_time(_audit, save_path=_plot_path)
                metadata.setdefault("target_temporal_audit", {}) \
                    .setdefault(str(target_type), {})[cur_target_name] = _audit.to_dict()
            except Exception as _audit_err:
                logger.warning(
                    "target_temporal_audit (per-target render) failed for "
                    "target='%s': %s. Training continues.",
                    cur_target_name, _audit_err,
                )

        if verbose:
            logger.info(f"select_target...")

        t0_select_target = timer()
        od_common_params, current_behavior_config = _build_common_params_for_target(
            common_params_dict=common_params_dict,
            trainset_features_stats=trainset_features_stats,
            plot_file=plot_file,
            train_od_idx=train_od_idx,
            val_od_idx=val_od_idx,
            current_train_target=current_train_target,
            current_val_target=current_val_target,
            outlier_detector=outlier_detector,
            behavior_config=behavior_config,
            fairness_subgroups=fairness_subgroups,
        )

        # Test set is never OD-filtered. train_df_size_bytes_cached is the pre-conversion Polars-side size
        # passed through so configure_training_params can skip a 3-min pandas memory_usage(deep=...) scan
        # on high-cardinality object columns; the OD-shrinkage approximation only feeds a GPU-RAM heuristic.
        common_params, models_params, rfecv_models_params, cpu_configs, gpu_configs = select_target(
            model_name=f"{target_name} {model_name} {cur_target_name}",
            target=cur_target_values,
            target_type=target_type,
            df=None,
            train_df=filtered_train_df,
            val_df=filtered_val_df,
            test_df=test_df_pd,
            train_idx=filtered_train_idx,
            val_idx=filtered_val_idx,
            test_idx=test_idx,
            train_details=train_details,
            val_details=val_details,
            test_details=test_details,
            group_ids=group_ids,
            cat_features=cat_features,
            text_features=text_features,
            embedding_features=embedding_features,
            hyperparams_config=hyperparams_config,
            behavior_config=current_behavior_config,
            common_params=od_common_params,
            mlframe_models=mlframe_models,
            linear_model_config=linear_model_config,
            train_df_size_bytes=train_df_size_bytes_cached,
            val_df_size_bytes=val_df_size_bytes_cached,
            multilabel_dispatch_config=multilabel_dispatch_config,
        )

        if verbose:
            logger.info("  select_target done in %s", _elapsed_str(t0_select_target))
            log_ram_usage()

        # Pack H: auto-pick MAE / Huber loss for heavy-tail regression
        # residuals. ``cur_target_values`` is the raw y for raw-target or
        # the composite residual T for composite-target paths; in both
        # cases the inner boosting fits this distribution directly, so
        # the auto-switch matches the actual signal-vs-noise regime.
        if _is_regression_target_type(target_type):
            _apply_loss_recommendation_in_place(
                models_params=models_params,
                target_values=cur_target_values,
                composite_name=cur_target_name,
                logger_=logger,
                verbose=verbose,
            )

        pre_pipelines, pre_pipeline_names = _build_pre_pipelines(
            use_ordinary_models=use_ordinary_models,
            rfecv_models=rfecv_models,
            rfecv_models_params=rfecv_models_params,
            use_mrmr_fs=use_mrmr_fs,
            mrmr_kwargs=mrmr_kwargs,
            custom_pre_pipelines=custom_pre_pipelines,
            rfecv_leakage_corr_threshold=feature_selection_config.rfecv_leakage_corr_threshold,
            rfecv_mbh_adaptive_threshold=feature_selection_config.rfecv_mbh_adaptive_threshold,
            use_boruta_shap=feature_selection_config.use_boruta_shap,
            boruta_shap_kwargs=feature_selection_config.boruta_shap_kwargs,
            use_sample_weights_in_fs=feature_selection_config.use_sample_weights_in_fs,
            mrmr_identity_cache=(
                ctx._mrmr_identity_cache
                if getattr(feature_selection_config, "mrmr_identity_cache_scope", "ctx") == "ctx"
                else None
            ),
        )
    else:
        # No mlframe_models means the downstream pre_pipeline loop must be a no-op; bind empty sequences
        # so callers that iterate ``zip(pre_pipelines, pre_pipeline_names)`` see zero iterations rather
        # than NameError on the unbound names.
        pre_pipelines, pre_pipeline_names = [], []

    # Custom transformers run AFTER preprocessing, so the preprocessing output is shared across
    # pre_pipelines of the same model-type bucket; one cache instance covers the whole sweep. Hoist
    # to ctx (PIPECACHE-PER-TGT) so multi-target suites share one cache across targets -- selector /
    # encoder fits done for target 1 are reusable for target 2 when the cache_key matches (only
    # changes when the feature set / strategy / kind / pp_name changes).
    if ctx._pipeline_cache is None:
        ctx._pipeline_cache = PipelineCache()
    pipeline_cache = ctx._pipeline_cache

    # Suite-scoped cache observability. ``finalize_suite`` aggregates these into
    # ``metadata["cache_stats"]``. Initialise once per call rather than per pre_pipeline so the inner
    # loop's HIT / MISS bumps accumulate across the whole target's training, and use ``setdefault``
    # at ctx level so cross-target calls (multi-target suites) keep counters monotonic across calls.
    if not hasattr(ctx, "_cache_stats") or ctx._cache_stats is None:
        ctx._cache_stats = {}

    for pre_pipeline, pre_pipeline_name in tqdmu_lazy_start(zip(pre_pipelines, pre_pipeline_names), desc="pre_pipeline", total=len(pre_pipelines)):
        # CatBoost + RFECV metamodel_func combination breaks sklearn.clone().
        if _should_skip_catboost_metamodel(pre_pipeline_name.strip(), target_type, behavior_config):
            continue

        # Skip identity-equivalent pre_pipelines: marker survives across targets, so a selector
        # that was a no-op on a prior target gets skipped here before any model trains.
        # Honour ``feature_selection_config.skip_identity_equivalent_pre_pipelines``: when False
        # the caller asked to retrain even on identity-equivalent pre_pipelines (e.g. for
        # ensembling-diversity-via-RNG-seed scenarios), so this early-exit must not fire.
        _pp_name_stripped = pre_pipeline_name.strip()
        if (
            _pp_name_stripped
            and feature_selection_config.skip_identity_equivalent_pre_pipelines
            and getattr(pre_pipeline, "_mlframe_identity_equivalent", False)
        ):
            logger.info(
                "[Dedup] Skipping pre_pipeline '%s' -- "
                "identity-equivalent to ordinary (cached from "
                "prior target/iteration); models already covered.",
                _pp_name_stripped,
            )
            continue
        ens_models = [] if use_mlframe_ensembles else None
        orig_pre_pipeline = pre_pipeline

        if sample_weights:
            weight_schemas = sample_weights
            # SW-LOG-PER-PP-PER-TGT: emit this banner once per suite, not once per (target x
            # pre_pipeline x weight). The weighting schema is suite-constant; identical lines
            # repeated K_targets x K_pp times bloat the log without adding info.
            if not ctx._sw_log_emitted:
                if "uniform" in sample_weights:
                    logger.info("Using %d weighting schema(s) from extractor: %s", len(weight_schemas), list(weight_schemas.keys()))
                else:
                    logger.info("Using %d weighting schema(s) from extractor: %s. Note: uniform weighting not included.", len(weight_schemas), list(weight_schemas.keys()))
                ctx._sw_log_emitted = True
        else:
            weight_schemas = {"uniform": None}
            if not ctx._sw_log_emitted:
                logger.info("No weighting schemas from extractor, defaulting to uniform weighting.")
                ctx._sw_log_emitted = True

        # Backward val placement + recency weighting cancel each other's drift-proxy intent
        # (val older than train, training biased to newest rows). Warn so the user picks one.
        # VAL-PLACE-WARN-PP: gate behind a per-suite latch so the warning fires once, not per PP.
        _val_placement = getattr(split_config, "val_placement", "forward")
        if _val_placement == "backward" and not ctx._val_placement_warn_emitted:
            _non_uniform = [k for k in weight_schemas.keys() if k != "uniform"]
            if _non_uniform:
                ctx._val_placement_warn_emitted = True
                logger.warning(
                    "  val_placement='backward' is combined with %d non-"
                    "uniform weighting schema(s) %s. Backward val is "
                    "designed to approximate DEPLOYMENT error under "
                    "drift by mirroring the val->train gap against the "
                    "train->prod gap, while recency-style weights bias "
                    "training toward the newest rows. Together they "
                    "optimise 'fit newest, validate on oldest' -- which "
                    "contradicts the drift-proxy intent of backward. "
                    "Consider disabling use_recency_weighting on the "
                    "extractor (runs will fall back to uniform only) "
                    "or switching back to val_placement='forward'.",
                    len(_non_uniform), _non_uniform,
                )

        # Models sorted by feature tier (richest first) so text/embedding columns are dropped once per tier.
        # Strategy lookup keyed by id() because estimators / tuples are not hashable, and identity-distinct
        # instances must stay distinct in the map. Pre-computed once per suite by setup_configuration;
        # reading off ctx here avoids the O(targets * pre_pipelines * models) re-evaluation that used to
        # rebuild this map per inner-loop iteration.
        strategy_by_model = ctx.strategy_by_model
        sorted_models = ctx.sorted_mlframe_models
        # Suite-scoped feature-side cache: tier_dfs / pl.Enum map / prepared polars frames carry
        # ACROSS targets (target-independent transforms) so only y / sample_weight differ inside
        # the inner loop. Both inner caches are scoped to the current ``pre_pipeline_name`` since
        # different pre_pipelines may keep different columns (MRMR / RFECV vs ordinary), and the
        # tier-DFs / Enum maps depend on the column set after pre-pipeline column trimming.
        _suite_feature_cache = _get_feature_side_cache(ctx)
        _per_pp_cache = _suite_feature_cache.setdefault(pre_pipeline_name, {})
        tier_dfs_cache: dict[tuple, dict[str, Any]] = _per_pp_cache.setdefault("tier_dfs", {})
        # Leak-free pl.Enum map built from train+val UNION only (test EXCLUDED to avoid label-time leakage).
        # Depends only on (feature_tier, strategy class) - target-independent so it carries cross-target.
        tier_enum_map_cache: dict[tuple, dict[str, Any] | None] = _per_pp_cache.setdefault("tier_enum_map", {})
        # Prepared polars frames + xgb_category_map per (tier, supports_polars, strategy_class).
        # Target-independent because _prep_polars_df / build_polars_enum_map do not touch y; the
        # text-features fill_null pass below is also target-independent. Carry cross-target.
        prepared_frames_cache: dict[tuple, dict[str, Any]] = _per_pp_cache.setdefault("prepared_frames", {})
        prev_tier = None

        # Neural max_time defaults to P95 of non-neural train times so MLP can't run 2h while boosters take 5min.
        # CODE-LOW-4: per-target reset is INTENTIONAL -- each target's neural budget is computed only from the
        # same target's non-neural runs, so an unusually fast/slow earlier target cannot widen or starve the
        # current target's neural budget. We rebind both the local AND ctx._non_neural_train_times to the
        # SAME fresh list so the writeback at end-of-function is a no-op (the dict the caller sees is the
        # one we just mutated) and downstream readers of ctx._non_neural_train_times observe the per-target
        # contents in-flight, not the previous target's tail.
        _non_neural_train_times = []
        ctx._non_neural_train_times = _non_neural_train_times

        _total_models_in_run = len(sorted_models)
        _model_idx_in_run = 0
        _break_model_loop = False
        for mlframe_model_name in tqdmu_lazy_start(sorted_models, desc="mlframe model"):
            if _should_skip_catboost_metamodel(mlframe_model_name, target_type, behavior_config):
                continue
            _model_idx_in_run += 1
            if verbose:
                # Per-model RSS sample is intentional: localising OOM-blame to a specific
                # model+target+pre_pipeline tuple in the verbose-suite log saves hours of post-mortem
                # log-correlation. The ~3ms/call Windows cost is dwarfed by per-model fit times.
                # PSUTIL-IMPORT-HOT: ``psutil`` is now imported at module level (``_ps_module``);
                # the prior in-loop import paid ImportError lookup costs on every iter.
                try:
                    _ram_gb_now = (
                        _ps_module.Process().memory_info().rss / (1024 ** 3)
                        if _ps_module is not None else 0.0
                    )
                except Exception:
                    _ram_gb_now = 0.0
                logger.info(
                    "  process_model(%s) START -- model %d/%d, RAM=%.1fGB",
                    mlframe_model_name,
                    _model_idx_in_run, _total_models_in_run,
                    _ram_gb_now,
                )

            if mlframe_model_name not in models_params:
                logger.warning(f"mlframe model {mlframe_model_name} not known, skipping...")
                continue

            # Cross-target dataset reuse: restore the prior target's _DATASET_REUSE_CACHE_ATTRS
            # snapshot onto the freshly-built model template BEFORE the weight loop's clone()
            # forward-transfer reads them. select_target() rebuilds models_params per target so
            # the cache attributes are absent on a virgin template - without this restore the
            # XGB/LGB shims would rebuild the binned dataset on target 2. The shim's
            # signature_of(X) check then matches against the same ctx-pinned train_df pointer
            # and triggers set_label / set_weight in place rather than a fresh build.
            _restore_dataset_reuse_cache(
                ctx, mlframe_model_name, models_params[mlframe_model_name]["model"],
                pp_name=pre_pipeline_name,
            )

            strategy = strategy_by_model[id(mlframe_model_name)]

            # Drop pre-pipeline Polars originals as soon as we hit the first non-Polars strategy. The
            # post-iteration release fires only on tier transitions, but same-tier siblings (e.g. XGB and
            # LGB share tier=(False,False)) would keep Polars frames alive into a lazy pandas conversion,
            # doubling peak RAM. Releasing upfront halves peak in mixed suites.
            if (
                not strategy.supports_polars
                and train_df_polars is not None
            ):
                # Drop locals AND ctx attributes -- ctx still pins the strong ref to the same frames
                # assigned via ctx.*_df_polars at function entry, so a bare ``del`` of the locals would
                # leave maybe_clean_ram_and_gpu with nothing to reclaim and turn the log line into a lie.
                del train_df_polars, val_df_polars, test_df_polars
                train_df_polars = val_df_polars = test_df_polars = None
                # Drop polars-tier entries only - pandas-tier entries hang on the SAME cache dicts
                # (these locals now reference suite-scoped dicts in _per_pp_cache) and must survive
                # the release. ``_invalidate_polars_feature_side_cache(ctx)`` runs further down the
                # _release_ctx_polars_frames path and does the same for the prepared_frames sub-
                # cache; the tier_dfs / tier_enum_map dicts that pre-date this hoist are scrubbed
                # here so a same-target pandas-tier sibling reads a clean enum-map slot.
                for _pl_only_key in [_k for _k in tier_dfs_cache if isinstance(_k, tuple) and len(_k) >= 2 and _k[1] == "pl"]:
                    tier_dfs_cache.pop(_pl_only_key, None)
                tier_enum_map_cache.clear()  # All entries are polars-only (populated only on the polars fastpath).
                baseline_rss_mb = _release_ctx_polars_frames(
                    ctx,
                    baseline_rss_mb,
                    df_size_mb,
                    verbose=verbose,
                    reason="non-polars-native strategy entry",
                )
                if verbose:
                    logger.info(
                        "  Released pre-pipeline Polars originals before %s (non-polars-native strategy).",
                        mlframe_model_name,
                    )

            # Clone the base_pipeline per model so each iteration gets a fresh, un-fitted selector. Sharing a
            # fitted MRMR/RFECV across strategies caused `_is_fitted` to misreport True for a partially-fit
            # pipeline (selector fitted but encoder/imputer/scaler not), tripping imputer.transform on a
            # feature-names mismatch.
            _base_for_strategy = orig_pre_pipeline
            if _base_for_strategy is not None:
                try:
                    _base_for_strategy = clone(_base_for_strategy)
                except Exception:
                    # Non-BaseEstimator custom pipelines don't clone; keep the original reference.
                    pass
            pre_pipeline = strategy.build_pipeline(
                base_pipeline=_base_for_strategy,
                cat_features=cat_features,
                category_encoder=category_encoder if cat_features else None,
                imputer=imputer,
                scaler=scaler,
            )
            # Cache key = strategy.cache_key + pre_pipeline_name + feature_tier + container kind + feature-list digest.
            # feature_tier is required because CB/LGB/XGB all share cache_key="tree" but have different
            # tiers; without it, CB's text/embedding-bearing frame would be served to LGB/XGB.
            # Kind suffix prevents Polars-native (XGB) and pandas-only (LGB) consumers from sharing entries
            # within a tier, which would otherwise undo the lazy pandas conversion downstream.
            # See _compute_pipeline_cache_key for the features-digest contract (frozenset, order-invariant).
            # Pass the polars train frame (if present) so dtype changes between targets / runs
            # invalidate the cache; pandas frames don't reach this branch typed-distinct enough to
            # need the suffix (handled upstream in split_features), so it's safe to skip there.
            _cache_key_train_df = train_df_polars if strategy.supports_polars else None
            cache_key = _compute_pipeline_cache_key(
                strategy.cache_key,
                pre_pipeline_name,
                strategy.feature_tier(),
                strategy.supports_polars,
                cat_features,
                text_features,
                embedding_features,
                train_df=_cache_key_train_df,
            )

            # Polars fastpath substitutes original Polars DataFrames for natively-Polars consumers
            # (CatBoost >= 1.2.7, HGB). Polars DFs are prepared once per model (outside the weight loop)
            # because prepare_polars_dataframe() allocates via .with_columns().
            polars_fastpath_active = train_df_polars is not None and strategy.supports_polars

            if polars_fastpath_active:
                if verbose:
                    logger.info("  Polars fastpath active for %s (strategy=%s)", mlframe_model_name, type(strategy).__name__)
                # MUST use the post-promotion `cat_features` (post-auto-detect reassignment), NOT the stale
                # `cat_features_polars` snapshot from before auto-detect ran - the latter would still list
                # text-promoted columns and trip CB's polars-categorical fastpath on String dtypes.
                _cat_features = list(cat_features or [])

                # Cross-target reuse: cache key is (feature_tier, supports_polars=True, strategy_class,
                # cb_text_pass) where cb_text_pass tracks whether the CB-only Categorical->String text-
                # column cast must be applied (CB requires it; other CB-tier polars-native models don't).
                # All target-independent so the prepared frames carry from target 1 to target N.
                _prep_key = (
                    strategy.feature_tier(),
                    True,
                    type(strategy).__name__,
                    bool(text_features and mlframe_model_name == "cb"),
                )
                _cached_prep = prepared_frames_cache.get(_prep_key)
                if _cached_prep is not None:
                    prepared_train = _cached_prep["prepared_train"]
                    prepared_val = _cached_prep["prepared_val"]
                    prepared_test = _cached_prep["prepared_test"]
                    _xgb_category_map = _cached_prep["xgb_category_map"]
                    if verbose:
                        logger.info(
                            "  feature-side cache hit for %s (strategy=%s, pp=%s): reusing prepared polars frames across targets",
                            mlframe_model_name, type(strategy).__name__, pre_pipeline_name or "<ordinary>",
                        )
                else:
                    tier_base = {
                        "train_df": train_df_polars,
                        "val_df": val_df_polars,
                        "test_df": test_df_polars,
                    }
                    tier_polars = _build_tier_dfs(
                        tier_base, strategy, text_features, embedding_features, tier_dfs_cache, verbose=verbose,
                    )

                    # Enum map: leak-free, train+val union only; cached by (tier, strategy class).
                    _enum_cache_key = (strategy.feature_tier(), type(strategy).__name__)
                    if _enum_cache_key in tier_enum_map_cache:
                        _xgb_category_map = tier_enum_map_cache[_enum_cache_key]
                    elif hasattr(strategy, "build_polars_enum_map"):
                        try:
                            _xgb_category_map = strategy.build_polars_enum_map(
                                tier_polars["train_df"],
                                tier_polars.get("val_df"),
                                _cat_features,
                            )
                        except Exception as _emb_exc:
                            logger.warning(
                                "build_polars_enum_map failed for %s; "
                                "falling back to per-DF Enum cast: %s",
                                type(strategy).__name__, _emb_exc,
                            )
                            _xgb_category_map = None
                        tier_enum_map_cache[_enum_cache_key] = _xgb_category_map
                    else:
                        _xgb_category_map = None
                        tier_enum_map_cache[_enum_cache_key] = None

                    prepared_train = _prep_polars_df(tier_polars["train_df"], strategy, _cat_features, _xgb_category_map)
                    prepared_val = _prep_polars_df(tier_polars.get("val_df"), strategy, _cat_features, _xgb_category_map)
                    prepared_test = _prep_polars_df(tier_polars.get("test_df"), strategy, _cat_features, _xgb_category_map)

                    # CatBoost's polars text-features path requires plain String with no nulls; cast Categorical/Enum
                    # text columns and fill_null. The dtype mismatch happens whenever auto-detect promotes a
                    # column from cat_features to text_features without changing its backing dtype.
                    if text_features and mlframe_model_name == "cb":
                        text_cols_present = filter_existing(prepared_train, text_features)
                        if text_cols_present:
                            # Determine which of the text columns need a dtype cast.
                            needs_cast = [
                                c for c in text_cols_present
                                if prepared_train.schema[c] == pl.Categorical
                                or isinstance(prepared_train.schema[c], pl.Enum)
                            ]
                            prep_exprs = []
                            for c in text_cols_present:
                                expr = pl.col(c)
                                if c in needs_cast:
                                    expr = expr.cast(pl.String)
                                prep_exprs.append(expr.fill_null(""))
                            prepared_train = prepared_train.with_columns(prep_exprs)
                            if prepared_val is not None:
                                prepared_val = prepared_val.with_columns(prep_exprs)
                            if prepared_test is not None:
                                prepared_test = prepared_test.with_columns(prep_exprs)
                            if needs_cast and verbose:
                                logger.info(
                                    "  Cast %d text feature(s) from Polars Categorical to String "
                                    "for CatBoost: %s",
                                    len(needs_cast), needs_cast,
                                )

                    # Null-in-Categorical fill is applied upstream once on train_df_polars/val/test (search:
                    # `_polars_fill_null_in_categorical`, marker "__MISSING__"); no per-model fill needed.

                    # Store REFERENCES only (no clones / no copies): a 100GB train_df_polars is shared
                    # with ctx.train_df_polars; the prepared variant is a polars LazyFrame-evaluation
                    # result that's already eager but immutable in our path. Carrying across targets
                    # costs ~one pointer per slot - never duplicates feature data.
                    prepared_frames_cache[_prep_key] = {
                        "prepared_train": prepared_train,
                        "prepared_val": prepared_val,
                        "prepared_test": prepared_test,
                        "xgb_category_map": _xgb_category_map,
                    }

            else:

                # Lazy pandas conversion for non-Polars-native strategies. The upfront _convert_dfs_to_pandas
                # is skipped when all blockers are non-native; per-strategy conversion happens here, which
                # preserves RAM when CB/XGB can run natively on polars. Two trigger cases get distinct log
                # messages: (a) strategy genuinely non-Polars-native; (b) strategy IS native but polars
                # originals were released earlier in the run.
                # CONV-MED-5: cache the polars->pandas view by id() of the source frame on ctx so two
                # non-Polars-native strategies sharing the same source polars frame pay one conversion
                # total, not one per strategy.
                _logged_lazy_conv = False
                _view_cache = ctx._pandas_view_cache
                for df_key in ("train_df", "val_df", "test_df"):
                    df_ = common_params.get(df_key)
                    if isinstance(df_, pl.DataFrame):
                        if not _logged_lazy_conv and verbose:
                            if strategy.supports_polars:
                                _reason = (
                                    "Polars originals released "
                                    "(common_params still carries "
                                    "polars frames; converting to "
                                    "pandas for inner predict path)"
                                )
                            else:
                                _reason = (
                                    f"non-Polars-native strategy "
                                    f"{type(strategy).__name__}"
                                )
                            logger.info(
                                "  Lazy pandas conversion for %s -- %s",
                                mlframe_model_name, _reason,
                            )
                            _logged_lazy_conv = True
                        _src_id = id(df_)
                        _pd_view = _view_cache.get(_src_id)
                        # Pandas-view cache stats: count one HIT per reuse (id() match) and one MISS
                        # per fresh conversion. Stamped on ctx so finalize_suite can read without
                        # touching the cache backend (a plain dict on ctx that exposes no counters).
                        _cs_pv = ctx._cache_stats.setdefault("pandas_view_cache", {"hits": 0, "misses": 0})
                        if _pd_view is None:
                            _cs_pv["misses"] += 1
                            _pd_view = get_pandas_view_of_polars_df(df_)
                            _view_cache[_src_id] = _pd_view
                            # VIEW-CACHE-NO-EVICT: bound to 4 entries (the suite has at most 3 frames
                            # active -- train/val/test -- plus one slack for in-flight tier swaps).
                            # Without this the cache grows monotonically across targets and pins
                            # pandas blockmgrs that the upstream polars frames thought they freed.
                            # OrderedDict not used: ctx slot is plain dict for back-compat, so a
                            # popitem() FIFO is good enough at this small bound.
                            while len(_view_cache) > 4:
                                _evict_k = next(iter(_view_cache))
                                _view_cache.pop(_evict_k, None)
                        else:
                            _cs_pv["hits"] += 1
                        common_params[df_key] = _pd_view

                # Defense-in-depth: after lazy conversion, every common_params DF must be non-polars.
                # Surfacing here (rather than at trainer.fit time) makes the cross-iteration leakage cause
                # visible with full strategy/common_params context.
                for df_key in ("train_df", "val_df", "test_df"):
                    df_ = common_params.get(df_key)
                    if isinstance(df_, pl.DataFrame):
                        raise RuntimeError(
                            f"Lazy pandas conversion produced incomplete "
                            f"state for non-Polars-native strategy "
                            f"{type(strategy).__name__} ({mlframe_model_name}): "
                            f"common_params[{df_key!r}] is still pl.DataFrame "
                            f"(shape={df_.shape}, id={id(df_)}). The lazy-"
                            f"conversion hook iterated over train/val/test but "
                            f"this key escaped. Likely cause: a ``common_params`` "
                            f"override between lazy-conversion and here, or "
                            f"pipeline_cache cross-stream leakage (see core.py "
                            f"kind-suffix in cache_key)."
                        )

                tier_pandas = _build_tier_dfs(
                    {"train_df": common_params.get("train_df"), "val_df": common_params.get("val_df"), "test_df": common_params.get("test_df")},
                    strategy, text_features, embedding_features, tier_dfs_cache, verbose=verbose,
                )

            # CODE-P1-10: compute input-schema fingerprint ONCE per (model, pre_pipeline) outside the
            # weight loop. The fingerprinted train_df is the same across all weight schemas (only
            # sample_weight changes inside the weight loop), so the previous per-iteration call was
            # pure waste. Cache key is purely feature-side (strategy+tier+kind+pp_name) - dropping
            # ``target_type`` / ``cur_target_name`` from the key was the per-target hoist: the
            # schema hash depends on column names/dtypes, NOT on y, so target N reuses target 1's
            # fingerprint without recomputation. Audit-checked vs compute_model_input_fingerprint:
            # signature takes train_df + cat/text/embedding_features only, no target.
            _fp_train_df_pre = prepared_train if polars_fastpath_active else tier_pandas["train_df"]
            # FP-KEY-OMITS-CONTENT: original key excluded the train_df identity, so when the same
            # strategy / tier / kind / pp_name combination was hit by two different per-target
            # frames (filtered_train_df rebuilt across targets), the cache would return target 1's
            # schema hash for target 2. Fold ``id(train_df)`` (strong-ref-pinned at this point) and
            # the schema column-count to disambiguate; full schema hash isn't needed here because a
            # mismatch in id alone forces recompute.
            _fp_train_df_id = id(_fp_train_df_pre) if _fp_train_df_pre is not None else 0
            _fp_train_df_ncols = (
                len(_fp_train_df_pre.columns)
                if _fp_train_df_pre is not None and hasattr(_fp_train_df_pre, "columns")
                else 0
            )
            _fp_cache_key = (
                id(strategy),
                strategy.feature_tier(),
                strategy.supports_polars,
                pre_pipeline_name,
                _fp_train_df_id,
                _fp_train_df_ncols,
            )
            # Fingerprint cache stats: HIT when the per-(strategy, tier, kind, pp_name) key is already
            # cached, MISS when we have to compute. Same proxy-counter pattern as the pandas-view
            # cache above (the underlying cache is a plain dict on ctx with no counters of its own).
            _cs_fp = ctx._cache_stats.setdefault("fingerprint_cache", {"hits": 0, "misses": 0})
            if _fp_cache_key in ctx._model_input_fingerprint_cache:
                _cs_fp["hits"] += 1
                _schema_hash, _input_schema = ctx._model_input_fingerprint_cache[_fp_cache_key]
            else:
                _cs_fp["misses"] += 1
                _schema_hash, _input_schema = compute_model_input_fingerprint(
                    _fp_train_df_pre,
                    cat_features=cat_features,
                    text_features=text_features,
                    embedding_features=embedding_features,
                )
                ctx._model_input_fingerprint_cache[_fp_cache_key] = (_schema_hash, _input_schema)

            for weight_name, weight_values in tqdmu_lazy_start(weight_schemas.items(), desc="weighting schema"):
                model_name_with_weight = common_params["model_name"]
                model_file_name=f"{mlframe_model_name}"
                if weight_name != "uniform":
                    model_name_with_weight += f" w={weight_name}"
                    model_file_name +=f"_{weight_name}"

                # Isolation copy: per-(model, weight) inner mutations (sample_weight, plot_file
                # decoration, lazy pandas conversion, fastpath frame swap) must not bleed into
                # the outer ``common_params`` template that the next iteration consumes. The
                # 4-deep nesting (target_type x target x pre_pipeline x model x weight) has been
                # verified across the suite -- removing the copy regresses the cross-weight
                # contamination tests. Do NOT inline.
                current_common_params = common_params.copy()
                current_common_params["sample_weight"] = weight_values

                if polars_fastpath_active:
                    current_common_params["train_df"] = prepared_train
                    if prepared_val is not None:
                        current_common_params["val_df"] = prepared_val
                    if prepared_test is not None:
                        current_common_params["test_df"] = prepared_test
                else:
                    current_common_params["train_df"] = tier_pandas["train_df"]
                    if tier_pandas.get("val_df") is not None:
                        current_common_params["val_df"] = tier_pandas["val_df"]
                    if tier_pandas.get("test_df") is not None:
                        current_common_params["test_df"] = tier_pandas["test_df"]
                if getattr(behavior_config, "model_file_hash_suffix", True):
                    model_file_name += f"__sch_{_schema_hash}"

                if weight_name != "uniform" and current_common_params.get("plot_file"):
                    current_common_params["plot_file"] = current_common_params["plot_file"] + weight_name + "_"

                cached_dfs = pipeline_cache.get(cache_key)

                # INTENTIONAL: clone() lives INSIDE the weight loop. Each weight schema produces a
                # different trained model stored separately in models[type][target]; without per-iteration
                # cloning all in-memory entries would alias to the same last-trained sklearn object and
                # only the .dump snapshots would be correct. Do NOT move clone() outside the loop.
                original_model = models_params[mlframe_model_name]["model"]
                try:
                    cloned_model = clone(original_model)
                except RuntimeError:
                    # CatBoost wraps custom eval_metric objects internally; sklearn's identity check fails.
                    # Direct constructor call with get_params() produces an equivalent unfitted instance.
                    cloned_model = type(original_model)(**original_model.get_params())
                except TypeError:
                    # NGBoost: get_params() exposes attributes the constructor doesn't accept.
                    # SIG-IN-EXCEPT: memoize the inspect.signature lookup so the TypeError branch
                    # isn't paying ~0.5-1ms per hit -- the cache lives at module scope keyed by
                    # ``id(cls)`` because ``cls.__init__`` is class-invariant.
                    _cls = type(original_model)
                    _sig_params = _cached_init_params(_cls)
                    _raw = original_model.get_params(deep=False)
                    cloned_model = _cls(**{k: v for k, v in _raw.items() if k in _sig_params})
                # sklearn.clone() strips non-param attributes; re-assert mlframe sticky flags so the
                # calibration directive and the polars-fastpath-broken marker survive each iteration.
                if getattr(original_model, "_mlframe_posthoc_calibrate", False):
                    try:
                        cloned_model._mlframe_posthoc_calibrate = True
                    except Exception as _attr_err:
                        logger.debug("Could not set _mlframe_posthoc_calibrate on clone: %s", _attr_err)
                if getattr(original_model, "_mlframe_polars_fastpath_broken", False):
                    try:
                        cloned_model._mlframe_polars_fastpath_broken = True
                    except Exception as _attr_err:
                        logger.debug("Could not set _mlframe_polars_fastpath_broken on clone: %s", _attr_err)
                # Hand the XGB DMatrix / LGB Dataset reuse caches forward across clone() so the
                # weight-schema loop (uniform -> recency on the same train_df) reuses the heavy binned
                # dataset in place via set_label / set_weight instead of rebuilding.
                _forward_dataset_reuse_cache(original_model, cloned_model)
                # Isolation copy: each weight iteration installs its own cloned_model and may
                # patch fit_params (CatBoost text/embedding fastpath); without copying we would
                # mutate the suite-level models_params template and the next target would inherit
                # this iteration's overrides.
                current_model_params = models_params[mlframe_model_name].copy()
                current_model_params["model"] = cloned_model

                # CatBoost is the only Polars-native consumer that accepts cat_features / text_features /
                # embedding_features at fit time; XGB and HGB auto-detect via enable_categorical=True.
                if polars_fastpath_active and mlframe_model_name == "cb" and "fit_params" in current_model_params:
                    extra_fit = {}
                    if _cat_features:
                        _valid_cat = _filter_polars_cat_features_by_dtype(
                            prepared_train, _cat_features
                        )
                        if _valid_cat:
                            extra_fit["cat_features"] = _valid_cat
                    if text_features:
                        cb_text = filter_existing(prepared_train, text_features)
                        if cb_text:
                            extra_fit["text_features"] = cb_text
                    if embedding_features:
                        cb_emb = filter_existing(prepared_train, embedding_features)
                        if cb_emb:
                            extra_fit["embedding_features"] = cb_emb
                    if extra_fit:
                        current_model_params["fit_params"] = {**current_model_params["fit_params"], **extra_fit}

                # Build process_model kwargs using helper
                process_model_kwargs = _build_process_model_kwargs(
                    model_file=model_file,
                    model_name_with_weight=model_name_with_weight,
                    model_file_name=model_file_name,
                    target_type=target_type,
                    pre_pipeline=pre_pipeline,
                    pre_pipeline_name=pre_pipeline_name,
                    cur_target_name=cur_target_name,
                    models=models,
                    model_params=current_model_params,
                    common_params=current_common_params,
                    ens_models=ens_models,
                    trainset_features_stats=trainset_features_stats,
                    verbose=verbose,
                    cached_dfs=cached_dfs,
                    # Per-strategy decision on whether preprocessing for this strategy is already done.
                    # Two sufficient conditions:
                    #   (1) the suite-level polars-ds pipeline ran AND this strategy consumes polars natively;
                    #   (2) the polars fastpath is active for this strategy (its frame is the polars native
                    #       one, so sklearn encoder/scaler/imputer would be redundant and crash anyway).
                    # Note: requires_encoding=True is NOT a re-run trigger (HGB declares it for pandas-fallback
                    # only; on the polars fastpath HGB consumes pl.Categorical natively). Only non-Polars
                    # strategies fall through to their own pre_pipeline run in trainer.py.
                    polars_pipeline_applied=(
                        (polars_pipeline_applied and strategy.supports_polars)
                        or polars_fastpath_active
                    ),
                    mlframe_model_name=mlframe_model_name,
                    metadata_columns=metadata.get("columns"),
                )

                _is_neural = is_neural_model(mlframe_model_name)
                _timeout = _compute_neural_max_time(_non_neural_train_times) if _is_neural else None
                if _timeout is not None:
                    _max_time_dict, _p95, _n = _timeout
                    # Reach into Pipeline(StandardScaler, TTR(PytorchLightningRegressor(...))) to find trainer_params.
                    _neural_model = current_model_params.get("model")
                    if _neural_model is not None:
                        _inner = getattr(_neural_model, "regressor", None)
                        if _inner is None and hasattr(_neural_model, "named_steps"):
                            for _step in _neural_model.named_steps.values():
                                if hasattr(_step, "regressor"):
                                    _inner = _step.regressor
                                    break
                        if _inner is not None and hasattr(_inner, "trainer_params"):
                            _inner.trainer_params["max_time"] = _max_time_dict
                            if verbose:
                                logger.info(
                                    "  [NeuralTimeout] %s max_time=%dh%02dm%02ds "
                                    "(P95 of %d prior non-neural train times: %.0fs)",
                                    mlframe_model_name,
                                    _max_time_dict["hours"], _max_time_dict["minutes"], _max_time_dict["seconds"],
                                    _n, _p95,
                                )

                t0_model = timer()
                try:
                    with phase("process_model", model=mlframe_model_name, weight=weight_name):
                        trainset_features_stats, pre_pipeline, train_df_transformed, val_df_transformed, test_df_transformed = process_model(
                            **process_model_kwargs
                        )
                except Exception as model_err:
                    # Skip-and-continue is opt-in. KeyboardInterrupt is intentionally not caught here;
                    # native SIGSEGV that kills the process won't be caught either.
                    if not behavior_config.continue_on_model_failure:
                        raise
                    logger.error(
                        f"  process_model({mlframe_model_name}, w={weight_name}) FAILED after "
                        f"{_elapsed_str(t0_model)} -- {type(model_err).__name__}: {model_err}. "
                        f"continue_on_model_failure=True -> skipping and moving on.",
                        exc_info=True,
                    )
                    metadata.setdefault("failed_models", []).append({
                        "model": mlframe_model_name,
                        "weighting": weight_name,
                        "error_type": type(model_err).__name__,
                        "error_message": str(model_err),
                    })
                    continue  # next weight_name in the inner loop
                if verbose:
                    logger.info("  process_model(%s, w=%s) done -- %s", mlframe_model_name, weight_name, _elapsed_str(t0_model))
                if not _is_neural and t0_model is not None:
                    _non_neural_train_times.append(timer() - t0_model)

                # After the first model trains, if the pre_pipeline is identity-equivalent (kept all
                # columns) AND the ordinary branch is in the suite, the remaining models would see
                # identical data - skip them.
                if (
                    _model_idx_in_run == 1
                    and _pp_name_stripped
                    and use_ordinary_models
                    and feature_selection_config.skip_identity_equivalent_pre_pipelines
                    and getattr(
                        pre_pipeline, "_mlframe_identity_equivalent", False
                    )
                ):
                    _skip_remaining = _total_models_in_run - 1
                    if _skip_remaining > 0:
                        logger.info(
                            "[Dedup] pre_pipeline '%s' is "
                            "identity-equivalent to ordinary (kept "
                            "all %d columns); skipping remaining "
                            "%d model(s) for this target.",
                            _pp_name_stripped,
                            train_df_transformed.shape[1]
                            if train_df_transformed is not None
                            else 0,
                            _skip_remaining,
                        )
                    _break_model_loop = True
                    break  # exit weight_schema loop

                # Hand the dataset-reuse cache from cloned_model back to the template so the next
                # weight-schema iteration's clone() carries it forward (symmetric to the forward-transfer
                # block above). Without this the cache would be born and die in a single iteration.
                _forward_dataset_reuse_cache(cloned_model, original_model, skip_none=True)

                # Persist this model's input-schema fingerprint in metadata so load-time can verify it
                # against the serving frame. Multi-output extensions (target_type / n_classes /
                # multilabel_strategy + schema_version) let load_mlframe_suite dispatch correctly;
                # legacy artifacts without these fields fall back to binary inference.
                _record = {
                    "schema_hash": _schema_hash,
                    "input_schema": _input_schema,
                    "mlframe_model": mlframe_model_name,
                    "weight_name": weight_name,
                    "target_type": str(target_type) if target_type is not None else None,
                    "schema_version": 2,  # 1=legacy, 2=multi-output-aware
                }
                train_y = (
                    cur_target_values[_train_idx]
                    if isinstance(cur_target_values, (np.ndarray, pl.Series))
                    else cur_target_values.iloc[_train_idx]
                )
                try:
                    if target_type == _TargetTypes.MULTILABEL_CLASSIFICATION:
                        _record["n_classes"] = (
                            int(train_y.shape[1])
                            if hasattr(train_y, "shape") and train_y.ndim == 2
                            else None
                        )
                        _record["multilabel_strategy"] = "native" if (
                            hasattr(strategy, "supports_native_multilabel") and strategy.supports_native_multilabel
                        ) else "wrapper"
                    elif target_type == _TargetTypes.MULTICLASS_CLASSIFICATION:
                        _record["n_classes"] = (
                            int(len(np.unique(np.asarray(train_y))))
                            if hasattr(train_y, "shape") else None
                        )
                        _record["multilabel_strategy"] = None
                    else:
                        _record["n_classes"] = None
                        _record["multilabel_strategy"] = None
                except Exception as _intro_err:
                    # Never fail the metadata write because of an introspection error on optional fields.
                    # Surface as warning since load_mlframe_suite dispatches on n_classes/multilabel_strategy.
                    logger.warning("n_classes/multilabel_strategy introspection failed for %s: %s", mlframe_model_name, _intro_err)

                # Per-model feature-selection report. ``pre_pipeline`` returned by ``process_model``
                # is the FITTED selector / pipeline (or None for the ordinary branch). ``train_df_
                # transformed.columns`` gives the post-FS surviving features for both pandas and
                # polars frames. The report is always stamped (selector_name=None for ordinary) so
                # downstream consumers can rely on the key existing.
                #
                # Cache the report at (target, pp_name, model_name, selector_params_hash, kept_cols)
                # because the fitted selector + kept columns are weight-invariant. The prior key used
                # id(pre_pipeline) which is Python's memory-address; once an object is GC'd its id can
                # be recycled, so a long-lived ``ctx._fs_report_cache`` could collide on a recycled
                # address across the per-(target, model) inner loops. ``_selector_params_hash`` is
                # content-derived and id-stable across recycling.
                try:
                    _kept_cols = None
                    if train_df_transformed is not None and hasattr(train_df_transformed, "columns"):
                        _kept_cols = list(train_df_transformed.columns)
                    _fsr_key = (
                        cur_target_name,
                        pre_pipeline_name,
                        mlframe_model_name,
                        _selector_params_hash(_unwrap_selector(pre_pipeline)),
                        tuple(_kept_cols) if _kept_cols is not None else None,
                    )
                    _fsr_cached = ctx._fs_report_cache.get(_fsr_key)
                    if _fsr_cached is None:
                        _fsr_cached = _build_feature_selection_report(
                            pre_pipeline=pre_pipeline,
                            pre_pipeline_name=pre_pipeline_name,
                            fitted_columns_in=None,
                            kept_columns=_kept_cols,
                        )
                        ctx._fs_report_cache[_fsr_key] = _fsr_cached
                    _record["feature_selection_report"] = _fsr_cached
                except Exception as _fsr_err:
                    logger.warning("feature_selection_report build failed for %s: %s", mlframe_model_name, _fsr_err)
                    _record["feature_selection_report"] = {
                        "selector_name": None,
                        "selector_params_hash": None,
                        "kept_features": None,
                        "dropped_features": None,
                        "scores": None,
                        "reason_per_feature": None,
                    }

                metadata.setdefault("model_schemas", {})[model_file_name] = _record

                if cached_dfs is None:
                    pipeline_cache.set(cache_key, train_df_transformed, val_df_transformed, test_df_transformed)

            # Preserve a fitted feature-selector across same-bucket tree iterations. Tree strategies return
            # just the base_pipeline from build_pipeline(); non-tree strategies wrap it in a full Pipeline
            # (encoder/imputer/scaler), which we do NOT want to reuse as the base for other model types.
            if cache_key.startswith("tree"):
                orig_pre_pipeline = pre_pipeline

            if _break_model_loop:
                break

            # Release dataset-reuse caches at strategy-iter end. Both shims park the heavy binned dataset
            # on ``_cached_train_*`` / ``_cached_val_*`` as a weight-schema-loop scratchpad; nothing
            # downstream reads them (.predict goes through _Booster, ensemble uses pre-computed probs,
            # save strips via __getstate__). Releasing here frees ~30% of peak RAM between strategies.
            # Capture the binned-dataset references off the template BEFORE clearing so the next
            # target's _restore_dataset_reuse_cache can re-attach them. Without this snapshot the
            # clear below frees the dataset and the cross-target hoist degrades to a no-op (same
            # behaviour as before the hoist). Storing references only - the binned dataset is
            # shared with whatever held it before; the clear merely drops the template's pointer.
            _capture_dataset_reuse_cache(ctx, mlframe_model_name, original_model, pp_name=pre_pipeline_name)
            _maybe_clear_shim_cache(original_model)
            # ens_models snapshots may also hold the cache by reference (forward-transfer at clone() copied
            # the reference rather than moving it); release on each so the binned dataset can be freed.
            if ens_models:
                for _ens_ns in ens_models:
                    _maybe_clear_shim_cache(getattr(_ens_ns, "model", None))

            # On a tier transition into a non-Polars strategy, release the pre-pipeline Polars originals.
            cur_tier = strategy.feature_tier()
            if prev_tier is not None and cur_tier != prev_tier and not strategy.supports_polars:
                if train_df_polars is not None:
                    # Same rationale as the entry-site release: locals AND ctx attributes must both drop their refs.
                    del train_df_polars, val_df_polars, test_df_polars
                    train_df_polars = val_df_polars = test_df_polars = None
                    # Selective drop: see same-shape comment at the non-polars-native entry site
                    # above for rationale. These cache references are suite-scoped now, so a blanket
                    # .clear() would also wipe pandas-tier entries that survived the polars release.
                    for _pl_only_key in [_k for _k in tier_dfs_cache if isinstance(_k, tuple) and len(_k) >= 2 and _k[1] == "pl"]:
                        tier_dfs_cache.pop(_pl_only_key, None)
                    tier_enum_map_cache.clear()
                    baseline_rss_mb = _release_ctx_polars_frames(ctx, baseline_rss_mb, df_size_mb, verbose=verbose, reason="tier transition")
                    if verbose:
                        logger.info("  Released pre-pipeline Polars originals (tier transition)")
            prev_tier = cur_tier

        if ens_models and len(ens_models) > 1:
            if verbose:
                logger.info(f"evaluating simple ensembles...")
            ens_n_features = train_df_transformed.shape[1] if train_df_transformed is not None else None
            # Name the ensemble by its members so log grep shows which models actually participated;
            # cap to 4 to keep headers readable. short_model_tag strips internal shim suffixes
            # (WithDMatrixReuse / WithDatasetReuse) so the tag is the bare family name.
            from .._format import short_model_tag as _short_tag_fn
            _member_tags = [_short_tag_fn(getattr(m, "model", m)) for m in ens_models]
            if len(_member_tags) <= 4:
                _members_label = "[" + "+".join(_member_tags) + "]"
            else:
                _members_label = f"[N={len(_member_tags)}]"
            # confidence_ensemble_quantile=0.0 disables the Conf Ensemble output entirely.
            _conf_q = float(getattr(behavior_config, "confidence_ensemble_quantile", 0.1))
            _ensembles = score_ensemble(
                models_and_predictions=ens_models,
                ensemble_name=f"{pre_pipeline_name}{_members_label} ",
                n_features=ens_n_features,
                uncertainty_quantile=_conf_q,
                **common_params,
            )
            # Persist the ensemble outputs so finalize_suite can serialise them and downstream
            # consumers (predict, reporting) see them. Pre-fix this return value was bound to a
            # local that nothing read, silently discarding every ensemble model the suite built.
            if _ensembles:
                ctx.ensembles.setdefault(target_type, {})[cur_target_name] = _ensembles
                # Mirror into the per-target model list (same slot the per-family training loop
                # uses) so any code iterating ``models[target_type][target_name]`` picks the
                # ensembles up without needing a separate dispatch.
                _target_models = models.setdefault(target_type, {}).setdefault(cur_target_name, [])
                for _ens_method, _ens_result in _ensembles.items():
                    _target_models.append(_ens_result)
                # Stamp the winning ensemble flavour so the predict path picks the same flavour the
                # training selection rule would have picked. Predict reads ``ensembles_chosen``
                # (see core/predict.py::_resolve_chosen_flavour) which expects the nested layout
                # ``{target_type: {target_name: flavour}}``. A None winner (no candidate exposed a
                # ranking metric) is intentionally NOT stamped so the predict-side fallback fires.
                try:
                    _chosen = _choose_ensemble_flavour(_ensembles)
                    if _chosen is not None:
                        # Sub-key per ensemble family: simple per-target ensembles live under
                        # ``ensembles_chosen["simple"]``; cross-target ensembles are stamped by
                        # _phase_composite_post under ``ensembles_chosen["cross_target"]``.
                        metadata.setdefault("ensembles_chosen", {}) \
                            .setdefault("simple", {}) \
                            .setdefault(target_type, {})[cur_target_name] = _chosen
                except Exception as _choose_err:
                    logger.warning("ensembles_chosen stamp failed for %s/%s: %s", target_type, cur_target_name, _choose_err)
                # C-P1-11: persist rrf_k (and a couple of other replay-critical params) into metadata
                # so predict-side ``_combine_probs`` replays the exact same blend a non-default-k
                # train was scored with. Pre-fix predict hard-coded k=60 -- a user setting rrf_k=10
                # at train time silently got k=60 at predict time. The default 60 mirrors
                # ``score_ensemble``'s default; common_params may override.
                try:
                    _rrf_k_used = int(common_params.get("rrf_k", 60))
                except (TypeError, ValueError):
                    _rrf_k_used = 60
                metadata.setdefault("ensembles_chosen_params", {}) \
                    .setdefault(str(target_type), {})[str(cur_target_name)] = {
                        "rrf_k": _rrf_k_used,
                    }

    ctx.models = models
    ctx.metadata = metadata
    ctx.trainset_features_stats = trainset_features_stats
    # Merge ``pipeline_cache`` HIT / MISS counters into the per-suite cache_stats accumulator.
    # PipelineCache itself is local to this function (one instance per pre_pipeline sweep) so the
    # only handoff to finalize is this stash; later targets create fresh PipelineCaches whose hits
    # accumulate via ``+=`` into the suite-wide running totals.
    try:
        _cs_pc = ctx._cache_stats.setdefault("pipeline_cache", {"hits": 0, "misses": 0})
        _cs_pc["hits"] += int(getattr(pipeline_cache, "n_hits", 0))
        _cs_pc["misses"] += int(getattr(pipeline_cache, "n_misses", 0))
    except Exception as _pc_stats_err:
        logger.debug("pipeline_cache stats merge failed: %s", _pc_stats_err)
    # CODE-LOW-2 + CODE-LOW-4: slug_to_original_target_{type,name} and _non_neural_train_times
    # are mutable containers we already rebound on ctx (the slugs are bound by reference at the top
    # of this function and mutated in place; _non_neural_train_times is rebound to a fresh list each
    # target with a matching ``ctx._non_neural_train_times = _non_neural_train_times`` at that point).
    # The earlier writeback of these three was a no-op.
    ctx.train_df_polars = train_df_polars
    ctx.val_df_polars = val_df_polars
    ctx.test_df_polars = test_df_polars
    ctx.train_df_pd = train_df_pd
    ctx.val_df_pd = val_df_pd
    ctx.test_df_pd = test_df_pd
    ctx.filtered_train_df = filtered_train_df
    ctx.filtered_val_df = filtered_val_df
    ctx.pipeline = pipeline
    ctx.defer_pandas_conv = defer_pandas_conv
    ctx.baseline_rss_mb = baseline_rss_mb
    ctx.train_df_size_bytes_cached = train_df_size_bytes_cached
    ctx.val_df_size_bytes_cached = val_df_size_bytes_cached
