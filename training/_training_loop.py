"""Training loop helpers extracted from ``trainer.py``.

Core training path: model fitting with CatBoost/LGB/XGB fallbacks,
early stopping, OOM recovery, and post-hoc probability calibration.
"""

from __future__ import annotations

import logging
from timeit import default_timer as timer
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from ._cb_pool import _maybe_get_or_build_cb_pool, _maybe_rewrite_eval_set_as_cb_pool
from ._predict_guards import _recover_cb_feature_names
from ._eval_helpers import _align_xgb_cat_categories
from ._pipeline_helpers import _passthrough_cols_fit_transform
from sklearn.isotonic import IsotonicRegression

from mlframe.helpers import get_model_best_iter
from mlframe.config import CATBOOST_MODEL_TYPES, XGBOOST_MODEL_TYPES

from ._predict_guards import _recover_cb_feature_names

logger = logging.getLogger(__name__)


def _handle_oom_error(model_obj, model_type_name: str) -> bool:
    """2026-05-12: migrated from _eval_helpers refactor fallout.
    Attempts to recover from an OOM error by clearing caches and
    returning True if the caller should retry the fit.
    """
    import gc
    gc.collect()
    # Clear LGB/XGB/CB internal caches if accessible.
    for _attr in ("_Booster", "_cached_train_features", "_cached_val_features"):
        if hasattr(model_obj, _attr):
            try:
                delattr(model_obj, _attr)
            except Exception:
                pass
    logger.warning(
        "OOM during %s.fit; cleared caches and will retry once.",
        model_type_name,
    )
    return True


class _SigmoidAdapter:
    """Thin adapter giving a fitted LogisticRegression an IsotonicRegression-
    style .predict() API that returns positive-class probabilities."""

    def __init__(self, lr):
        self.lr = lr

    def predict(self, x):
        import numpy as _np

        return self.lr.predict_proba(_np.asarray(x).reshape(-1, 1))[:, 1]


class _PostHocCalibratedModel:
    """Transparent wrapper that applies isotonic post-hoc calibration to
    predict_proba outputs of a fitted binary classifier.

    Added 2026-04-15 to make ``prefer_calibrated_classifiers=True`` actually
    calibrate tree classifiers. Prior behavior only swapped the early-stopping
    eval_metric, which was a no-op when early stopping did not trigger -- so
    calibrated and uncalibrated runs produced bit-identical probabilities.

    The wrapper delegates every attribute to the underlying ``base`` model
    except ``predict_proba``, which runs the base classifier and then maps
    the positive-class probability through a fitted IsotonicRegression.
    """

    def __init__(self, base, calibrator):
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "_calibrator", calibrator)

    def __getattr__(self, name):  # delegate unknown attrs to base
        # During unpickling __getattr__ may fire before __dict__ is populated.
        # Guard against that to avoid infinite recursion / KeyError.
        if name in ("base", "_calibrator", "__setstate__", "__getstate__", "__reduce__", "__reduce_ex__"):
            raise AttributeError(name)
        try:
            base = object.__getattribute__(self, "__dict__")["base"]
        except KeyError:
            raise AttributeError(name)
        return getattr(base, name)

    def __getstate__(self):
        return {"base": self.base, "_calibrator": self._calibrator}

    def __setstate__(self, state):
        object.__setattr__(self, "base", state["base"])
        object.__setattr__(self, "_calibrator", state["_calibrator"])

    def predict_proba(self, X):
        import numpy as _np

        raw = self.base.predict_proba(X)
        raw = _np.asarray(raw)
        if raw.ndim == 2 and raw.shape[1] == 2:
            p1 = self._calibrator.predict(raw[:, 1])
            p1 = _np.clip(p1, 0.0, 1.0)
            out = _np.column_stack([1.0 - p1, p1])
            return out
        return raw

    def predict(self, X):
        import numpy as _np

        probs = self.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] == 2:
            classes = getattr(self.base, "classes_", _np.array([0, 1]))
            return classes[(probs[:, 1] >= 0.5).astype(int)]
        return self.base.predict(X)


class _PerClassIsotonicCalibrator:
    """Multi-output post-hoc calibrator: K independent IsotonicRegression fits.

    Added 2026-04-24 Session 4 to unblock calibration on
    ``MULTICLASS_CLASSIFICATION`` and ``MULTILABEL_CLASSIFICATION``
    target types (previously raised ``NotImplementedError`` in
    ``evaluation.post_calibrate_model``).

    Semantics per target_type:
      - **MULTICLASS** (exclusive labels, softmax output): fit one
        isotonic per class on ``probs[:, k]`` vs ``(y_true == k)``.
        At predict time, each column is mapped independently through
        its own isotonic; then re-normalised row-wise so probabilities
        sum to 1 (preserves the exclusive-class invariant).
      - **MULTILABEL** (independent binary outputs, per-label sigmoid):
        fit one isotonic per label on ``probs[:, k]`` vs ``y_true[:, k]``.
        At predict time, each column is mapped independently; no
        re-normalisation (labels are independent).

    Numerical guards:
      - Each per-class isotonic needs >=2 samples of both classes in
        training; if a class is near-constant in the calibration set,
        we skip that class's calibrator (identity mapping applied).
      - Output clipped to [0, 1] post-isotonic (isotonic can over/
        undershoot at boundaries).

    Wrapped in _PostHocCalibratedModel for transparent predict_proba /
    predict delegation. Stored as a dict {class_idx: IsotonicRegression}
    plus a boolean mode flag (exclusive vs independent).
    """

    def __init__(self, calibrators, is_exclusive: bool, n_classes: int):
        """
        calibrators: dict {class_idx: IsotonicRegression or None (skip)}
        is_exclusive: True for MULTICLASS softmax, False for MULTILABEL sigmoid
        n_classes: K
        """
        self.calibrators = calibrators
        self.is_exclusive = is_exclusive
        self.n_classes = n_classes

    @classmethod
    def fit(cls, probs_NK, y_true, target_type):
        """Fit K independent isotonic regressions on the calibration set.

        Parameters
        ----------
        probs_NK : np.ndarray (N, K)
            Canonical (N, K) probability matrix (use
            ``_canonical_predict_proba_shape`` to coerce first).
        y_true : np.ndarray
            - MULTICLASS: shape (N,) with int labels 0..K-1
            - MULTILABEL: shape (N, K) binary indicator matrix
        target_type : TargetTypes
        """
        import numpy as _np
        from sklearn.isotonic import IsotonicRegression
        from .configs import TargetTypes

        probs = _np.asarray(probs_NK, dtype=_np.float64)
        K = probs.shape[1]
        is_exclusive = target_type == TargetTypes.MULTICLASS_CLASSIFICATION
        y = _np.asarray(y_true)

        calibrators = {}
        for k in range(K):
            # Per-class binary target
            if is_exclusive:
                y_k = (y == k).astype(_np.int8)
            else:
                # Multilabel: y is (N, K)
                y_k = y[:, k].astype(_np.int8)
            # Guard: skip constant-label calibrators (1-class or near-so)
            n_pos = int(y_k.sum())
            if n_pos < 2 or n_pos >= (len(y_k) - 1):
                calibrators[k] = None  # identity mapping
                continue
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(probs[:, k], y_k)
            calibrators[k] = iso

        return cls(calibrators, is_exclusive, K)

    def predict_proba(self, probs_NK):
        """Apply per-class isotonic to the (N, K) probability matrix.

        Returns a new (N, K) array with each column independently
        calibrated. For MULTICLASS, row-normalise so rows sum to 1.
        """
        import numpy as _np

        probs = _np.asarray(probs_NK, dtype=_np.float64)
        out = _np.empty_like(probs)
        for k in range(self.n_classes):
            iso = self.calibrators.get(k)
            if iso is None:
                out[:, k] = probs[:, k]  # identity
            else:
                out[:, k] = _np.clip(iso.predict(probs[:, k]), 0.0, 1.0)
        if self.is_exclusive:
            # Softmax-space: re-normalise rows to sum to 1. Guard against
            # all-zero rows (rare but possible after clip).
            row_sums = out.sum(axis=1, keepdims=True)
            row_sums = _np.where(row_sums == 0.0, 1.0, row_sums)
            out = out / row_sums
        return out


class _PostHocMultiCalibratedModel:
    """Multi-output variant of _PostHocCalibratedModel.

    Wraps ``base`` classifier + per-class isotonic calibrator.
    ``predict_proba(X)`` runs the base model and routes through
    ``_PerClassIsotonicCalibrator.predict_proba`` for (N, K) output.

    Uses ``_canonical_predict_proba_shape`` to normalise ``MultiOutputClassifier``'s
    List[(N, 2)] output to (N, K) before calibration.
    """

    def __init__(self, base, calibrator: "_PerClassIsotonicCalibrator", target_type, classes_=None):
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "_calibrator", calibrator)
        object.__setattr__(self, "_target_type", target_type)
        object.__setattr__(self, "_classes", classes_)

    def __getattr__(self, name):
        if name in ("base", "_calibrator", "_target_type", "_classes", "__setstate__", "__getstate__", "__reduce__", "__reduce_ex__"):
            raise AttributeError(name)
        try:
            base = object.__getattribute__(self, "__dict__")["base"]
        except KeyError:
            raise AttributeError(name)
        return getattr(base, name)

    def __getstate__(self):
        return {
            "base": self.base,
            "_calibrator": self._calibrator,
            "_target_type": self._target_type,
            "_classes": self._classes,
        }

    def __setstate__(self, state):
        for k, v in state.items():
            object.__setattr__(self, k, v)

    def predict_proba(self, X):
        from .helpers import _canonical_predict_proba_shape

        raw = self.base.predict_proba(X)
        classes_ = getattr(self.base, "classes_", self._classes)
        probs_NK = _canonical_predict_proba_shape(raw, classes_=classes_)
        return self._calibrator.predict_proba(probs_NK)

    def predict(self, X):
        from .helpers import _predict_from_probs

        probs = self.predict_proba(X)
        return _predict_from_probs(probs, self._target_type, classes_=self._classes)


def _maybe_apply_posthoc_calibration(model, fit_params, model_type_name, verbose=False):
    """If the fitted estimator was tagged for post-hoc calibration and an
    eval_set is available, fit an IsotonicRegression on (val_preds, val_y)
    and return a wrapped model. Otherwise return the model unchanged.
    """
    try:
        inner = model.steps[-1][1] if hasattr(model, "steps") else model
    except Exception:
        inner = model

    want_calib = getattr(inner, "_mlframe_posthoc_calibrate", False) or getattr(model, "_mlframe_posthoc_calibrate", False)
    if not want_calib:
        return model
    # Post-hoc calibration hook is now a no-op. Calibration is handled
    # pre-fit by wrapping classifiers in CalibratedClassifierCV (see
    # _configure_*_params). This avoids the val-set overfitting problem.
    return model


def _train_model_with_fallback(
    model: Any,
    model_obj: Any,
    model_type_name: str,
    train_df: Union[pd.DataFrame, np.ndarray],
    train_target: Union[pd.Series, np.ndarray],
    fit_params: Dict[str, Any],
    verbose: bool = False,
) -> Tuple[Any, Optional[int]]:
    """Train model with automatic GPU->CPU fallback on OOM errors.

    Parameters
    ----------
    model : Any
        Model to train (may be a Pipeline).
    model_obj : Any
        The actual estimator object (extracted from Pipeline if needed).
    model_type_name : str
        Name of the model type (e.g., 'CatBoostClassifier').
    train_df : pd.DataFrame or np.ndarray
        Training features.
    train_target : pd.Series or np.ndarray
        Training target values.
    fit_params : dict
        Additional parameters for model.fit().
    verbose : bool, default=False
        Whether to log verbose output.

    Returns
    -------
    tuple
        (trained_model, best_iteration) where best_iteration may be None.
    """
    t0_fit = timer()
    # Fix 9.4.3 (CB only, 2026-04-21): reuse a single ``catboost.Pool``
    # across weight schemas and same-target_type targets by mutating the
    # Pool's label/weight in place instead of letting the sklearn wrapper
    # rebuild from X on every fit. Gated on:
    #   * model is CatBoost-family;
    #   * installed CatBoost exposes ``Pool.set_label`` and
    #     ``Pool.set_weight`` (callable);
    #   * ``CatBoostClassifier.fit(X=Pool)`` is the idiomatic native path
    #     (short-circuits rebuild in ``_build_train_pool``).
    # XGB/LGB are not covered this round -- their sklearn wrappers don't
    # accept pre-built DMatrix/Dataset yet (upstream FRs drafted in
    # ``D:\Machine Learning\3rdParty\reproducers\upstream_feature_requests\``).
    # Only the per-build logging from Fix 9.4.1 makes their rebuild cost visible.
    _cb_pool = _maybe_get_or_build_cb_pool(
        model_type_name=model_type_name,
        model=model,
        train_df=train_df,
        train_target=train_target,
        fit_params=fit_params,
    )
    # Fix Orch-1 (2026-04-21): also reuse the val Pool across fits.
    # Rewrites fit_params['eval_set'] from (val_df, val_target) to a
    # cached Pool so CB's sklearn wrapper short-circuits the val-side
    # rebuild too. Only fires when _cb_pool is active (train-side reuse
    # succeeded) -- otherwise the mixed-container path (train=df,
    # eval_set=pool) confuses CB's fit signature.
    if _cb_pool is not None and model_type_name in CATBOOST_MODEL_TYPES:
        _maybe_rewrite_eval_set_as_cb_pool(fit_params)
    # Diagnostic: log the type+module of train_df right before model.fit so
    # silent type drift is visible in the log (Polars vs pandas vs numpy).
    # Critical: type(pl.DataFrame).__name__ == "DataFrame" -- same as pandas --
    # so we log the module too, otherwise "DataFrame" can hide a Polars frame
    # that should have been converted upstream.
    _is_polars = isinstance(train_df, pl.DataFrame)
    _is_pandas = isinstance(train_df, pd.DataFrame)
    try:
        if _is_polars:
            _kind = "pl.DataFrame"
        elif _is_pandas:
            _kind = "pd.DataFrame"
        elif isinstance(train_df, np.ndarray):
            _kind = f"np.ndarray(dtype={train_df.dtype})"
        else:
            _kind = type(train_df).__name__
        if hasattr(train_df, "dtypes") and hasattr(train_df, "columns"):
            _dtype_summary = ", ".join(f"{c}={train_df[c].dtype}" for c in list(train_df.columns)[:5])
            if len(train_df.columns) > 5:
                _dtype_summary += f", ... ({len(train_df.columns)} cols total)"
        else:
            _dtype_summary = ""
        logger.info("  [pre-fit] train_df type=%s, %s", _kind, _dtype_summary)
    except Exception:
        pass

    # Polars-frame contract: only CatBoost, XGBoost, and HistGradientBoosting
    # accept a Polars frame natively at fit time -- their strategies carry
    # ``supports_polars=True``. Everyone else (LGB, sklearn, linear, ridge,
    # ...) MUST arrive with pandas; if a pl.DataFrame gets here for them, the
    # upstream lazy-conversion -> pipeline_cache -> process_model chain has a
    # leak. Previously the trainer silently ran a second polars->pandas
    # conversion as a "self-heal" -- which hid the 2026-04-23 regression where
    # ``pipeline_cache`` crossed streams between XGB (polars-native,
    # ``cache_key="tree" + tier(False,False)``) and LGB (same key) -- LGB kept
    # pulling XGB's polars frame out of cache and paying a duplicate 224 s
    # conversion. The pipeline_cache fix (container-kind in key, core.py) is
    # the real fix; this raise is the guard that ensures future leaks are
    # caught at the trainer boundary instead of being papered over.
    _POLARS_NATIVE_FIT_MODEL_PREFIXES = (
        "CatBoost",  # CatBoostClassifier / CatBoostRegressor / CatBoost
        "XGB",  # XGBClassifier / XGBRegressor / XGBRanker
        "HistGradient",  # HistGradientBoostingClassifier / Regressor
    )
    # Look through MultiOutputClassifier wrapper for the polars-native check.
    # The wrapper's `estimator` is the per-label base; if the base is polars-native
    # (e.g. HGB), each per-label fit will accept polars too.
    # _ChainEnsemble is the multilabel chain ensemble -- its inner is exposed
    # as `base_estimator` (cloned per chain at fit time).
    _effective_model_type_name = model_type_name
    if model_type_name in ("MultiOutputClassifier", "MultiOutputRegressor", "ClassifierChain"):
        inner = getattr(model, "estimator", None)
        if inner is not None:
            _effective_model_type_name = type(inner).__name__
    elif model_type_name == "_ChainEnsemble":
        inner = getattr(model, "base_estimator", None)
        if inner is not None:
            _effective_model_type_name = type(inner).__name__
    if _is_polars and not any(_effective_model_type_name.startswith(p) for p in _POLARS_NATIVE_FIT_MODEL_PREFIXES):
        raise RuntimeError(
            f"{model_type_name} received pl.DataFrame at fit time "
            f"(shape={train_df.shape}, id={id(train_df)}). Only Polars-native "
            f"strategies (CatBoost, XGBoost, HistGradientBoosting) may receive "
            f"polars -- everyone else needs pandas via the core.py lazy-"
            f"conversion path. Most likely cause: ``pipeline_cache`` returned "
            f"a polars frame cached by a polars-native strategy under a "
            f"``cache_key`` that collides with this strategy's key (see the "
            f"2026-04-23 kind-suffix fix in core.py). Diagnose via "
            f"pipeline_cache keys + id() -- do NOT add another silent "
            f"self-heal."
        )

    # Defensive null-fill for pandas categorical features handed to CatBoost.
    # The polars-native path's ``_polars_fill_null_in_categorical`` plus
    # ``prepare_df_for_catboost`` cover most cases, but the multilabel
    # codepath (MultiOutputClassifier wrapping) re-slices the frame after
    # those fills run -- by the time the per-label CB fit lands here, val
    # / test rows may carry raw NaN in cat columns and CB raises ``Invalid
    # type for cat_feature ... =NaN`` (fuzz c0062). Mirror the polars
    # __MISSING__ sentinel for the pandas surface so the bug is patched
    # at the trainer boundary regardless of which upstream path led here.
    if _is_pandas and model_type_name in CATBOOST_MODEL_TYPES and "cat_features" in fit_params and fit_params["cat_features"]:
        _cat_cols = [c for c in fit_params["cat_features"] if c in train_df.columns]
        if _cat_cols:
            for _c in _cat_cols:
                _s = train_df[_c]
                if _s.isna().any():
                    train_df = train_df.copy() if not getattr(train_df, "_mlframe_filled", False) else train_df
                    train_df[_c] = _s.astype("string").fillna("__MISSING__").astype("category")
                    train_df._mlframe_filled = True
            # Symmetric fill on eval_set so val/test slices don't trip the
            # same NaN check during early-stopping evaluation.
            _eval_set = fit_params.get("eval_set")
            if _eval_set:
                _new_eval_set = []
                for pair in _eval_set:
                    if isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], pd.DataFrame):
                        _eval_df, _eval_y = pair
                        _eval_df_filled = _eval_df
                        for _c in _cat_cols:
                            if _c in _eval_df_filled.columns and _eval_df_filled[_c].isna().any():
                                if not getattr(_eval_df_filled, "_mlframe_filled", False):
                                    _eval_df_filled = _eval_df_filled.copy()
                                _eval_df_filled[_c] = _eval_df_filled[_c].astype("string").fillna("__MISSING__").astype("category")
                                _eval_df_filled._mlframe_filled = True
                        _new_eval_set.append((_eval_df_filled, _eval_y))
                    else:
                        _new_eval_set.append(pair)
                fit_params["eval_set"] = _new_eval_set

    # Dynamic CB ``text_processing`` calibration: scale ``occurrence_lower_bound``
    # to the training-row count whenever this is a CatBoost fit with text
    # features. Default CB OLB=50 hangs on small folds (RFECV inner CV +
    # outlier-detection trim, fuzz c0056/c0070); ``compute_cb_text_processing``
    # returns None (no-op) when the row count is high enough to leave the
    # default in place. Skipped when the user has explicitly set
    # ``text_processing`` via cb_kwargs (already on the estimator's params).
    if model_type_name in CATBOOST_MODEL_TYPES:
        _has_text = bool(fit_params.get("text_features")) or (_cb_pool is not None and bool(getattr(_cb_pool, "_mlframe_text_features", None)))
        if _has_text:
            _cb_n_rows = train_df.shape[0] if hasattr(train_df, "shape") else (len(_cb_pool) if _cb_pool is not None and hasattr(_cb_pool, "__len__") else None)
            _user_text_proc = None
            if hasattr(model, "get_params"):
                try:
                    _user_text_proc = model.get_params().get("text_processing")
                except Exception:
                    _user_text_proc = None
            if _user_text_proc is None:
                _tp = compute_cb_text_processing(_cb_n_rows) if _cb_n_rows is not None else None
                if _tp is not None and hasattr(model, "set_params"):
                    try:
                        model.set_params(text_processing=_tp)
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "[%s] scaled CB text_processing.occurrence_lower_bound to %s " "(n_train=%s, default would be %s).",
                                model_type_name,
                                _tp["dictionaries"][0]["occurrence_lower_bound"],
                                _cb_n_rows,
                                CB_DEFAULT_OCCURRENCE_LOWER_BOUND,
                            )
                    except Exception as _tp_exc:
                        # Non-fatal: if CB version rejects this shape we
                        # fall back to the post-fit "Dictionary size is 0"
                        # recovery path below.
                        logger.warning(
                            "[%s] failed to set scaled CB text_processing (%s); " "falling back to post-fit recovery.",
                            model_type_name,
                            _tp_exc,
                        )

    try:
        # 2026-04-28 (batch 4): final cat alignment right before fit, by
        # which point any upstream polars->pandas conversion has run.
        # Targets the seed=2024 c0060 flake: c0009 (polars_nullable
        # multilabel) leaves the polars_nullable->pandas conversion in
        # a state where c0060's pandas frame ends up with a
        # pd.CategoricalDtype whose categories list disagrees between
        # train and val/test. Re-align here so XGB's stored cat index
        # matches at predict.
        _eval_set_for_align = fit_params.get("eval_set")
        _val_df_from_eval = None
        if _eval_set_for_align:
            _pairs = _eval_set_for_align if isinstance(_eval_set_for_align, list) else [_eval_set_for_align]
            for _p in _pairs:
                if isinstance(_p, tuple) and len(_p) == 2 and isinstance(_p[0], pd.DataFrame):
                    _val_df_from_eval = _p[0]
                    break
        if isinstance(train_df, pd.DataFrame) and _val_df_from_eval is not None:
            train_df, _aligned_val, _ = _align_xgb_cat_categories(
                model_type_name,
                train_df,
                val_df=_val_df_from_eval,
                test_df=None,
            )
            # Refresh eval_set if val_df was modified (set_categories
            # returns a new Series; the eval_set tuple needs the new
            # frame reference).
            if _aligned_val is not None and _aligned_val is not _val_df_from_eval:
                if isinstance(_eval_set_for_align, list):
                    fit_params["eval_set"] = [
                        (_aligned_val, _p[1]) if isinstance(_p, tuple) and _p[0] is _val_df_from_eval else _p for _p in _eval_set_for_align
                    ]
                elif isinstance(_eval_set_for_align, tuple):
                    fit_params["eval_set"] = (_aligned_val, _eval_set_for_align[1])

        with phase(
            "model.fit",
            model=model_type_name,
            n_rows=(train_df.shape[0] if hasattr(train_df, "shape") else None),
            n_cols=(train_df.shape[1] if hasattr(train_df, "shape") else None),
        ):
            if _cb_pool is not None:
                # Reuse path: X=Pool, y omitted (label already on the Pool).
                # fit_params still carries sample_weight, which CB's wrapper
                # ignores when X is a Pool (the Pool already has weight).
                # Filter it explicitly so downstream assertion paths don't
                # flag a "sample_weight with Pool" mismatch.
                _reuse_fit_params = {k: v for k, v in fit_params.items() if k not in ("sample_weight", "cat_features", "text_features", "embedding_features")}
                _ensure_cb_multilabel_loss(model, train_target, pool=_cb_pool)
                _ensure_xgb_classification_objective(model, train_target)
                model = _maybe_wrap_for_2d_target(model, train_target)
                model.fit(_cb_pool, **_reuse_fit_params)
            else:
                _ensure_cb_multilabel_loss(model, train_target)
                _ensure_xgb_classification_objective(model, train_target)
                _model_pre_wrap_type = type(model).__name__
                model = _maybe_wrap_for_2d_target(model, train_target)
                # 2026-04-29: when ``_maybe_wrap_for_2d_target`` introduced a
                # MultiOutputClassifier wrapper, strip ``eval_set`` from
                # fit_params - MOC doesn't slice eval_set per label, so the
                # inner estimator would see a 2-D val y and raise
                # ``y should be a 1d array``. The inner HGB / LGB / Linear
                # classifiers don't accept eval_set anyway. Surfaced 3-way
                # fuzz c0036 / c0041 / c0045 / c0056 (cb_hgb_lgb_linear*xgb /
                # multilabel + eval_set passed through).
                if type(model).__name__ == "MultiOutputClassifier" and _model_pre_wrap_type != "MultiOutputClassifier":
                    # Strip val-injected fit_params - MOC doesn't slice
                    # them per label, so the inner estimator's
                    # ``_validate_data`` chokes on the 2-D ``y_val``
                    # with ``y should be a 1d array``. Surfaced 3-way
                    # fuzz c0036 / c0041 / c0045 / c0056 (mixed-model
                    # multilabel suites where ``_setup_eval_set``
                    # injected ``X_val`` / ``y_val`` for inner
                    # gradient-boosting val-set support).
                    _strip_keys = ("eval_set", "X_val", "y_val", "validation_data")
                    fit_params = {k: v for k, v in fit_params.items() if k not in _strip_keys}
                model.fit(train_df, train_target, **fit_params)
    except Exception as e:
        try_again = False
        error_str = str(e)

        if "out of memory" in error_str:
            try_again = _handle_oom_error(model_obj, model_type_name)

        elif "User defined callbacks are not supported for GPU" in error_str:
            if "callbacks" in fit_params:
                logger.warning(e)
                try_again = True
                del fit_params["callbacks"]

        elif "CUDA Tree Learner" in error_str:
            logger.warning("CUDA is not enabled in this LightGBM build. Falling back to CPU.")
            model.set_params(device_type="cpu")
            try_again = True
        elif "pandas dtypes must be int, float or bool" in error_str:
            logger.warning(f"Model {model} skipped due to error 'pandas dtypes must be int, float or bool, got {train_df.dtypes}'")
            return None, None

        elif model_type_name in CATBOOST_MODEL_TYPES and "Dictionary size is 0" in error_str:
            # CatBoost's text feature estimator failed to build a TF-IDF
            # vocabulary -- the column's non-null samples, after the
            # occurrence_lower_bound filter, leave an empty dictionary.
            # Root cause (seen 2026-04-19 in prod): columns auto-promoted
            # to text_features that have >99.9% null rows (e.g.
            # _raw_countries, job_post_source with 6-20 non-null
            # strings out of 810_000). Proactive guard in
            # _auto_detect_feature_types now blocks these at promotion
            # time, but this is the defensive fallback: on the exact
            # CB error, drop text_features from fit_params and retry
            # without text processing. The columns stay in the frame
            # (CB will treat them as plain categorical-by-name or
            # ignore).
            text_feat = fit_params.get("text_features") or []
            if text_feat:
                logger.warning(
                    "CatBoost raised 'Dictionary size is 0' on text_features %s -- "
                    "the column(s) have too few non-null samples for CB's TF-IDF "
                    "estimator to build a vocabulary. Dropping text_features from "
                    "fit_params and retrying. Fix upstream: block promotion of "
                    "sparse columns in _auto_detect_feature_types (see the "
                    "min_non_null_for_text_promotion guard), or increase "
                    "non-null coverage of these columns in your feature "
                    "extraction.",
                    text_feat,
                )
                fit_params = {k: v for k, v in fit_params.items() if k != "text_features"}
                try_again = True
            else:
                # Raise -- same error without text_features in params is
                # an unexpected variant, not our problem.
                pass

        elif (
            model_type_name in CATBOOST_MODEL_TYPES
            and isinstance(train_df, pl.DataFrame)
            and (
                "No matching signature found" in error_str
                # Catch *both* "Categorical for a numerical feature column" and
                # "Categorical for a text feature column" (the latter surfaces
                # when a column auto-promoted from cat_features -> text_features
                # is still pl.Categorical in the df). Upstream fix casts those
                # columns to pl.String before CB.fit; this is a safety net for
                # any future variant of the same error family.
                or "Unsupported data type Categorical" in error_str
            )
        ):
            # CatBoost's native-Polars fastpath (_set_features_order_data_polars_*)
            # can reject certain categorical column layouts with opaque messages --
            # either "No matching signature found" (fused cpdef dispatch miss) or
            # the categorical/numeric type mismatch above. Fall back to the pandas
            # path: zero-copy Arrow view + `prepare_df_for_catboost` preserves
            # dtypes (post-2026-04-18) and CatBoost's pandas path accepts a wider
            # range of category backings.
            # Full last-line for the one-line message, plus a structured
            # schema dump so the NEXT occurrence is diagnosable from the
            # first log line (prev. we only had the truncated error str,
            # and for opaque dispatch misses that's useless).
            last_line = error_str.splitlines()[-1] if error_str else "<empty>"
            logger.warning(
                "CatBoost Polars fastpath rejected the data (%s); " "converting to pandas and retrying.",
                last_line[:240],
            )
            # Mark the model "Polars-broken" so subsequent predict_proba /
            # predict_log_proba calls via _predict_with_fallback go straight
            # to the pandas path -- avoids the same Cython dispatch miss on
            # every VAL/TEST/ensemble scoring (one WARN + one ~2s retry per
            # call saved). See the symmetric short-circuit in
            # _predict_with_fallback.
            try:
                model._mlframe_polars_fastpath_broken = True
            except Exception:
                pass
            schema_dump = _polars_schema_diagnostic(
                train_df,
                cat_features=fit_params.get("cat_features"),
                text_features=fit_params.get("text_features"),
            )
            logger.warning("CB Polars fastpath failure -- schema context:\n%s", schema_dump)
            from mlframe.training.utils import get_pandas_view_of_polars_df
            from mlframe.preprocessing import prepare_df_for_catboost as _prep_cb

            cat_feat = list(fit_params.get("cat_features") or [])
            text_feat = list(fit_params.get("text_features") or [])

            def _decategorize_text_cols(df):
                """CatBoost's pandas path rejects columns that are pd.Categorical
                but not in cat_features with "column 'X' has dtype 'category' but
                is not in cat_features list". Columns auto-promoted from
                cat_features -> text_features keep a pd.Categorical dtype after
                the Polars->pandas zero-copy conversion. Cast those to plain
                object to keep CB happy (and preserve the string content).
                """
                if not text_feat:
                    return df
                for col in text_feat:
                    if col in df.columns and isinstance(df[col].dtype, pd.CategoricalDtype):
                        df[col] = df[col].astype("object").fillna("")
                return df

            # Per-step timing for the fallback: a production run showed this
            # entire path consumed >1 hour on a 1M x 98 frame with 4
            # high-cardinality text columns -- without timing it was impossible
            # to tell which step (Polars->pandas vs. prep_cb vs. decategorize)
            # was responsible. The timer log writes to the trainer logger so
            # the lines interleave with surrounding INFO output.
            t0_fb = timer()
            shape_str = f"{train_df.shape[0]:_}x{train_df.shape[1]}" if hasattr(train_df, "shape") else "?"

            t0 = timer()
            train_df = get_pandas_view_of_polars_df(train_df)
            logger.info("  [fallback] polars->pandas(train) %s in %.1fs", shape_str, timer() - t0)

            # IMPORTANT: decategorize text columns BEFORE prepare_df_for_catboost.
            # Otherwise prep_cb hits the pd.Categorical text columns (auto-promoted
            # from cat to text earlier) and runs
            #   df[col].astype(str).fillna("").astype("category")
            # which on a high-cardinality column like skills_text (81k unique
            # values over 810k rows) takes many minutes per column -- the
            # production-reproduced hang that motivated this reorder.
            t0 = timer()
            train_df = _decategorize_text_cols(train_df)
            logger.info("  [fallback] decategorize text cols(train) in %.1fs", timer() - t0)

            t0 = timer()
            train_df = _prep_cb(train_df, cat_features=cat_feat, text_features=text_feat)
            logger.info("  [fallback] prepare_df_for_catboost(train) in %.1fs", timer() - t0)

            # eval_set carries the val split for CB -- rewrite it too.
            eval_set = fit_params.get("eval_set")
            if eval_set is not None:
                t0_es = timer()
                pairs = eval_set if isinstance(eval_set, list) else [eval_set]
                new_pairs = []
                for pair in pairs:
                    X_val, y_val = pair
                    if isinstance(X_val, pl.DataFrame):
                        X_val = get_pandas_view_of_polars_df(X_val)
                        # Decategorize BEFORE prep_cb (see train_df comment above).
                        X_val = _decategorize_text_cols(X_val)
                        X_val = _prep_cb(X_val, cat_features=cat_feat, text_features=text_feat)
                    else:
                        X_val = _decategorize_text_cols(X_val) if isinstance(X_val, pd.DataFrame) else X_val
                    new_pairs.append((X_val, y_val))
                fit_params["eval_set"] = new_pairs if isinstance(eval_set, list) else new_pairs[0]
                logger.info("  [fallback] eval_set rewrite in %.1fs", timer() - t0_es)

            logger.info("  [fallback] total pandas prep for CB in %.1fs", timer() - t0_fb)
            try_again = True

        elif "unexpected keyword argument" in error_str and any(param in error_str for param in ("X_val", "y_val", "eval_set")):
            # Older sklearn versions don't support validation set in HistGradientBoosting
            val_params = ["X_val", "y_val", "eval_set"]
            removed = [p for p in val_params if p in fit_params]
            if removed:
                logger.warning(
                    f"This sklearn version doesn't support validation set parameters ({', '.join(removed)}) "
                    f"for {model_type_name}. Training without early stopping validation."
                )
                for param in val_params:
                    fit_params.pop(param, None)
                try_again = True

        if try_again:
            _maybe_clean_ram()
            with phase(
                "model.fit",
                model=model_type_name,
                n_rows=(train_df.shape[0] if hasattr(train_df, "shape") else None),
                n_cols=(train_df.shape[1] if hasattr(train_df, "shape") else None),
                retry=True,
            ):
                model.fit(train_df, train_target, **fit_params)
        else:
            raise e

    _maybe_clean_ram()
    fit_elapsed = timer() - t0_fit
    if verbose:
        shape_str = f"{train_df.shape[0]:_}x{train_df.shape[1]}" if hasattr(train_df, "shape") else ""
        logger.info("  model.fit(%s) done -- %s, %.1fs", model_type_name, shape_str, fit_elapsed)

    # Apply post-hoc isotonic calibration to binary classifiers that were
    # tagged with ``_mlframe_posthoc_calibrate=True``. Fix 2026-04-15 for the
    # long-standing no-op behavior of ``prefer_calibrated_classifiers=True``
    # on tree models.
    try:
        model = _maybe_apply_posthoc_calibration(model, fit_params, model_type_name, verbose=verbose)
    except Exception as _calib_err:
        logger.warning(f"Post-hoc calibration hook raised: {_calib_err}")

    best_iter = None
    if model is not None:
        try:
            best_iter = get_model_best_iter(model_obj)
            if best_iter and verbose:
                logger.info("es_best_iter: %d", best_iter)
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Could not get best iteration: {e}")

    return model, best_iter


