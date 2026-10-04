"""Per-method feature-importance computations behind ``get_feature_importances`` (one function per ``importance_getter`` name).

``get_feature_importances`` resolves the requested name here and falls back to an attribute read (``'auto'``, ``'coef_'``, a dotted path). Each function
returns the raw importance vector; the caller validates its length and NaN content. Names that live in ``_helpers_importance`` (the scorer factory, the
conditional-permutation routine, the finite-fold check) are imported at call time, so a patch of them on that module is seen here and the two modules
carry no import cycle.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["named_importance"]


def _permutation(model: object, data: Any, target: Any, train_data: Any, *, n_repeats: int, random_state: int, **_: Any) -> Any:
    """sklearn ``permutation_importance`` mean over ``n_repeats`` shuffles, scored through the fast default scorer."""
    from ._helpers_importance import _fold_is_all_finite, _make_fast_default_scorer

    if target is None:
        raise ValueError("importance_getter='permutation' requires target (y_test) " "to score against. Pass target= explicitly.")
    from sklearn.inspection import permutation_importance
    # sklearn's permutation_importance shuffles a column in place (``X_permuted[:, col] = ...``); a read-only
    # ndarray backing (polars/Arrow zero-copy view, pandas copy-on-write, or a loky memmap) makes that raise
    # "assignment destination is read-only". Hand it a writeable copy of the (test-fold-sized, not full-frame)
    # data so the shuffle always has somewhere to write.
    # sklearn's permutation_importance reuses one ``X_permuted`` buffer and shuffles a column in place across
    # ``n_repeats``. CatBoost.predict() flips its input ndarray's writeable flag to False, so once the scorer
    # runs on ``X_permuted`` the NEXT iteration's in-place shuffle raises "assignment destination is read-only".
    # Score through a wrapper that hands the estimator a private copy, leaving sklearn's ``X_permuted``
    # writeable. The wrapper preserves the exact default metric (R2 / accuracy) used pre-fix.
    #
    # Perf P1: the per-call ``estimator.score()`` re-runs sklearn's full target-type
    # validation (``_check_targets`` -> ``type_of_target``) on the IDENTICAL ``_y`` every call. On the
    # scene 2407x299 bench that validation is the dominant hotspot: cProfile cumtime ``_check_targets``
    # ~43s + ``type_of_target`` ~23s vs the irreducible ``predict`` matmul ~27s (the permutation FI loop
    # issues p*n_repeats scorer calls/fold). ``estimator.score()`` for a classifier is
    # ``accuracy_score(y, predict(X))`` and for a regressor ``r2_score(y, predict(X))``; both have a
    # bit-identical closed form (``np.mean(pred==y)`` / ``1 - ss_res/ss_tot``) that skips the redundant
    # re-validation. We latch the fast path per fold ONLY after a baseline self-check proves it returns
    # the EXACT same float as ``estimator.score()`` (gate-the-win-on-its-safe-condition); any mismatch,
    # exception, multioutput target, or non-clf/reg estimator falls back to ``estimator.score()`` for
    # that fold, so the selected set stays bit-identical. The defensive ``copy`` is unchanged (keeps the
    # CatBoost / read-only-buffer safety exactly).
    scorer = _make_fast_default_scorer(model)
    # Perf P3: the estimator's ``predict`` re-validates the permuted X on every one of
    # the p*n_repeats scorer calls - ``check_array`` -> ``_assert_all_finite`` rescans the whole
    # (test-fold-sized) array for NaN/inf each time even though permutation only RESHUFFLES already-
    # validated finite values within a column. sklearn's own ``assume_finite=True`` config skips that
    # rescan WITHOUT touching any numeric result. We enable it ONLY after verifying the fold's
    # (data, target) are actually all-finite ONCE up front: if they are, skipping the per-call
    # re-checks is bit-identical (the check would have passed every time); if they are NOT (NaN in X,
    # which LightGBM/tree models accept), we keep the default context so sklearn validates / raises
    # exactly as before. Bit-identity verified (np.array_equal, max|diff| 0.0).
    _assume_finite = _fold_is_all_finite(data) and _fold_is_all_finite(target)
    if _assume_finite:
        import sklearn as _sklearn
        with _sklearn.config_context(assume_finite=True):
            pi = permutation_importance(
                model, data, target,
                scoring=scorer,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=1,
            )
    else:
        pi = permutation_importance(
            model, data, target,
            scoring=scorer,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=1,
        )
    # Cross-repeat aggregator stays the arithmetic mean: median / 20%-trimmed-mean were benched
    # (bench_perm_fi_repeat_aggregator.py) and REJECTED - neither beats the mean on spearman-vs-true-
    # relevance at the realistic small n_repeats (3, 5); with so few repeats a robust aggregator just
    # discards the averaging that suppresses per-permutation noise. Revisit only at n_repeats >= 15.
    return pi.importances_mean


def _conditional_permutation(
    model: object, data: Any, target: Any, train_data: Any, *, n_repeats: int, cpi_max_depth: Union[int, None], cpi_min_samples_leaf: int, random_state: int, **_: Any
) -> Any:
    """Strobl 2008 conditional permutation importance (permute X_j within leaves of a shallow tree on X_{-j})."""
    from ._helpers_importance import _conditional_permutation_importance

    if target is None:
        raise ValueError("importance_getter='conditional_permutation' requires target (y_test) " "to score against. Pass target= explicitly.")
    # F10: cpi_max_depth=None lets the auxiliary tree grow
    # until min_samples_leaf constraint kicks in (Strobl 2008 recommendation).
    # random_state forwarded (default 0 preserves legacy behaviour); lets
    # the caller thread a fold-derived seed so CPI shuffles vary per fold
    # and the run stays reproducible, matching the 'permutation' path.
    return _conditional_permutation_importance(
        model, data, target,
        n_repeats=n_repeats,
        max_depth=cpi_max_depth,
        min_samples_leaf=cpi_min_samples_leaf,
        random_state=random_state,
    )


def _drop_column(model: object, data: Any, target: Any, train_data: Any, **_: Any) -> Any:
    """Refit ``model`` with each column individually dropped and measure the score drop against the full-X baseline."""
    # Drop-column importance: refit ``model`` on data with each column
    # individually dropped, measure score drop vs full-X baseline.
    # O(p * full_fit_time) - infeasible on p>=1000. Useful as a
    # ground-truth oracle when benchmarking other importance methods.
    if data is None or target is None:
        raise ValueError("importance_getter='drop_column' requires data (X) and target (y) at the call site.")
    from sklearn.base import clone as _clone
    _Xnp = data.to_numpy(copy=False) if hasattr(data, "to_numpy") else np.asarray(data)
    _baseline = float(model.score(data, target))  # type: ignore[attr-defined]  # model: object at signature (accepts any estimator-like with .score)
    _scores = np.zeros(_Xnp.shape[1], dtype=float)
    for _j in range(_Xnp.shape[1]):
        _X_drop = np.delete(_Xnp, _j, axis=1)
        if hasattr(data, "columns"):
            _X_drop = pd.DataFrame(_X_drop, columns=[c for i, c in enumerate(data.columns) if i != _j])
        _m = _clone(model)
        try:
            _m.fit(_X_drop, target)
            _scores[_j] = _baseline - float(_m.score(_X_drop, target))
        except Exception as e:
            logger.debug("drop-column importance fit/score failed for column %d, recording NaN: %s", _j, e)
            _scores[_j] = np.nan  # a 0.0 would outrank every column whose removal IMPROVED the score
    return _scores


def _boruta(model: object, data: Any, target: Any, train_data: Any, **_: Any) -> Any:
    """Classical Boruta shadow-feature importance (Pure-Gini variant), refitting ``model`` on ``[X, X_shadow]``."""
    # Classical Boruta (Kursa & Rudnicki 2010, JSS-36): pair each real
    # feature with a SHADOW (shuffled copy) and judge importance vs
    # the max-shadow importance. No external dep; uses the supplied
    # ``model``'s feature_importances_ / coef_ after refitting on
    # [X, X_shadow]. Pure-Gini variant - biased on high-cardinality
    # categoricals. Use 'boruta_shap' for the SHAP-based unbiased
    # version when shap is available.
    if data is None or target is None:
        raise ValueError("importance_getter='boruta' requires data (X) and target (y) at the call site.")
    from sklearn.base import clone as _clone
    rng = np.random.default_rng(0)
    _Xnp = data.to_numpy(copy=False) if hasattr(data, "to_numpy") else np.asarray(data)
    _p = _Xnp.shape[1]
    # Shadow = column-wise shuffled copy.
    _Xshadow = _Xnp.copy()
    for _j in range(_p):
        rng.shuffle(_Xshadow[:, _j])
    _Xjoint = np.hstack([_Xnp, _Xshadow])
    _model_clone = _clone(model)
    try:
        _model_clone.fit(_Xjoint, target)
    except Exception as _exc:
        raise RuntimeError(f"Boruta refit on [X, shadow] failed for {type(model).__name__}: {_exc}") from _exc
    # Read importances of joint model.
    if hasattr(_model_clone, "feature_importances_"):
        _imps = np.asarray(_model_clone.feature_importances_)
    elif hasattr(_model_clone, "coef_"):
        _imps = np.abs(np.asarray(_model_clone.coef_))
        if _imps.ndim > 1:
            _imps = _imps.max(axis=0)
    else:
        raise AttributeError(
            f"'boruta' importance_getter requires feature_importances_ or coef_ on the refit model; " f"{type(_model_clone).__name__} has neither."
        )
    _real = _imps[:_p]
    _shadow_max = float(_imps[_p:].max()) if len(_imps) > _p else 0.0
    # Per-feature score = real importance MINUS shadow-max threshold.
    # Positive = beats shadow; negative = noise. Caller's downstream
    # consumer treats it like any FI vector.
    return _real - _shadow_max


def _boruta_shap(model: object, data: Any, target: Any, train_data: Any, *, current_features: list, **_: Any) -> Any:
    """L1: Boruta-SHAP via the optional ``BorutaShap`` package: 1.0 accepted, 0.5 tentative, 0.0 rejected per current feature."""
    # L1: Boruta-SHAP via the optional
    # ``BorutaShap`` package. Returns per-feature shadow-relative
    # importance: positive => beats max-shadow at the configured
    # p-value level; zero => indistinguishable from shadow.
    if target is None:
        raise ValueError("importance_getter='boruta_shap' requires target (y_test).")
    # data (X) is required: BorutaShap.fit needs the feature matrix.
    # Pre-fix this path fell back to ``X=target`` when data was None,
    # feeding y in as the feature matrix and silently producing a
    # nonsensical fit instead of a clear error.
    if data is None:
        raise ValueError("importance_getter='boruta_shap' requires data (X) at the call site.")
    try:
        from BorutaShap import BorutaShap as _BorutaShap
    except ImportError as _exc2:
        # No arfs/GrootCV fallback: GrootCV's constructor + fit signature
        # (GrootCV(objective=, cutoff=).fit(X, y) -> .selected_features_)
        # is incompatible with the BorutaShap call shape below, so
        # aliasing it would crash rather than degrade gracefully.
        raise ImportError("importance_getter='boruta_shap' requires the optional ``BorutaShap`` package. " "Install via ``pip install BorutaShap``.") from _exc2
    try:
        _bs = _BorutaShap(model=model, importance_measure="shap", classification=hasattr(model, "classes_"))
        _bs.fit(X=data, y=target, n_trials=15, random_state=0, verbose=False)
        # Output: BorutaShap stores accepted/rejected lists; build dense importance with shadow-relative scores.
        _accepted = set(getattr(_bs, "accepted", []) or [])
        _tentative = set(getattr(_bs, "tentative", []) or [])
        return np.array([1.0 if c in _accepted else (0.5 if c in _tentative else 0.0) for c in current_features], dtype=float)
    except Exception as _exc:
        raise RuntimeError(f"BorutaShap failed: {_exc}") from _exc


def _powershap(model: object, data: Any, target: Any, train_data: Any, *, current_features: list, **_: Any) -> Any:
    """L2: PowerSHAP via the optional ``powershap`` package: 1.0 for selected features, 0.0 otherwise."""
    if target is None:
        raise ValueError("importance_getter='powershap' requires target (y_test).")
    try:
        from powershap import PowerShap as _PowerShap
    except ImportError as _exc:
        raise ImportError("importance_getter='powershap' requires the optional ``powershap`` package. " "Install via ``pip install powershap``.") from _exc
    try:
        _ps = _PowerShap(model=model)
        _ps.fit(data, target)
        # _ps stores _processed_shaps_df with p-values per feature; treat selected -> 1, else 0.
        _sel = set(_ps.selected_features_) if hasattr(_ps, "selected_features_") else set()
        return np.array([1.0 if c in _sel else 0.0 for c in current_features], dtype=float)
    except Exception as _exc:
        raise RuntimeError(f"PowerSHAP failed: {_exc}") from _exc


def _shap(model: object, data: Any, target: Any, train_data: Any, *, importance_getter: str, **_: Any) -> Any:
    """Mean ``|SHAP|`` per feature; ``'shap_oof'`` is an explicit alias for ``'shap'``."""
    # L4: 'shap_oof' is an explicit alias for
    # 'shap'. The standard RFECV fold path fits the model on
    # X_train then calls this with data=X_test (held-out fold),
    # so the resulting mean(|SHAP|) is already an OOF importance.
    # The alias name makes that semantic explicit for callers who
    # want SHAP-OOF elimination without having to read the source.
    try:
        import shap as _shap
    except ImportError as _exc:
        raise ImportError(f"importance_getter={importance_getter!r} requires the optional " f"``shap`` package. Install via ``pip install shap``.") from _exc
    try:
        explainer = _shap.Explainer(model, data)
        shap_values = explainer(data, check_additivity=False)
        vals = shap_values.values
        if vals.ndim > 2:
            vals = np.abs(vals).mean(axis=tuple(range(2, vals.ndim)))
        return np.abs(vals).mean(axis=0)
    except Exception as _exc:
        raise RuntimeError(f"shap.Explainer failed for {type(model).__name__}: {_exc}. " f"Try importance_getter='permutation' instead.") from _exc


def _unwrap_estimator(m: Any) -> Any:
    """Walk up to 8 wrapper hops to the innermost fitted estimator (Pipeline / TransformedTargetRegressor /
    a search-CV's ``best_estimator_``), so ``'auto'`` can find ``feature_importances_``/``coef_`` through
    common wrapper chains rather than only on the outermost object."""
    # Walk to the innermost fitted estimator: Pipeline -> _final_estimator;
    # TransformedTargetRegressor -> regressor_; ColumnTransformer-style
    # wrappers fall back to themselves.
    for _ in range(8):
        if hasattr(m, "_final_estimator") and m._final_estimator is not m:
            m = m._final_estimator
            continue
        if hasattr(m, "regressor_") and getattr(m, "regressor_") is not m:
            m = m.regressor_
            continue
        if hasattr(m, "best_estimator_") and getattr(m, "best_estimator_") is not m:
            m = m.best_estimator_
            continue
        break
    return m


def _resolve_getter(obj: Any, dotted: str) -> Any:
    """Resolve a dotted attribute path (e.g. ``'regressor_.coef_'``, ``'named_steps.lr.coef_'``) on ``obj``."""
    # operator.attrgetter handles the dotted-path traversal.
    from operator import attrgetter
    return attrgetter(dotted)(obj)


def _read_attribute(model: object, importance_getter: str) -> tuple[Any, str]:
    """The raw attribute value and the attribute NAME it came from (the last dotted segment), for ``'auto'``, a plain name or a dotted path."""
    # 2026-05-28 sklearn-parity: ``importance_getter`` may be a dotted
    # path such as ``regressor_.coef_`` (for TransformedTargetRegressor)
    # or ``named_steps.lr.coef_`` (for Pipeline). Resolve via
    # operator.attrgetter so the legacy single-attr behaviour AND the
    # dotted path both work. ``auto`` also unwraps Pipelines and other
    # wrappers to the underlying ``_final_estimator`` / ``regressor_``
    # before searching for feature_importances_ / coef_.
    if importance_getter == "auto":
        inner = _unwrap_estimator(model)
        if hasattr(inner, "feature_importances_"):
            return inner.feature_importances_, "feature_importances_"
        if hasattr(inner, "coef_"):
            return inner.coef_, "coef_"
        raise AttributeError(
            f"importance_getter='auto' could not find feature_importances_ or coef_ "
            f"on a fitted {type(model).__name__} (unwrapped to {type(inner).__name__})."
        )
    # Dotted path (sklearn convention).
    if "." in importance_getter:
        try:
            res = _resolve_getter(model, importance_getter)
        except (AttributeError, KeyError) as _attr_exc:
            raise AttributeError(
                f"importance_getter={importance_getter!r}: could not resolve dotted "
                f"path on {type(model).__name__}. Verify each step exists on the "
                f"fitted estimator. Underlying error: {_attr_exc}"
            ) from _attr_exc
        # Normalise getter_attr to the LAST segment for downstream coef_-scaling logic.
        return res, importance_getter.rsplit(".", 1)[-1]
    return getattr(model, importance_getter), importance_getter


def _scaled_coef(res: Any, data: Any, train_data: Any, multiclass_coef_aggregation: str, coef_scale_source: str) -> Any:
    """``|coef_|`` collapsed over classes and rescaled by the feature stds (train stds by default)."""
    # F5: multi-class collapse. 'max' (default)
    # uses max(|coef_class|, axis=0) -> a feature important for ANY
    # class is important. Pre-fix sum(|coef|) over OvR rows mixed
    # class-specific signals: a single-class discriminator looked
    # like a mid-relevance feature for every class. 'sum' is opt-in.
    res = np.abs(res)
    if res.ndim > 1:
        if multiclass_coef_aggregation == "max":
            res = res.max(axis=0)
        else:
            res = res.sum(axis=0)
    # F4: scale correction with TRAIN stds.
    # Pre-fix used X_test stds -> leaks test variance into FI on small
    # folds. Use train_data when provided; fall back to data only if
    # train_data is absent (callable importance_getter path) and
    # coef_scale_source != 'none'.
    _scale_src = coef_scale_source
    if _scale_src == "none":
        return res
    _src_data = train_data if (train_data is not None and _scale_src == "train") else data
    if _src_data is not None:
        try:
            if hasattr(_src_data, "values"):
                _Xarr = _src_data.values
            else:
                _Xarr = np.asarray(_src_data)
            _stds = np.nanstd(_Xarr, axis=0)
            # Avoid blow-up on near-constant cols (stds ~ 0).
            _stds = np.where(_stds > 1e-12, _stds, 1.0)
            if len(_stds) == len(res):
                res = res * _stds
        except (TypeError, ValueError):
            # Non-numeric data (object cols, mixed pl frames): skip scaling.
            pass
    return res


def _attribute(model: object, data: Any, target: Any, train_data: Any, *, importance_getter: str, multiclass_coef_aggregation: str, coef_scale_source: str, **_: Any) -> Any:
    """Read ``feature_importances_`` / ``coef_`` (or any named / dotted attribute) off the fitted model."""
    res, getter_attr = _read_attribute(model, importance_getter)
    if getter_attr == "coef_":
        return _scaled_coef(res, data, train_data, multiclass_coef_aggregation, coef_scale_source)
    if res.ndim > 1:
        # Tree-based feature_importances_ stays 1-D normally; this branch
        # handles unusual estimators (e.g. multi-output) and uses sum
        # because we can't distinguish "OvR class" from "output" here.
        res = res.sum(axis=0)
    return res


_NAMED_METHODS: dict[str, Callable[..., Any]] = {
    "permutation": _permutation,
    "conditional_permutation": _conditional_permutation,
    "drop_column": _drop_column,
    "boruta": _boruta,
    "boruta_shap": _boruta_shap,
    "powershap": _powershap,
    "shap": _shap,
    "shap_oof": _shap,
}


def named_importance(
    model: object,
    current_features: list,
    importance_getter: str,
    data: Any,
    target: Any,
    train_data: Any,
    *,
    multiclass_coef_aggregation: str,
    coef_scale_source: str,
    cpi_max_depth: Union[int, None],
    cpi_min_samples_leaf: int,
    n_repeats: int,
    random_state: int,
) -> Any:
    """Importance vector for a string ``importance_getter``: a named method from the table, else an attribute read (``'auto'`` included)."""
    method = _NAMED_METHODS.get(importance_getter, _attribute)
    return method(
        model, data, target, train_data,
        current_features=current_features,
        importance_getter=importance_getter,
        multiclass_coef_aggregation=multiclass_coef_aggregation,
        coef_scale_source=coef_scale_source,
        cpi_max_depth=cpi_max_depth,
        cpi_min_samples_leaf=cpi_min_samples_leaf,
        n_repeats=n_repeats,
        random_state=random_state,
    )
