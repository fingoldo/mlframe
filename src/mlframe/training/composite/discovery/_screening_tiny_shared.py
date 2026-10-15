"""Shared tiny-CV helpers for the composite discovery screen, kept in a leaf module.

Leaf module: the per-bin screen and the screen both import these, and this module imports nothing from the screen.
"""

from __future__ import annotations

import contextlib
import logging
import re
import threading
from collections import OrderedDict
import warnings
from typing import Any
import numpy as np
from ._splitter import make_discovery_splitter

logger = logging.getLogger("mlframe.training.composite.discovery._screening_tiny")


_KFOLD_SPLIT_CACHE: "OrderedDict[tuple[int, int, int], list[tuple[np.ndarray, np.ndarray]]]" = OrderedDict()


_KFOLD_SPLIT_CACHE_MAX = 8


_KFOLD_SPLIT_CACHE_LOCK = threading.Lock()  # parallel screening workers share the cache; clear-then-insert must not interleave


def _cached_kfold_splits(n_rows: int, cv_folds: int, random_state: int):
    """Return the shuffled-KFold ``(train_idx, val_idx)`` list for this
    ``(n_rows, cv_folds, random_state)``, building+caching it on first use.

    Bit-identical to ``list(KFold(n_splits=cv_folds, shuffle=True,
    random_state=random_state).split(x))`` for any ``x`` with ``n_rows`` rows --
    the split is a pure function of those three ints. Cached arrays are only
    READ downstream (fancy-indexed into copies), so reuse is safe without a copy."""
    key = (int(n_rows), int(cv_folds), int(random_state))
    with _KFOLD_SPLIT_CACHE_LOCK:
        cached = _KFOLD_SPLIT_CACHE.get(key)
        if cached is not None:
            _KFOLD_SPLIT_CACHE.move_to_end(key)
            return cached
        kf = make_discovery_splitter(cv_folds, random_state=random_state)[0]  # the one place a shuffled discovery KFold is built
        splits = list(kf.split(np.empty(n_rows, dtype=np.uint8)))
        _KFOLD_SPLIT_CACHE[key] = splits
        while len(_KFOLD_SPLIT_CACHE) > _KFOLD_SPLIT_CACHE_MAX:
            # evict-ok: the splits are a pure function of the key; a miss rebuilds the identical list
            _KFOLD_SPLIT_CACHE.popitem(last=False)
    return splits


_silence_tiny_state = threading.local()


_FEATURE_NAMES_RE = re.compile(".*feature names.*", re.I)


_SKIPPING_FEATURES_RE = re.compile(".*Skipping features without any observed values.*", re.I)


def _family_uses_lgb(family: str | None) -> bool:
    """True when ``family`` actually drives LightGBM. Only then is the
    lightgbm-logger level bump (logging.Manager._clear_cache, which walks the
    whole logger tree twice) worth paying. ``None`` keeps the legacy
    always-bump behaviour for callers that don't pass a family."""
    if family is None:
        return True
    return family.lower() in ("lgb", "lightgbm")


@contextlib.contextmanager
def _silence_tiny_model_output(family: str | None = None):
    """Context manager: silence the per-fold tiny-model fit/predict
    noise without changing the numeric path (no DataFrame->ndarray
    conversion; we keep the user's frame as-is for performance).

    ``family`` gates the lightgbm-logger level bump: for linear/ridge/cb/xgb
    families no LightGBM is involved, so the ``setLevel`` (which triggers
    ``logging.Manager._clear_cache`` over the whole logger tree, twice per
    fold) is skipped. ``None`` (default) keeps the legacy always-bump path for
    external callers that don't know the family.

    Suppressed:
    - sklearn ``UserWarning`` for "X has feature names, but X was
      fitted without feature names" / vice versa (we mix ndarray-fit
      with DataFrame-predict on the cross-target ensemble path).
    - sklearn ``ConvergenceWarning`` from Ridge / linear models on
      degenerate folds.
    - ``RuntimeWarning`` from numpy on near-singular regressors.
    - LightGBM "No further splits with positive gain" info messages
      that escape ``verbose=-1`` via the C library (silenced through
      ``logging.getLogger('lightgbm')`` level bump).

    Not touched: errors, structured logging from mlframe itself,
    catboost / xgboost (already silenced via their own kwargs).
    """
    import logging as _logging

    _bump_lgb = _family_uses_lgb(family)
    _lgb_logger = _logging.getLogger("lightgbm") if _bump_lgb else None
    # setLevel ALWAYS calls Manager._clear_cache() over the whole logger tree (even when the level is unchanged); on a large process logger tree that is ~100-150us per call, and it fires twice per CV fold (bump + restore).
    # Across a screening run the lightgbm logger sits at NOTSET and we always bump-to-ERROR / restore-to-NOTSET, so consecutive folds keep re-clearing the cache for no net level change. Use a reentrancy depth so only the OUTERMOST silence bump touches the level: nested / sequential-within-an-outer-silence entries skip both setLevel calls.
    # Numerically identical: throughout the yield the effective lightgbm level is ERROR (INFO/DEBUG suppressed) exactly as before, and the level is restored to its original value when the outermost context exits.
    _do_bump = False
    if _lgb_logger is not None:
        _depth = getattr(_silence_tiny_state, "lgb_depth", 0)
        if _depth == 0:
            _silence_tiny_state.lgb_prev_level = _lgb_logger.level
            _lgb_logger.setLevel(_logging.ERROR)
            _do_bump = True
        _silence_tiny_state.lgb_depth = _depth + 1
    try:
        from sklearn.exceptions import ConvergenceWarning
    except ImportError:  # pragma: no cover - sklearn always installed in our deps
        ConvergenceWarning = UserWarning
    with warnings.catch_warnings():
        # Prepend the four ignore filters directly using precompiled message regexes (see ``_FEATURE_NAMES_RE`` note above). ``catch_warnings.__enter__`` has just copied the global filters into a fresh ``warnings.filters`` list, so a plain prepend (newest-first) reproduces the exact order four ``filterwarnings("ignore", ...)`` calls would leave -- last-inserted (RuntimeWarning) first. The second message filter squelches sklearn SimpleImputer's "Skipping features without any observed values" no-op warning (fires per CV fold * candidate spec in tiny-rerank); the actual remediation (drop fully-NaN columns) lives in the auto-base per-column NaN gate.
        warnings.filters[:0] = [  # type: ignore[index]  # warnings.filters is a private CPython list, not the Sequence the stub declares
            ("ignore", None, RuntimeWarning, None, 0),
            ("ignore", None, ConvergenceWarning, None, 0),
            ("ignore", _SKIPPING_FEATURES_RE, UserWarning, None, 0),
            ("ignore", _FEATURE_NAMES_RE, UserWarning, None, 0),
        ]
        warnings._filters_mutated()  # type: ignore[attr-defined]  # private CPython API (no typeshed stub); invalidates the warnings filter cache after the manual prepend above
        try:
            yield
        finally:
            if _lgb_logger is not None:
                _silence_tiny_state.lgb_depth -= 1
                if _do_bump:
                    _lgb_logger.setLevel(_silence_tiny_state.lgb_prev_level)


def cap_inner_n_jobs(model: Any, outer_n_jobs: int) -> None:
    """Cap a fold model's own threading at 1 when the folds already run in parallel; a model with nothing to cap is left alone.

    Only parameters the estimator actually exposes are set -- ``n_jobs`` itself, or ``<step>__n_jobs`` inside a pipeline.
    The linear family is ``Pipeline(SimpleImputer, Ridge)``, which has no ``n_jobs`` anywhere: ``set_params(n_jobs=1)``
    on it raised, and the handler logged an oversubscription warning for a model that cannot oversubscribe -- 189 of the
    240 warnings in one production run. A model that does expose the knob and still rejects it is a real risk (every
    fold's inner model then oversubscribes against the outer dispatch, 4-8x discovery wall time), so that case warns.
    """
    if outer_n_jobs <= 1 or not hasattr(model, "get_params"):
        return
    try:
        keys = [k for k in model.get_params(deep=True) if k == "n_jobs" or k.endswith("__n_jobs")]
    except Exception as exc:  # best-effort: an estimator that cannot list its params cannot be capped either; nothing to warn about
        logger.debug("cannot list params of %s to cap n_jobs: %s", type(model).__name__, exc)
        return
    if not keys:
        return
    try:
        model.set_params(**{k: 1 for k in keys})
    except Exception as err:
        logger.warning(
            "composite_screening: failed to cap %s=1 on inner %s under outer n_jobs=%d (parallel oversubscription risk; "
            "discovery wallclock may regress 4-8x): %s: %s",
            ",".join(keys),
            type(model).__name__,
            outer_n_jobs,
            type(err).__name__,
            err,
        )


def _build_tiny_model(
    family: str, *, n_estimators: int, num_leaves: int, learning_rate: float, random_state: int, deterministic: bool = False, inner_n_jobs: int = -1
) -> Any:
    """Lazy-build a tiny regressor for the requested family. Lazy
    imports keep the discovery module light when those libraries
    aren't installed.

    No-actionable-speedup (bench: _benchmarks/bench_build_tiny_model_per_fold.py): a fresh per-fold construction is
    ~3.7us (ridge) / ~6-7us (lgb/xgb) / ~28us (cb) warm -- the lazy import is cached after first touch, so a 5-fold
    build sweep is sub-millisecond, orders below the fold FIT (ms-to-s). Pooling/reusing instances across folds saves
    nothing and risks cross-fold/spec state leakage; a fresh unfitted estimator per fold is the correct contract.

    When ``deterministic=True``, inject the well-known per-family
    determinism flags so run-to-run results are bit-exact at a
    5-10% per-fit cost. See ``deterministic_screening_models`` config
    field for the rationale.

    LightGBM determinism set:
    - ``deterministic=True``: forces deterministic histograms +
      bin-construction + tree-learner.
    - ``force_row_wise=True``: row-wise histogram aggregation is
      deterministic; the column-wise default is faster but uses
      atomic adds whose order varies.
    - ``force_col_wise=False``: explicitly OFF; otherwise it overrides
      ``force_row_wise``.

    XGBoost determinism set:
    - ``tree_method="hist"``: explicit hist; the auto-pick may flip
      to ``"approx"`` with non-deterministic atomic ops.
    - ``predictor="auto"``: keep -- predict path is already deterministic.
    - XGB doesn't expose a single ``deterministic`` switch the way
      LGB does; ``hist`` is the deterministic path.

    CatBoost determinism set:
    - ``boosting_type="Plain"``: the ``Ordered`` default is faster
      but uses random ordering which differs run-to-run; ``Plain``
      is deterministic.

    Linear (Ridge) is already deterministic by construction.
    """
    family_lower = family.lower()
    if family_lower in ("lgb", "lightgbm"):
        import lightgbm as lgb

        kwargs: dict[str, Any] = dict(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=inner_n_jobs,
            verbose=-1,
            force_col_wise=True,
        )
        if deterministic:
            # ``force_col_wise`` + ``force_row_wise`` are mutually
            # exclusive in LightGBM; flip the pair when going
            # deterministic.
            kwargs["force_col_wise"] = False
            kwargs["force_row_wise"] = True
            kwargs["deterministic"] = True
        return lgb.LGBMRegressor(**kwargs)
    if family_lower in ("xgb", "xgboost"):
        import xgboost as xgb

        xgb_kwargs: dict[str, Any] = dict(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=inner_n_jobs,
            verbosity=0,
        )
        if deterministic:
            xgb_kwargs["tree_method"] = "hist"
        return xgb.XGBRegressor(**xgb_kwargs)
    if family_lower in ("cb", "catboost"):
        import catboost as cb

        cb_kwargs: dict[str, Any] = dict(
            iterations=n_estimators,
            depth=4,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=False,
            allow_writing_files=False,  # parallel fold/spec fits race on catboost_info/ (Windows file-lock -> silent NaN folds) and litter CWD
        )
        if deterministic:
            cb_kwargs["boosting_type"] = "Plain"
        return cb.CatBoostRegressor(**cb_kwargs)
    if family_lower in ("linear", "ridge"):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline

        # Wrap Ridge in a SimpleImputer pipeline because Ridge raises on
        # NaN inputs (observed in prod: tens of thousands of warnings
        # spammed per discovery run when ``tiny_screening_families``
        # included "linear" alongside "lightgbm" -- LGBM handles NaN
        # natively, Ridge doesn't). Mean-imputing the screening matrix
        # for the proxy is sound: this isn't the production model, just
        # a screening proxy for "would a downstream linear model find
        # this composite useful". Ridge is deterministic by construction;
        # the random_state kwarg accepted for call-signature uniformity.
        return Pipeline(
            [
                ("imp", SimpleImputer(strategy="mean")),
                ("ridge", Ridge(alpha=1.0, random_state=random_state)),
            ]
        )
    raise ValueError(f"_build_tiny_model: unknown family '{family}'. " "Supported: lightgbm, xgboost, catboost, linear / ridge.")
