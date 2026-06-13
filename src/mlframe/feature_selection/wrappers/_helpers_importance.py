"""Feature-importance computation + vote-based ranking helpers for the RFECV wrapper.

Carved from ``_helpers.py``; the parent re-exports these names so legacy
``from ._helpers import get_feature_importances`` call sites keep working.
"""
from __future__ import annotations

import logging
from typing import Callable, Union

import numpy as np
import pandas as pd

from mlframe.votenrank import Leaderboard

from ._enums import VotesAggregation


logger = logging.getLogger(__name__)

# Cell budget (n_rows * n_cols of the per-fold held-out set) below which the unspecified ('auto')
# importance default routes to PERMUTATION (the accuracy winner on the FS bench); above it 'auto' falls
# back to impurity for speed. ~40k x 100; tune via the dispatcher rather than hardcoding per call site.
_PERM_AUTO_CELL_CAP = 4_000_000


def _fold_is_all_finite(arr) -> bool:
    """True iff every element of ``arr`` is finite (no NaN / inf). Used by P3 to decide whether the
    permutation-FI call may run under ``assume_finite=True``. One O(n*p) scan replaces the per-scorer-call
    rescans sklearn would otherwise do. Returns False (conservative -> keep sklearn validation) for any
    array it can't cheaply check as a numeric ndarray (object/categorical/non-array)."""
    if arr is None:
        return False
    try:
        a = arr.to_numpy(copy=False) if hasattr(arr, "to_numpy") else np.asarray(arr)
    except Exception:
        return False
    if a.dtype.kind in ("f", "c"):
        return bool(np.isfinite(a).all())
    if a.dtype.kind in ("i", "u", "b"):
        return True  # integer / bool arrays are always finite
    return False  # object / string / datetime: don't assume; let sklearn validate


def _make_fast_default_scorer(model: object) -> Callable:
    """Build a permutation-FI scorer that reproduces ``estimator.score()`` bit-identically but
    skips its redundant per-call target-type validation.

    perf P1 (2026-06-08): ``estimator.score()`` is the permutation-importance hotspot because it
    re-runs sklearn's ``_check_targets`` / ``type_of_target`` validation on the SAME ``_y`` for every
    one of the ``p * n_repeats`` scorer calls per fold. For the standard single-output case the default
    score has a closed form that is bit-identical to ``estimator.score()``:
      - classifier -> ``accuracy_score(y, pred)`` == ``np.mean(pred == y)``  (verified ==, diff 0.0)
      - regressor  -> ``r2_score(y, pred)``       == ``1 - ss_res / ss_tot`` (verified ==, diff 0.0)

    Safety (gate-the-win-on-its-safe-condition): the returned scorer latches the fast path ONLY after a
    one-shot baseline self-check proves the fast value equals ``estimator.score()`` EXACTLY on the
    unpermuted X. The first scorer call permutation_importance makes is the baseline (unpermuted), so the
    self-check sees clean data. Any of the following pins the fold to the original ``estimator.score()``
    path (so the elimination ranking, and thus the selected set, is unchanged):
      - estimator is neither a classifier nor a regressor (``is_classifier``/``is_regressor`` both False),
      - the target is multi-output (y.ndim > 1 with >1 column),
      - the closed-form value does not bit-match ``estimator.score()`` on the baseline,
      - any exception in the fast path.

    perf P2 (2026-06-08): the per-call defensive ``np.array(_X, copy=True)`` is only required for
    estimators whose ``predict`` flips the input ndarray's writeable flag to False (CatBoost), which would
    make sklearn's NEXT in-place column shuffle of its reused ``X_permuted`` buffer raise
    "assignment destination is read-only". For every estimator that does NOT flip the flag (sklearn
    linear / tree / ensemble, LightGBM, XGBoost, ...) the copy is pure waste -- it was ~36k array copies on
    the scene bench. We detect the flip on the baseline call: predict on a copy, then inspect that copy's
    writeable flag. If still writeable the estimator is flag-safe and we latch ``need_copy=False`` (read
    sklearn's ``X_permuted`` directly); if flipped (or the buffer was already read-only) we keep the copy.
    Either way the values fed to ``predict`` are identical, so the score -- and the selected set -- is
    bit-identical.
    """
    from sklearn.base import is_classifier, is_regressor

    _is_clf = bool(is_classifier(model))
    _is_reg = bool(is_regressor(model))

    # mode:      -1 = baseline pending; 0 = safe (estimator.score); 1 = fast closed-form.
    # need_copy: True = defensive copy each call (writeable-flip / read-only buffer); False = read _X directly.
    # _ycache:   y-derived invariants keyed by id(_y) -- see _fast_value.
    state = {"mode": -1, "need_copy": True, "_ycache": None, "_yid": None}

    def _y_invariants(_y):
        """Return (y_arr, regressor (yt, ss_tot)) for ``_y``, computing once and reusing while ``_y`` is unchanged.

        ``permutation_importance`` permutes only X across its ``p*n_repeats`` scorer calls and feeds the IDENTICAL
        ``_y`` every time. The classifier path then re-ran ``np.asarray(_y)`` and the regressor path re-ran
        ``_y.astype(float64)`` + ``np.mean(_y)`` + ``ss_tot=sum((y-mean)**2)`` -- all functions of ``_y`` alone --
        on every call. Hoisting them behind an ``id(_y)`` cache is bit-identical (same float, same NaN handling)
        and removes ~3x of the regressor-path per-call cost (18.3us -> 6.0us in isolation). The caller holds ``_y``
        alive for the whole permutation loop, so ``id(_y)`` cannot alias a freed object mid-loop.
        """
        if state["_yid"] == id(_y) and state["_ycache"] is not None:
            return state["_ycache"]
        _y_arr = np.asarray(_y)
        reg = None
        if not _is_clf and _y_arr.ndim == 1:
            yt = _y_arr.astype(np.float64, copy=False)
            ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
            reg = (yt, ss_tot)
        cached = (_y_arr, reg)
        state["_yid"] = id(_y)
        state["_ycache"] = cached
        return cached

    def _fast_value(_est, _Xc, _y):
        """Closed-form default score; returns None if the case isn't supported (-> fall back)."""
        _y_arr, _reg = _y_invariants(_y)
        if _y_arr.ndim > 1 and _y_arr.shape[-1] > 1:
            return None  # multi-output: defer to estimator.score
        pred = _est.predict(_Xc)
        pred = np.asarray(pred)
        if _is_clf:
            # accuracy: fraction of exact matches. Bit-identical to sklearn accuracy_score for 1d targets.
            if pred.shape != _y_arr.shape:
                return None
            return float(np.mean(pred == _y_arr))
        # regressor: r2 with default multioutput='uniform_average'; single-output closed form.
        if _reg is None:
            return None  # multi-output regressor (y_arr.ndim>1): defer to estimator.score.
        yt, ss_tot = _reg
        pred = pred.astype(np.float64, copy=False)
        if pred.shape != yt.shape:
            return None
        if ss_tot == 0.0:
            return None  # constant target: sklearn r2 has special-case semantics; defer.
        ss_res = float(np.sum((yt - pred) ** 2))
        return 1.0 - ss_res / ss_tot

    def _scorer(_est, _X, _y):
        mode = state["mode"]
        if mode != -1:
            # Latched. Copy only if the estimator was found to flip the writeable flag (or non-ndarray X).
            if state["need_copy"] and isinstance(_X, np.ndarray):
                _Xc = np.array(_X, copy=True)
            else:
                _Xc = _X
            return _fast_value(_est, _Xc, _y) if mode == 1 else _est.score(_Xc, _y)

        # Undecided: this is the baseline (unpermuted) call. Always copy HERE so the probe can't corrupt
        # sklearn's reused ``X_permuted`` buffer; then decide whether subsequent calls may skip the copy.
        _Xc = np.array(_X, copy=True) if isinstance(_X, np.ndarray) else _X
        safe_val = _est.score(_Xc, _y)

        # need_copy decision: if predicting through ``_est.score`` left our private copy still writeable,
        # the estimator does not flip the flag -> subsequent calls can read sklearn's buffer directly.
        # A non-ndarray X (pandas/polars) keeps the historical pass-through (no copy was made anyway).
        if isinstance(_X, np.ndarray):
            state["need_copy"] = not bool(getattr(_Xc.flags, "writeable", True))
        else:
            state["need_copy"] = False

        if not (_is_clf or _is_reg):
            state["mode"] = 0
            return safe_val
        try:
            fast_val = _fast_value(_est, _Xc, _y)
        except Exception:
            fast_val = None
        # Bit-identity gate: latch fast ONLY on an exact match (handles NaN equally on both sides).
        if fast_val is not None and (fast_val == safe_val or (fast_val != fast_val and safe_val != safe_val)):
            state["mode"] = 1
        else:
            state["mode"] = 0
        return safe_val

    return _scorer


def _conditional_permutation_importance(
    model,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_repeats: int = 5,
    max_depth: Union[int, None] = None,
    min_samples_leaf: int = 10,
    random_state: int = 0,
) -> np.ndarray:
    """Strobl, Boulesteix, Zeileis, Hothorn 2008 conditional permutation importance.

    Vanilla permutation (Breiman 2001) shuffles X_j independently of X_{-j},
    creating out-of-distribution combinations on correlated feature sets and
    inflating measured importance. This conditional variant fits a shallow
    decision tree X_{-j} -> X_j, then permutes X_j WITHIN each leaf, which
    preserves P(X_j | X_{-j}) and removes the correlation-induced bias.

    F10 (Wave 3, 2026-05-28): max_depth=None grows the tree until
    min_samples_leaf binds. The pre-fix max_depth=5 cap under-conditioned on
    >5 correlated features and silently degenerated to vanilla permutation
    (Strobl 2008 recommends >=5 samples per leaf, no depth cap).

    Cost: ~2-3x vanilla permutation (per-feature tree fit + n_repeats
    leaf-grouped shuffles).

    Returns
    -------
    importances : ndarray of shape (p,)
        Per-feature mean score loss (baseline - permuted). Higher = more important.
    """
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

    is_dataframe = isinstance(X, pd.DataFrame)
    if is_dataframe:
        X_arr = X.to_numpy()
        cols = X.columns
        idx = X.index
    else:
        X_arr = np.asarray(X)
        cols = None
        idx = None

    if X_arr.ndim != 2:
        raise ValueError(f"conditional_permutation expects 2D X, got shape {X_arr.shape}")

    n, p = X_arr.shape
    rng = np.random.default_rng(random_state)
    baseline = float(model.score(X, y))
    importances = np.zeros(p, dtype=float)

    def _is_discrete(col: np.ndarray) -> bool:
        # Integer dtype OR <=10 unique non-null values: pragmatic proxy.
        if np.issubdtype(col.dtype, np.integer):
            return True
        try:
            mask = ~np.isnan(col.astype(float, copy=False))
            uniq = np.unique(col[mask])
        except (TypeError, ValueError):
            uniq = np.unique(col)
        return uniq.size <= 10

    def _is_discrete_v2(col: np.ndarray) -> bool:
        # F11 (Wave 3, 2026-05-28): tighter discrete detection.
        # Integer dtype is canonical discrete. For floats, require BOTH (a) low
        # unique count AND (b) cardinality << n_rows. Decile-binned continuous
        # variables (10 unique values across 100k rows) now correctly route to
        # regression instead of classification.
        if np.issubdtype(col.dtype, np.integer):
            return True
        try:
            mask = ~np.isnan(col.astype(float, copy=False))
            uniq = np.unique(col[mask])
        except (TypeError, ValueError):
            uniq = np.unique(col)
        _n = max(int(mask.sum()) if hasattr(mask, "sum") else len(col), 1)
        return uniq.size <= max(5, int(np.sqrt(_n))) and uniq.size <= 0.5 * _n

    for j in range(p):
        Xj = X_arr[:, j]
        Xnotj = np.delete(X_arr, j, axis=1)

        if Xnotj.shape[1] == 0:
            # Single-feature case: no conditioning set; fall back to vanilla shuffle.
            score_losses = []
            for _ in range(n_repeats):
                X_perm = X_arr.copy()
                X_perm[:, j] = rng.permutation(X_arr[:, j])
                X_for_score = (
                    pd.DataFrame(X_perm, columns=cols, index=idx) if is_dataframe else X_perm
                )
                score_losses.append(baseline - float(model.score(X_for_score, y)))
            importances[j] = float(np.mean(score_losses))
            continue

        # F10/F11: pass max_depth + min_samples_leaf. max_depth=None grows the tree
        # until min_samples_leaf binds (recommended by Strobl 2008 on >=5).
        # F11 (Wave 3, 2026-05-28): _is_discrete heuristic improved to require
        # integer-dtype OR n_unique<=max(5, sqrt(n)) so decile-binned continuous
        # variables don't trigger Classifier mis-detection.
        if _is_discrete_v2(Xj):
            tree = DecisionTreeClassifier(
                max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )
        else:
            tree = DecisionTreeRegressor(
                max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )

        try:
            tree.fit(Xnotj, Xj)
            leaves = tree.apply(Xnotj)
        except (ValueError, TypeError, MemoryError, RuntimeError):
            # E11 (Wave 4, 2026-05-28): widen the except to catch MemoryError
            # on 1M-row Xnotj and RuntimeError from custom-estimator paths
            # raising AttributeError-like wrapped exceptions. Conditioning
            # fit failed (constant Xj, all-NaN row, etc.); skip.
            importances[j] = 0.0
            continue

        score_losses = []
        for _ in range(n_repeats):
            X_perm = X_arr.copy()
            for leaf_id in np.unique(leaves):
                in_leaf = np.where(leaves == leaf_id)[0]
                if in_leaf.size <= 1:
                    continue
                shuffled_positions = rng.permutation(in_leaf)
                X_perm[in_leaf, j] = X_arr[shuffled_positions, j]
            X_for_score = (
                pd.DataFrame(X_perm, columns=cols, index=idx) if is_dataframe else X_perm
            )
            # E11 ext: wrap model.score in try/except so a custom scorer crash
            # on the permuted X doesn't kill the whole CPI loop. NaN signals
            # the failure to the consumer.
            try:
                score_losses.append(baseline - float(model.score(X_for_score, y)))
            except Exception:
                score_losses.append(np.nan)
        importances[j] = float(np.nanmean(score_losses)) if any(not np.isnan(s) for s in score_losses) else 0.0

    return importances


def get_feature_importances(
    model: object,
    current_features: list,
    importance_getter: str | Callable,
    data: pd.DataFrame | np.ndarray | None = None,
    reference_data: pd.DataFrame | np.ndarray | None = None,
    target: pd.Series | np.ndarray | None = None,
    train_data: pd.DataFrame | np.ndarray | None = None,
    multiclass_coef_aggregation: str = "max",
    coef_scale_source: str = "train",
    cpi_max_depth: Union[int, None] = None,
    cpi_min_samples_leaf: int = 10,
    n_repeats: int = 5,
    random_state: int = 0,
) -> dict:
    """Compute per-feature importance for a fitted model.

    importance_getter:
        - 'auto': inspect model's attributes (feature_importances_ -> coef_)
        - 'feature_importances_' / 'coef_' / any other attr name
        - 'permutation': sklearn.inspection.permutation_importance
        - 'conditional_permutation' (Strobl 2008): permute X_j WITHIN
          leaves of a shallow tree X_{-j} -> X_j; preserves P(X_j | X_{-j}) so
          correlated-feature pairs no longer inflate each other's importance.
        - 'shap': shap.Explainer mean-abs values
        - Callable: importance_getter(model, data, reference_data, target)

    Accuracy-first default (CLAUDE.md "Variant defaults: most accurate first"): when the caller leaves
    importance unspecified (None / 'auto') AND a held-out (data, target) is available, resolve to
    PERMUTATION importance. On the wide FS bench (6 scenarios x 2 seeds) permutation beat impurity on
    10/12 cells (best downstream LightGBM AUC 0.795 vs 0.790, and far cleaner: 2.5 vs 6.2 noise features
    kept), because impurity importance is in-bag and inflates high-variance / structure-bearing noise
    columns, whereas permutation measures held-out predictive degradation (noise -> ~0). SHAP main-effect
    importance behaved like impurity (kept noise) and was the slowest, so it is NOT the default. Cost gate:
    above ``_PERM_AUTO_CELL_CAP`` cells the per-fold permutation cost is prohibitive, so 'auto' falls back
    to impurity (speed) where it is the only affordable option. Pass importance_getter='feature_importances_'
    to force impurity, or 'permutation' to force it regardless of size.
    """
    if importance_getter is None:
        importance_getter = "auto"
    if importance_getter == "auto" and target is not None and data is not None:
        try:
            _shape = getattr(data, "shape", None)
            _cells = int(_shape[0]) * (int(_shape[1]) if len(_shape) > 1 else 1) if _shape else 0
        except Exception:
            _cells = 0
        if 0 < _cells <= _PERM_AUTO_CELL_CAP:
            importance_getter = "permutation"  # accuracy winner; below the cost cap
    if isinstance(importance_getter, str):
        if importance_getter == "permutation":
            if target is None:
                raise ValueError(
                    "importance_getter='permutation' requires target (y_test) "
                    "to score against. Pass target= explicitly."
                )
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
            # perf P1 (2026-06-08): the per-call ``estimator.score()`` re-runs sklearn's full target-type
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
            # perf P3 (2026-06-08): the estimator's ``predict`` re-validates the permuted X on every one of
            # the p*n_repeats scorer calls -- ``check_array`` -> ``_assert_all_finite`` rescans the whole
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
            # (bench_perm_fi_repeat_aggregator.py) and REJECTED -- neither beats the mean on spearman-vs-true-
            # relevance at the realistic small n_repeats (3, 5); with so few repeats a robust aggregator just
            # discards the averaging that suppresses per-permutation noise. Revisit only at n_repeats >= 15.
            res = pi.importances_mean
        elif importance_getter == "conditional_permutation":
            if target is None:
                raise ValueError(
                    "importance_getter='conditional_permutation' requires target (y_test) "
                    "to score against. Pass target= explicitly."
                )
            # F10 (Wave 3, 2026-05-28): cpi_max_depth=None lets the auxiliary tree grow
            # until min_samples_leaf constraint kicks in (Strobl 2008 recommendation).
            # random_state forwarded (default 0 preserves legacy behaviour); lets
            # the caller thread a fold-derived seed so CPI shuffles vary per fold
            # and the run stays reproducible, matching the 'permutation' path.
            res = _conditional_permutation_importance(
                model, data, target,
                n_repeats=n_repeats,
                max_depth=cpi_max_depth,
                min_samples_leaf=cpi_min_samples_leaf,
                random_state=random_state,
            )
        elif importance_getter == "drop_column":
            # Drop-column importance: refit ``model`` on data with each column
            # individually dropped, measure score drop vs full-X baseline.
            # O(p * full_fit_time) -- infeasible on p>=1000. Useful as a
            # ground-truth oracle when benchmarking other importance methods.
            if data is None or target is None:
                raise ValueError(
                    "importance_getter='drop_column' requires data (X) and target (y) at the call site."
                )
            from sklearn.base import clone as _clone
            _Xnp = data.to_numpy(copy=False) if hasattr(data, "to_numpy") else np.asarray(data)
            _baseline = float(model.score(data, target))
            _scores = np.zeros(_Xnp.shape[1], dtype=float)
            for _j in range(_Xnp.shape[1]):
                _X_drop = np.delete(_Xnp, _j, axis=1)
                if hasattr(data, "columns"):
                    _X_drop = pd.DataFrame(_X_drop, columns=[c for i, c in enumerate(data.columns) if i != _j])
                _m = _clone(model)
                try:
                    _m.fit(_X_drop, target)
                    _scores[_j] = _baseline - float(_m.score(_X_drop, target))
                except Exception:
                    _scores[_j] = 0.0
            res = _scores
        elif importance_getter == "boruta":
            # Classical Boruta (Kursa & Rudnicki 2010, JSS-36): pair each real
            # feature with a SHADOW (shuffled copy) and judge importance vs
            # the max-shadow importance. No external dep; uses the supplied
            # ``model``'s feature_importances_ / coef_ after refitting on
            # [X, X_shadow]. Pure-Gini variant -- biased on high-cardinality
            # categoricals. Use 'boruta_shap' for the SHAP-based unbiased
            # version when shap is available.
            if data is None or target is None:
                raise ValueError(
                    "importance_getter='boruta' requires data (X) and target (y) at the call site."
                )
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
                raise RuntimeError(
                    f"Boruta refit on [X, shadow] failed for {type(model).__name__}: {_exc}"
                ) from _exc
            # Read importances of joint model.
            if hasattr(_model_clone, "feature_importances_"):
                _imps = np.asarray(_model_clone.feature_importances_)
            elif hasattr(_model_clone, "coef_"):
                _imps = np.abs(np.asarray(_model_clone.coef_))
                if _imps.ndim > 1:
                    _imps = _imps.max(axis=0)
            else:
                raise AttributeError(
                    f"'boruta' importance_getter requires feature_importances_ or coef_ on the refit model; "
                    f"{type(_model_clone).__name__} has neither."
                )
            _real = _imps[:_p]
            _shadow_max = float(_imps[_p:].max()) if len(_imps) > _p else 0.0
            # Per-feature score = real importance MINUS shadow-max threshold.
            # Positive = beats shadow; negative = noise. Caller's downstream
            # consumer treats it like any FI vector.
            res = _real - _shadow_max
        elif importance_getter == "boruta_shap":
            # L1 (Wave 5, 2026-05-28): Boruta-SHAP via the optional
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
                raise ImportError(
                    "importance_getter='boruta_shap' requires the optional ``BorutaShap`` package. "
                    "Install via ``pip install BorutaShap``."
                ) from _exc2
            try:
                _bs = _BorutaShap(model=model, importance_measure="shap", classification=hasattr(model, "classes_"))
                _bs.fit(X=data, y=target, n_trials=15, random_state=0, verbose=False)
                # Output: BorutaShap stores accepted/rejected lists; build dense importance with shadow-relative scores.
                _accepted = set(getattr(_bs, "accepted", []) or [])
                _tentative = set(getattr(_bs, "tentative", []) or [])
                res = np.array([1.0 if c in _accepted else (0.5 if c in _tentative else 0.0)
                                for c in current_features], dtype=float)
            except Exception as _exc:
                raise RuntimeError(f"BorutaShap failed: {_exc}") from _exc
        elif importance_getter == "powershap":
            # L2 (Wave 5, 2026-05-28): PowerSHAP via optional ``powershap`` pkg.
            if target is None:
                raise ValueError("importance_getter='powershap' requires target (y_test).")
            try:
                from powershap import PowerShap as _PowerShap
            except ImportError as _exc:
                raise ImportError(
                    "importance_getter='powershap' requires the optional ``powershap`` package. "
                    "Install via ``pip install powershap``."
                ) from _exc
            try:
                _ps = _PowerShap(model=model)
                _ps.fit(data, target)
                # _ps stores _processed_shaps_df with p-values per feature; treat selected -> 1, else 0.
                _sel = set(_ps.selected_features_) if hasattr(_ps, "selected_features_") else set()
                res = np.array([1.0 if c in _sel else 0.0 for c in current_features], dtype=float)
            except Exception as _exc:
                raise RuntimeError(f"PowerSHAP failed: {_exc}") from _exc
        elif importance_getter in ("shap", "shap_oof"):
            # L4 (Wave 5, 2026-05-28): 'shap_oof' is an explicit alias for
            # 'shap'. The standard RFECV fold path fits the model on
            # X_train then calls this with data=X_test (held-out fold),
            # so the resulting mean(|SHAP|) is already an OOF importance.
            # The alias name makes that semantic explicit for callers who
            # want SHAP-OOF elimination without having to read the source.
            try:
                import shap as _shap
            except ImportError as _exc:
                raise ImportError(
                    f"importance_getter={importance_getter!r} requires the optional "
                    f"``shap`` package. Install via ``pip install shap``."
                ) from _exc
            try:
                explainer = _shap.Explainer(model, data)
                shap_values = explainer(data, check_additivity=False)
                vals = shap_values.values
                if vals.ndim > 2:
                    vals = np.abs(vals).mean(axis=tuple(range(2, vals.ndim)))
                res = np.abs(vals).mean(axis=0)
            except Exception as _exc:
                raise RuntimeError(
                    f"shap.Explainer failed for {type(model).__name__}: {_exc}. "
                    f"Try importance_getter='permutation' instead."
                ) from _exc
        else:
            # 2026-05-28 sklearn-parity: ``importance_getter`` may be a dotted
            # path such as ``regressor_.coef_`` (for TransformedTargetRegressor)
            # or ``named_steps.lr.coef_`` (for Pipeline). Resolve via
            # operator.attrgetter so the legacy single-attr behaviour AND the
            # dotted path both work. ``auto`` also unwraps Pipelines and other
            # wrappers to the underlying ``_final_estimator`` / ``regressor_``
            # before searching for feature_importances_ / coef_.
            def _unwrap_estimator(m):
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

            def _resolve_getter(obj, dotted: str):
                # operator.attrgetter handles the dotted-path traversal.
                from operator import attrgetter
                return attrgetter(dotted)(obj)

            if importance_getter == "auto":
                inner = _unwrap_estimator(model)
                if hasattr(inner, "feature_importances_"):
                    res = inner.feature_importances_
                    getter_attr = "feature_importances_"
                elif hasattr(inner, "coef_"):
                    res = inner.coef_
                    getter_attr = "coef_"
                else:
                    raise AttributeError(
                        f"importance_getter='auto' could not find feature_importances_ or coef_ "
                        f"on a fitted {type(model).__name__} (unwrapped to {type(inner).__name__})."
                    )
            else:
                getter_attr = importance_getter
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
                    getter_attr = importance_getter.rsplit(".", 1)[-1]
                else:
                    res = getattr(model, getter_attr)
            if getter_attr == "coef_":
                # F5 (Wave 3, 2026-05-28): multi-class collapse. 'max' (default)
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
                # F4 (Wave 3, 2026-05-28): scale correction with TRAIN stds.
                # Pre-fix used X_test stds -> leaks test variance into FI on small
                # folds. Use train_data when provided; fall back to data only if
                # train_data is absent (callable importance_getter path) and
                # coef_scale_source != 'none'.
                _scale_src = coef_scale_source
                if _scale_src == "none":
                    pass
                else:
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
            elif res.ndim > 1:
                # Tree-based feature_importances_ stays 1-D normally; this branch
                # handles unusual estimators (e.g. multi-output) and uses sum
                # because we can't distinguish "OvR class" from "output" here.
                res = res.sum(axis=0)
    else:
        try:
            res = importance_getter(model=model, data=data, reference_data=reference_data, target=target)
        except TypeError:
            res = importance_getter(model=model, data=data, reference_data=reference_data)

    if len(res) != len(current_features):
        raise ValueError(f"Feature importances length {len(res)} doesn't match current_features length {len(current_features)}")

    try:
        res_arr = np.asarray(res, dtype=float)
        n_nan = int(np.isnan(res_arr).sum()) if res_arr.size else 0
    except (TypeError, ValueError):
        n_nan = 0
    if n_nan:
        logger.warning(
            "get_feature_importances: %d / %d importance value(s) are NaN from %s.",
            n_nan, res_arr.size, type(model).__name__,
        )
    return {feature_index: feature_importance for feature_index, feature_importance in zip(current_features, res)}


def select_appropriate_feature_importances(
    feature_importances: dict,
    nfeatures: int,
    n_original_features: int,
    use_all_fi_runs: bool = True,
    use_last_fi_run_only: bool = False,
    use_one_freshest_fi_run: bool = False,
    use_fi_ranking: bool = False,
    votes_aggregation_method: Union[VotesAggregation, None] = None,
) -> dict:
    if use_last_fi_run_only:
        fi_to_consider = {key: value for key, value in feature_importances.items() if len(value) == n_original_features}
    else:
        if use_all_fi_runs:
            fi_to_consider = {key: value for key, value in feature_importances.items() if len(value) > 1} if n_original_features > 1 else feature_importances
        else:
            if use_one_freshest_fi_run:
                # Upper bound is inclusive (n_original_features + 1) so the
                # FI run on the full feature set is also considered.
                fi_to_consider = {}
                for possible_nfeatures in range(nfeatures + 1, n_original_features + 1):
                    for key, value in feature_importances.items():
                        if len(value) == possible_nfeatures:
                            fi_to_consider[key] = value
                    if fi_to_consider:
                        print(f"using freshest FI of {possible_nfeatures} features for nfeatures={nfeatures}")
                        break
            else:
                fi_to_consider = {key: value for key, value in feature_importances.items() if (len(value) > nfeatures and len(value) != 1)}
    if use_fi_ranking:
        # F12 (Wave 3, 2026-05-28): rank-based aggregation rules (Borda /
        # Copeland / Dowdall / Minimax / Plurality) internally rank the
        # input table anyway; pre-ranking it here is a no-op for them and
        # only adds tiebreaker-method drift (.rank default 'average' vs
        # Leaderboard's 'min'/'max'). Skip pre-ranking when the downstream
        # aggregator is itself rank-based.
        _rank_based = {
            VotesAggregation.Borda,
            VotesAggregation.Copeland,
            VotesAggregation.Dowdall,
            VotesAggregation.Minimax,
            VotesAggregation.Plurality,
        } if votes_aggregation_method is not None else set()
        if votes_aggregation_method in _rank_based:
            pass  # downstream Leaderboard handles ranking
        else:
            fi_to_consider = {key: pd.Series(value).rank(ascending=True, pct=True).to_dict() for key, value in fi_to_consider.items()}
    return fi_to_consider


def _impute_ragged_fi_table(table: pd.DataFrame, policy: str) -> pd.DataFrame:
    """F1+F2+F3 (Wave 1, 2026-05-28): impute missing per-run FI entries in a ragged voting table BEFORE handing it to Leaderboard.

    The historical RFECV vote let pandas' NaN propagate into Borda / Dowdall / Copeland / Minimax / Plurality with skipna=True semantics
    that systematically biased toward late-surviving features (a feature voting in 30/30 runs sums over 30 columns vs a feature voting in
    3/30 runs that sums over 3). AM/GM/OG already fill with the column median upstream; the other rules did not. Different rules + different
    pre-fixes = unpredictable user-facing behaviour. We now normalise the ragged table at the WRAPPER layer so every rule sees the same
    completed input.

    policy:
        'worst'  : missing -> min(col) - eps for each column. A feature absent from run K is treated as "ranked LAST in run K" by every
                   downstream rule. This matches the operator intuition that elimination at iter N means the feature lost the iter-N comparison.
        'median' : missing -> column median. Pre-2026 default for AM/GM/OG, generalised. Lets re-appearing features keep "average" treatment.
        'skip'   : raw pre-fix table (back-compat A/B path).
    """
    if policy == "skip":
        return table
    if not isinstance(table, pd.DataFrame) or table.empty or not table.isna().to_numpy().any():
        return table
    if policy == "worst":
        # For each column, the imputed value sits strictly below the smallest observed FI so the "missing -> last rank" guarantee
        # holds even on ties: the eps is scaled to the column range so a constant column doesn't collapse the gap.
        out = table.copy()
        col_min = out.min(axis=0, skipna=True)
        col_max = out.max(axis=0, skipna=True)
        col_range = (col_max - col_min).fillna(0.0)
        # Treat zero-range columns (every present value identical) as needing a finite eps so the imputed value still sorts strictly below.
        col_eps = col_range.where(col_range > 0.0, other=1.0) * 1e-3
        fill = col_min - col_eps
        # Any column whose every value is NaN cannot be imputed from itself; fall back to a global floor below the table-wide min.
        all_nan_cols = fill.isna()
        if all_nan_cols.any():
            global_floor = float(table.min(skipna=True).min(skipna=True))
            if not np.isfinite(global_floor):
                global_floor = 0.0
            fill = fill.where(~all_nan_cols, other=global_floor - 1.0)
        out = out.fillna(fill)
        return out
    if policy == "median":
        return table.fillna(table.median())
    raise ValueError(f"_impute_ragged_fi_table: unknown policy={policy!r}")


def get_actual_features_ranking(feature_importances: dict, votes_aggregation_method: VotesAggregation, fi_missing_policy: str = "worst", run_weights: Union[dict, None] = None) -> list:
    """Vote-based rank of features given per-run importances.

    Borda/AM/GM/Dowdall use only ranks (cheap). Copeland needs majority_graph,
    which Leaderboard builds lazily.

    Args:
        feature_importances: dict[run_key -> dict[feature -> importance]].
        votes_aggregation_method: rule from VotesAggregation enum.
        fi_missing_policy: how to complete ragged-NaN table (see _impute_ragged_fi_table).
        run_weights: optional dict[run_key -> weight] for F8 exponential-decay
            over FI history. When None, all runs vote with equal weight
            (legacy). Leaderboard normalises so the absolute scale doesn't
            matter; the RATIO between newer and older runs is what shifts
            the final ranking.

    F7 (Wave 3, 2026-05-28) tie-breaker: when two features end the rule with
    identical Leaderboard scores (very common on tree FI with many zeros),
    fall back to lexicographic ordering by feature name so the output is
    fully deterministic across Python set/dict iteration orders.
    """
    table = pd.DataFrame(feature_importances)
    table = _impute_ragged_fi_table(table, policy=fi_missing_policy)
    # F8: forward run_weights into Leaderboard. Leaderboard normalises by sum
    # so we just pass the float weights; the rule code multiplies per-column.
    if run_weights:
        lb = Leaderboard(table=table, weights=dict(run_weights))
    else:
        lb = Leaderboard(table=table)
    if votes_aggregation_method == VotesAggregation.Borda:
        ranks = lb.borda_ranking()
    elif votes_aggregation_method == VotesAggregation.AM:
        ranks = lb.mean_ranking(mean_type="arithmetic")
    elif votes_aggregation_method == VotesAggregation.GM:
        ranks = lb.mean_ranking(mean_type="geometric")
    elif votes_aggregation_method == VotesAggregation.Copeland:
        ranks = lb.copeland_ranking()
    elif votes_aggregation_method == VotesAggregation.Dowdall:
        ranks = lb.dowdall_ranking()
    elif votes_aggregation_method == VotesAggregation.Minimax:
        ranks = lb.minimax_ranking()
    elif votes_aggregation_method == VotesAggregation.OG:
        ranks = lb.optimality_gap_ranking(gamma=1)
    elif votes_aggregation_method == VotesAggregation.Plurality:
        ranks = lb.plurality_ranking()
    else:
        raise NotImplementedError(
            f"votes_aggregation_method={votes_aggregation_method!r} not handled"
        )
    # F7 tie-breaker: lexicographic on feature name. Without this the order
    # of equal-rank features depends on Leaderboard's internal sort and on
    # the order of the original dict keys -> different runs of the SAME
    # input pick different "top N" features on tie-clusters.
    _scores = ranks.to_dict()
    out = sorted(_scores.keys(), key=lambda k: (-float(_scores[k]) if np.isfinite(_scores[k]) else 0.0, str(k)))
    return out
