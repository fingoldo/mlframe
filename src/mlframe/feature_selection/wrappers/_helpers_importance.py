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
from ._importance_methods import named_importance

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
    except Exception as exc:
        logger.debug("_fold_is_all_finite: coercion failed, conservatively assuming not-finite: %s", exc)
        return False
    if a.dtype.kind in ("f", "c"):
        return bool(np.isfinite(a).all())
    if a.dtype.kind in ("i", "u", "b"):
        return True  # integer / bool arrays are always finite
    return False  # object / string / datetime: don't assume; let sklearn validate


def _make_fast_default_scorer(model: object) -> Callable:
    """Build a permutation-FI scorer that reproduces ``estimator.score()`` bit-identically but
    skips its redundant per-call target-type validation.

    Perf P1: ``estimator.score()`` is the permutation-importance hotspot because it
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

    Perf P2: the per-call defensive ``np.array(_X, copy=True)`` is only required for
    estimators whose ``predict`` flips the input ndarray's writeable flag to False (CatBoost), which would
    make sklearn's NEXT in-place column shuffle of its reused ``X_permuted`` buffer raise
    "assignment destination is read-only". For every estimator that does NOT flip the flag (sklearn
    linear / tree / ensemble, LightGBM, XGBoost, ...) the copy is pure waste - it was ~36k array copies on
    the scene bench. We detect the flip on the baseline call: predict on a copy, then inspect that copy's
    writeable flag. If still writeable the estimator is flag-safe and we latch ``need_copy=False`` (read
    sklearn's ``X_permuted`` directly); if flipped (or the buffer was already read-only) we keep the copy.
    Either way the values fed to ``predict`` are identical, so the score - and the selected set - is
    bit-identical.
    """
    from sklearn.base import is_classifier, is_regressor

    _is_clf = bool(is_classifier(model))
    _is_reg = bool(is_regressor(model))

    # mode:      -1 = baseline pending; 0 = safe (estimator.score); 1 = fast closed-form.
    # need_copy: True = defensive copy each call (writeable-flip / read-only buffer); False = read _X directly.
    # _ycache:   y-derived invariants keyed by id(_y) - see _fast_value.
    state: dict = {"mode": -1, "need_copy": True, "_ycache": None, "_yid": None}

    def _y_invariants(_y):
        """Return (y_arr, regressor (yt, ss_tot)) for ``_y``, computing once and reusing while ``_y`` is unchanged.

        ``permutation_importance`` permutes only X across its ``p*n_repeats`` scorer calls and feeds the IDENTICAL
        ``_y`` every time. The classifier path then re-ran ``np.asarray(_y)`` and the regressor path re-ran
        ``_y.astype(float64)`` + ``np.mean(_y)`` + ``ss_tot=sum((y-mean)**2)`` - all functions of ``_y`` alone -
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
        """``permutation_importance``-compatible scoring callable: latches to the closed-form fast path (mode 1)
        or falls back to ``_est.score`` (mode 0) after a one-shot baseline bit-identity self-check; see the
        enclosing docstring for the exact latch/copy safety conditions."""
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
        except Exception as e:
            logger.debug("_fast_value computation failed: %s", e)
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

    F10: max_depth=None grows the tree until
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

    if isinstance(X, pd.DataFrame):
        is_dataframe = True
        X_arr = X.to_numpy()
        cols = X.columns
        idx = X.index
    else:
        is_dataframe = False
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
        """Superseded by ``_is_discrete_v2`` below (kept for reference / A-B); pragmatic proxy: integer dtype OR
        <=10 unique non-null values routes the conditioning tree to a classifier."""
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
        """Tighter discrete-vs-continuous detection than ``_is_discrete``: integer dtype is canonical discrete; for
        floats, both low unique-count AND cardinality far below the row count are required, so decile-binned
        continuous columns route to the regressor conditioning tree instead of mis-triggering the classifier."""
        # F11: tighter discrete detection. Integer dtype is canonical discrete. For floats, require BOTH (a) low unique count AND (b) cardinality << n_rows.
        # Decile-binned continuous variables (10 unique values across 100k rows) now correctly route to regression instead of classification.
        if np.issubdtype(col.dtype, np.integer):
            return True
        try:
            mask = ~np.isnan(col.astype(float, copy=False))
            uniq = np.unique(col[mask])
        except (TypeError, ValueError):
            uniq = np.unique(col)
        _n = max(int(mask.sum()) if hasattr(mask, "sum") else len(col), 1)
        return uniq.size <= max(5, int(np.sqrt(_n))) and uniq.size <= 0.5 * _n

    # Copy the matrix ONCE; every iteration mutates only column j and restores it in a
    # try/finally so the buffer re-enters each iteration identical to X_arr. The prior
    # per-repeat ``X_arr.copy()`` allocated a full p-column copy on every (j x repeat).
    X_perm = X_arr.copy()

    # Conditioning set X_{-j} buffer, reused across j. The prior per-feature
    # ``np.delete(X_arr, j, axis=1)`` allocated a fresh (n, p-1) array every iteration
    # (O(n*p^2) allocation/copy over the loop). A single C-contiguous (n, p-1) scratch
    # refilled by two contiguous block-copies (cols [0:j] and [j+1:p]) yields exactly the
    # same bytes np.delete would produce, with one allocation total. sklearn's tree fit
    # re-reads the buffer each call, so reuse is safe (no aliasing across iterations).
    Xnotj_buf = np.empty((n, p - 1), dtype=X_arr.dtype) if p > 1 else None
    for j in range(p):
        Xj = X_arr[:, j]
        if Xnotj_buf is None:
            Xnotj = X_arr[:, :0]
        else:
            if j > 0:
                Xnotj_buf[:, :j] = X_arr[:, :j]
            if j < p - 1:
                Xnotj_buf[:, j:] = X_arr[:, j + 1 :]
            Xnotj = Xnotj_buf

        if Xnotj.shape[1] == 0:
            # Single-feature case: no conditioning set; fall back to vanilla shuffle.
            score_losses = []
            orig_col = X_arr[:, j].copy()
            try:
                for _ in range(n_repeats):
                    X_perm[:, j] = rng.permutation(orig_col)
                    X_for_score = pd.DataFrame(X_perm, columns=cols, index=idx) if is_dataframe else X_perm
                    # E11 ext (mirrors the general p>1 branch below): wrap model.score in try/except so a
                    # custom scorer crash on the permuted X doesn't abort the whole per-fold FI computation.
                    # NaN signals the failure to the consumer instead of propagating the exception.
                    try:
                        score_losses.append(baseline - float(model.score(X_for_score, y)))
                    except Exception as e:
                        logger.debug("permutation-importance score() failed (single-feature branch), recording NaN: %s", e)
                        score_losses.append(np.nan)
            finally:
                X_perm[:, j] = orig_col
            # NaN, not 0.0, when nothing could be measured: real importances are ``baseline - score`` and routinely
            # negative, so a neutral 0.0 outranks every feature measured to be harmful.
            importances[j] = float(np.nanmean(score_losses)) if any(not np.isnan(s) for s in score_losses) else np.nan
            continue

        # F10/F11: pass max_depth + min_samples_leaf. max_depth=None grows the tree
        # until min_samples_leaf binds (recommended by Strobl 2008 on >=5).
        # F11: _is_discrete heuristic improved to require
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
            # E11: widen the except to catch MemoryError
            # on 1M-row Xnotj and RuntimeError from custom-estimator paths
            # raising AttributeError-like wrapped exceptions. Conditioning
            # fit failed (constant Xj, all-NaN row, etc.); skip. NaN = "not measured", never a rank-competitive 0.0.
            importances[j] = np.nan
            continue

        score_losses = []
        orig_col = X_arr[:, j].copy()
        unique_leaves = np.unique(leaves)
        try:
            for _ in range(n_repeats):
                for leaf_id in unique_leaves:
                    in_leaf = np.where(leaves == leaf_id)[0]
                    if in_leaf.size <= 1:
                        continue
                    shuffled_positions = rng.permutation(in_leaf)
                    X_perm[in_leaf, j] = orig_col[shuffled_positions]
                X_for_score = pd.DataFrame(X_perm, columns=cols, index=idx) if is_dataframe else X_perm
                # E11 ext: wrap model.score in try/except so a custom scorer crash
                # on the permuted X doesn't kill the whole CPI loop. NaN signals
                # the failure to the consumer.
                try:
                    score_losses.append(baseline - float(model.score(X_for_score, y)))
                except Exception as e:
                    logger.debug("permutation-importance score() failed, recording NaN: %s", e)
                    score_losses.append(np.nan)
        finally:
            X_perm[:, j] = orig_col
        # Same as the single-feature branch: NaN when every repeat failed, never a 0.0 that outranks harmful features.
        importances[j] = float(np.nanmean(score_losses)) if any(not np.isnan(s) for s in score_losses) else np.nan

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
    importance_getter = _get_feature_importan_step1_importance_getter_none(importance_getter, target, data)
    if isinstance(importance_getter, str):
        res = named_importance(
            model, current_features, importance_getter, data, target, train_data,
            multiclass_coef_aggregation=multiclass_coef_aggregation,
            coef_scale_source=coef_scale_source,
            cpi_max_depth=cpi_max_depth,
            cpi_min_samples_leaf=cpi_min_samples_leaf,
            n_repeats=n_repeats,
            random_state=random_state,
        )
    else:
        try:
            res = importance_getter(model=model, data=data, reference_data=reference_data, target=target)
        except TypeError:
            res = importance_getter(model=model, data=data, reference_data=reference_data)

    if len(res) != len(current_features):
        raise ValueError(f"Feature importances length {len(res)} doesn't match current_features length {len(current_features)}")

    res_arr = None
    try:
        res_arr = np.asarray(res, dtype=float)
        n_nan = int(np.isnan(res_arr).sum()) if res_arr.size else 0
    except (TypeError, ValueError):
        # Non-numeric importances (object / mixed): NaN detection isn't meaningful, skip it.
        n_nan = 0
        logger.debug("get_feature_importances: skipping NaN-detection on non-numeric importances from %s.", type(model).__name__)
    if n_nan and res_arr is not None:
        logger.warning(
            "get_feature_importances: %d / %d importance value(s) are NaN from %s.",
            n_nan, res_arr.size, type(model).__name__,
        )
    return {feature_index: feature_importance for feature_index, feature_importance in zip(current_features, res)}


def _get_feature_importan_step1_importance_getter_none(importance_getter, target, data):
    """Step 1 of get_feature_importances: lines starting at ``if importance_getter is None:``."""
    if importance_getter is None:
        importance_getter = "auto"
    if importance_getter == "auto" and target is not None and data is not None:
        try:
            _shape = getattr(data, "shape", None)
            _cells = int(_shape[0]) * (int(_shape[1]) if len(_shape) > 1 else 1) if _shape else 0
        except Exception as e:
            logger.debug("cell-count computation failed: %s", e)
            _cells = 0
        if 0 < _cells <= _PERM_AUTO_CELL_CAP:
            importance_getter = "permutation"  # accuracy winner; below the cost cap
    return importance_getter


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
    """Filter the per-RFECV-iteration ``feature_importances`` history down to the runs the voting rule should see.

    ``feature_importances`` maps a run key -> per-feature importance dict, with runs recorded at every elimination
    step (a shrinking feature-set size per run). The four selection modes:
      - ``use_last_fi_run_only``: keep only runs whose feature-set size equals the FULL original feature count.
      - ``use_all_fi_runs`` (default): keep every run with more than 1 feature (all history votes).
      - ``use_one_freshest_fi_run``: walk feature-set sizes from ``nfeatures+1`` up to ``n_original_features``
        (inclusive) and take the first size that has any runs recorded - the "freshest" (smallest, most-refined)
        run set that still covers at least ``nfeatures`` features.
      - else: keep runs with more than ``nfeatures`` features (excluding the degenerate size-1 case).

    When ``use_fi_ranking`` is set, importances are converted to a percentile RANK per run BEFORE voting, except for
    aggregators that are themselves rank-based (Borda / Copeland / Dowdall / Minimax / Plurality), where a pre-rank
    would only add tiebreak drift against the aggregator's own internal ranking.
    """
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
                        logger.debug("using freshest FI of %d features for nfeatures=%d", possible_nfeatures, nfeatures)
                        break
            else:
                fi_to_consider = {key: value for key, value in feature_importances.items() if (len(value) > nfeatures and len(value) != 1)}
    if use_fi_ranking:
        # F12: rank-based aggregation rules (Borda /
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
    """F1+F2+F3: impute missing per-run FI entries in a ragged voting table BEFORE handing it to Leaderboard.

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

    F7 tie-breaker: when two features end the rule with
    identical Leaderboard scores (very common on tree FI with many zeros),
    fall back to lexicographic ordering by feature name so the output is
    fully deterministic across Python set/dict iteration orders.

    Returns:
        Features ordered best to worst by the chosen voting rule; an empty list when ``feature_importances`` is empty.
    """
    if not feature_importances:
        # No run ever produced FI (e.g. every CV fold was skipped upstream, such as a
        # min_train_size above every fold's train slice) -- there is nothing to vote on.
        # Leaderboard requires a nonempty weight sum, so return no ranking rather than crash;
        # the caller's `ranks[:next_nfeatures_to_check]` already tolerates an empty list.
        return []
    table = pd.DataFrame(feature_importances)
    table = _impute_ragged_fi_table(table, policy=fi_missing_policy)
    # "skip" deliberately hands the raw, still-partial table to Leaderboard (see
    # _impute_ragged_fi_table's own docstring: a back-compat A/B path reproducing the pre-fix
    # pandas skipna=True bias) -- bypass the F6 partial-table guard ONLY for this intentional case.
    _allow_partial = fi_missing_policy == "skip"
    # F8: forward run_weights into Leaderboard. Leaderboard normalises by sum
    # so we just pass the float weights; the rule code multiplies per-column.
    if run_weights:
        lb = Leaderboard(table=table, weights=dict(run_weights), allow_partial=_allow_partial)
    else:
        lb = Leaderboard(table=table, allow_partial=_allow_partial)
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
        raise NotImplementedError(f"votes_aggregation_method={votes_aggregation_method!r} not handled")
    # F7 tie-breaker: lexicographic on feature name. Without this the order
    # of equal-rank features depends on Leaderboard's internal sort and on
    # the order of the original dict keys -> different runs of the SAME
    # input pick different "top N" features on tie-clusters.
    _scores = ranks.to_dict()
    out = sorted(_scores.keys(), key=lambda k: (-float(_scores[k]) if np.isfinite(_scores[k]) else 0.0, str(k)))
    return out
