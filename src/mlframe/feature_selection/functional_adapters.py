"""sklearn fit/transform adapters over the FUNCTIONAL feature-selection utilities.

``forward_select`` / ``greedy_backward_elimination`` / ``iterative_zero_importance_pruning`` /
``cascade_select`` are plain functions that return a selected-column LIST (or a report dict for
``cascade_select``), not sklearn estimators -- unlike MRMR / BorutaShap / ShapProxiedFS / ACE, which
already expose the fit/get_support/transform contract the training suite's pre-pipeline slot drives
selectors through (see ``ace.ACESelector`` for the template this module mirrors). Each adapter here
runs its wrapped function once in ``fit`` and materialises ``support_`` / ``selected_features_`` /
``feature_names_in_`` in INPUT-column order so the suite can treat it exactly like the existing
selectors.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .cascade_select import cascade_select
from .forward_select import forward_select
from .greedy_backward_elimination import greedy_backward_elimination
from .zero_importance_pruning import iterative_zero_importance_pruning


def _is_classification_target(y: np.ndarray) -> bool:
    """Same low-cardinality heuristic as ``ace._default_estimator`` -- integer/label y with a
    small number of unique values is treated as classification, everything else as regression."""
    y_arr = np.asarray(y)
    n = y_arr.shape[0]
    if y_arr.dtype.kind in ("i", "u", "b", "O", "U", "S"):
        return bool(np.unique(y_arr).size <= max(20, int(np.sqrt(max(n, 1)))))
    return False


def _default_tree_estimator(y: np.ndarray, random_state: int = 0):
    """Task-appropriate RandomForest default, mirrors ``ace._default_estimator`` verbatim."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if _is_classification_target(y):
        return RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=random_state, max_features="sqrt")
    return RandomForestRegressor(n_estimators=120, n_jobs=-1, random_state=random_state, max_features="sqrt")


def _default_pointwise_scoring(y: np.ndarray):
    """Default ``scoring(y_true, y_pred) -> float`` (HIGHER is better) for the callable-scoring functions
    (``greedy_backward_elimination`` / ``iterative_zero_importance_pruning``): accuracy for classification,
    R2 for regression -- auto-derived so an operator opting into these selectors needs no extra wiring."""
    from sklearn.metrics import accuracy_score, r2_score

    return accuracy_score if _is_classification_target(y) else r2_score


def _default_sklearn_scorer_name(y: np.ndarray) -> str:
    """Default sklearn scorer NAME for ``forward_select`` (which takes a scorer string, not a callable)."""
    return "accuracy" if _is_classification_target(y) else "r2"


def _support_from_selected(feature_names: list, selected: list) -> np.ndarray:
    """Boolean support mask (input-column order) marking columns present in ``selected``."""
    selected_set = set(str(c) for c in selected)
    return np.asarray([str(c) in selected_set for c in feature_names], dtype=bool)


def _numeric_view_for_selection(X):
    """Return an all-numeric view of ``X`` (same column order/names) for the tree-estimator-driven selection loop.

    ``forward_select`` / ``greedy_backward_elimination`` / ``iterative_zero_importance_pruning`` / ``cascade_select``
    all fit the estimator on raw candidate-column subsets of ``X`` -- with the default ``_default_tree_estimator``
    (plain sklearn RandomForest) this crashes with "could not convert string to float" on ANY non-numeric column,
    e.g. a CatBoost-native categorical kept as a raw string (``skip_categorical_encoding=True``): caught live via a
    fuzz combo where ``use_forward_select_fs`` (default-ON) silently dropped the whole cb model from the suite.
    Ordinal-encode (``pd.factorize``) any non-numeric column so the default/any bare-numeric-only estimator can
    consume it; a fully-numeric ``X`` is returned unchanged (no copy). Only affects the internal selection fit --
    ``_finalize`` still reads column NAMES off the original ``X``, so the selected features returned to the caller
    are unaffected.
    """
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        non_numeric = [c for c in X.columns if X[c].dtype.kind not in "iufb"]
        if not non_numeric:
            return X
        out = X.copy()
        for c in non_numeric:
            codes, _ = pd.factorize(out[c], use_na_sentinel=True)
            out[c] = np.where(codes < 0, np.nan, codes).astype(np.float64)
        return out
    if hasattr(X, "dtypes") and hasattr(X, "with_columns"):
        # polars DataFrame: encoding must stay a polars frame (never bare np.asarray(X), which silently
        # strips column names -- forward_select/cascade_select need those to name candidate subsets, and
        # dropping them was caught live as a correctness regression, not just a cosmetic type mismatch).
        non_numeric = [c for c, dt in zip(X.columns, X.dtypes) if not dt.is_numeric()]
        if not non_numeric:
            return X
        import polars as pl

        out = X
        for c in non_numeric:
            codes, _ = pd.factorize(out[c].to_numpy(), use_na_sentinel=True)
            out = out.with_columns(pl.Series(c, np.where(codes < 0, np.nan, codes).astype(np.float64)))
        return out
    if isinstance(X, np.ndarray) and X.dtype.kind in "iufb":
        return X
    arr = np.asarray(X)
    if arr.dtype.kind in "iufb":
        return arr
    out = np.empty(arr.shape, dtype=np.float64)
    for j in range(arr.shape[1]):
        codes, _ = pd.factorize(pd.Series(arr[:, j]), use_na_sentinel=True)
        out[:, j] = np.where(codes < 0, np.nan, codes).astype(np.float64)
    return out


class _FunctionalSelectorBase(BaseEstimator, TransformerMixin):
    """Shared fit/transform/get_support plumbing for the functional-utility adapters below."""

    def transform(self, X):
        from sklearn.exceptions import NotFittedError

        if not hasattr(self, "support_"):
            raise NotFittedError(f"{type(self).__name__}.transform called before fit.")
        idx = np.where(self.support_)[0]
        if hasattr(X, "iloc"):
            return X.iloc[:, idx]
        return np.asarray(X)[:, self.support_]

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False):
        from sklearn.exceptions import NotFittedError

        if not hasattr(self, "support_"):
            raise NotFittedError(f"{type(self).__name__}.get_support called before fit.")
        return np.where(self.support_)[0] if indices else self.support_

    def get_feature_names_out(self, input_features=None):
        from sklearn.exceptions import NotFittedError

        if not hasattr(self, "selected_features_"):
            raise NotFittedError(f"{type(self).__name__}.get_feature_names_out called before fit.")
        return np.asarray(self.selected_features_, dtype=object)

    def _finalize(self, X, selected: list) -> None:
        names = [str(c) for c in X.columns] if hasattr(X, "columns") else [f"x{i}" for i in range(np.asarray(X).shape[1])]
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(names)
        self.support_ = _support_from_selected(names, selected)
        self.selected_features_ = [str(c) for c in selected]


class ForwardSelectSelector(_FunctionalSelectorBase):
    """sklearn-compatible adapter over :func:`forward_select` for the training suite's pre-pipeline slot."""

    def __init__(
        self,
        estimator=None,
        *,
        scoring=None,
        cv: int = 5,
        max_features=None,
        min_improvement: float = 0.0,
        patience=None,
        significance_level: float = 0.05,
        random_state: int = 0,
    ):
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.max_features = max_features
        self.min_improvement = min_improvement
        self.patience = patience
        self.significance_level = significance_level
        self.random_state = random_state

    def fit(self, X, y=None):
        from sklearn.base import clone

        base_estimator = self.estimator if self.estimator is not None else _default_tree_estimator(y, self.random_state)
        scoring = self.scoring if self.scoring is not None else _default_sklearn_scorer_name(y)

        def estimator_factory():
            return clone(base_estimator)

        selected = forward_select(
            _numeric_view_for_selection(X), y, estimator_factory, scoring=scoring, cv=self.cv, max_features=self.max_features,
            min_improvement=self.min_improvement, patience=self.patience, significance_level=self.significance_level,
            return_report=False,
        )
        assert isinstance(selected, list)
        self._finalize(X, selected)
        return self


class GreedyBackwardEliminationSelector(_FunctionalSelectorBase):
    """sklearn-compatible adapter over :func:`greedy_backward_elimination` for the training suite's pre-pipeline slot."""

    def __init__(
        self,
        estimator=None,
        *,
        scoring=None,
        cv=None,
        min_features: int = 1,
        tol: float = 0.0,
        n_repeats: int = 1,
        seed_base: int = 0,
        random_state: int = 0,
    ):
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.min_features = min_features
        self.tol = tol
        self.n_repeats = n_repeats
        self.seed_base = seed_base
        self.random_state = random_state

    def fit(self, X, y=None):
        base_estimator = self.estimator if self.estimator is not None else _default_tree_estimator(y, self.random_state)
        scoring = self.scoring if self.scoring is not None else _default_pointwise_scoring(y)

        selected = greedy_backward_elimination(
            base_estimator, _numeric_view_for_selection(X), y, scoring, cv=self.cv, min_features=self.min_features, tol=self.tol,
            n_repeats=self.n_repeats, seed_base=self.seed_base,
        )
        self._finalize(X, selected)
        return self


class ZeroImportancePruningSelector(_FunctionalSelectorBase):
    """sklearn-compatible adapter over :func:`iterative_zero_importance_pruning` for the training suite's pre-pipeline slot."""

    def __init__(
        self,
        estimator=None,
        *,
        scoring=None,
        cv=None,
        importance_threshold: float = 0.0,
        max_rounds: int = 20,
        importance_fn=None,
        random_state: int = 0,
    ):
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.importance_threshold = importance_threshold
        self.max_rounds = max_rounds
        self.importance_fn = importance_fn
        self.random_state = random_state

    def fit(self, X, y=None):
        base_estimator = self.estimator if self.estimator is not None else _default_tree_estimator(y, self.random_state)
        scoring = self.scoring if self.scoring is not None else _default_pointwise_scoring(y)

        selected = iterative_zero_importance_pruning(
            base_estimator, _numeric_view_for_selection(X), y, scoring, cv=self.cv, importance_threshold=self.importance_threshold,
            max_rounds=self.max_rounds, importance_fn=self.importance_fn,
        )
        self._finalize(X, selected)
        return self


class CascadeSelectSelector(_FunctionalSelectorBase):
    """sklearn-compatible adapter over :func:`cascade_select` (Boruta screen -> forward select -> RFECV backward elimination)."""

    def __init__(
        self,
        estimator=None,
        *,
        n_boruta_iterations: int = 20,
        boruta_alpha: float = 0.05,
        forward_max_features=None,
        forward_min_improvement: float = 0.0,
        cv: int = 5,
        scoring=None,
        random_state: int = 42,
        rfecv_kwargs=None,
    ):
        self.estimator = estimator
        self.n_boruta_iterations = n_boruta_iterations
        self.boruta_alpha = boruta_alpha
        self.forward_max_features = forward_max_features
        self.forward_min_improvement = forward_min_improvement
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.rfecv_kwargs = rfecv_kwargs

    def fit(self, X, y=None):
        from sklearn.base import clone

        base_estimator = self.estimator if self.estimator is not None else _default_tree_estimator(y, self.random_state)
        scoring = self.scoring if self.scoring is not None else _default_sklearn_scorer_name(y)

        def estimator_factory():
            return clone(base_estimator)

        result = cascade_select(
            _numeric_view_for_selection(X), y, estimator_factory, n_boruta_iterations=self.n_boruta_iterations, boruta_alpha=self.boruta_alpha,
            forward_max_features=self.forward_max_features, forward_min_improvement=self.forward_min_improvement,
            cv=self.cv, scoring=scoring, random_state=self.random_state, rfecv_kwargs=self.rfecv_kwargs,
        )
        self.cascade_result_ = result
        self._finalize(X, result.get("final_selected") or [])
        return self


__all__ = [
    "ForwardSelectSelector",
    "GreedyBackwardEliminationSelector",
    "ZeroImportancePruningSelector",
    "CascadeSelectSelector",
]
