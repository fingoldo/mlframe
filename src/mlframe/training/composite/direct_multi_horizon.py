"""``DirectMultiHorizonEnsemble``: one model per forecast-horizon block, trained directly (no recursion).

Source: M5 Forecasting Accuracy 4th place -- "Not to use complex and recursive features ... Recursive
features lead to error accumulation" -- instead partition the forecast horizon into blocks (e.g. weeks:
F01-F07, F08-F14, ...) and train one model PER BLOCK, each predicting its own horizon steps directly from
features available at the forecast origin (t=0) -- never from a PRIOR block's own predictions. This avoids
the classic recursive-forecasting failure mode: a one-step-ahead model applied repeatedly, feeding each
step's noisy prediction back in as the next step's lag feature, compounds error multiplicatively over the
horizon (a small per-step bias/variance snowballs by horizon 14 into a large one).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


def _contiguous_blocks(n_horizons: int, block_size: int) -> list[list[int]]:
    """Partition ``range(n_horizons)`` into contiguous chunks of ``block_size`` (last chunk may be shorter)."""
    return [list(range(i, min(i + block_size, n_horizons))) for i in range(0, n_horizons, block_size)]


def _pooled_rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


class DirectMultiHorizonEnsemble(BaseEstimator, RegressorMixin):
    """One independently-fit model per horizon block, each predicting directly from origin-time features.

    Parameters
    ----------
    estimator_factory
        Zero-arg callable returning a fresh unfitted estimator. Called once per horizon block; each block's
        model is trained ONLY on that block's own horizon columns of ``Y``, never on another block's
        predictions -- the defining property of the "direct" (non-recursive) multi-horizon strategy.
    horizon_blocks
        Sequence of horizon-index groups, e.g. ``[[0,1,2,3,4,5,6], [7,8,9,10,11,12,13], ...]`` for
        week-sized blocks over a 14-day horizon. Every index must appear in exactly one block. Required
        unless ``auto_block_search=True``, in which case it may be left ``None`` and the block partition is
        discovered by CV grid search instead.
    auto_block_search
        Opt-in. When ``True`` and ``horizon_blocks`` is ``None``, ``fit`` tries every contiguous-block-size
        candidate in ``block_size_grid`` (K-fold CV pooled RMSE) and keeps the winner. Default behavior when
        this flag is omitted (or ``horizon_blocks`` is given explicitly) is bit-for-bit unchanged from before
        this feature existed -- ``horizon_blocks`` always wins when both are set.
    block_size_grid
        Candidate contiguous block sizes to try under auto search. Defaults to divisor-friendly sizes
        ``1, 2, 3, 4, 6, 8, 12`` capped at ``n_horizons``, plus ``n_horizons`` itself (single block).
    cv_splits
        Number of ``KFold`` splits used to score each candidate block size under auto search.
    random_state
        Passed to ``KFold(shuffle=True, random_state=...)`` during auto search.
    compute_block_diagnostics
        Opt-in. When ``True``, after the final fit, extracts a per-block feature-importance vector
        (``feature_importances_`` or ``coef_``, when the fitted estimator exposes one) and the cosine
        similarity between each pair of horizon-adjacent blocks' importance vectors -- a high similarity
        between neighbouring blocks suggests they could share one model (block too fine); a low similarity
        within a block that spans genuinely distinct dynamics suggests it should be split (block too coarse).

    Attributes
    ----------
    block_models_
        One fitted clone per block, in ``horizon_blocks_`` order.
    horizon_blocks_
        The block partition actually used to fit -- either the caller-supplied ``horizon_blocks`` verbatim,
        or the winning candidate from auto search.
    block_search_report_
        Only set when auto search ran: list of ``{"block_size": int, "n_blocks": int, "cv_rmse": float}``,
        one entry per candidate, in ``block_size_grid`` order.
    block_feature_importances_, block_importance_similarity_
        Only set when ``compute_block_diagnostics=True``: per-block importance vectors (``None`` entries for
        estimators exposing neither ``feature_importances_`` nor ``coef_``) and the list of cosine
        similarities between consecutive blocks (length ``len(horizon_blocks_) - 1``).
    """

    def __init__(
        self,
        estimator_factory: Callable[[], Any],
        horizon_blocks: Optional[Sequence[Sequence[int]]] = None,
        *,
        auto_block_search: bool = False,
        block_size_grid: Optional[Sequence[int]] = None,
        cv_splits: int = 3,
        random_state: Optional[int] = None,
        compute_block_diagnostics: bool = False,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.horizon_blocks = horizon_blocks
        self.auto_block_search = auto_block_search
        self.block_size_grid = block_size_grid
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.compute_block_diagnostics = compute_block_diagnostics

    def _validate_blocks(self, blocks: Sequence[Sequence[int]], n_horizons: int) -> None:
        seen: set[int] = set()
        for block in blocks:
            for h in block:
                if h in seen:
                    raise ValueError(f"horizon index {h} appears in more than one block.")
                seen.add(h)
        if seen != set(range(n_horizons)):
            raise ValueError(f"horizon_blocks must partition range(n_horizons={n_horizons}) exactly; got indices {sorted(seen)}.")

    def _default_block_size_grid(self, n_horizons: int) -> list[int]:
        candidates = sorted({b for b in (1, 2, 3, 4, 6, 8, 12) if b <= n_horizons} | {n_horizons})
        return candidates

    def _fit_blocks(self, X: Any, Y_arr: np.ndarray, blocks: Sequence[Sequence[int]]) -> list[Any]:
        models: list[Any] = []
        for block in blocks:
            model = clone(self.estimator_factory())
            block_idx = list(block)
            Y_block = Y_arr[:, block_idx]
            model.fit(X, Y_block[:, 0] if Y_block.shape[1] == 1 else Y_block)
            models.append(model)
        return models

    def _predict_blocks(self, X: Any, blocks: Sequence[Sequence[int]], models: Sequence[Any], n_horizons: int) -> np.ndarray:
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        out = np.zeros((n, n_horizons), dtype=np.float64)
        for block, model in zip(blocks, models):
            block_idx = list(block)
            pred = np.asarray(model.predict(X), dtype=np.float64)
            if pred.ndim == 1:
                out[:, block_idx[0]] = pred
            else:
                out[:, block_idx] = pred
        return out

    def _cv_score_block_size(self, X: Any, Y_arr: np.ndarray, block_size: int) -> float:
        n_horizons = Y_arr.shape[1]
        blocks = _contiguous_blocks(n_horizons, block_size)
        n = Y_arr.shape[0]
        kf = KFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        has_iloc = hasattr(X, "iloc")
        fold_errors = []
        for train_idx, test_idx in kf.split(np.arange(n)):
            X_train = X.iloc[train_idx] if has_iloc else X[train_idx]
            X_test = X.iloc[test_idx] if has_iloc else X[test_idx]
            models = self._fit_blocks(X_train, Y_arr[train_idx], blocks)
            pred = self._predict_blocks(X_test, blocks, models, n_horizons)
            fold_errors.append((pred - Y_arr[test_idx]) ** 2)
        return float(np.sqrt(np.mean(np.concatenate([e.ravel() for e in fold_errors]))))

    def _search_horizon_blocks(self, X: Any, Y_arr: np.ndarray) -> list[list[int]]:
        n_horizons = Y_arr.shape[1]
        grid = list(self.block_size_grid) if self.block_size_grid is not None else self._default_block_size_grid(n_horizons)
        report = []
        best_block_size = grid[0]
        best_rmse = np.inf
        for block_size in grid:
            rmse = self._cv_score_block_size(X, Y_arr, block_size)
            blocks = _contiguous_blocks(n_horizons, block_size)
            report.append({"block_size": block_size, "n_blocks": len(blocks), "cv_rmse": rmse})
            if rmse < best_rmse:
                best_rmse = rmse
                best_block_size = block_size
        self.block_search_report_ = report
        return _contiguous_blocks(n_horizons, best_block_size)

    def _compute_block_diagnostics(self, n_horizons: int) -> None:
        importances: list[Optional[np.ndarray]] = []
        for model in self.block_models_:
            vec: Optional[np.ndarray] = None
            if hasattr(model, "feature_importances_"):
                vec = np.asarray(model.feature_importances_, dtype=np.float64).ravel()
            elif hasattr(model, "coef_"):
                coef = np.asarray(model.coef_, dtype=np.float64)
                vec = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
            importances.append(vec)
        self.block_feature_importances_ = importances

        similarities: list[Optional[float]] = []
        for a, b in zip(importances, importances[1:]):
            if a is None or b is None or a.shape != b.shape:
                similarities.append(None)
                continue
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            similarities.append(float(np.dot(a, b) / denom) if denom > 0 else None)
        self.block_importance_similarity_ = similarities

    def fit(self, X: Any, Y: np.ndarray) -> "DirectMultiHorizonEnsemble":
        """``Y``: ``(n, n_horizons)`` -- one column per forecast step, all measured from the SAME origin t=0
        (never a previous block's prediction)."""
        Y_arr = np.asarray(Y, dtype=np.float64)
        if Y_arr.ndim != 2:
            raise ValueError(f"Y must be 2-D (n, n_horizons), got shape {Y_arr.shape}")

        if self.horizon_blocks is not None:
            self._validate_blocks(self.horizon_blocks, Y_arr.shape[1])
            self.horizon_blocks_: list[list[int]] = [list(block) for block in self.horizon_blocks]
        elif self.auto_block_search:
            self.horizon_blocks_ = self._search_horizon_blocks(X, Y_arr)
        else:
            raise ValueError("horizon_blocks must be given, or auto_block_search=True to discover it via CV grid search.")

        self.block_models_ = self._fit_blocks(X, Y_arr, self.horizon_blocks_)
        self._n_horizons_ = Y_arr.shape[1]

        if self.compute_block_diagnostics:
            self._compute_block_diagnostics(Y_arr.shape[1])

        return self

    def predict(self, X: Any) -> np.ndarray:
        return self._predict_blocks(X, self.horizon_blocks_, self.block_models_, self._n_horizons_)


__all__ = ["DirectMultiHorizonEnsemble"]
