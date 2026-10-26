"""Host (numpy) batched-Newton cross-validated logloss of the usability greedy's shortlist, binary targets.

``usability_greedy`` scores every shortlisted candidate of a step by refitting ``StandardScaler -> LogisticRegression`` on each of the folds, one sklearn fit per (candidate, fold): a step over a shortlist of 40
with 4 folds is 160 lbfgs fits that dominate a small-n MRMR fit. The fits are independent and the objective is strictly convex (L2 penalty, unpenalised intercept), so its optimum is unique; the device twin
(``_usability_greedy_clf_batched``) already finds it for all candidates of a fold at once with a batched Newton iteration. That module takes the array namespace as an argument, so the same math runs here on numpy.

A fold whose Newton fit is singular or does not converge makes the whole call raise ``HostBatchedFallbackError``, and the caller keeps the exact per-candidate sklearn refit, so the fast path never decides a selection
the exact path would not.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ._usability_greedy_clf_batched import cv_candidates_binary
from ._usability_greedy_clf_gpu_resident import _C, _LOGLOSS_EPS, _NEWTON_MAX_ITER, _NEWTON_TOL


class HostBatchedFallbackError(Exception):
    """A fold fit was singular or did not converge: the caller falls back to the exact per-candidate refit."""


class _NumpyNamespace:
    """numpy with the one cupy-only call the batched module makes (``asnumpy``)."""

    @staticmethod
    def asnumpy(a: Any) -> np.ndarray:
        """Identity on a host array."""
        return np.asarray(a)

    def __getattr__(self, name: str) -> Any:
        """Everything else is numpy's."""
        return getattr(np, name)


_NP = _NumpyNamespace()


def cv_candidates_binary_host(pv: Callable[[int], np.ndarray], selected: "list[int]", cand_idx: "list[int]", folds: np.ndarray, n_folds: int, y_enc: np.ndarray) -> "dict[int, np.ndarray]":
    """``{candidate: per-fold validation logloss}`` of ``selected + [candidate]`` for every candidate, from one batched Newton fit per fold.

    ``pv(i)`` returns the values of pool column ``i``; ``folds`` holds the fold id of every row; ``y_enc`` the 0/1 labels. Raises ``HostBatchedFallbackError`` when a fit is singular or does not converge."""
    cols = [*selected, *cand_idx]
    # Intentional float64: the exact path fits in double and the Newton stop (1e-9 on the weights) needs it
    V = np.column_stack([np.asarray(pv(i), dtype=np.float64) for i in cols])
    sel_local = list(range(len(selected)))
    cand_local = list(range(len(selected), len(cols)))
    tr_masks = [folds != fo for fo in range(n_folds)]
    va_masks = [folds == fo for fo in range(n_folds)]
    specs = [(tr_masks[fo], va_masks[fo], bool(np.unique(y_enc[tr_masks[fo]]).size >= 2)) for fo in range(n_folds)]
    # Intentional float64: the design above is double, so the fit and its logloss accumulator are double too
    fit_dtype = np.float64
    out = cv_candidates_binary(
        _NP, V, sel_local, cand_local, specs, y_enc, dtype=fit_dtype, newton_max_iter=_NEWTON_MAX_ITER, tol=_NEWTON_TOL, lam=1.0 / _C, logloss_eps=_LOGLOSS_EPS, fallback_error=HostBatchedFallbackError,
    )
    return {i: out[j] for j, i in enumerate(cand_idx)}


__all__ = ["HostBatchedFallbackError", "cv_candidates_binary_host"]
