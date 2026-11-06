"""Start the surrogate-GBM interaction seeder's LightGBM fits at the top of the first FE step, beside the screening, instead of inside the operand-pool build.

The ten shallow LightGBM fits of ``surrogate_gbm_interaction_seeds`` take about 13 s of a 100000 x 100 fit and depend only on the discretised matrix, the target
codes and the raw numeric candidate columns, all known when the first FE step starts. ``apply_surrogate_gbm_seeder`` used to run them after the step's screening;
the fits now start on a worker thread at the step's entry and the seeder takes their result when its own arguments are the ones the prefetch used (the same
call, so the same seeds). Measured: the join still waits about 11 s, so the overlap hides about 2.5 s of the fits.

The candidate list and the keyword arguments come from :func:`seeder_candidates` / :func:`seeder_kwargs`, shared by the prefetch and the seeder, so the two cannot
disagree. The pending call lives in a module map keyed by the estimator's id (not on the estimator: a pickled estimator must not carry a thread handle); any
mismatch, error or missing entry leaves the seeder to call the function itself. ``MLFRAME_GBM_SEEDER_PREFETCH=0`` turns the prefetch off.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_PENDING: "dict[int, tuple[Any, Any, Any, tuple, Future]]" = {}
_LOCK = threading.Lock()
_EXECUTOR: "Optional[ThreadPoolExecutor]" = None


def seeder_candidates(self: Any, cols: Sequence[str], target_indices: Any, categorical_vars: Any, non_numeric_idx: Any) -> "Optional[list[int]]":
    """The raw numeric columns the seeder scores (sorted indices into ``cols``), or ``None`` when there are fewer than ``fe_gbm_seeder_min_features``."""
    raw_names = set(getattr(self, "feature_names_in_", []))
    targets = {int(t) for t in np.atleast_1d(target_indices)}
    cats = set(categorical_vars)
    idx = {i for i, nm in enumerate(cols) if nm in raw_names and i not in targets and i not in cats}
    idx -= set(non_numeric_idx)
    if self.factors_to_use is not None:
        idx &= set(self.factors_to_use)
    if self.factors_names_to_use is not None:
        idx &= {cols.index(n) for n in self.factors_names_to_use if n in cols}
    if len(idx) < int(getattr(self, "fe_gbm_seeder_min_features", 30)):
        return None
    return sorted(idx)


def seeder_kwargs(self: Any) -> dict:
    """Keyword arguments of the seeder call, from the estimator's knobs."""
    return dict(
        is_classification=True,
        top_k_pairs=int(getattr(self, "fe_gbm_seeder_top_k_pairs", 12)),
        top_k_triples=int(getattr(self, "fe_gbm_seeder_top_k_triples", 8)),
        n_estimators=int(getattr(self, "fe_gbm_seeder_n_estimators", 150)),
        max_depth=int(getattr(self, "fe_gbm_seeder_max_depth", 4)),
        self_gate_margin=float(getattr(self, "fe_gbm_seeder_self_gate_margin", 0.0)),
        self_gate_reps=int(getattr(self, "fe_gbm_seeder_self_gate_reps", 3)),
        self_gate_min_z=float(getattr(self, "fe_gbm_seeder_self_gate_min_z", 2.0)),
        random_seed=int(getattr(self, "random_seed", 0) or 0),
    )


def _key(data: Any, classes_y: np.ndarray, cand: Sequence[int], kwargs: dict) -> tuple:
    """Identity of one seeder call: the matrix and target objects, the candidates and the arguments."""
    return (id(data), id(classes_y), tuple(cand), tuple(sorted(kwargs.items())))


def prefetch_surrogate_seeds(self: Any, *, num_fs_steps: int, data: Any, cols: Sequence[str], X: Any, target_indices: Any, categorical_vars: Any, classes_y: Any) -> None:
    """Start the seeder call of the first FE step on a worker thread when the seeder will run (enabled, first step, enough candidates)."""
    global _EXECUTOR
    import os

    if num_fs_steps != 0 or not bool(getattr(self, "fe_gbm_seeder_enable", False)) or os.environ.get("MLFRAME_GBM_SEEDER_PREFETCH", "1").strip() == "0":
        return
    try:
        from ._mrmr_fe_step._helpers import _non_numeric_column_indices
        from ._surrogate_interaction_seeder import surrogate_gbm_interaction_seeds

        cand = seeder_candidates(self, cols, target_indices, categorical_vars, _non_numeric_column_indices(X, cols))
        if cand is None:
            return
        y = np.ascontiguousarray(classes_y)
        kwargs = seeder_kwargs(self)
        with _LOCK:
            if _EXECUTOR is None:
                _EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gbm_seeder")
            _PENDING[id(self)] = (self, data, classes_y, _key(data, classes_y, cand, kwargs), _EXECUTOR.submit(surrogate_gbm_interaction_seeds, data, y, cand, **kwargs))
    except Exception as exc:
        logger.debug("surrogate seeder prefetch not started (%s); the seeder runs in place", exc)


def take_prefetched(self: Any, data: Any, classes_y: Any, cand: Sequence[int], kwargs: dict) -> Any:
    """The result of the prefetched call when it is the call the seeder is about to make, else ``None`` (the entry is dropped either way)."""
    with _LOCK:
        entry = _PENDING.pop(id(self), None)
    if entry is None:
        return None
    owner, owner_data, owner_y, key, fut = entry
    if owner is not self or owner_data is not data or owner_y is not classes_y or key != _key(data, classes_y, cand, kwargs):
        fut.cancel()
        return None
    try:
        return fut.result()
    except Exception as exc:
        logger.debug("surrogate seeder prefetch failed (%s); the seeder runs in place", exc)
        return None


def drop_prefetched(self: Any) -> None:
    """Forget a pending call of this estimator (the step ended without using it)."""
    with _LOCK:
        entry = _PENDING.get(id(self))
        if entry is None or entry[0] is not self:
            return
        del _PENDING[id(self)]
    entry[-1].cancel()


__all__ = ["drop_prefetched", "prefetch_surrogate_seeds", "seeder_candidates", "seeder_kwargs", "take_prefetched"]
