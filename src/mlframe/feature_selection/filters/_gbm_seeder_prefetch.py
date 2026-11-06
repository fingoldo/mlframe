"""Start the surrogate-GBM interaction seeder's LightGBM fits at the top of the first FE step, beside the screening, instead of inside the operand-pool build.

The ten shallow LightGBM fits of ``surrogate_gbm_interaction_seeds`` take about 10 to 13 s of a 100000 x 100 fit and depend only on the discretised matrix, the target
codes and the raw numeric candidate columns. ``apply_surrogate_gbm_seeder`` used to run them after the step's screening; the fits now start on a worker thread and the
seeder takes their result when its own arguments carry the same values as the ones the prefetch used (the same call, so the same seeds).

Where the start is: the step's entry comes after the first screening, and the seeder needs the result right there (measured at 100000 x 100: the prefetch call and
``take_prefetched`` in the same second, 54.8 s, the fits ending at 64.5 s with the main thread blocked and the device idle). ``prefetch_surrogate_seeds_early`` starts
them at the top of the first round, before the screening, so they run beside it (the cascades before that are not an option: the discretised matrix does not exist yet).
The match is made on the VALUES (a hash of the candidate columns and of the target codes), not on object identity, because the screening may hand the step a new
matrix object that still carries the same raw columns.

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
    """Identity of one seeder call by value: a hash of the candidate columns of the matrix and of the target codes, the candidates and the arguments."""
    from ._content_hasher import new_content_hasher

    h = new_content_hasher()
    h.update(np.ascontiguousarray(np.asarray(data)[:, list(cand)]).data)
    h.update(np.ascontiguousarray(classes_y).data)
    return (h.hexdigest(), tuple(cand), tuple(sorted(kwargs.items())))


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
        key = _key(data, classes_y, cand, kwargs)
        with _LOCK:
            pending = _PENDING.get(id(self))
            if pending is not None and pending[0] is self and pending[1] == key:
                return  # the early start (before the screening) is this very call
            if _EXECUTOR is None:
                _EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gbm_seeder")
            _PENDING[id(self)] = (self, key, _EXECUTOR.submit(surrogate_gbm_interaction_seeds, data, y, cand, **kwargs))
    except Exception as exc:
        logger.debug("surrogate seeder prefetch not started (%s); the seeder runs in place", exc)


def prefetch_surrogate_seeds_early(self: Any, *, data: Any, cols: Sequence[str], X: Any, target_indices: Any, categorical_vars: Any, nbins: Any, dtype: Any) -> None:
    """Start the seeder call of the first FE step before the first screening: the target codes are the ones the screening will compute (``merge_vars`` of the target
    columns of the matrix), the rest are the step's own arguments. A call the seeder does not make (different values by then) is simply not used."""
    if not bool(getattr(self, "fe_gbm_seeder_enable", False)):
        return
    try:
        from .info_theory import merge_vars

        classes_y, _freqs, _ = merge_vars(factors_data=data, vars_indices=target_indices, var_is_nominal=None, factors_nbins=nbins, dtype=dtype)
        prefetch_surrogate_seeds(self, num_fs_steps=0, data=data, cols=cols, X=X, target_indices=target_indices, categorical_vars=categorical_vars, classes_y=classes_y)
    except Exception as exc:
        logger.debug("early surrogate seeder prefetch not started (%s)", exc)


def take_prefetched(self: Any, data: Any, classes_y: Any, cand: Sequence[int], kwargs: dict) -> Any:
    """The result of the prefetched call when it is the call the seeder is about to make, else ``None`` (the entry is dropped either way)."""
    with _LOCK:
        entry = _PENDING.pop(id(self), None)
    if entry is None:
        return None
    owner, key, fut = entry
    if owner is not self or key != _key(data, classes_y, cand, kwargs):
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


__all__ = ["drop_prefetched", "prefetch_surrogate_seeds", "prefetch_surrogate_seeds_early", "seeder_candidates", "seeder_kwargs", "take_prefetched"]
