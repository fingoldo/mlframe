"""Small stage helpers for ``score_ensemble``: the gate-state bundle and the joblib worker-count resolver."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pyutilz.parallel import cpu_count_physical
from pyutilz.pythonlib import is_jupyter_notebook


@dataclass
class _GateState:
    """Mutable bundle of the member lists / gate thresholds threaded through the score_ensemble quality-gate stages."""

    members: Any
    gate_preds: Any
    source_split: Any
    member_tags: Any
    short_tags: Any
    ensemble_name: str
    max_mae: float
    max_std: float
    max_mae_relative: float
    max_std_relative: float
    gate_stats: Any = None
    early_return: bool = False


def _resolve_effective_n_jobs(first_pred: Any, n_methods: int, n_jobs: Optional[int], min_samples_for_parallel: int) -> int:
    """Pick the joblib worker count: explicit ``n_jobs`` wins, else parallel only for large samples outside notebooks."""
    # Determine sample count for parallelization decision
    if first_pred.val_probs is not None:
        n_samples = len(first_pred.val_probs)
    elif first_pred.val_preds is not None:
        n_samples = len(first_pred.val_preds)
    else:
        n_samples = 0

    # Determine n_jobs if not specified
    effective_n_jobs = n_jobs
    if effective_n_jobs is None:
        if n_samples >= min_samples_for_parallel and not is_jupyter_notebook():
            effective_n_jobs = min(n_methods, cpu_count_physical())
        else:
            effective_n_jobs = 1
    return int(effective_n_jobs)
