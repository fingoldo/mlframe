"""Score the tiny-model rerank's specs in worker PROCESSES, and one implementation shared by every execution mode.

A production Jupyter kernel died three times out of three with a heap corruption (0xc0000374) while the rerank ran 16
threads, each training LightGBM in the same process; the serial rerank passed. Serialising LightGBM's dataset
construction on both construction sites did not stop it. A worker process gives every spec's fits their own heap:
nothing native is shared between concurrently running fits, and if a native library does fault, the worker dies and
the kernel does not -- the remaining specs are then scored in the calling process, serially, which is the mode that
has been seen to survive.

The per-spec work is a plain function of a task dict (no ``self``), so the serial, threaded and process paths run the
same code. The transform and its fitted parameters travel as ``cloudpickle`` blobs: 11 of the 51 registered transforms
are closures that the standard pickle cannot serialise, and cloudpickle also carries a transform registered at runtime
into a child that never ran the registration. Large arrays stay plain ``ndarray`` fields, which joblib hands to the
workers as memory maps instead of copies.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

# Module level, not inside score_spec: a joblib-delayed callee's lazy ``from X import name`` races on a partially
# initialised module across threads and leaves the local name unbound.
from ._screening_tiny import _tiny_cv_rmse_y_scale_multiseed

logger = logging.getLogger(__name__)


def _cloudpickle() -> Any:
    """The cloudpickle module: the standalone package, which joblib>=1.6 depends on instead of vendoring it under
    ``joblib.externals`` (the old path raises ImportError there), else the copy older joblib releases vendor."""
    try:
        import cloudpickle
    except ImportError:
        from joblib.externals import cloudpickle
    return cloudpickle


def make_spec_task(spec: Any, transform: Any, common: dict, skip: bool, base_screen: Any, *, x_matrix: Any = None,
                   x_full: Any = None, drop_idx: Optional[list] = None) -> dict:
    """Everything one spec's scoring needs, detached from the discovery object so it can cross a process boundary.

    Give either the base's own ``x_matrix`` (in-process modes, from the bounded per-base cache) or ``x_full`` plus the
    ``drop_idx`` a worker process uses to gather that matrix itself.
    """
    cloudpickle = _cloudpickle()

    return {
        "name": spec.name,
        "skip": bool(skip),
        "transform_blob": cloudpickle.dumps(transform),
        "fitted_params_blob": cloudpickle.dumps(spec.fitted_params),
        "base_screen": base_screen,
        "x_matrix": x_matrix,
        "x_full": x_full,
        "drop_idx": list(drop_idx or []),
        **common,
    }


def make_worker_task(spec: Any, transform: Any, common: dict, skip: bool, per_base_cache: Any) -> dict:
    """A task for a worker PROCESS: the full screen matrix plus the columns to drop, never a pre-gathered base matrix."""
    base_screen, x_full, drop_idx = per_base_cache.worker_inputs(spec.base_column)
    return make_spec_task(spec, transform, common, skip, base_screen, x_full=x_full, drop_idx=drop_idx)


RERANK_BACKENDS = ("auto", "processes", "threads")


def rerank_backend(configured: Any) -> str:
    """``"processes"`` or ``"threads"`` for a parallel rerank; ``"auto"`` (and anything unrecognised) means processes.

    Threads are kept as an explicit choice: they share one process and start instantly, which is fine where the
    native libraries are stable under concurrency. The default is the mode that cannot take the kernel down.
    """
    value = str(configured or "auto").strip().lower()
    if value not in RERANK_BACKENDS:
        logger.warning("[CompositeTargetDiscovery] tiny_rerank_backend=%r is not one of %s; using processes.", configured, RERANK_BACKENDS)
        return "processes"
    return "threads" if value == "threads" else "processes"


def score_spec(task: dict) -> tuple:
    """``(name, family_rmses, per_seed_by_family, per_bin_first_or_none)`` for one spec, in whatever process runs it."""
    if task["skip"]:
        # Honest-OOF already measured this spec and will set its score; the CV fits would be discarded.
        return task["name"], {}, {}, None
    cloudpickle = _cloudpickle()

    transform = cloudpickle.loads(task["transform_blob"])
    fitted_params = cloudpickle.loads(task["fitted_params_blob"])
    x_matrix = task["x_matrix"]
    if x_matrix is None:
        x_full = task["x_full"]
        x_matrix = np.delete(x_full, task["drop_idx"], axis=1) if task["drop_idx"] else np.asarray(x_full)
    fam_rmses: dict[str, float] = {}
    per_seed_by_family: dict[str, np.ndarray] = {}
    per_bin_first: Optional[np.ndarray] = None
    families = task["families"]
    for family in families:
        # Per-bin is captured alongside the RMSE in the SAME pass, for the first family only.
        want_per_bin = bool(task["per_bin_enabled"] and family == families[0])
        kwargs = dict(
            y_train=task["y_screen"], base_train=task["base_screen"], transform=transform, fitted_params=fitted_params,
            x_train_matrix=x_matrix, family=family, n_estimators=task["n_estimators"], num_leaves=task["num_leaves"],
            learning_rate=task["learning_rate"], cv_folds=task["cv_folds"], n_jobs=task["fold_n_jobs"],
            deterministic=task["deterministic"], n_seed_repeats=task["n_seed_repeats"], base_random_state=task["random_state"],
            inner_n_jobs=task["inner_n_jobs"], return_per_bin=want_per_bin, n_bins=task["per_bin_n_bins"],
            time_aware=task["time_aware"], groups=task["groups"], cv_selector_mode=task["cv_selector_mode"],
            cv_selector_alpha=task["cv_selector_alpha"], cv_selector_confidence=task["cv_selector_confidence"],
            cv_selector_quantile_level=task["cv_selector_quantile_level"],
        )
        if task["use_wilcoxon"]:
            result = _tiny_cv_rmse_y_scale_multiseed(return_per_seed=True, **kwargs)
            rmse, per_seed = result[0], result[-1]
            if want_per_bin:
                per_bin_first = result[1]
            per_seed_by_family[family] = per_seed
        else:
            result = _tiny_cv_rmse_y_scale_multiseed(seed_early_stop_threshold=task["early_stop_threshold"], **kwargs)
            if want_per_bin and isinstance(result, tuple):
                rmse, per_bin_first = result[0], result[1]
            else:
                rmse = result
        fam_rmses[family] = rmse
    return task["name"], fam_rmses, per_seed_by_family, per_bin_first


def score_specs_in_processes(tasks: list[dict], n_jobs: int) -> list[tuple]:
    """Score ``tasks`` in ``n_jobs`` worker processes, in task order; a dead worker falls back to scoring in this process.

    The fallback reruns every task serially rather than only the unfinished ones: joblib does not return partial
    results, and the serial path is deterministic, so the scores are the same ones the workers would have produced.
    """
    from joblib import Parallel, delayed

    try:
        return list(Parallel(n_jobs=n_jobs, backend="loky")(delayed(score_spec)(t) for t in tasks))
    except Exception as exc:
        from joblib.externals.loky.process_executor import BrokenProcessPool

        # BrokenProcessPool covers a worker that died (TerminatedWorkerError is its subclass) and a task a worker could not
        # un-serialize, which a paging-file shortage on Windows produces; either way the pool is gone and the specs are not.
        if not isinstance(exc, BrokenProcessPool):
            raise
        logger.warning(
            "[CompositeTargetDiscovery] the rerank worker pool broke (%s). The kernel is unaffected; the %d spec(s) of this "
            "rerank are rescored serially in this process, the mode that survived where 16 threads in one process did not.",
            str(exc).splitlines()[0][:200], len(tasks),
        )
        return [score_spec(t) for t in tasks]
