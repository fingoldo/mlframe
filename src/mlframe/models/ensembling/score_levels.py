"""The per-level ensembling loop of ``score_ensemble``: build every requested flavour for each ensembling level.

Carved out of ``score.py`` (which the split tests cap at 500 lines). ``process_fn`` and ``parallel_run_fn`` are
injected by the caller rather than imported here, so a caller or test that patches
``mlframe.models.ensembling.score._process_single_ensemble_method`` / ``.parallel_run`` still controls what runs.
"""

from __future__ import annotations

import logging
from typing import Callable

from joblib import delayed

from mlframe.system import callable_looks_gpu_bound
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger("mlframe.models.ensembling")  # same logger as score.py: callers filter log records by name


def run_ensembling_levels(
    *,
    res: dict,
    level_models_and_predictions: list,
    ensembling_methods: list,
    max_ensembling_level: int,
    effective_n_jobs: int,
    base_params: dict,
    process_fn: Callable,
    parallel_run_fn: Callable,
) -> None:
    """Fill ``res`` with one entry per (level, flavour) (plus ``"<flavour> conf"`` entries), feeding each level's
    ensembles to the next as members. ``base_params`` are the level-invariant kwargs of ``process_fn``; a fallback
    to sequential processing (unpicklable or GPU-bound custom metrics) sticks for the remaining levels."""
    custom_ice_metric = base_params.get("custom_ice_metric")
    custom_rice_metric = base_params.get("custom_rice_metric")
    for ensembling_level in range(max_ensembling_level):

        next_level_models_and_predictions = []

        # Common parameters for all ensemble methods
        common_params = dict(base_params, level_models_and_predictions=level_models_and_predictions, ensembling_level=ensembling_level)

        if len(ensembling_methods) > 1 and effective_n_jobs > 1:
            # loky pickles kwargs across worker boundaries; closure-captured metrics/lambdas
            # blow up in workers. Pre-check so we can fall back to sequential with a clear warning.
            try:
                import pickle  # nosec B403 - pickle used only for trusted same-process/dev-local round-trips, see call sites in this file

                pickle.dumps((custom_ice_metric, custom_rice_metric, base_params.get("kwargs")))
            except (pickle.PicklingError, AttributeError, TypeError) as exc:
                log_throttle(
                    logger,
                    "ensembling_fallback_sequential_unpicklable",
                    logging.WARNING,
                    "ensembling: falling back to sequential -- one of " "custom_ice_metric / custom_rice_metric / kwargs is not picklable: %s",
                    exc,
                )
                effective_n_jobs = 1

        if len(ensembling_methods) > 1 and effective_n_jobs > 1:
            # Each loky worker is a separate process; if custom_ice_metric/custom_rice_metric is GPU-bound
            # (torch/cupy), every worker independently contends for the single physical GPU device instead of
            # running in parallel -- process isolation prevents corruption but the run gets slower than serial.
            if callable_looks_gpu_bound(custom_ice_metric) or callable_looks_gpu_bound(custom_rice_metric):
                log_throttle(
                    logger,
                    "ensembling_fallback_sequential_gpu_bound",
                    logging.WARNING,
                    "ensembling: falling back to sequential -- custom_ice_metric / custom_rice_metric looks "
                    "GPU-bound (torch/cupy reference detected); process-pool fan-out would contend for the "
                    "single GPU device across workers instead of parallelising.",
                )
                effective_n_jobs = 1

        if len(ensembling_methods) > 1 and effective_n_jobs > 1:
            # Parallel processing -- loky + tiny max_nbytes keeps arrays in-memory (no spill) per pre-existing tuning
            results = parallel_run_fn(
                [delayed(process_fn)(ensemble_method=method, **common_params) for method in ensembling_methods],
                n_jobs=effective_n_jobs,
                backend="loky",
                max_nbytes="1K",
                verbose=0,
            )
            for internal_method, next_ens_results, conf_results in results:
                res[internal_method] = next_ens_results
                next_level_models_and_predictions.append(next_ens_results)
                if conf_results is not None:
                    res[internal_method + " conf"] = conf_results
        else:
            # Sequential processing
            for ensemble_method in ensembling_methods:
                internal_method, next_ens_results, conf_results = process_fn(
                    ensemble_method=ensemble_method, **common_params  # type: ignore[arg-type]  # common_params is a heterogeneous dict(...) blob shared with the parallel loky path
                )
                res[internal_method] = next_ens_results
                next_level_models_and_predictions.append(next_ens_results)
                if conf_results is not None:
                    res[internal_method + " conf"] = conf_results

        level_models_and_predictions = next_level_models_and_predictions
