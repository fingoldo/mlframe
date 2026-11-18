"""Kernel-tuning entry of the TreeSHAP interaction backend choice (``shap_proxy_treeshap_interactions_backend``): the numba pair-table kernel against the device pair-table kernel.

The cost of both grows with ``rows x features^2 x trees`` but the device kernel pays a fixed upload and table-build cost, so the crossover depends on the size of the tensor (``cells = rows x features^2``) and of the ensemble.
The sweep times the two over a (cells, trees) grid on small XGBoost ensembles of depth 4 over 30 features; the measured floor ``_FALLBACK_MIN_CELLS`` decides until the sweep has run on the host. The device
kernel module is ``_shap_proxy_treeshap_interactions_table_gpu`` (when the ensemble does not fit its tables the conditioned-scan fallback ``_shap_proxy_treeshap_interactions_gpu`` answers), the host twin ``_shap_proxy_treeshap_interactions_table``.
"""

from __future__ import annotations

import threading

import logging
from typing import Any

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_ktc import make_sweep_ensemble, numba_gpu_sweep

logger = logging.getLogger(__name__)

_FEATURES = 30
_DEPTH = 4
_SWEEP_CELLS = [_FEATURES * _FEATURES * rows for rows in (200, 1_000, 4_000)]
_SWEEP_TREES = [50, 200]
_SALT = 1
_FALLBACK_MIN_CELLS = 2_000_000  # between the measured points: the host kernel wins at 900 000 cells (1000 rows x 30 features), the device kernel at 3 600 000 (4000 rows); the old hand floor was 4 800 000

_ENSEMBLES: dict = {}
_ENSEMBLES_LOCK = threading.RLock()  # serialises the get-or-build of the cache above (one build per key even with concurrent fits)


_ensemble = make_sweep_ensemble(_ENSEMBLES, _ENSEMBLES_LOCK, _FEATURES, _DEPTH, 3_000)  # a flat ensemble of ``trees`` depth-``_DEPTH`` trees over ``_FEATURES`` features, built once per size


def _sweep_inputs(dims: dict) -> tuple:
    """``(ensemble, X)`` of ``dims["trees"]`` trees and ``dims["cells"] / features^2`` rows."""
    rng = np.random.default_rng(1)
    rows = max(1, int(dims["cells"]) // (_FEATURES * _FEATURES))
    return (_ensemble(int(dims["trees"])), rng.standard_normal((rows, _FEATURES)).astype(np.float32))


def _numba(ens, X) -> np.ndarray:
    """The host pair-table kernel."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions_table import interaction_tensor_table

    out = interaction_tensor_table(ens, X)
    assert out is not None
    return np.asarray(out[0])


def _gpu(ens, X) -> np.ndarray:
    """The device pair-table kernel (synchronised by the download)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions_table_gpu import interaction_tensor_table_gpu

    out = interaction_tensor_table_gpu(ens, X)
    assert out is not None
    return np.asarray(out[0])


_sweep = numba_gpu_sweep(_numba, _gpu, {"cells": _SWEEP_CELLS, "trees": _SWEEP_TREES}, _sweep_inputs)  # per (cells, trees)


def _fallback(cells: int, trees: int) -> str:
    """Before any sweep: the measured cell floor."""
    return "gpu" if int(cells) >= _FALLBACK_MIN_CELLS else "numba"


def _register() -> Any:
    """Register the ``kernel_tuner`` entry; ``None`` when the registry is unavailable."""
    try:
        from pyutilz.performance.kernel_tuning.registry import kernel_tuner

        return kernel_tuner(
            kernel_name="shap_proxy_treeshap_interactions_backend",
            variant_fns=(),
            tuner=_sweep,
            axes={"cells": list(_SWEEP_CELLS), "trees": list(_SWEEP_TREES)},
            fallback=_fallback,
            gpu_capable=True,
            salt=_SALT,
            cli_label="shap_proxy_treeshap_interactions_backend",
        )
    except Exception as e:  # best-effort: without a tuner entry the cell floor decides
        logger.warning("kernel_tuner registration of shap_proxy_treeshap_interactions_backend failed: %s", e)
        return None


_SPEC: Any = _register()


def use_gpu(cells: int, trees: int) -> bool:
    """Whether an interaction tensor of ``cells`` = rows x features^2 cells from ``trees`` trees goes to the device kernel: the per-host tuning cache, else the cell floor."""
    if _SPEC is None:
        return _fallback(cells, trees) == "gpu"
    try:
        return bool(str(_SPEC.choose(cells=int(cells), trees=int(trees))) == "gpu")
    except Exception as exc:
        logger.debug("shap_proxy_treeshap_interactions_backend: KTC choose() failed (%s); cell floor", exc)
        return _fallback(cells, trees) == "gpu"
