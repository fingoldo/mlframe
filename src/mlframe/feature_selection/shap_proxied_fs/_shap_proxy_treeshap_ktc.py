"""Kernel-tuning entry of the TreeSHAP backend choice (``shap_proxy_treeshap_backend``): the numba table kernel against the device table kernel (``_shap_proxy_treeshap_table_gpu``, reached through ``treeshap_phi_base_gpu``).

The device kernel pays a fixed upload and table-build cost and the host kernel scales with ``rows x leaves``, so the crossover depends on both the row count and the ensemble size. The sweep times the
two over a (rows, trees) grid on small XGBoost ensembles of depth 5; the measured crossover on the development host (GTX 1050 Ti, 4 cores / 8 threads) is in ``_FALLBACK_MIN_WORK``, which decides until
the sweep has run on the host.
"""

from __future__ import annotations

import threading

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_SWEEP_ROWS = [100, 400, 1_500, 6_000, 25_000]
_SWEEP_TREES = [20, 100]
_DEPTH = 5
_FEATURES = 40
_SALT = 1
# rows x leaves at which the device kernel starts to win on the development host: (20 trees, depth 4) breaks even near 3 000 rows, (100 trees, depth 6) near 700 rows
_FALLBACK_MIN_WORK = 2_500_000

_ENSEMBLES: dict = {}
_ENSEMBLES_LOCK = threading.RLock()  # serialises the get-or-build of the cache above (one build per key even with concurrent fits)


def _ensemble(trees: int):
    """A flat ensemble of ``trees`` depth-``_DEPTH`` trees (built once per size)."""
    with _ENSEMBLES_LOCK:
        ens = _ENSEMBLES.get(trees)
        if ens is None:
            import xgboost as xgb

            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble

            rng = np.random.default_rng(0)
            X = rng.standard_normal((4_000, _FEATURES)).astype(np.float32)
            y = 1e-3 * (X[:, 0] * 2 + X[:, 1] * X[:, 2] + np.sin(X[:, 3]))  # leaf values of order 1e-3: the float32 kernel's absolute error (about 1e-5 for order-1 leaves) is then ~1e-8, inside the shared 1e-6 sweep tolerance cap, while a wrong kernel is off by the size of phi (~1e-4)
            ens = _ENSEMBLES[trees] = extract_ensemble(xgb.XGBRegressor(n_estimators=trees, max_depth=_DEPTH, tree_method="hist", verbosity=0, n_jobs=2).fit(X, y))
        return ens


def _sweep_inputs(dims: dict) -> tuple:
    """``(ensemble, X)`` of ``dims["trees"]`` trees and ``dims["n_rows"]`` rows."""
    rng = np.random.default_rng(1)
    return (_ensemble(int(dims["trees"])), rng.standard_normal((int(dims["n_rows"]), _FEATURES)).astype(np.float32))


def _numba(ens, X) -> np.ndarray:
    """The host table kernel."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import treeshap_phi_base_numba

    return treeshap_phi_base_numba(ens, X)[0]


def _gpu(ens, X) -> np.ndarray:
    """The device table kernel (synchronised by the download)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_gpu import treeshap_phi_base_gpu

    return treeshap_phi_base_gpu(ens, X)[0]


def numba_gpu_sweep(numba: Any, gpu: Any, axes: dict, inputs: Any) -> Any:
    """The tuner of a numba-versus-device TreeSHAP kernel pair: times both over ``axes`` on ``inputs(dims)`` and requires the device result to agree with the numba one to 1e-6 (the device
    kernels are float32 and the sweep models are scaled so that their error sits under that cap)."""

    def _sweep() -> list:
        """Time the two kernels per point of the axes."""
        from pyutilz.dev.benchmarking import sweep_backend_grid

        return list(sweep_backend_grid({"numba": numba, "gpu": gpu}, axes, inputs, reference="numba", repeats=3, equiv_rtol=1e-6, equiv_atol=1e-6))

    return _sweep


_sweep = numba_gpu_sweep(_numba, _gpu, {"n_rows": _SWEEP_ROWS, "trees": _SWEEP_TREES}, _sweep_inputs)  # per (rows, trees)


def _fallback(n_rows: int, trees: int) -> str:
    """Before any sweep: the measured work mark (rows x trees x 2^depth leaves)."""
    return "gpu" if int(n_rows) * int(trees) * (1 << _DEPTH) >= _FALLBACK_MIN_WORK else "numba"


def _register() -> Any:
    """Register the ``kernel_tuner`` entry; ``None`` when the registry is unavailable."""
    try:
        from pyutilz.performance.kernel_tuning.registry import kernel_tuner

        return kernel_tuner(
            kernel_name="shap_proxy_treeshap_backend",
            variant_fns=(),
            tuner=_sweep,
            axes={"n_rows": list(_SWEEP_ROWS), "trees": list(_SWEEP_TREES)},
            fallback=_fallback,
            gpu_capable=True,
            salt=_SALT,
            cli_label="shap_proxy_treeshap_backend",
        )
    except Exception as e:  # best-effort: without a tuner entry the work mark decides
        logger.warning("kernel_tuner registration of shap_proxy_treeshap_backend failed: %s", e)
        return None


_SPEC: Any = _register()


def use_gpu(n_rows: int, trees: int) -> bool:
    """Whether ``n_rows`` rows of an ensemble of ``trees`` trees go to the device kernel: the per-host tuning cache, else the work mark."""
    if _SPEC is None:
        return _fallback(n_rows, trees) == "gpu"
    try:
        return bool(str(_SPEC.choose(n_rows=int(n_rows), trees=int(trees))) == "gpu")
    except Exception as exc:
        logger.debug("shap_proxy_treeshap_backend: KTC choose() failed (%s); work mark", exc)
        return _fallback(n_rows, trees) == "gpu"
