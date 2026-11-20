"""Host or device for the scoring of one generation of the pair-polynomial search, chosen per call from the kernel-tuning entry ``polynom_generation_backend``.

``optimizer="cupy_kernel"`` (the default of the hermite / orthogonal pair search) scores a generation of coefficient candidates with the device pipeline of ``_cupy_polynom_optimizer``
(two GEMMs, the fused ``bf_apply`` kernel, the radix-edge binner and the tile MI kernel); ``random_batch`` scores the same generation on the host (``_eval_coef_pair_batch``: BLAS products,
numba binary functions, the numba batched plug-in MI). The device pays a fixed upload and launch cost and the host does not, so the winner depends on the row count; the sweep times one
generation of 100 candidates on both paths over the swept row counts and requires the same candidates at the top of both rankings (``selection_view``: the two bin differently by tied values, so
the scores differ in the last digits while the selection is the same). A device fault or an unusable device keeps the host path.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, cast

import numpy as np

from mlframe.utils.log_throttle import log_throttle

from ._ktc_fallback import host_device_sweep, rows_fallback, selection_view

logger = logging.getLogger(__name__)

__all__ = ["polynom_device_chosen"]

DEVICE_FALLBACK_ROWS = 20_000  # rows from which the device scores a generation before a sweep has run
_SWEEP_ROWS = [5_000, 20_000, 100_000]
_SALT = 1
_BF_NAMES = ["mul", "add", "sub", "div", "atan2", "logabs"]
_DEGREE = 3
_CANDIDATES = 100
_BINS = 20


def _sweep_inputs(dims: dict) -> tuple:
    """``dims["n_samples"]`` rows: two standard-normal operands, a binary target depending on their product, 100 coefficient candidates per side (degree 3, four coefficients)."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    za = rng.standard_normal(n).astype(np.float32)
    zb = rng.standard_normal(n).astype(np.float32)
    y = ((za * zb + 0.3 * rng.standard_normal(n)) > 0).astype(np.int64)
    ca = rng.uniform(-2.0, 2.0, (_CANDIDATES, _DEGREE + 1))
    cb = rng.uniform(-2.0, 2.0, (_CANDIDATES, _DEGREE + 1))
    return (za, zb, y, ca, cb)


def _sweep_host(za: np.ndarray, zb: np.ndarray, y: np.ndarray, ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """The host path: ``_eval_coef_pair_batch`` over the Chebyshev basis matrices, as the ``random_batch`` search scores a generation."""
    from numpy.polynomial import chebyshev

    from ._hermite_fe_optimise import _eval_coef_pair_batch
    from .hermite_fe.shared import DEFAULT_BIN_FUNCS as _DEFAULT_BIN_FUNCS

    z_a, z_b = za.astype(np.float64), zb.astype(np.float64)  # Intentional float64: the host evaluator works in double
    B_a = chebyshev.chebvander(z_a, _DEGREE)
    B_b = chebyshev.chebvander(z_b, _DEGREE)
    scores, _raws, _idxs = _eval_coef_pair_batch(
        ca, cb, z_a=z_a, z_b=z_b, eval_func=lambda z, c: chebyshev.chebval(z, c), bf_callables=[cast(Callable[..., Any], _DEFAULT_BIN_FUNCS[b]) for b in _BF_NAMES], bf_names=_BF_NAMES,
        y=y, y_njit=y, mi_estimator="plugin", plugin_n_bins=_BINS, n_neighbors=3, discrete_target=True, l2_penalty=0.01, B_a=B_a, B_b=B_b,
    )
    return selection_view(scores)


def _sweep_device(za: np.ndarray, zb: np.ndarray, y: np.ndarray, ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """The device path: basis matrices, then ``_score_generation_gpu``; a failed launch raises, so the sweep never records a host fallback as a device time."""
    import cupy as cp

    from ._cupy_polynom_optimizer import _BASIS_IDS, _basis_matrix_gpu, _score_generation_gpu

    xa, xb = cp.asarray(za), cp.asarray(zb)
    Ba = _basis_matrix_gpu(cp, xa, _DEGREE, _BASIS_IDS["chebyshev"])
    Bb = _basis_matrix_gpu(cp, xb, _DEGREE, _BASIS_IDS["chebyshev"])
    y_codes = cp.asarray(y - int(y.min()))
    scores, _raws, _bfs = _score_generation_gpu(
        cp, Ba, Bb, cp.asarray(ca.astype(np.float32)), cp.asarray(cb.astype(np.float32)), y_codes, int(y.max() - y.min()) + 1, _BINS, 0.01, False, _BF_NAMES,
    )
    return selection_view(scores)


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="polynom_generation_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-9, atol=1e-12),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=rows_fallback(__name__),
        gpu_capable=True,
        salt=_SALT,
        cli_label="polynom_generation_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the floor decides
    logger.warning("kernel_tuner spec build for polynom_generation_backend failed: %s", e)
    _SPEC = None


def polynom_device_chosen(n_rows: int) -> bool:
    """Whether a generation of ``n_rows`` rows is scored on the device: the per-host cache by the row count, else the ``DEVICE_FALLBACK_ROWS`` floor."""
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        log_throttle(logger, "_polynom_search_backend.polynom_device_chosen", logging.WARNING, "polynom_generation_backend: KTC choose() failed, using the row floor of %d: %s", DEVICE_FALLBACK_ROWS, exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
