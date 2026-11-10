"""Device scoring of the MI-greedy candidates: build every ``transform(source columns)`` candidate on the GPU and score its plug-in MI there, so only the winners come back to the host.

The host path (``_mi_greedy_fe.generate_mi_greedy_features``) materialises each candidate as a numpy column (about 0.8 ms each, 1084 of them at five seed columns), stacks them in a DataFrame and scores
the DataFrame; of that frame only ``top_k`` columns survive. Here the (at most ten) distinct source columns are uploaded once, every candidate is one fused expression of resident arrays written into a
column block of the candidate matrix, the block goes straight to the resident plug-in MI batch (``_plugin_mi_classif_batch_cuda_resident``, no transfer), and the winners are materialised on the host
by the unchanged ``_materialise_candidate`` (so the engineered columns, the recipes and ``transform`` replay are the host's, bit for bit).

The device transforms are the host transforms written with cupy at the criterion dtype (``{REAL}``): the same formulas, the same NaN / Inf scrub (``nan_to_num``) and the same constant-column drop
(``std <= 1e-12``). Elementwise transcendentals (``log1p``, ``tanh``, ``sin``, ``cos``, ``expm1``) can differ from numpy's in the last float32 bit; the MI is computed on quantile codes, so the
only effect is a boundary row moving to the neighbouring bin, which is selection-equivalent, not bit-identical (``tests/feature_selection/fe/test_mi_greedy_device_matches_host.py``).

Columns go to the MI kernel in blocks that fit ``_BUDGET_FRACTION`` of the free device memory; any device fault returns ``None`` and the caller runs the host path. The host/device choice is the
kernel-tuning cache entry ``fe_mi_greedy_candidates_device`` (fallback: the host).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ._device_sort_safe import device_room
from ._fe_usability_signal import _crit_np_dtype
from ._ktc_fallback import device_from_rows
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

_BUDGET_FRACTION = 0.25
DEVICE_FALLBACK_ROWS = 10**9  # the tuner decides; until a sweep says the device wins the host path runs
_EPS = 1e-12


def _device_unary(cp: Any) -> dict:
    """The unary transforms of ``_mi_greedy_fe.UNARY_TRANSFORMS`` / ``TRIG_BOUNDED_TRANSFORMS`` on cupy arrays of the criterion dtype."""

    def reciprocal(a: Any) -> Any:
        """``1 / (x + eps * sign)`` with ``sign(0)`` taken as +1."""
        sgn = cp.sign(a)
        sgn = cp.where(sgn == 0.0, 1.0, sgn).astype(a.dtype)
        return 1.0 / (a + _EPS * sgn)

    return {
        "log_abs": lambda a: cp.log1p(cp.abs(a)),
        "sqrt_abs": lambda a: cp.sqrt(cp.abs(a)),
        "square": lambda a: a * a,
        "cube": lambda a: a * a * a,
        "reciprocal_safe": reciprocal,
        "tanh": cp.tanh,
        "expm1_clip": lambda a: cp.expm1(cp.clip(a, -20.0, 20.0)),
        "abs": cp.abs,
        "sin": cp.sin,
        "cos": cp.cos,
    }


def _device_binary(cp: Any) -> dict:
    """The binary transforms of ``_mi_greedy_fe.BINARY_TRANSFORMS`` on cupy arrays of the criterion dtype."""

    def div_safe(a: Any, b: Any) -> Any:
        """``a / (b + eps * sign(b))`` with ``sign(0)`` taken as +1."""
        sgn = cp.sign(b)
        sgn = cp.where(sgn == 0.0, 1.0, sgn).astype(b.dtype)
        return a / (b + _EPS * sgn)

    return {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": div_safe,
        "max": cp.maximum,
        "min": cp.minimum,
        "abs_diff": lambda a, b: cp.abs(a - b),
        "ratio_log": lambda a, b: cp.log1p(cp.abs(a)) - cp.log1p(cp.abs(b)),
    }


def _named_transform(src: tuple, tname: str, unary: dict, binary: dict) -> "Optional[tuple[str, Any]]":
    """``(engineered name, device function)`` of the candidate ``tname(src)``, or ``None`` for an unknown transform or an unsupported arity."""
    from ._mi_greedy_fe import engineered_name_binary, engineered_name_unary

    if len(src) == 1:
        fn = unary.get(tname)
        return None if fn is None else (engineered_name_unary(src[0], tname), fn)
    if len(src) == 2:
        fn = binary.get(tname)
        return None if fn is None else (engineered_name_binary(src[0], src[1], tname), fn)
    return None


def score_candidates_device(
    X: pd.DataFrame, candidates: Sequence[tuple], y_codes: np.ndarray, nbins: int
) -> "Optional[tuple[list[str], list[tuple], np.ndarray]]":
    """``(names, parsed, mi)`` of the candidates the host would keep (distinct names, non-constant), scored on the device; ``None`` when the device cannot take it.

    ``names`` / ``parsed`` are in candidate order, aligned with ``mi`` (plug-in MI in nats, ``nbins`` quantile bins against the integer labels ``y_codes``)."""
    if not candidates:
        return None
    try:
        import cupy as cp

        from ._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident

        dt = _crit_np_dtype()
        unary, binary = _device_unary(cp), _device_binary(cp)
        sources = sorted({c for src, _ in candidates for c in src})
        resident = {c: cp.asarray(np.ascontiguousarray(X[c].to_numpy(), dtype=dt)) for c in sources}
        n = len(X)
        free = device_room(cp)
        block = int(max(1, min(len(candidates), (_BUDGET_FRACTION * free) // (4 * n * np.dtype(dt).itemsize))))
        names: list[str] = []
        parsed: list[tuple] = []
        mi: list[float] = []
        yd = cp.asarray(np.ascontiguousarray(y_codes, dtype=np.int64))
        n_cls = int(y_codes.max()) + 1 if y_codes.size else 1
        seen: set[str] = set()
        pending: list[Any] = []
        pending_meta: list[tuple] = []

        def flush() -> None:
            """Score the pending columns as one resident ``(k, n)`` block."""
            if not pending:
                return
            blk = cp.nan_to_num(cp.stack(pending), nan=0.0, posinf=0.0, neginf=0.0)  # the host's scrub, once for the block
            live = np.flatnonzero(cp.asnumpy(blk.std(axis=1) > 1e-12))  # one reduction and one transfer for the whole block: the constant candidates the host drops
            if live.size == 0:
                pending.clear()
                pending_meta.clear()
                return
            if live.size < blk.shape[0]:
                blk = blk[cp.asarray(live)]
            vals = np.asarray(_plugin_mi_classif_batch_cuda_resident(blk, yd, int(nbins), y_min=0, n_classes=n_cls, keep_dtype=True, x_is_cm=True), dtype=np.float64)  # Intentional float64: MI statistic
            for i, v in zip(live, vals):
                names.append(pending_meta[int(i)][0])
                parsed.append(pending_meta[int(i)][1])
                mi.append(float(v))
            pending.clear()
            pending_meta.clear()

        for src, tname in candidates:
            named = _named_transform(src, tname, unary, binary)
            if named is None or named[0] in seen:
                continue
            name, fn = named
            seen.add(name)
            pending.append(fn(*(resident[c] for c in src)).astype(dt, copy=False))
            pending_meta.append((name, (src, tname)))
            if len(pending) >= block:
                flush()
        flush()
        if not names:
            return None
        return names, parsed, np.asarray(mi, dtype=np.float64)  # Intentional float64: MI statistic
    except Exception as exc:
        log_throttle(logger, "mi_greedy_device.failed", logging.WARNING, "device MI-greedy scoring failed (%s: %s); the host path scores the candidates", type(exc).__name__, exc)
        return None


def _sweep_inputs(dims: dict) -> tuple:
    """A frame of ``dims["n_samples"]`` rows with five signal-bearing float32 columns, all unary and binary candidates over them, and a binary target."""
    from ._mi_greedy_fe import iter_candidates

    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    X = pd.DataFrame(rng.standard_normal((n, 5)).astype(np.float32), columns=[f"x{i}" for i in range(5)])
    y = ((X["x0"] * X["x1"] + 0.5 * X["x2"] + rng.standard_normal(n)) > 0).astype(np.int64).to_numpy()
    return (X, list(iter_candidates(X, cols=list(X.columns))), y, 10)


def _host(X: pd.DataFrame, candidates: list, y: np.ndarray, nbins: int) -> np.ndarray:
    """The host path's MI per kept candidate."""
    from ._mi_greedy_fe import generate_mi_greedy_features
    from ._orthogonal_univariate_fe import mi_classif_batch_chunked

    engineered, _parsed = generate_mi_greedy_features(X, candidates)
    return np.asarray(mi_classif_batch_chunked(engineered, y, nbins=nbins), dtype=np.float64)  # Intentional float64: MI statistic


def _device(X: pd.DataFrame, candidates: list, y: np.ndarray, nbins: int) -> np.ndarray:
    """The device path's MI per kept candidate; a failed launch raises so the sweep does not record a host fallback as a device time."""
    got = score_candidates_device(X, candidates, y, nbins)
    if got is None:
        raise RuntimeError("device MI-greedy scoring unavailable")
    return got[2]


def _run_sweep() -> list:
    """Time the host and the device path per row count; the device must agree with the host to the selection-equivalence tolerance."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return sweep_backend_grid({"host": _host, "device": _device}, {"n_samples": _SWEEP_ROWS}, _sweep_inputs, reference="host", repeats=3, equiv_rtol=1e-3, equiv_atol=1e-4)


_SWEEP_ROWS = [5_000, 30_000, 100_000]
_SALT = 1


def _fallback_choice(n_samples: int) -> str:
    """Pre-sweep fallback: the host unless ``DEVICE_FALLBACK_ROWS`` is lowered."""
    return device_from_rows(n_samples, DEVICE_FALLBACK_ROWS)


try:
    from pyutilz.performance.kernel_tuning.registry import TunerSpec, kernel_tuner

    _SPEC: "TunerSpec | None" = kernel_tuner(
        kernel_name="fe_mi_greedy_candidates_device",
        variant_fns=(),  # GPU-only cupy path; covered by salt
        tuner=_run_sweep,
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_mi_greedy_candidates_device",
    )
except Exception as e:
    logger.warning("kernel_tuner spec build for fe_mi_greedy_candidates_device failed: %s", e)
    _SPEC = None


def use_device(n_rows: int) -> bool:
    """Whether ``n_rows`` rows take the device path: the per-host kernel-tuning cache decides (fallback ``DEVICE_FALLBACK_ROWS``, the host until a sweep says otherwise)."""
    from ._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled():
        return False
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("fe_mi_greedy_candidates_device: KTC choose() failed, using the fallback: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS


__all__ = ["score_candidates_device", "use_device"]
