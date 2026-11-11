"""Held-out uplift of MANY engineered columns against their raw sources in one batched pass on the device.

The accuracy gate probes each candidate on its own: 10 folds of a 5000-row subsample, each fold a standardised logistic (or ridge) fit and a held-out score, about 26 ms per candidate
(``audits/2026-10-22/profiles/_fe_accuracy_gate.md``). At 1000 columns that is the largest host block of the fit, about 65 thread-seconds, and the profile note already names the lever:
one batched Newton over all candidates and folds.

Batching applies because every candidate sees the SAME rows and the SAME folds: ``keep_engineered_over_source`` draws its subsample with ``default_rng(seed)`` and one seed for the whole
gate, and ``_probe_splits`` builds the folds with ``StratifiedKFold(..., random_state=seed)`` over ``y`` alone (the design enters only by its row count). Only the columns differ: the base
design is the raw source, the augmented one the source plus the engineered column. So a fold is a stack of ``(mb, n_train, k)`` designs of ONE width, which cupy solves as a batch
(``einsum`` for the gradient, ``matmul`` for the Hessian, ``cholesky``/``solve`` for the step).

The arithmetic mirrors ``_fe_accuracy_gate_newton.fit_logistic_ridge_newton_numpy`` term for term, including the gradient-norm stop test and the backtracking line search, and the held-out
score mirrors ``mlframe.metrics.core.fast_roc_auc`` (mid-ranks on ties). The batch reduces in a different ORDER than the per-candidate path, so the uplift of a candidate can differ in the
last bits; the bar for this family is an unchanged SELECTION, which ``tests/feature_selection/fe/test_accuracy_gate_batched.py`` pins by comparing the kept set against the host path.
A candidate whose batched fit does not converge is handed back as ``nan`` so the caller probes it on the host, exactly as a non-converged host fit falls through to the scikit-learn fit.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from ._fe_accuracy_gate_newton import _GRAD_TOL, _MAX_ITER
from ._fe_usability_signal import _crit_np_dtype

logger = logging.getLogger(__name__)

#: Measured fallback (no KTC entry yet, or the probe is off the sweep's grid): below this many candidates the per-candidate host probe on the edge-jobs pool wins (``_fe_accuracy_gate_kernel.md``:
#: at 96 designs x 10 folds x 5000 rows, alone in a quiet process, the fused kernel is slower than an 8-way host pool; it wins once the pool is shared with other concurrent FE lanes, which
#: a static threshold measured in isolation cannot see - the KTC sweep below is run with the pool genuinely busy so its regions reflect the pool-contended case the gate actually runs in).
_MIN_CANDIDATES_FALLBACK = 24


def _gate_batch_fallback_choice(n_designs: int) -> str:
    """Fallback when the KTC has no entry for ``n_designs`` yet: the measured crossover of :data:`_MIN_CANDIDATES_FALLBACK`."""
    return "device" if int(n_designs) >= _MIN_CANDIDATES_FALLBACK else "host"


def _make_gate_batch_inputs(dims: dict) -> tuple:
    """``(srcs, engs, y)`` for the sweep: ``dims["n_designs"]`` source/engineered column pairs of 5000 rows, a binary target, half the engineered columns carrying real signal."""
    n = 5000
    rng = np.random.default_rng(0)
    k = int(dims["n_designs"])
    srcs = [rng.standard_normal(n).astype(np.float32) for _ in range(k)]
    y = (rng.random(n) < 0.45).astype(np.int64)
    engs = [(s**2 if i % 2 == 0 else rng.standard_normal(n)).astype(np.float32) for i, s in enumerate(srcs)]
    return srcs, engs, y


def _gate_batch_host_variant(srcs, engs, y):
    """The per-candidate host probe of every (source, engineered) pair, on the edge-jobs pool, as the gate's host path runs it."""
    from ._fe_accuracy_gate import measure_feature_uplift

    return np.array([measure_feature_uplift(s[:, None], e[:, None], y, classification=True) for s, e in zip(srcs, engs)])


def _gate_batch_device_variant(srcs, engs, y):
    """The fused-kernel probe of the same pairs, forced on regardless of :data:`_MIN_CANDIDATES_FALLBACK`."""
    from ._fe_accuracy_gate import _probe_splits

    probe = _probe_splits(np.zeros((len(y), 1)), y, classification=True, n_splits=10, seed=0, groups=None)
    it, y_enc, _binary, _ns = probe
    folds = [(tr, va) for tr, va in it if np.unique(y_enc[tr]).size >= 2]
    return batched_uplift(list(srcs), list(engs), y_enc.astype(np.float64), folds=folds, base_of=list(range(len(srcs))))


def _sweep_gate_batch() -> list:
    """The actual host-vs-device grid sweep (kept separate from :func:`_run_gate_batch_sweep` so a test can call it without the KTC import chain)."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {"host": lambda *a: _gate_batch_host_variant(*a), "device": lambda *a: _gate_batch_device_variant(*a)}
    return list(
        sweep_backend_grid(
            variants, {"n_designs": [8, 24, 64, 160, 400]}, _make_gate_batch_inputs, reference="host", repeats=2, equiv_rtol=5e-2, equiv_atol=5e-3
        )
    )


_GATE_BATCH_SPEC: "Any | None" = None
try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _GATE_BATCH_SPEC = kernel_tuner(
        kernel_name="fe_accuracy_gate_batch",
        variant_fns=(),  # GPU-only cupy/RawKernel path; covered by salt
        tuner=_sweep_gate_batch,
        axes={"n_designs": [8, 24, 64, 160, 400]},
        fallback=_gate_batch_fallback_choice,
        gpu_capable=True,
        salt=1,
        cli_label="fe_accuracy_gate_batch",
    )
except Exception as e:  # the dispatcher falls back to the measured heuristic when the registry is unavailable
    logger.warning("kernel_tuner spec build for fe_accuracy_gate_batch failed: %s", e)
    _GATE_BATCH_SPEC = None


def _use_device(n_designs: int) -> bool:
    """Whether ``n_designs`` candidates should take the fused device probe: the KTC's tuned choice, or the measured fallback when no entry exists yet."""
    if _GATE_BATCH_SPEC is None:
        return _gate_batch_fallback_choice(n_designs) == "device"
    try:
        return str(_GATE_BATCH_SPEC.choose(n_designs=int(n_designs))) == "device"
    except Exception as exc:
        logger.debug("fe_accuracy_gate_batch: KTC choose() failed, using the fallback: %s", exc)
        return _gate_batch_fallback_choice(n_designs) == "device"


def batched_enabled() -> bool:
    """Whether the batched device probe may run (``MLFRAME_GATE_BATCHED=0`` keeps the per-candidate host path)."""
    return os.environ.get("MLFRAME_GATE_BATCHED", "1").strip() not in ("0", "false", "off", "no")


def _device_block_designs(cp, n_rows: int, k: int, itemsize: int) -> int:
    """Designs held on the device at once: a third of the free VRAM over one design's bytes, at least one (CUDA.md section 5 sizes a block from free memory, never a constant)."""
    try:
        free_b = int(cp.cuda.runtime.memGetInfo()[0])
    except Exception as exc:  # no device information: a single modest block
        logger.debug("gate batch: free VRAM unreadable (%r); one 64-design block", exc)
        return 64
    per = max(1, int(n_rows) * max(1, int(k)) * int(itemsize))
    return max(1, int((free_b // 3) // per))


def _fold_assignment(n: int, folds: Sequence[tuple]) -> "tuple[np.ndarray, np.ndarray]":
    """``(fold_of, eval_folds)``: the fold whose held-out part owns each row (-1 for a row no fold holds out), and the fold ids to evaluate."""
    fold_of = np.full(int(n), -1, dtype=np.int8)
    for f, (_tr, va) in enumerate(folds):
        fold_of[np.asarray(va, dtype=np.int64)] = f
    return fold_of, np.arange(len(folds), dtype=np.int32)


def _fold_resident(cp: Any, y_host: np.ndarray, fold_of: np.ndarray, eval_folds: np.ndarray, real: np.dtype) -> tuple:
    """Upload ``(y, fold_of, eval_folds)`` once: every ``_fold_auc_device`` call in a chunk loop reuses the SAME fold
    assignment, so repeating this upload per chunk would be a pure, avoidable per-iteration cost."""
    return (
        cp.asarray(np.ascontiguousarray(y_host, dtype=real)),
        cp.asarray(np.ascontiguousarray(fold_of)),
        cp.asarray(np.ascontiguousarray(eval_folds)),
    )


def _fold_auc_launch(cp: Any, designs_host: np.ndarray, fold_resident: tuple, eval_folds: np.ndarray, lam: float, out: Any) -> np.ndarray:
    """``_fold_auc_device``'s launch, into a CALLER-OWNED ``out`` buffer (``(>=n_designs, n_folds)``): a chunk loop over several
    design blocks passes the SAME reusable buffer (sliced to ``n_designs``) every call instead of a fresh ``cp.empty`` each time."""
    from ._fe_accuracy_gate_kernel import BLOCK, kernel

    real = designs_host.dtype
    n_designs, k, n = (int(v) for v in designs_host.shape)
    fn = kernel(cp, real)
    d_designs = cp.asarray(designs_host)
    d_y, d_fold, d_eval = fold_resident
    out = out[:n_designs]
    fn(  # launch-ok: BLOCK=128 (4 warps) is fixed by the kernel's own shared-memory tree-fold reduction over WARPS=BLOCK//32 slots, not a tuned size
        (n_designs, int(eval_folds.shape[0])),
        (BLOCK,),
        (
            d_designs,
            d_y,
            d_fold,
            d_eval,
            np.int64(n),
            np.int32(k),
            np.int32(n_designs),
            np.int32(int(eval_folds.shape[0])),
            np.float64(float(lam)),
            np.float64(float(_GRAD_TOL)),
            np.int32(int(_MAX_ITER)),
            np.int32(1 if real != np.float64 else 0),
            out,
        ),  # Intentional float64: the kernel's {ACC} scalars
    )
    return np.asarray(cp.asnumpy(out))


def _fold_auc_device(cp: Any, designs_host: np.ndarray, fold_resident: tuple, eval_folds: np.ndarray, lam: float) -> np.ndarray:
    """``_fold_auc_launch`` for a one-off call: allocates its own output buffer (a chunk loop calling this
    repeatedly should call ``_fold_auc_launch`` directly with one buffer it owns and reuses instead)."""
    out = cp.empty((int(designs_host.shape[0]), int(eval_folds.shape[0])), dtype=cp.float64)  # Intentional float64: the kernel's {ACC} output
    return _fold_auc_launch(cp, designs_host, fold_resident, eval_folds, lam, out)


def batched_uplift(
    base_cols: Sequence[np.ndarray], eng_cols: Sequence[np.ndarray], y: np.ndarray, *, folds: Sequence[tuple], base_of: Sequence[int], lam: float = 1.0
) -> "Optional[np.ndarray]":
    """``(n_candidates,)`` mean held-out AUC uplift of each engineered column over its base column, or ``None`` when the device path does not apply.

    ``base_cols`` holds the DISTINCT base columns (each ``(n,)``), ``eng_cols`` the candidates' engineered columns and ``base_of[i]`` the index in ``base_cols`` of candidate ``i``'s base, so a
    base shared by several candidates is fitted once. ``folds`` are the ``(train rows, validation rows)`` index arrays the host probe would use. A candidate whose fit did not converge on some
    fold comes back ``nan``: the caller probes that one on the host, as it does for a host fit that returns ``None``. Designs go in blocks sized from the free VRAM, and any device fault returns
    ``None`` so the caller keeps the host probe (CUDA.md section 5).
    """
    if not batched_enabled() or not eng_cols:
        return None
    try:
        import cupy as cp

        from ._gpu_policy import cuda_available_for_run

        if not cuda_available_for_run():
            return None
        from ._fe_accuracy_gate_kernel import MAX_K, MAX_VA

        real = np.dtype(_crit_np_dtype())
        n = int(np.asarray(y).ravel().shape[0])
        if n == 0 or any(int(np.asarray(c).shape[0]) != n for c in base_cols) or any(int(np.asarray(c).shape[0]) != n for c in eng_cols):
            return None
        if 2 > MAX_K:  # pragma: no cover - the design width is 1 (base) or 2 (augmented)
            return None
        fold_of, eval_folds = _fold_assignment(n, folds)
        if max(int(np.count_nonzero(fold_of == f)) for f in range(len(folds))) > MAX_VA:
            return None  # a held-out fold the block cannot reduce in shared memory
        y_host = np.ascontiguousarray(np.asarray(y).ravel(), dtype=real)
        fold_resident = _fold_resident(cp, y_host, fold_of, eval_folds, real)  # one upload for every chunk below, not one per chunk
        bases = np.ascontiguousarray(np.stack([np.asarray(c, dtype=real).ravel() for c in base_cols])[:, None, :])  # (n_base, 1, n) feature-major
        base_auc = _fold_auc_device(cp, bases, fold_resident, eval_folds, lam)
        del bases
        block = min(len(eng_cols), _device_block_designs(cp, n, 2, real.itemsize))
        out_buf = cp.empty((block, int(eval_folds.shape[0])), dtype=cp.float64)  # one reusable AUC buffer for every chunk below, not one per chunk
        uplift = np.empty(len(eng_cols), dtype=np.float64)  # Intentional float64: the uplift is a mean of double AUC differences
        for start in range(0, len(eng_cols), block):
            idx = list(range(start, min(len(eng_cols), start + block)))
            aug = np.empty((len(idx), 2, n), dtype=real)  # feature-major: column 0 the base, column 1 the engineered candidate
            for row, i in enumerate(idx):
                aug[row, 0, :] = np.asarray(base_cols[base_of[i]], dtype=real).ravel()
                aug[row, 1, :] = np.asarray(eng_cols[i], dtype=real).ravel()
            aug_auc = _fold_auc_launch(cp, np.ascontiguousarray(aug), fold_resident, eval_folds, lam, out_buf)
            delta = aug_auc - base_auc[np.asarray([base_of[i] for i in idx], dtype=np.int64)]
            uplift[start : start + len(idx)] = np.mean(delta, axis=1)  # NaN on any fold makes the candidate NaN, so the host probes it
        return uplift
    except Exception as exc:  # any device fault: the caller keeps the per-candidate host probe
        logger.warning("batched accuracy-gate probe not used (%s: %s); the per-candidate host probe decides", type(exc).__name__, exc)
        return None


def batched_uplifts_for_pairs(
    src_arrays: Sequence[np.ndarray],
    eng_arrays: Sequence[np.ndarray],
    base_keys: Sequence[Any],
    y: np.ndarray,
    *,
    seed: int = 0,
    max_probe_n: int = 5000,
    n_splits: int = 10,
) -> "Optional[np.ndarray]":
    """``(n_candidates,)`` uplift of each ``(source, engineered)`` pair, ``nan`` for a pair the batch does not cover, or ``None`` when it does not apply at all.

    Covers the pairs the per-candidate probe would decide with the Newton fit on the SHARED rows and folds: a binary classification target, a pair whose values are all finite over the
    subsample (``measure_feature_uplift`` drops non-finite rows per pair, which would give that pair its own rows), and enough rows for the folds. Everything else comes back ``nan`` and the
    caller probes it as before. ``base_keys[i]`` names candidate ``i``'s source so a source shared by several candidates is fitted once.
    """
    from ._fe_accuracy_gate import _newton_probe_enabled, _probe_splits, infer_classification

    if not batched_enabled() or not eng_arrays or not _newton_probe_enabled():
        return None
    y_arr = np.asarray(y).ravel()
    n_full = int(y_arr.shape[0])
    if n_full == 0 or not infer_classification(y_arr):
        return None
    rows = np.arange(n_full)
    if n_full > int(max_probe_n):
        rows = np.random.default_rng(int(seed)).choice(
            n_full, int(max_probe_n), replace=False
        )  # the per-candidate probe draws this same subsample from the same seed
    y_sub = y_arr[rows]
    if y_sub.size < 40:
        return None
    classes = np.unique(y_sub)
    if classes.size != 2:  # a multiclass probe scores the hit rate of a scikit-learn fit, which this batch does not reproduce
        return None
    probe = _probe_splits(np.zeros((y_sub.size, 1)), y_sub, classification=True, n_splits=int(n_splits), seed=int(seed), groups=None)
    if probe is None:
        return None
    split_iter, y_enc, binary, _n_splits = probe
    if not binary:
        return None
    folds = [(tr, va) for tr, va in split_iter if np.unique(y_enc[tr]).size >= 2]
    if not folds:
        return None
    out = np.full(len(eng_arrays), np.nan)  # Intentional float64: the uplift is a mean of double AUC differences
    # the per-candidate probe reads its columns at the criterion dtype (``measure_feature_uplift`` casts them), so the batch starts from the same values
    crit = _crit_np_dtype()
    subs = [np.asarray(a, dtype=crit).ravel()[rows] for a in src_arrays]
    engs = [np.asarray(a, dtype=crit).ravel()[rows] for a in eng_arrays]
    covered = [i for i in range(len(engs)) if np.isfinite(subs[i]).all() and np.isfinite(engs[i]).all()]
    if not covered:
        return out
    order: "dict[Any, int]" = {}
    for i in covered:
        order.setdefault(base_keys[i], len(order))
    base_cols: "list[np.ndarray]" = [np.empty(0)] * len(order)
    for i in covered:
        base_cols[order[base_keys[i]]] = subs[i]
    got = batched_uplift(
        base_cols, [engs[i] for i in covered], y_enc.astype(np.float64), folds=folds, base_of=[order[base_keys[i]] for i in covered]
    )  # Intentional float64: double labels for the Newton solve
    if got is None:
        return None
    out[np.asarray(covered, dtype=np.int64)] = got
    return out


__all__ = ["batched_enabled", "batched_uplift", "batched_uplifts_for_pairs"]
