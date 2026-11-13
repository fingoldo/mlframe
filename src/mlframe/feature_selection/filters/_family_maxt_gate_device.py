"""Device-resident family gate: the family-wise permutation test of ``family_pass_batches`` run on the GPU codes, with no host copy of them.

The host gate (``_family_maxt_gate``) reads the family's code matrix on the host: a transposed stack of the batches, the observed MI of every column, the occupied bins of every column for the
chi-square triage, and the column bit sets the null kernel counts against. When the batches are resident device arrays (``LazyHostCodes`` over cupy arrays) every one of those is available on
the device already. This module packs the codes into bin bit sets on the device (the packing kernel of the bit-set noise gate), scores the observed MI and every block of label permutations with
the same popcount kernel (``family_null_bits_v2``), and hands the host orchestration (analytic triage, early-stopping permutation loop) only small vectors: the observed MI and the occupied
bins of every column, and the null maxima of each block. The decisions are the host gate's: the orchestration is the same code, only the arrays come from the device.

``family_pass_batches_device`` returns ``None`` whenever the device path does not apply (a batch that is not a device array, more bins or classes than the bit-set kernels hold, bit sets that
would not fit the card), and the caller then runs the host gate.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for
from ._fe_usability_signal import _crit_np_dtype
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)


class _NotApplicableError(Exception):
    """The family cannot be scored by the bit-set kernels on this device."""


def _device_codes(batches: Sequence[Any]) -> "Optional[list]":
    """The device array behind every batch (a ``LazyHostCodes``), or ``None`` when any batch is a plain host array."""
    out = []
    for b in batches:
        dev = getattr(b, "dev", None)
        if dev is None or type(dev).__module__.split(".", 1)[0] != "cupy" or getattr(dev, "ndim", 0) != 2:
            return None
        out.append(dev)
    return out


def _observed_from_noise_gate(batches: Sequence[Any], y: np.ndarray) -> "Optional[np.ndarray]":
    """The observed MI of every one-column batch as the noise gate already computed it for the chunk the columns come from, or ``None`` (every column must be a lazy column of a chunk whose
    resident codes the noise gate scored against these labels). The same kernel produced the numbers, so they are the values ``_FamilyOnDevice.observed`` would compute and copy."""
    try:
        from ._batch_mi_noise_gate_bits import labels_key, recall_observed
        from ._feature_engineering_pairs._pairs_chunk_codes import LazyCodeColumn

        if not all(isinstance(b, LazyCodeColumn) for b in batches):  # an isinstance test: ``getattr`` on a ``LazyHostCodes`` batch would copy the whole array to the host
            return None
        key = labels_key(y)
        by_chunk: dict = {}
        out = np.empty(len(batches), dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
        for i, b in enumerate(batches):
            dev, j = b.chunk._dev, b.j
            if dev is None:
                return None
            vec = by_chunk.get(id(dev))
            if vec is None:
                vec = recall_observed(dev, key)
                if vec is None or int(vec.shape[0]) != int(dev.shape[1]):
                    return None
                by_chunk[id(dev)] = vec
            out[i] = vec[int(j)]
        return out
    except Exception as exc:
        log_throttle(logger, "_family_maxt_gate_device.except_65", logging.WARNING, "the noise gate's observed MI is not usable for the family gate (%s: %s); computing it", type(exc).__name__, exc)
        return None


class _FamilyOnDevice:
    """The column bit sets of one family on the device, with the observed-MI and null-maximum evaluations over them."""

    def __init__(self, cp: Any, devs: "list", y: np.ndarray, n_y: int) -> None:
        from ._batch_mi_noise_gate_bits import _BUDGET_CAP_BYTES, _C_TYPES, MAX_BINS, _pack_module
        from ._family_maxt_cuda_bits import _v2_applies
        from ._family_maxt_null_v2_njit import xlogx_table

        self.cp, self.y, self.n_y = cp, y, int(n_y)
        n = int(devs[0].shape[0])
        if any(int(d.shape[0]) != n for d in devs):
            raise _NotApplicableError("batches of different length")
        # The batches are packed one by one at a column offset: concatenating them first copied the whole (n, k) code matrix (57 MB a call, 22 ms; 2.4 GB over a default fit) only so that one launch could read it.
        parts = [cp.ascontiguousarray(d) for d in devs]
        parts = [d if d.dtype in _C_TYPES else d.astype(cp.int64) for d in parts]
        k_total = int(sum(int(d.shape[1]) for d in parts))
        self.n, self.k_total = n, k_total
        n_x = cp.concatenate([(d.max(axis=0) + 1).astype(cp.int64) for d in parts]) if k_total else cp.zeros(0, dtype=cp.int64)  # per-column level counts: k_total int64, not the code matrix
        # The host needs only the largest level count (the bit-set depth): one scalar crosses the bus, not the per-column vector (the same vector was copied once per family).
        cap = int(n_x.max()) if k_total else 0
        words = (n + 63) // 64
        if n == 0 or k_total == 0 or not 1 <= cap <= MAX_BINS or not _v2_applies(cp, self.n_y, words):
            raise _NotApplicableError("shape outside the bit-set kernels")
        free_bytes = int(cp.cuda.runtime.memGetInfo()[0])
        if cap * words * k_total * 8 > min(_BUDGET_CAP_BYTES, free_bytes // 3):
            raise _NotApplicableError("bit sets do not fit the card")
        self.cap, self.words = cap, words
        self.n_x = n_x
        table_host = xlogx_table(n)
        self.table = cp.asarray(table_host)
        counts = np.bincount(y, minlength=self.n_y)
        self.py_const = float(table_host[n] - sum(float(table_host[int(c)]) for c in counts))
        group = max(1, int(min(_BUDGET_CAP_BYTES, free_bytes // 4) // (cap * words * 8)))
        self.groups = []
        occupied = []
        # One group per ``group`` columns of the whole family, however many device batches they come from: a launch per batch (hundreds of batches of a few columns) made a default fit
        # launch the null kernel and its maximum 1.2 million times; the batches are packed into a group at their column offsets instead.
        starts = np.cumsum([0, *[int(c.shape[1]) for c in parts]])
        for g0 in range(0, k_total, group):
            g1 = min(g0 + group, k_total)
            kg = g1 - g0
            bits = cp.empty((cap, words, kg), dtype=cp.uint64)
            px = cp.zeros((kg, cap), dtype=cp.int64)
            for codes, p0, p1 in zip(parts, starts[:-1], starts[1:]):
                lo, hi = max(g0, int(p0)), min(g1, int(p1))
                if lo >= hi:
                    continue
                pack_bins = _pack_module(cp, cap, self.n_y, _C_TYPES[codes.dtype]).get_function("pack_bins")
                width = hi - lo
                bins_block = block_size_for(pack_bins, work_items=width)
                pack_bins(
                    ((width + bins_block - 1) // bins_block, words),
                    (bins_block,),
                    (codes, np.int64(n), np.int32(codes.shape[1]), np.int32(lo - int(p0)), np.int32(width), np.int32(words), np.int32(kg), np.int32(lo - g0), bits, px),
                )
            self.groups.append((g0, g1, bits, px, self.table[px].sum(axis=1)))
            occupied.append((px > 0).sum(axis=1))
        self.nx_occ = cp.asnumpy(cp.concatenate(occupied).astype(cp.int16)).astype(np.int64)  # occupied-level counts are at most MAX_BINS: int16 over the bus

    def _parts(self, cls_host: np.ndarray):
        """The ``(B, kg)`` MI block of every column group against the ``B`` label bit-set rows of ``cls_host``."""
        from ._family_maxt_cuda_bits import launch_family_null

        cp = self.cp
        b = int(cls_host.shape[0])
        cls = cp.asarray(cls_host)
        const_b = cp.full(b, self.py_const, dtype=self.table.dtype)
        for lo, hi, bits, px, c_a in self.groups:
            kg = hi - lo
            part = cp.empty((b, kg), dtype=self.table.dtype)
            launch_family_null(cp, self.n_y, self.words, bits, self.n_x[lo:hi], cls, self.table, c_a, px, const_b, self.n, self.cap, kg, b, part)
            yield part

    def observed(self) -> np.ndarray:
        """The plug-in MI of every column against the original labels (host double precision)."""
        from ._family_maxt_null_v2_njit import label_bitsets_from_index

        cls = label_bitsets_from_index(self.y, np.arange(self.n, dtype=np.int32)[None, :], self.n_y, self.words)
        cp = self.cp
        # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
        return np.concatenate([cp.asnumpy(p[0]).astype(np.float64) for p in self._parts(cls)])

    def null_max(self, m: int, rng: np.random.Generator) -> np.ndarray:
        """The next ``m`` null maxima over every column of the family (host double precision), the permutations drawn from ``rng`` exactly as the host gate draws them."""
        from ._family_maxt_gate import _permutation_block
        from ._family_maxt_null_v2_njit import label_bitsets_from_index

        out = np.empty(int(m), dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
        block = 256
        cp = self.cp
        for start in range(0, int(m), block):
            mm = min(block, int(m) - start)
            cls = label_bitsets_from_index(self.y, _permutation_block(rng, self.y, mm), self.n_y, self.words)
            best = None
            for part in self._parts(cls):
                top = part.max(axis=1)
                best = top if best is None else cp.maximum(best, top)
            out[start : start + mm] = cp.asnumpy(best).astype(_crit_np_dtype())
        return out


def family_pass_batches_device(batches: Sequence[Any], y_codes: np.ndarray, alpha: float, n_perm: int, seed: int) -> "Optional[list]":
    """Family-wise pass flags of resident device batches (see ``family_pass_batches``), or ``None`` when the device path does not apply."""
    devs = _device_codes(batches)
    if devs is None:
        return None
    try:
        import cupy as cp

        from ._family_maxt_gate import _analytic_triage_core, _permutation_pass_flags_with

        y = np.ascontiguousarray(np.asarray(y_codes, dtype=np.int64).ravel())
        n_y = int(y.max()) + 1
        fam = _FamilyOnDevice(cp, devs, y, n_y)
        widths = [int(d.shape[1]) for d in devs]
        starts = np.cumsum([0, *widths])
        observed = _observed_from_noise_gate(batches, y)
        if observed is None:
            observed = fam.observed()
        n_batches = len(widths)
        # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
        observed_max = np.array([float(observed[starts[j] : starts[j + 1]].max()) if widths[j] else 0.0 for j in range(n_batches)], dtype=np.float64)
        flags = np.zeros(n_batches, dtype=np.bool_)
        status = _analytic_triage_core(fam.n, fam.nx_occ, y, n_y, observed, starts, alpha)
        if status is None:
            flags[:] = _permutation_pass_flags_with(fam.null_max, observed_max, alpha, n_perm, seed)
            return [bool(f) for f in flags]
        flags[status > 0] = True
        gray = status == 0
        if gray.any():
            flags[gray] = _permutation_pass_flags_with(fam.null_max, observed_max[gray], alpha, n_perm, seed)
        return [bool(f) for f in flags]
    except _NotApplicableError as exc:
        logger.debug("device family gate does not apply (%s); the host gate runs", exc)
        return None
    except Exception as exc:  # any device failure: the host gate gives the decisions
        logger.warning("device family gate failed (%s: %s); the host gate runs", type(exc).__name__, exc)
        return None
