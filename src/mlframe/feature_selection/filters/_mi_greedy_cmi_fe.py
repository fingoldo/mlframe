"""CMI-greedy feature constructor (Layer 60, 2026-05-31).

Sibling to :mod:`_mi_greedy_fe` (Layer 26). Where Layer 26 ranks the same
candidate transform pool by MARGINAL ``MI(candidate; y)`` and de-duplicates
selected winners post-hoc via Spearman, THIS module ranks by CONDITIONAL
``MI(candidate; y | current_support)`` - i.e. each step directly measures
the NEW information the candidate adds on top of the already-selected
columns.

Why CMI ranking matters
-----------------------

Marginal MI ranks ``log_abs(x)``, ``square(x)``, ``abs(x)`` all near the top
when ``y = sign(x^2 - 1)`` because each is monotone in ``|x|`` and so
captures the same signal. The marginal-MI greedy path then picks all three
and the downstream Spearman dedup drops two of them post-hoc - waste.
CMI ranking sees that once ``square(x)`` is in the support, ``CMI(abs(x); y |
square(x))`` is near zero, so ``abs(x)`` is never picked.

Algorithm
---------

1. Materialise the candidate library via :func:`_mi_greedy_fe.iter_candidates`
   over the top-N seed columns (same enumeration as Layer 26).
2. Quantile-bin every candidate column to ``nbins`` integer bins once.
3. Quantile-bin the seed columns identically.
4. Seed the support with the top-``seed_cols_count`` raw columns by marginal
   MI(x; y).
5. Greedy loop: at each step compute
   ``CMI(candidate; y | joint_support)`` for every remaining candidate,
   pick the one with the highest CMI provided it clears ``min_cmi_gain``.
   Stop when no candidate clears the gate or ``top_k`` winners are seated.
6. Emit recipes of kind ``"mi_greedy_transform"`` (same as Layer 26) so
   transform-time replay is shared infrastructure.

The conditional joint Z is the per-row class id of the cross-product of
the currently-selected binned columns - collapsed via the densely-renumbered
contingency table so the joint stays computable even at d=8+ support cols
(memory dominated by ``n``, not by the cartesian bin space).
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional, Sequence, cast

from mlframe._numba_parallel_guard import parallel_kernel_entry

import numpy as np
from ._fe_usability_signal import _crit_np_dtype
import pandas as pd

try:
    from numba import njit
    from numba.core import types as _nb_types
    from numba.typed import Dict as _NbDict
    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - numba is a hard dep in practice
    _NUMBA_AVAILABLE = False
    _nb_types = None
    _NbDict = None

    from mlframe._numba_fallback import njit

from mlframe.utils.log_throttle import log_throttle
from types import SimpleNamespace as _SimpleNamespace

# GPU quantile-bin crossover (synchronized micro-bench of _quantile_bin_gpu incl. code D2H, GTX
# 1050 Ti, nbins=10): a single host column round-tripped to the device for equi-frequency binning (H2D +
# cp.percentile sort + cp.searchsorted + code D2H) only beats the host introselect-partition np.quantile path
# well above the launch/transfer floor - n=20k CPU 0.92ms vs GPU 1.67ms; n=35k near-tie 1.53 vs 1.71ms; n=100k
# GPU 2.10 vs CPU 4.12ms = 2x; n=300k GPU 3.06 vs CPU 14.1ms = 4.6x. The gate is set at 50k (clear of the 35k
# near-tie) so every routed call is a decisive win; the small (3k/20k) gate-redundancy columns stay on the
# host, where the fixed ~1.7ms device round-trip overhead loses to numpy.
_GPU_QBIN_MIN_ROWS = 50_000

__all__ = [
    "score_candidates_by_cmi",
    "greedy_cmi_fe_construct",
    "greedy_cmi_fe_construct_with_recipes",
]

# ---------------------------------------------------------------------------
# Binning + entropy helpers (self-contained — mirrors ``_mi_classif_batch``'s
# equi-frequency binning so CMI numbers are directly comparable to the
# marginal-MI numbers Layer 26 reports).
# ---------------------------------------------------------------------------

# Direct-array factorize is used while the joint's max id keeps the ``seen``
# lookup buffer under this many int64 entries (~128 MB at the cap). Above it
# (cartesian blow-up: a high-cardinality support col times a large running
# class count) we fall back to the hash path so memory stays bounded.
# Parallel-memset crossover for the dense ``seen`` buffer in ``_combine_factorize_njit``. The first-seen
# factorize WALK is irreducibly sequential (each id depends on the running ``seen``/``nc`` state), but its
# ``np.full(kmax+1, -1)`` initialisation is an independent fill that prange-splits across threads. The fill
# is only worth the thread spin-up once it is large: synchronized micro-bench (4 threads, GTX
# box, n=1M) put the crossover near kmax~500k - below it parallel loses (kmax~40k 0.99x, ~200k 0.92x),
# above it wins and scales (500k 1.08x, 1M 1.06x, 2M 1.09x, 4M 1.12x, 16M 1.24x). At the cap (16M int64 =
# 128 MB) the fill alone is ~28ms of the ~55ms call, so parallelising it is the single largest safe win on
# this sequential kernel. Below the gate the fill stays serial -> bit-identical, zero regression.
from ._mi_greedy_cmi_fe_steps import (  # noqa: F401  -- carved helpers
    logger,
    _qbin_float_dtype,
    _sync_free_qbin_codes,
    _quantile_bin_gpu,
    _quantile_bin_gpu_resident,
    _quantile_bin,
    _FAC_ARRAY_CAP,
    _factorize_dense_njit,
    _FAC_PAR_MEMSET_MIN,
    _combine_factorize_njit,
    _renumber_two_dense_njit,
    _renumber_joint,
    _dense_renumber_device,
    _renumber_joint_gpu,
    _entropy_from_classes_njit,
    _joint_entropy_two_dense_njit,
    _joint_entropy_two,
    _entropy_from_classes,
    precompute_marginal_y_terms,
    marginal_mi_binned_fixed_y,
    precompute_cmi_yz_terms,
    cmi_from_binned_fixed_yz,
    _cmi_candidate_batch_on_device,
    _cmi_gpu_enabled,
    _CARD_MAX_CACHE,
    _CARD_MAX_CACHE_LOCK,
    _cached_card,
    _cmi_from_binned_fixed_yz_cupy,
    _greedy_cmi_fe_constr_step1_content_hash_already,
    _greedy_cmi_fe_constr_step2_rng_raw_small,
    _greedy_cmi_fe_constr_step3_while_st_remaining,
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .engineered_recipes import EngineeredRecipe

@njit(cache=True)
def _combine_factorize_serial_njit(joint: np.ndarray, c: np.ndarray, mult: int) -> tuple:
    """Fully-serial reference form of :func:`_combine_factorize_njit` (the bit-identical baseline + the
    numba-absent fallback). Kept per the repo "keep all kernel versions" rule and used by the parity test."""
    n = joint.size
    if n == 0:
        return joint, 0
    kmax = 0
    for i in range(n):
        v = joint[i] + c[i] * mult
        if v > kmax:
            kmax = v
    inv = np.empty(n, dtype=np.int64)
    nc = 0
    if 0 <= kmax < _FAC_ARRAY_CAP:
        seen = np.full(kmax + 1, -1, dtype=np.int64)
        for i in range(n):
            v = joint[i] + c[i] * mult
            s = seen[v]
            if s >= 0:
                inv[i] = s
            else:
                seen[v] = nc
                inv[i] = nc
                nc += 1
    else:
        d = _NbDict.empty(key_type=_nb_types.int64, value_type=_nb_types.int64)
        for i in range(n):
            v = joint[i] + c[i] * mult
            s = d.get(v, -1)
            if s >= 0:
                inv[i] = s
            else:
                d[v] = nc
                inv[i] = nc
                nc += 1
    return inv, nc

def _cmi_from_binned(
    x: np.ndarray, y: np.ndarray, z_joint: Optional[np.ndarray], return_cards: bool = False,
    kx: int = 0, kz: int = 0,
):
    """``CMI(X; Y | Z) = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z)`` from binned
    integer arrays. Miller-Madow bias correction applied: each plug-in
    entropy is reduced by ``(K-1)/(2n)`` where K is the number of
    occupied cells. The four entropy bias terms combine in CMI to
    ``-(K_xz + K_yz - K_z - K_xyz) / (2n)`` (subtracted from the plug-in
    CMI). On noise data this drives the CMI estimate toward zero where
    the unbiased MLE would inflate to e.g. 0.005 - 0.02 nats and admit
    false positives. On signal data the bias term is dwarfed by the
    true CMI so the correction is benign.

    When ``z_joint is None`` (empty support), reduces to marginal
    ``MI(X; Y) = H(X) + H(Y) - H(X, Y)`` (also Miller-Madow corrected).

    ``return_cards`` (conditional path only): also return the OCCUPIED-cell cards ``(k_z, k_xz, k_yz, k_xyz)``
    already computed here, so a same-(x,y,z) analytic CMI null can reuse them (``precomp_cards``) instead of
    recomputing the four joints. Returns ``(cmi, cards)``; ``cards`` is ``None`` on the marginal path.
    """
    # GPU route: same partition-entropy-via-cp.unique offload as cmi_from_binned_fixed_yz,
    # covering the general CMI callers (conditional-perm null, candidate scoring). value-order densify ->
    # same partition -> same CMI; selection-identical. Gated (STRICT / MLFRAME_CMI_GPU), default CPU.
    #
    # ``x`` may ALREADY be device-resident (e.g. _step_score.py's DEVICE-BORN marginal-MI path binning a
    # candidate via _quantile_bin_gpu_resident before ever calling here) - the shape-based gate below only
    # decides whether it's worth PROACTIVELY uploading, it has no way to know the array is already on-device.
    # An already-device x MUST take the cupy path regardless of that heuristic (np.ascontiguousarray on a
    # cupy array raises TypeError - cupy forbids the implicit host conversion __array__ would trigger), and
    # a cupy failure there must explicitly pull x/y/z_joint to host before falling back (the swallow-and-
    # retry-on-CPU pattern below assumes host arrays).
    _x_device = hasattr(x, "__cuda_array_interface__")
    if _x_device or _cmi_gpu_enabled(n=int(x.size), p=1):
        try:
            return _cmi_from_binned_cupy(x, y, z_joint, return_cards=return_cards, kx=kx, kz=kz)
        except Exception as e:
            # A cupy import error, a kernel-shape miss, transient GPU contention and a genuine numeric
            # regression in the cupy path all land here, so the message has to name which kernel and at what
            # shape. The CPU recomputation below keeps the answer correct, which is exactly why a real kernel
            # regression would otherwise never be noticed.
            log_throttle(
                logger,
                "cmi_gpu_kernel_fallback",
                logging.WARNING,
                "_cmi_from_binned_cupy failed (%s: %s) at n=%d; recomputing this CMI on the CPU path. Correctness is preserved, the GPU cost is not.",
                type(e).__name__,
                e,
                int(getattr(x, "size", 0)),
            )
            if _x_device:
                import cupy as cp
                x = cp.asnumpy(x)
                if hasattr(y, "__cuda_array_interface__"):
                    y = cp.asnumpy(y)
                if z_joint is not None and hasattr(z_joint, "__cuda_array_interface__"):
                    z_joint = cp.asnumpy(z_joint)
    x_i = np.ascontiguousarray(x, dtype=np.int64)
    y_i = np.ascontiguousarray(y, dtype=np.int64)
    n = float(max(1, x_i.size))
    if z_joint is None or z_joint.size == 0:
        h_x, k_x = _entropy_from_classes(x_i)
        h_y, k_y = _entropy_from_classes(y_i)
        h_xy, k_xy = _joint_entropy_two(x_i, y_i)  # fused densify+entropy; xy labels discarded
        mi_plugin = h_x + h_y - h_xy
        # Plug-in MLE entropy underestimates the true entropy by
        # ``(K-1)/(2n)`` (Miller 1955). MI = H(X) + H(Y) - H(XY)
        # therefore OVERESTIMATES the true MI by ``(K_xy - K_x - K_y + 1)/(2n)``. The Miller-Madow
        # correction ``(K_x + K_y - K_xy - 1)/(2n)`` is that bias with its sign flipped (it is negative
        # for nearly any table), so it is ADDED to the plug-in.
        mi_correction = (k_x + k_y - k_xy - 1) / (2.0 * n)
        mi = max(0.0, mi_plugin + mi_correction)
        return (mi, None) if return_cards else mi
    z_i = np.ascontiguousarray(z_joint, dtype=np.int64)
    # Fused densify+entropy for the joints (labels feed only the entropy). yz is materialised DENSE once
    # and reused for H(X,Y,Z): partition(x, part(y,z)) == partition(x,y,z) -> identical counts -> the
    # 3-column renumber(x,y,z) collapses to a 2-array densify against yz_dense.
    yz_dense, _ = _renumber_joint(y_i, z_i)
    h_z, k_z = _entropy_from_classes(z_i)
    h_xz, k_xz = _joint_entropy_two(x_i, z_i)
    h_yz, k_yz = _entropy_from_classes(yz_dense)
    h_xyz, k_xyz = _joint_entropy_two(x_i, yz_dense)
    cmi_plugin = h_xz + h_yz - h_z - h_xyz
    # Plug-in CMI = H(XZ) + H(YZ) - H(Z) - H(XYZ). Each plug-in entropy
    # is biased low by (K-1)/(2n). The CMI bias from combining them
    # (with signs +H_xz +H_yz -H_z -H_xyz, where each contributes
    # -(K-1)/(2n) to the plug-in vs true entropy) is:
    #   E[CMI_plugin] - CMI_true
    #   = -((k_xz-1) + (k_yz-1) - (k_z-1) - (k_xyz-1))/(2n)
    #   = (k_xyz + k_z - k_xz - k_yz)/(2n).
    # On noise frames k_xyz - k_xz dominates (XYZ has many empty cells
    # filled by noise) so plug-in CMI is biased UP - subtract the
    # bias to denoise.
    cmi_bias = (k_xyz + k_z - k_xz - k_yz) / (2.0 * n)
    cmi = max(0.0, cmi_plugin - cmi_bias)
    return (cmi, (int(k_z), int(k_xz), int(k_yz), int(k_xyz))) if return_cards else cmi

# Split out to `_mi_greedy_cmi_fe_gpu.py` for module size; re-exported by name so existing imports keep working.
from ._mi_greedy_cmi_fe_gpu import (
    _ENT_NNZ_KERNEL as _ENT_NNZ_KERNEL,
    _ENT_NNZ_SRC as _ENT_NNZ_SRC,
    _ENT_RK as _ENT_RK,
    _NNZ_RK as _NNZ_RK,
    _YZ_CARD_CACHE as _YZ_CARD_CACHE,
    _YZ_CARD_CACHE_LOCK as _YZ_CARD_CACHE_LOCK,
    _cmi_from_binned_cupy as _cmi_from_binned_cupy,
    _ent_from_counts as _ent_from_counts,
    _get_ent_nnz_kernel as _get_ent_nnz_kernel,
    _nnz_from_counts as _nnz_from_counts,
    joint_cardinalities_cupy as joint_cardinalities_cupy,
)

# ---------------------------------------------------------------------------
# Public CMI scorer
# ---------------------------------------------------------------------------

_PARALLEL_CMI_MIN_CANDS = 16  # below this many candidates the pool costs more than it saves


_CMI_CHUNK_LOCK = threading.Lock()
_CMI_CHUNK_START: dict[int, int] = {}  # candidate count -> chunk width that last fitted on the device
_CMI_CHUNK_BY_SHAPE: dict[tuple, int] = {}  # (rows, conditioned) -> the last chunk width that had to be reduced to fit


def _conditioning_is_all_singletons(z_joint: Any, n_rows: int) -> bool:
    """Whether the conditioning support ``z_joint`` holds a distinct value in every one of the ``n_rows`` rows (``False`` for no support, a wrong length or any fault)."""
    if z_joint is None or n_rows < 2:
        return False
    try:
        z = np.asarray(z_joint.get() if hasattr(z_joint, "get") else z_joint).ravel()
        return int(z.shape[0]) == n_rows and int(np.unique(z).size) == n_rows
    except Exception as exc:
        logger.debug("conditioning-support check failed (%s); scoring on the device", exc)
        return False


def _batched_cmi_in_chunks(codes: Any, y_bin: np.ndarray, z_joint: Any, batched_cmi_gpu: Any) -> np.ndarray:
    """``batched_cmi_gpu`` over all columns of ``codes``; when the device cannot hold the whole batch's joint histograms (a wide conditioning support makes them ``K x Kx x Kyz`` cells) the
    columns are scored in chunks, halving the chunk until it fits. Every column is scored independently, so the values are the whole-batch values."""
    import cupy as cp

    k = int(codes.shape[1])
    if _conditioning_is_all_singletons(z_joint, int(codes.shape[0])):
        # every row is its own conditioning cell: each cell holds one row, so the conditional MI is exactly 0 for every column (the kernel returns rounding noise of about 1e-15 there, far below every floor the scores are compared with)
        return np.zeros(k, dtype=np.float64)  # Intentional float64: score output contract kept f64 (exact comparison)

    def attempt(step: int) -> "np.ndarray | None":
        """All columns scored in chunks of ``step``; ``None`` when the device ran out of memory."""
        try:
            parts = [batched_cmi_gpu(codes[:, s : s + step], y_bin, z_joint, codes_trusted=True) for s in range(0, k, step)]
        except cp.cuda.memory.OutOfMemoryError:
            logger.debug("batched CMI of %d columns in chunks of %d ran out of device memory", k, step)
            return None
        scored: np.ndarray = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return scored

    # The chunk that fitted last time for this width, else the one that fitted last for this row count and conditioning: a failed whole-batch attempt costs an allocation sweep and a pool reclaim.
    shape_key = (int(codes.shape[0]), z_joint is None)
    with _CMI_CHUNK_LOCK:
        step = min(k, _CMI_CHUNK_START.get(k, _CMI_CHUNK_BY_SHAPE.get(shape_key, k)))
    while True:
        out = attempt(step)
        if out is not None:
            with _CMI_CHUNK_LOCK:
                _CMI_CHUNK_START[k] = step
                if step < k:
                    _CMI_CHUNK_BY_SHAPE[shape_key] = step
            return out
        if step <= 1:
            whole: np.ndarray = batched_cmi_gpu(codes, y_bin, z_joint, codes_trusted=True)  # nothing left to halve: let the error reach the caller's CPU fallback
            return whole
        step = max(1, step // 2)
        log_throttle(
            logger, "cmi_batch_oom_chunking", logging.WARNING,
            "batched CMI of %d candidates ran out of device memory; scoring in chunks of %d columns", k, step,
        )


def _support_scorer(y_bin: np.ndarray, z_joint: np.ndarray) -> "Callable":
    """``codes_block -> CMI(codes; y | z)`` per row of the block, with the y/z terms computed once; ``-1`` marks a column the kernel leaves to the generic path."""
    from ._numba_utils import cmi_columns_given_support

    y_i = np.ascontiguousarray(y_bin, dtype=np.int64)
    z_i = np.ascontiguousarray(z_joint, dtype=np.int64)
    yz_dense, _ = _renumber_joint(y_i, z_i)
    h_z, k_z = _entropy_from_classes(z_i)
    h_yz, k_yz = _entropy_from_classes(yz_dense)
    yz_dense = np.ascontiguousarray(yz_dense, dtype=np.int64)

    def score(codes: np.ndarray) -> np.ndarray:
        """The CMI of every row of ``codes`` given the support."""
        out = np.empty(codes.shape[0], dtype=np.float64)  # Intentional float64: score output contract kept f64 (exact comparison)
        with parallel_kernel_entry():
            cmi_columns_given_support(codes, z_i, yz_dense, float(h_z), int(k_z), float(h_yz), int(k_yz), out)
        return out

    return score


def quantile_bin_columns(X: np.ndarray, nbins: int) -> "list[np.ndarray]":
    """``_quantile_bin`` of every column of the host matrix ``X`` ``(n, k)``, as one int64 vector per column (the codes are consumed on the host).

    The all-finite columns are binned in parallel blocks by ``quantile_bin_block`` (the dedupe and two-class rules of ``_quantile_bin``, equal codes); a column with a NaN or infinity takes the scalar
    host ``_quantile_bin``. Looping ``_quantile_bin`` over the columns ran one full device round trip (upload, two sorts, ~15 launches, download) per column on the strict GPU path."""
    from ._numba_utils import quantile_bin_block

    n, k = int(X.shape[0]), int(X.shape[1])
    out: "list[np.ndarray | None]" = [None] * k
    dt = _crit_np_dtype()
    block = max(1, (128 << 20) // (np.dtype(dt).itemsize * max(n, 1)))
    for j0 in range(0, k, block):
        j1 = min(k, j0 + block)
        v = np.ascontiguousarray(np.asarray(X[:, j0:j1], dtype=dt).T)
        ok = np.isfinite(v).all(axis=1)
        if ok.any():
            codes = np.empty((int(ok.sum()), n), dtype=np.int16)
            with parallel_kernel_entry():
                quantile_bin_block(np.ascontiguousarray(v[ok]), int(nbins), codes)
            r = 0
            for j in range(j1 - j0):
                if ok[j]:
                    out[j0 + j] = codes[r].astype(np.int64)
                    r += 1
        for j in range(j1 - j0):
            if out[j0 + j] is None:
                out[j0 + j] = _quantile_bin(np.ascontiguousarray(v[j]), nbins=int(nbins), host_only=True)
    return cast("list[np.ndarray]", out)


def _prebinned_candidates(X_cand: Any, cand_cols: list, nbins: int, score: "Callable | None" = None) -> dict:
    """``{column: codes}`` of the finite candidate columns, binned in parallel blocks exactly as ``_quantile_bin`` would bin each (empty under the strict-resident GPU path, which bins on the device).

    With ``score`` the values are ``score(codes_block)[row]`` instead (one value per column), so a block of codes is dropped as soon as it is scored."""
    from ._gpu_strict_fe import fe_gpu_strict_resident_enabled
    from ._numba_utils import quantile_bin_block

    if len(cand_cols) < _PARALLEL_CMI_MIN_CANDS or fe_gpu_strict_resident_enabled():
        return {}
    n_rows = len(X_cand)
    dt = _crit_np_dtype()
    block = max(1, (128 << 20) // (np.dtype(dt).itemsize * max(n_rows, 1)))
    out: dict = {}
    for j0 in range(0, len(cand_cols), block):
        names = cand_cols[j0 : j0 + block]
        v = np.empty((len(names), n_rows), dtype=dt)
        for r, c in enumerate(names):
            v[r] = X_cand[c].to_numpy()
        ok = np.isfinite(v).all(axis=1)
        if not ok.any():
            continue
        v = v[ok]
        codes = np.empty(v.shape, dtype=np.int16)
        with parallel_kernel_entry():
            quantile_bin_block(v, nbins, codes)
        vals = score(codes) if score is not None else codes
        for r, c in enumerate(np.asarray(names, dtype=object)[ok]):
            out[c] = vals[r]
    return out


def _binned_cmi_in_column_blocks(cp: Any, X_float: np.ndarray, nbins: int, y_bin: np.ndarray, z_joint: Any, batched_cmi_gpu: Any, batched_quantile_bin_gpu: Any) -> np.ndarray:
    """``CMI(candidate; y | z)`` of every column of the host matrix ``X_float`` ``(n, k)``: the columns are binned and scored on the device in blocks sized from the free VRAM.

    Binning the whole candidate matrix at once converts it to a double-precision device copy plus the binner's temporaries (1920 candidates of 100000 rows: a 1.5 GB allocation against a 2.6 GB pool limit),
    so the device path failed with an out-of-memory error and the whole batch fell to the per-candidate CPU loop. A block that still does not fit is halved and retried; a single column that does not fit raises."""
    n, k = int(X_float.shape[0]), int(X_float.shape[1])
    if _conditioning_is_all_singletons(z_joint, n):
        return np.zeros(k, dtype=np.float64)  # Intentional float64: score output contract kept f64; no column needs binning or uploading when every row is its own conditioning cell
    per_col = max(1, n * 36)  # f32 upload + f64 copy + the binner's temporaries + int64 codes
    free = int(cp.cuda.runtime.memGetInfo()[0])
    block = int(max(1, min(k, (free // 3) // per_col)))
    parts: list = []
    j0 = 0
    while j0 < k:
        j1 = min(k, j0 + block)
        try:
            codes_dev = batched_quantile_bin_gpu(cp.asarray(np.ascontiguousarray(X_float[:, j0:j1])), nbins)
            parts.append(np.asarray(_batched_cmi_in_chunks(codes_dev, y_bin, z_joint, batched_cmi_gpu)))
            del codes_dev
            j0 = j1
        except cp.cuda.memory.OutOfMemoryError:
            cp.get_default_memory_pool().free_all_blocks()
            if block <= 1:
                raise
            block = max(1, block // 2)
    # Intentional float64: score output contract kept f64 (exact comparison)
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)


def score_candidates_by_cmi(
    X_cand: pd.DataFrame,
    y: np.ndarray,
    X_support: Optional[pd.DataFrame] = None,
    *,
    nbins: int = 10,
) -> pd.Series:
    """Score every candidate column by ``CMI(candidate; y | support_joint)``.

    Parameters
    ----------
    X_cand : DataFrame
        Remaining candidate columns.
    y : ndarray
        Target; promoted to int64 if not already integer-typed.
    X_support : DataFrame or None
        Currently-selected support columns. ``None`` (or empty) -> CMI
        reduces to marginal ``MI(candidate; y)`` and the function behaves
        as a batch-MI scorer (useful for the seed step).
    nbins : int
        Bins per column for equi-frequency quantile binning.

    Returns
    -------
    pd.Series indexed by ``X_cand.columns`` holding the CMI value for each.
    """
    if X_cand.empty:
        return pd.Series(dtype=_crit_np_dtype())
    # This used to `.astype(np.int64)` (TRUNCATE) any
    # non-integer y BEFORE the np.unique densify below - for a continuous y confined to one integer
    # bucket (e.g. a [0,1) probability), truncation collapses every distinct value to the SAME integer
    # first, so the subsequent np.unique can no longer recover the distinctness (the exact B-18 bug class
    # already fixed in 7 sibling orth-scoring files via _coerce_y_int64, but never applied here). Densify
    # directly via np.unique on the RAW y instead - safe for an already-integer y too (same result).
    y_arr = np.asarray(y)
    # Bin y by unique-value remap (renumbers to dense 0..K-1; the caller's y may already be class-typed,
    # in which case this is a no-op renumber, or a raw continuous target, in which case this is what
    # actually turns it into usable class codes without truncation).
    _, y_bin = np.unique(y_arr, return_inverse=True)
    y_bin = y_bin.astype(np.int64)

    if X_support is None or X_support.shape[1] == 0:
        z_joint: Optional[np.ndarray] = None
    else:
        sup_bins = quantile_bin_columns(X_support.to_numpy(), nbins)
        z_joint, _ = _renumber_joint(*sup_bins)

    cand_cols = list(X_cand.columns)
    # BATCHED born-on-device path under STRICT (default OFF -> per-candidate CPU loop, byte-identical):
    # bin all candidates into one (n, K) code matrix and score CMI for EVERY candidate in ONE device
    # workload (batched_cmi_gpu), instead of a per-candidate cp.unique CMI. Parity-pinned selection-equiv.
    if (_cmi_gpu_enabled(n=int(y_bin.shape[0]), p=len(cand_cols)) or _cmi_candidate_batch_on_device(n=int(y_bin.shape[0]), p=len(cand_cols))) and len(cand_cols) > 1:
        try:
            from ._fe_batched_mi import batched_cmi_gpu, batched_quantile_bin_gpu
            # filled as contiguous rows of the transpose (a column write into a C-ordered matrix strides over every row); the F-ordered view keeps every column block contiguous
            X_rows = np.empty((len(cand_cols), y_bin.shape[0]), dtype=_crit_np_dtype())
            for j, c in enumerate(cand_cols):
                X_rows[j] = X_cand[c].to_numpy()
            X_float = X_rows.T
            if np.isfinite(X_float).all():
                # Born-on-device: bin the whole candidate matrix on the GPU (one batched cp.percentile
                # sort) and keep the codes RESIDENT, scoring CMI on them without a code H2D round-trip.
                import cupy as cp
                # codes_trusted: binner-produced (batched_quantile_bin_gpu) + dense renumbered y/z are 0-based ->
                # the range guard cannot fire; skip its 2 blocking min/max syncs on the resident hot path (FIX1).
                cmis = _binned_cmi_in_column_blocks(cp, X_float, nbins, y_bin, z_joint, batched_cmi_gpu, batched_quantile_bin_gpu)
            else:
                # Non-finite columns -> host equi-freq binning (handles nan/inf), then device CMI.
                X_codes = np.empty((y_bin.shape[0], len(cand_cols)), dtype=np.int64)
                for j in range(len(cand_cols)):
                    X_codes[:, j] = _quantile_bin(X_float[:, j], nbins=nbins)
                cmis = _batched_cmi_in_chunks(X_codes, y_bin, z_joint, batched_cmi_gpu)  # host equi-freq binner -> 0-based
            # Intentional float64: score output contract kept f64 (exact comparison)
            return pd.Series({c: float(cmis[j]) for j, c in enumerate(cand_cols)}, dtype=np.float64)
        except Exception as e:  # nosec B110 - optional/best-effort path, rationale documented
            log_throttle(
                logger, "cmi_gpu_batched_failed", logging.WARNING,
                "GPU-resident batched CMI of %d candidates failed (%s: %s); scoring them one by one on the CPU", len(cand_cols), type(e).__name__, e,
            )
    pre_scores: dict = {}
    pre_codes: dict = {}
    if z_joint is not None and not _cmi_gpu_enabled(n=int(y_bin.shape[0]), p=1):
        pre_scores = _prebinned_candidates(X_cand, cand_cols, nbins, score=_support_scorer(y_bin, z_joint))
        pre_scores = {c: v for c, v in pre_scores.items() if v >= 0.0}
    else:
        pre_codes = _prebinned_candidates(X_cand, cand_cols, nbins)

    def _score(c: str) -> float:
        """CMI of one candidate: its quantile bins against ``y`` given the support."""
        if c in pre_scores:
            return float(pre_scores[c])
        codes = pre_codes.get(c)
        return float(_cmi_from_binned(codes.astype(np.int64) if codes is not None else _quantile_bin(X_cand[c].to_numpy(), nbins=nbins), y_bin, z_joint))

    # The candidates are independent and each score is a pure function of its own column, so a thread pool returns the serial values (the numpy partition and the njit CMI kernel release the GIL:
    # 300 columns of 20000 rows, 0.39 s -> 0.22 s on 8 threads). The first candidate runs on this thread so numba compiles once.
    from ._adaptive_nbins import _resolve_edge_jobs

    jobs = _resolve_edge_jobs(None, len(cand_cols) - len(pre_scores))
    if jobs > 1 and len(cand_cols) - len(pre_scores) >= _PARALLEL_CMI_MIN_CANDS:
        from concurrent.futures import ThreadPoolExecutor

        first = _score(cand_cols[0])
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            rest = list(pool.map(_score, cand_cols[1:]))
        scores = [first, *rest]
    else:
        scores = [_score(c) for c in cand_cols]
    # Intentional float64: score output contract kept f64 (exact comparison)
    return pd.Series(dict(zip(cand_cols, scores)), dtype=np.float64)

# ---------------------------------------------------------------------------
# End-to-end greedy CMI constructor
# ---------------------------------------------------------------------------

def greedy_cmi_fe_construct(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    seed_cols_count: int = 4,
    top_k: int = 5,
    include_unary: bool = True,
    include_binary: bool = True,
    include_trig_on_bounded: bool = True,
    min_cmi_gain: float = 0.005,
    nbins: int = 10,
    seed: int = 0xC011,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end CMI-greedy feature constructor.

    ``seed``: seeds the noise-floor permutation RNG
    (previously hardcoded to ``0xC011`` with no way to vary it). Defaults to the historical constant so
    existing pinned tests/behaviour stay byte-identical when the caller does not override it; pass
    ``self.random_seed`` (or any other seed) to decorrelate the FE admission gate across nominally-
    independent bootstrap/multi-seed replicates (stability selection, seed ensembles).

    Pipeline:

    1. Enumerate UNARY candidates over the FULL numeric column pool (NOT a
       top-N seed pool). The whole point of CMI ranking is that columns
       with near-zero marginal ``MI(x; y)`` can still emit transforms that
       carry the signal - on ``y = sign(x^2 - 1)``, ``MI(x; y) ~= 0``
       because ``x`` is symmetric, yet ``square(x)`` is perfectly
       informative. Restricting unary enumeration to the top-N raw-MI
       cols would discard exactly the signal Layer 60 is designed to
       recover. BINARY candidates are enumerated only over the
       top-``seed_cols_count`` raw cols by marginal ``MI(x; y)`` because
       the pair explosion is O(N^2 * |BINARY_TRANSFORMS|) and quickly
       exceeds the gain.
    2. Materialise every candidate; drop near-constants.
    3. Start the conditioning support Z EMPTY (Z grows step-by-step from
       the greedy loop). This avoids the fragmentation trap of dumping
       several raw cols into Z up front: when several raw cols enter Z,
       the joint Z cardinality climbs into the hundreds and the CMI of
       any candidate collapses toward noise (cells average < 5 samples).
       The greedy loop itself caps Z growth with the contingency budget
       below.
    4. Greedy loop: compute ``CMI(cand; y | support)`` for every remaining
       candidate, pick the highest, add it to support if it clears
       ``min_cmi_gain``; otherwise stop. Z is grown ONLY when the resulting
       joint cardinality stays under ``n / 5`` (chi-squared rule of
       thumb: cells must average >= 5 samples for CMI to be stable).
       Past that cap, the winner is still appended but Z is frozen, so
       subsequent CMI gains stay measurable.
    5. Append winners to X; return (X_augmented, scores) where ``scores``
       is a DataFrame with one row per appended column ordered by
       selection sequence.
    """
    _ret_greedy_cmi_fe_construct_part1 = _greedy_cmi_fe_construct_part1(
        cols, X, y, nbins, seed_cols_count, include_unary, include_trig_on_bounded, include_binary,
    )
    if _ret_greedy_cmi_fe_construct_part1[0]:
        _early_greedy_cmi_fe_construct_part1: tuple[pd.DataFrame, pd.DataFrame] = _ret_greedy_cmi_fe_construct_part1[1]
        return _early_greedy_cmi_fe_construct_part1
    st, y_bin, engineered, z_joint, z_joint_dev, cand_names, _bin_fingerprint = _ret_greedy_cmi_fe_construct_part1[1:]

    # RESIDENT candidate codes: when GPU-strict-resident is on, bin every candidate ONCE on-device
    # (``cand_bins_dev``) and fingerprint via a device-side reduction instead of the host ``_quantile_bin`` +
    # ``.tobytes()`` pair below. This closes the confirmed residency gap (``_fe_gpu_strict.py`` docstring, "KNOWN
    # NON-PRODUCTION RESIDENCY GAP"): the host path forces one bulk (n,) D2H PER CANDIDATE just to hash its
    # bytes. ``cand_bins`` (host) stays LAZY here - populated on demand only for the small subset of names the
    # Z-fold step actually touches (at most ``top_k`` winners, via ``_host_bins`` below), not eagerly for every
    # candidate. Falls back to the fully-eager host path (unchanged, byte-identical) on any cupy failure or when
    # GPU-strict-resident is off - selection-identical either way (fingerprint is a dedup key, not a scored
    # quantity; a 64-bit reduction hash is the same collision-domain acceptance the resident cache
    # (``_content_hash``) already uses elsewhere in this codebase).
    cand_bins_dev: dict = {}
    st.cand_fp = {}
    st._resident_fp_ok = False
    _host_bins, cand_bins_dev, rng_floor = _greedy_cmi_fe_constr_step1_content_hash_already(cand_names, engineered, nbins, st, _bin_fingerprint, seed, cand_bins_dev)

    _noise_floor_for_current_z = _greedy_cmi_fe_constr_step2_rng_raw_small(cand_names, rng_floor, y_bin, z_joint_dev, z_joint, cand_bins_dev, _host_bins, st)
    _greedy_cmi_fe_constr_step3_while_st_remaining(st, top_k, _noise_floor_for_current_z, min_cmi_gain, z_joint_dev, z_joint, y_bin, cand_bins_dev, _host_bins, nbins)

    st.scores = pd.DataFrame(st.rows, columns=[
        "engineered_col", "transform", "source_cols",
        "cmi_at_selection", "step",
    ])
    if st.winners:
        st.X_aug = pd.concat([X, engineered[st.winners]], axis=1)
    else:
        st.X_aug = X.copy(deep=False)
    return st.X_aug, st.scores


def _greedy_cmi_fe_construct_part1(
    cols: Optional[Sequence[str]], X: pd.DataFrame, y: np.ndarray, nbins: int, seed_cols_count: int, include_unary: bool, include_trig_on_bounded: bool,
    include_binary: bool,
) -> tuple[Any, ...]:
    """Extracted block of ``greedy_cmi_fe_construct`` (part 1); returns the names the caller still uses."""
    st = _SimpleNamespace()  # long-lived locals of this function (see the stage helpers below)
    from ._mi_greedy_fe import (
        generate_mi_greedy_features,
        iter_candidates,
    )
    from ._orthogonal_univariate_fe import _mi_classif_batch

    st.empty_scores = pd.DataFrame(columns=[
        "engineered_col", "transform", "source_cols",
        "cmi_at_selection", "step",
    ])

    st._cols_source = cols if cols is not None else X.columns
    st.candidates_pool = [c for c in st._cols_source if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    if not st.candidates_pool:
        return True, (X, st.empty_scores)

    # This used to `.astype(np.int64)` (TRUNCATE) any
    # non-integer y here, BEFORE both the raw_mi scoring below AND the np.unique densify further down -
    # for a continuous y confined to one integer bucket, truncation collapses every distinct value to the
    # SAME integer first, so densification downstream can no longer recover the distinctness (the B-18 bug
    # class). Densify via np.unique ONCE, up front, and reuse the dense y_bin everywhere below - safe for
    # an already-integer/already-dense y too (pure renumber, no-op if already 0..K-1).
    st.y_arr = np.asarray(y)
    _, y_bin = np.unique(st.y_arr, return_inverse=True)
    y_bin = y_bin.astype(np.int64)

    # 1. Pick the top-N raw cols by marginal MI as the BINARY-pair source
    #    pool (controls the O(N^2 * |BINARY_TRANSFORMS|) explosion).
    #    Unary candidates still enumerate over the full pool below.
    st.raw_arr = X[st.candidates_pool].to_numpy(dtype=_crit_np_dtype())
    st.raw_mi = _mi_classif_batch(st.raw_arr, y_bin, nbins=nbins)
    st.order = np.argsort(-st.raw_mi, kind="stable")  # plug-in MI is quantised, so ties are real: break them by position, reproducibly
    st.binary_seed_cols = [st.candidates_pool[i] for i in st.order[: int(seed_cols_count)]] if int(seed_cols_count) > 0 else list(st.candidates_pool)

    # 2. Enumerate candidates. UNARY over the full pool (so transforms on
    #    symmetric / interaction-only cols are never silently dropped),
    #    BINARY only over the seeded subset (pair explosion control).
    st.cands = []
    if include_unary:
        st.cands.extend(iter_candidates(
            X, cols=st.candidates_pool,
            include_unary=True, include_binary=False,
            include_trig_on_bounded=include_trig_on_bounded,
        ))
    if include_binary:
        st.cands.extend(iter_candidates(
            X, cols=st.binary_seed_cols,
            include_unary=False, include_binary=True,
            include_trig_on_bounded=False,
        ))
    engineered, st.parsed = generate_mi_greedy_features(X, st.cands)
    if engineered.empty:
        return True, (X, st.empty_scores)

    # 3. y_bin was already densified up front (reuses it here, no re-densify
    #    needed). Start Z EMPTY. Z grows step-by-step from greedy picks (under the fragmentation cap
    #    below). Starting with several raw cols in Z up front pushes joint Z cardinality past the
    #    chi-squared contingency budget and collapses every candidate's CMI toward noise - defeats the
    #    purpose of CMI ranking.
    st.n_samples = int(y_bin.size)
    st.frag_cap = max(2, st.n_samples // 5)
    # RESIDENT fit-constant y for the GPU-strict greedy hot path: upload y_bin to the device ONCE here and hand
    # the SAME cupy array to every per-round ``batched_cmi_gpu`` call (it uses an already-resident y as-is, no
    # re-upload). Routed through ``resident_operand`` only to upload (and to be cleared at FE teardown); kept on a
    # DISTINCT role from the per-permutation shuffled y so the transient null-draw y can never evict it. None when
    # GPU is off (host path uses y_bin directly, byte-identical).
    st.y_bin_dev = None
    if _cmi_gpu_enabled(n=st.n_samples):
        try:
            from ._fe_resident_operands import resident_code_operand as _resident_code_operand
            st.y_bin_dev = _resident_code_operand(y_bin, "cmi_greedy_y_fixed")
        except Exception as e:
            logger.warning("resident_code_operand for y_bin failed, falling back to the per-call upload path: %s", e)
            st.y_bin_dev = None
    z_joint: Optional[np.ndarray] = None
    z_joint_dev = None  # resident twin of z_joint, maintained alongside it once cand_bins_dev is available
    st.z_card = 1

    # 4. Bin every engineered candidate up front. Compute a sortable
    #    bin fingerprint (tuple of the sorted unique bin counts) used
    #    below for monotone-equivalence dedup against already-picked
    #    winners - when Z hits the fragmentation cap (frozen Z),
    #    monotone-equivalent candidates would otherwise tie at the same
    #    plug-in CMI and all get picked.
    cand_names = list(engineered.columns)
    st.name_to_parsed = dict(zip(cand_names, st.parsed))

    def _bin_fingerprint(b: np.ndarray) -> bytes:
        """Hashable fingerprint of a binned candidate array; monotone-equivalent candidates (identical bin assignment under equi-frequency quantization) collapse to the same fingerprint for dedup against already-picked winners."""
        return b.tobytes()
    return False, st, y_bin, engineered, z_joint, z_joint_dev, cand_names, _bin_fingerprint

def greedy_cmi_fe_construct_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    seed_cols_count: int = 4,
    top_k: int = 5,
    include_unary: bool = True,
    include_binary: bool = True,
    include_trig_on_bounded: bool = True,
    min_cmi_gain: float = 0.005,
    nbins: int = 10,
    seed: int = 0xC011,
) -> tuple[pd.DataFrame, pd.DataFrame, list[EngineeredRecipe]]:
    """Same as :func:`greedy_cmi_fe_construct` but additionally returns a list
    of ``EngineeredRecipe`` objects (one per appended column) so MRMR.transform
    can replay each column on test data without re-running CMI scoring AND
    without referencing y.

    Recipes reuse kind ``"mi_greedy_transform"`` (same as Layer 26) so the
    replay code path is shared infrastructure.

    ``seed``: see :func:`greedy_cmi_fe_construct`'s matching parameter.
    """
    from ._mi_greedy_fe import _parse_binary_name, _parse_unary_name
    from .engineered_recipes import build_mi_greedy_transform_recipe

    X_aug, scores = greedy_cmi_fe_construct(
        X, y,
        cols=cols, seed_cols_count=seed_cols_count, top_k=top_k,
        include_unary=include_unary, include_binary=include_binary,
        include_trig_on_bounded=include_trig_on_bounded,
        min_cmi_gain=min_cmi_gain, nbins=nbins, seed=seed,
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    recipes = []
    for name in appended:
        parsed_bin = _parse_binary_name(name)
        parsed_un = _parse_unary_name(name)
        if parsed_bin is not None:
            tname, col_i, col_j = parsed_bin
            recipes.append(build_mi_greedy_transform_recipe(
                name=name, transform=tname, src_names=(col_i, col_j),
            ))
        elif parsed_un is not None:
            tname, col = parsed_un
            recipes.append(build_mi_greedy_transform_recipe(
                name=name, transform=tname, src_names=(col,),
            ))
        else:
            log_throttle(
                logger,
                "mi_greedy_cmi_fe_cannot_parse_column",
                logging.WARNING,
                "greedy_cmi_fe_construct_with_recipes: cannot parse " "engineered column %r back to (transform, source); skipping " "recipe.",
                name,
            )
    return X_aug, scores, recipes
