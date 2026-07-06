"""Backend-dispatched subset-ranking for the exact SHAP-coalition proxy scan.

The subset-ranking step scores many candidate feature subsets by the additive proxy margin
``base[i] + sum_{j in S} phi[i, j]`` reduced through a proper loss (Brier / log-loss / RMSE / MAE),
and keeps the top-N. It is an embarrassingly-parallel map(subset -> loss) + reduce(argmin) -- the GPU
kernel in ``_shap_proxy_gpu`` runs one thread per subset.

This module adds the missing two halves of the "fastest path is the default, routed by a HW-aware
dispatcher" contract (CLAUDE.md numerical-kernel ladder):

  1. ``_subset_loss_scan_njit`` -- a CPU njit REFERENCE that mirrors the GPU kernel's per-subset
     layout exactly (naive full re-sum per subset, no incremental-prefix trick), so it is
     bit-identical to the GPU kernel BY CONSTRUCTION and serves as the safe fallback + the unit
     oracle. The production CPU path stays the incremental ``brute_force_top_n`` (faster); this naive
     scan exists for cross-backend bit-identity verification and as the dispatcher's GPU mirror.

  2. ``brute_force_top_n_dispatch`` -- a stateless, size-aware dispatcher. It DEFAULTS TO CPU and only
     routes to the GPU kernel when (a) the caller opts in (``prefer_gpu``), (b) cupy + a device are
     available, and (c) the kernel_tuning_cache crossover (or its measured fallback) says the subset
     count is past the GPU break-even. On ANY cupy unavailability / OOM / kernel error it auto-falls
     back to the CPU incremental kernel (catch + log ONCE) so a flaky GPU host never fails a fit.

HOST CAVEAT (this dev box): importing cupy under contention native-segfaults the training process, so
the dispatcher default is CPU here regardless of the measured GPU win (1.04-1.96x, bit-identical, see
``_benchmarks/bench_shap_subsetrank_backends.py``). The GPU route stays gated off until a stable host
measures its own crossover into the kernel_tuning_cache. The win is real and hardware-relative -- kept,
not deleted (REJECTED != DELETED): a datacenter GPU box flips it on by setting ``prefer_gpu=True``.
"""

from __future__ import annotations

import logging
import math
import os

import numpy as np
from numba import njit, prange

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import METRIC_CODES, resolve_metric, score_margin
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import brute_force_top_n, generate_combinations, _merge_topn

logger = logging.getLogger(__name__)

# Measured GPU break-even on the dev box (RTX, cupy 14, see bench): the GPU kernel ties the parallel
# CPU kernel around ~3e4 subsets and wins ~2x past ~2.5e5. Conservative fallback used ONLY when the
# kernel_tuning_cache has no per-HW region; never hardcoded into the route when the cache is present.
_DEFAULT_GPU_MIN_SUBSETS = 250_000

_fallback_logged = False


@njit(cache=True)
def _subset_loss_scan_njit(phi, base, y, combos, metric_code, out):
    """CPU reference: one subset per outer iteration, full re-sum -- mirrors the GPU thread layout.

    ``phi`` row-major (n, f), ``combos`` shape (C, r); writes the per-subset loss into ``out`` (len C).
    Deliberately NON-incremental (recomputes ``base + sum phi[:, comb]`` from scratch each subset) so it
    is bit-identical to the cupy kernel by construction -- the dispatcher's GPU/CPU cross-check oracle.
    Serial; the production CPU path is the faster incremental ``brute_force_top_n``."""
    C = combos.shape[0]
    r = combos.shape[1]
    n = phi.shape[0]
    margin = np.empty(n, dtype=np.float64)
    for c in range(C):
        comb = combos[c]
        for t in range(n):
            m = base[t]
            for jj in range(r):
                m += phi[t, comb[jj]]
            margin[t] = m
        out[c] = score_margin(margin, y, metric_code)


@njit(cache=True, parallel=True)
def _subset_loss_scan_njit_parallel(phi, base, y, combos, metric_code, out):
    """prange twin of :func:`_subset_loss_scan_njit` (one thread per subset, private margin buffer).

    Each subset is independent so the prange is a pure map with no cross-iteration state; the per-thread
    ``margin`` buffer is allocated inside the loop body. Same naive full re-sum -> same bit-identity to
    the GPU kernel. Used by the dispatcher as the GPU mirror when verifying / as a CPU fallback that
    matches the GPU numerics exactly."""
    C = combos.shape[0]
    r = combos.shape[1]
    n = phi.shape[0]
    for c in prange(C):
        comb = combos[c]
        margin = np.empty(n, dtype=np.float64)
        for t in range(n):
            m = base[t]
            for jj in range(r):
                m += phi[t, comb[jj]]
            margin[t] = m
        out[c] = score_margin(margin, y, metric_code)


def brute_force_top_n_cpu_ref(
    phi: np.ndarray,
    base: np.ndarray,
    y: np.ndarray,
    *,
    classification: bool,
    metric: str | None = None,
    min_card: int = 1,
    max_card: int | None = None,
    top_n: int = 30,
    parallel: bool = True,
) -> list[tuple[float, tuple[int, ...]]]:
    """Top-N via the naive per-subset CPU scan (GPU-mirror numerics). Same contract as
    ``brute_force_top_n``; used to assert GPU/CPU bit-identity and as the OOM fallback that matches the
    GPU kernel exactly. The faster default CPU path remains the incremental ``brute_force_top_n``."""
    metric_name = resolve_metric(classification, metric)
    if metric_name == "auc":
        raise ValueError("subset scan does not support metric='auc' (use brier/logloss); AUC is Python-path only.")
    metric_code = METRIC_CODES[metric_name]
    phi = np.ascontiguousarray(phi, dtype=np.float64)
    base = np.ascontiguousarray(base, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    f = phi.shape[1]
    max_card = f if max_card is None else min(max_card, f)
    seq = np.arange(f, dtype=np.int32)
    scan = _subset_loss_scan_njit_parallel if parallel else _subset_loss_scan_njit
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for r in range(min_card, max_card + 1):
        if math.comb(f, r) == 0:
            continue
        combos = generate_combinations(seq, r)
        C = combos.shape[0]
        if C == 0:
            continue
        out = np.empty(C, dtype=np.float64)
        scan(phi, base, y, combos, metric_code, out)
        # SR1: argpartition ordering is undefined with NaN, so a NaN loss (degenerate single-class slice)
        # could be selected as "top" (lowest). Map non-finite losses to +inf (worst) so they sink, never
        # win; lower=better, so +inf is correctly the least-preferred and is dropped downstream.
        out[~np.isfinite(out)] = np.inf
        k = min(top_n, C)
        sel = np.argpartition(out, k - 1)[:k]
        sel = sel[np.argsort(out[sel])]
        for ci in sel:
            candidates.append((float(out[ci]), tuple(int(x) for x in combos[ci])))
    merged = _merge_topn(candidates, top_n)
    if metric_name == "rmse":
        merged = [(float(np.sqrt(loss)), comb) for loss, comb in merged]
    return merged


def _gpu_min_subsets() -> int:
    """GPU break-even subset count from kernel_tuning_cache (key ``shap_proxy_subsetrank`` ->
    ``gpu_min_subsets``), falling back to the measured ``_DEFAULT_GPU_MIN_SUBSETS``.

    Never hardcoded into the route: a stable GPU host writes its own crossover; on this contended box
    the lookup misses and we keep the conservative measured default, which (with the CPU-default policy)
    means the GPU path is opt-in only. Env override ``MLFRAME_SHAP_SUBSETRANK_GPU_MIN_SUBSETS`` wins for
    quick A/B without a cache write."""
    env = os.environ.get("MLFRAME_SHAP_SUBSETRANK_GPU_MIN_SUBSETS", "")
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    try:
        from mlframe.feature_selection.filters import get_kernel_tuning_cache

        ktc = get_kernel_tuning_cache()
        if ktc is not None:
            entry = ktc.lookup("shap_proxy_subsetrank")
            if isinstance(entry, dict) and entry.get("gpu_min_subsets"):
                return int(entry["gpu_min_subsets"])
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logger.debug("suppressed in _shap_proxy_subsetrank.py:169: %s", e)
        pass
    return _DEFAULT_GPU_MIN_SUBSETS


def _total_subsets(n_features: int, min_card: int, max_card: int | None) -> int:
    max_card = n_features if max_card is None else min(max_card, n_features)
    return int(sum(math.comb(n_features, r) for r in range(min_card, max_card + 1)))


def brute_force_top_n_dispatch(
    phi: np.ndarray,
    base: np.ndarray,
    y: np.ndarray,
    *,
    classification: bool,
    metric: str | None = None,
    min_card: int = 1,
    max_card: int | None = None,
    top_n: int = 30,
    parallel: bool = True,
    prefer_gpu: bool = False,
    force_backend: str | None = None,
) -> list[tuple[float, tuple[int, ...]]]:
    """Size-aware subset-ranking dispatcher; DEFAULT backend is CPU (safe on every host).

    Routes to the cupy GPU kernel ONLY when all hold: ``prefer_gpu`` (or ``force_backend='gpu'``),
    cupy + a device are present, and the subset count is past the kernel_tuning_cache crossover
    (``_gpu_min_subsets``). On any cupy import / device / OOM / kernel error it auto-falls back to the
    CPU incremental kernel and logs ONCE. ``force_backend in {'cpu','cpu_ref','gpu'}`` overrides the
    route for tests / A/B (``cpu_ref`` = the naive per-subset scan that mirrors the GPU numerics).

    The CPU default is deliberate: this kernel is the cheapest pipeline stage and the local GPU host
    native-segfaults importing cupy under contention, so a measured-real GPU win (see bench) stays
    opt-in until a stable host tunes its own crossover into the cache. Bit-identical across backends."""
    global _fallback_logged
    backend = (force_backend or "").lower()
    n_features = phi.shape[1]
    n_sub = _total_subsets(n_features, min_card, max_card)
    want_gpu = backend == "gpu" or (backend == "" and prefer_gpu and n_sub >= _gpu_min_subsets())

    if backend == "cpu_ref":
        return brute_force_top_n_cpu_ref(phi, base, y, classification=classification, metric=metric,
                                         min_card=min_card, max_card=max_card, top_n=top_n, parallel=parallel)

    if want_gpu:
        try:
            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_gpu import brute_force_top_n_gpu, gpu_available

            if backend == "gpu" or gpu_available():
                return brute_force_top_n_gpu(phi, base, y, classification=classification, metric=metric, min_card=min_card, max_card=max_card, top_n=top_n)
            if not _fallback_logged:
                logger.warning("ShapProxiedFS subset-rank: GPU requested but no CUDA device; using CPU kernel.")
                _fallback_logged = True
        except Exception as exc:  # cupy missing / OOM / kernel compile failure -> CPU, never fail the fit
            if not _fallback_logged:
                logger.warning("ShapProxiedFS subset-rank: GPU backend unavailable (%s); using CPU kernel.", exc)
                _fallback_logged = True

    return brute_force_top_n(phi, base, y, classification=classification, metric=metric, min_card=min_card, max_card=max_card, top_n=top_n, parallel=parallel)
