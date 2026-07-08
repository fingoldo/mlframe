"""Unified kNN helper: hnswlib (HNSW) drop-in for sklearn.neighbors.NearestNeighbors.

Why: sklearn's NearestNeighbors falls back to ball_tree at d>20 with O(N^(1-1/d)) query
cost — at N=1M, d=64 this is 2-5 min total query phase. hnswlib uses HNSW graph index
with O(log N) per query — same workload runs in 50-200ms. ~10-50x speedup at d>20, N>500k.

Used by cdist, local_lift, RSD, BGM (the kNN-bottlenecked mechanisms).

Falls back to sklearn NearestNeighbors if hnswlib isn't installed. Returns ``(dists, ids)``
identical in shape and dtype to ``sklearn.NearestNeighbors.kneighbors(return_distance=True)``,
so callers can swap in without other code changes.

The HNSW-or-sklearn switch is automatic via ``prefer_hnsw_at_n`` (default 50000): below this
threshold sklearn's ball_tree is competitive and we skip the HNSW index-build overhead;
above it we get the speedup. The threshold matches the empirical crossover on a 16-core AVX2
box at d=20-100.

hnswlib parameters chosen for high-recall / acceptable build time on tabular FE workloads:
M=16 (graph degree), ef_construction=200, ef_search=max(k+8, 64). Empirically gives ~99% recall
vs exact kNN at these sizes; we don't validate against exact every call because the downstream
FE features are statistical aggregates and 99% recall is plenty for the signal they capture.
"""
from __future__ import annotations

import logging
import os
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

_HNSW_AVAILABLE: bool | None = None


def _check_hnsw_available() -> bool:
    """Cached check for hnswlib import.

    Opt-out via ``MLFRAME_DISABLE_HNSW=1``: forces the exact sklearn ``NearestNeighbors`` path and
    skips ``import hnswlib`` entirely. This is the escape hatch for hosts where hnswlib's native
    extension is unstable -- on some Windows / conda boxes ``import hnswlib`` triggers an MSVC-runtime
    / DLL access violation (a hard C-level crash uncatchable by ``try/except``) when its DLL is loaded
    into a process that already has other native extensions (cupy CUDA runtime, MKL/OpenBLAS) resident.
    The sklearn-exact path is the deterministic backend anyway (hnswlib is approximate), so disabling
    hnswlib only costs the large-N speedup, never correctness. Bit-identity-neutral when unset.
    """
    global _HNSW_AVAILABLE
    if _HNSW_AVAILABLE is None:
        if os.environ.get("MLFRAME_DISABLE_HNSW", "").strip() not in ("", "0", "false", "False"):
            _HNSW_AVAILABLE = False
            logger.info("[_knn_helper] MLFRAME_DISABLE_HNSW set; using exact sklearn NearestNeighbors " "(hnswlib import skipped).")
            return _HNSW_AVAILABLE
        try:
            import hnswlib  # noqa: F401
            _HNSW_AVAILABLE = True
        except ImportError:
            _HNSW_AVAILABLE = False
            logger.info(
                "[_knn_helper] hnswlib not installed; falling back to sklearn NearestNeighbors. "
                "Install hnswlib (pip install hnswlib) for 10-50x speedup at N>500k, d>20."
            )
    return _HNSW_AVAILABLE


def knn_search(
    X_subset: np.ndarray,
    X_query: np.ndarray,
    k: int,
    *,
    metric: str = "l2",
    prefer_hnsw_at_n: int = 50_000,
    hnsw_M: int = 16,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find k nearest neighbours of each X_query row in X_subset.

    Returns
    -------
    dists : (n_query, min(k, n_subset)) float32, EUCLIDEAN distances (sqrt of squared L2)
        Compatible with sklearn ``NearestNeighbors(...).kneighbors()`` default output.
    ids : (n_query, min(k, n_subset)) int64, row indices into X_subset

    Notes
    -----
    Uses hnswlib if available and N_subset >= prefer_hnsw_at_n. Otherwise sklearn
    NearestNeighbors with algorithm="auto" (which picks kd_tree at d<20, ball_tree else).

    For X_subset.shape[0] == 0 returns sentinel arrays (dists=1e6, ids=0) of shape
    (n_query, k) so callers can downstream-process without an empty-set branch.
    """
    n_sub = X_subset.shape[0]
    n_q = X_query.shape[0]
    if n_sub == 0:
        return (
            np.full((n_q, k), 1e6, dtype=np.float32),
            np.zeros((n_q, k), dtype=np.int64),
        )
    k_request = min(k, n_sub)

    # The prefer_hnsw_at_n=50_000 default is the documented empirical crossover.
    # When left at that sentinel, defer to the per-host tuned backend_choice (exact
    # sklearn vs approximate hnsw) from the kernel_tuning_cache via get_or_tune --
    # AVX-512 boxes cross over earlier, low-core ARM later. An explicit
    # prefer_hnsw_at_n override still wins (keeps the threshold semantics).
    if prefer_hnsw_at_n == 50_000:
        use_hnsw = _check_hnsw_available() and _KNN_SPEC.choose(n_subset=n_sub, d=int(X_subset.shape[1])) == "hnsw"
    else:
        use_hnsw = n_sub >= prefer_hnsw_at_n and _check_hnsw_available()
    if use_hnsw:
        import hnswlib
        d = X_subset.shape[1]
        space = "l2" if metric == "l2" else metric
        idx = hnswlib.Index(space=space, dim=d)
        idx.init_index(max_elements=n_sub, ef_construction=hnsw_ef_construction, M=hnsw_M)
        idx.add_items(np.ascontiguousarray(X_subset, dtype=np.float32), np.arange(n_sub))
        idx.set_ef(max(k_request + 8, hnsw_ef_search))
        try:
            labels, sq_dists = idx.knn_query(np.ascontiguousarray(X_query, dtype=np.float32), k=k_request)
        except Exception as exc:
            logger.warning("[_knn_helper] hnswlib knn_query failed (%s); falling back to sklearn.", exc)
            return _sklearn_fallback(X_subset, X_query, k_request)
        # hnswlib's 'l2' space returns SQUARED L2; take sqrt for sklearn-compat Euclidean.
        dists = np.sqrt(np.maximum(sq_dists, 0.0, dtype=np.float32)).astype(np.float32)
        return dists, labels.astype(np.int64)

    return _sklearn_fallback(X_subset, X_query, k_request)


def _sklearn_fallback(X_subset: np.ndarray, X_query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1).fit(X_subset)
    dists, ids = nn.kneighbors(X_query)
    return dists.astype(np.float32, copy=False), ids.astype(np.int64, copy=False)


# Per-host tuned crossover for exact (sklearn kd/ball-tree) vs approximate (hnsw)
# kNN. The two backends are NOT equivalent by design (hnsw is approximate -- we
# accept the recall tradeoff above the crossover), so the sweep's equivalence
# gate is DISABLED (equiv tol = inf); only wall-time decides. CPU-only (no GPU).
# Spans the ~50k crossover; capped (n<=50k, d<=32) so the one-time cold-start
# sweep is ~15s, not minutes -- sklearn ball_tree query at d=50/n=200k is slow.
# The catch-all extrapolates the largest-cell winner beyond the grid.
_KNN_SWEEP_N = [2_000, 10_000, 50_000]
_KNN_SWEEP_D = [8, 32]
_KNN_SALT = 1


def _make_knn_inputs(dims: dict):
    """(X_subset, X_query, k) for the crossover sweep at ``n_subset`` rows x ``d`` dims."""
    rng = np.random.default_rng(0)
    n_sub = int(dims["n_subset"])
    d = int(dims["d"])
    X_subset = rng.standard_normal((n_sub, d)).astype(np.float32)
    X_query = rng.standard_normal((min(n_sub, 1000), d)).astype(np.float32)
    return (X_subset, X_query, 10)


def _run_knn_sweep() -> list:
    """Full (n_subset x d) grid sweep, exact vs hnsw, fastest per cell. The
    variants reuse ``knn_search`` with a forced ``prefer_hnsw_at_n`` (1 = hnsw,
    huge = exact), which also dodges recursion into _knn_backend_choice (those
    values != the 50_000 sentinel). Equivalence gate OFF -- hnsw is approximate."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {"exact": lambda xs, xq, k: knn_search(xs, xq, k, prefer_hnsw_at_n=10**12)[0]}
    if _check_hnsw_available():
        variants["hnsw"] = lambda xs, xq, k: knn_search(xs, xq, k, prefer_hnsw_at_n=1)[0]
    return list(
        sweep_backend_grid(
            variants, {"n_subset": _KNN_SWEEP_N, "d": _KNN_SWEEP_D}, _make_knn_inputs,
            reference="exact", equiv_rtol=float("inf"), equiv_atol=float("inf"), repeats=3,
        )
    )


def _knn_fallback_choice(n_subset: int, d: int) -> str:
    """Pre-sweep heuristic (the spec's dynamic fallback callable): hnsw at/above the
    documented 50k-row crossover, else exact."""
    return "hnsw" if int(n_subset) >= 50_000 else "exact"


# Register with the kernel-tuner registry. CPU-only (gpu_capable=False); approximate
# hnsw vs exact sklearn, so retune_all / mlframe-tune-kernels can tune the crossover.
from pyutilz.performance.kernel_tuning.registry import kernel_tuner

_KNN_SPEC = kernel_tuner(
    kernel_name="knn_hnsw_crossover",
    variant_fns=(knn_search, _sklearn_fallback),  # both bodies -> edits auto-invalidate
    tuner=_run_knn_sweep,
    axes={"n_subset": list(_KNN_SWEEP_N), "d": list(_KNN_SWEEP_D)},
    fallback=_knn_fallback_choice,  # callable (n_subset, d) -> str
    gpu_capable=False,
    salt=_KNN_SALT,
    cli_label="knn_hnsw_crossover",
)
