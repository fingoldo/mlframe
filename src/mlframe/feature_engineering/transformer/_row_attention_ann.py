"""ANN (approximate nearest neighbour) backend for row-attention.

Default: **pynndescent** (pure-Python + numba; Windows-friendly; numpy 2 compatible). hnswlib was the original choice but the wheel commonly segfaults on Windows
when numpy 2 is in the env (the C extension was built against numpy 1.x). pynndescent has no C extension so the same issue cannot arise.

If a user has a working hnswlib in their environment, they can opt back into it via the ``ann_backend="hnswlib"`` parameter where exposed; the default
``ann_backend="auto"`` picks pynndescent first.

pynndescent at ``n_neighbors=k+5, random_state=42`` on 10M points with head_dim=8 builds in ~3-5 min on a 16-core Linux box; the build is parallelised across
cores via joblib. Query throughput is comparable to hnswlib at the same recall.

Reference:
    Dong, McCarthy & Cox 2011 - "Efficient k-Nearest Neighbor Graph Construction for Generic Similarity Measures" (NN-Descent algorithm).
    pynndescent implementation by Leland McInnes (also umap-learn maintainer).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_ann_available(backend: Literal["pynndescent", "hnswlib"]) -> Any:
    """Import the requested ANN backend lazily; raise a clear error if it's missing.

    Caller dispatches via ``mlframe[transformer-ann]`` extra (pynndescent) or via manual hnswlib install.
    """
    if backend == "pynndescent":
        try:
            import pynndescent  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "pynndescent is required for the default ANN backend. Install via `pip install 'mlframe[transformer-ann]'` "
                "(or `pip install pynndescent>=0.5` directly)."
            ) from exc
        return pynndescent
    if backend == "hnswlib":
        try:
            import hnswlib  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "hnswlib is required when ann_backend='hnswlib'. Note: hnswlib wheels on Windows with numpy 2 frequently segfault at import "
                "(the C extension is built against numpy 1.x). Default ann_backend='auto' uses pynndescent which avoids this issue."
            ) from exc
        return hnswlib
    raise ValueError(f"Unknown ANN backend: {backend!r}")


def _resolve_backend(backend: Literal["auto", "pynndescent", "hnswlib", "sklearn"], n_train: int = 0) -> Literal["pynndescent", "hnswlib", "sklearn"]:
    """Pick a concrete backend when ``auto`` is requested.

    Order when ``auto``:
        - n_train < 10000: ``sklearn`` (exact kNN, ~ms for thousands of rows, always works on Windows; pynndescent's numba thread-allocator can OOM on
          mid-size N under repeated calls in the same process).
        - n_train >= 10000: ``pynndescent`` (approximate, scales to 10M+).

    Auto never selects hnswlib because of the Windows numpy-2 segfault issue documented above; users who specifically want hnswlib pass it explicitly.
    """
    if backend == "auto":
        return "sklearn" if n_train < 10_000 else "pynndescent"
    if backend not in ("pynndescent", "hnswlib", "sklearn"):
        raise ValueError(f"backend must be 'auto', 'pynndescent', 'hnswlib', or 'sklearn'; got {backend!r}.")
    return backend


def _resolve_num_threads(num_threads: Optional[int]) -> int:
    """Default to physical core count (``psutil``) when caller passes None; respect explicit override otherwise.

    Physical cores (not logical) is the right default for ANN: SMT siblings contend on the same FP units and slow distance computation.
    """
    if num_threads is not None and num_threads > 0:
        return int(num_threads)
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except ImportError:
        pass
    return max(1, os.cpu_count() or 1)


class _AnnIndex:
    """Thin uniform wrapper over hnswlib / pynndescent so the rest of the code uses one query API.

    Holds the underlying backend object plus the metric / dimensions for the disk-cache fingerprint.
    """
    __slots__ = ("backend", "obj", "metric", "head_dim")

    def __init__(self, backend: str, obj: Any, metric: str, head_dim: int) -> None:
        self.backend = backend
        self.obj = obj
        self.metric = metric
        self.head_dim = head_dim


def build_hnsw_index(
    k_proj: np.ndarray,
    *,
    space: str = "cosine",
    M: int = 16,
    ef_construction: int = 200,
    num_threads: Optional[int] = None,
    progress_log_every: int = 0,
    ann_backend: Literal["auto", "pynndescent", "hnswlib"] = "auto",
    random_state: int = 42,
) -> _AnnIndex:
    """Build an ANN index over a single head's projected K-bank. Function name kept ``build_hnsw_index`` for back-compat; the underlying backend is
    ``pynndescent`` by default (Windows-safe; numpy-2 compatible). Pass ``ann_backend='hnswlib'`` if you have a working hnswlib wheel.

    Shapes: ``k_proj`` is (n_train, head_dim), expected float32 (cast on the fly if not). Returns an opaque ``_AnnIndex`` wrapper that ``query_topk`` knows how
    to query.

    Parameters:
        ``space``           - cosine / l2 / ip; default cosine matches the L2-normalised projections.
        ``M``               - graph degree at construction. For hnswlib this is the standard M; for pynndescent it maps to ``n_neighbors`` (we use ``M+5`` to
                              match hnswlib recall at typical k=32).
        ``ef_construction`` - search width during build (hnswlib only; pynndescent doesn't expose an equivalent knob).
        ``num_threads``     - None -> physical cores; explicit override otherwise.
        ``progress_log_every`` - 0 = no progress logging (hnswlib only; pynndescent is one-shot).
        ``ann_backend``     - "auto" (= pynndescent), "pynndescent", or "hnswlib".
        ``random_state``    - reproducibility seed for pynndescent (ignored by hnswlib).
    """
    n_train, head_dim = k_proj.shape
    backend = _resolve_backend(ann_backend, n_train=n_train)
    n_threads = _resolve_num_threads(num_threads)

    if k_proj.dtype != np.float32 or not k_proj.flags["C_CONTIGUOUS"]:
        k_proj = np.ascontiguousarray(k_proj, dtype=np.float32)

    if backend == "sklearn":
        from sklearn.neighbors import NearestNeighbors
        # Exact kNN. For n_train up to ~20k, query time per row is dominated by gather, not search. Always works on Windows; no numba thread races.
        metric_map = {"cosine": "cosine", "l2": "euclidean", "ip": "euclidean"}
        sk_metric = metric_map.get(space, "euclidean")
        # n_neighbors here is just a sensible upper bound for later query_topk calls; sklearn doesn't pre-commit to a graph degree.
        index = NearestNeighbors(n_neighbors=max(M + 5, 16), metric=sk_metric, n_jobs=n_threads, algorithm="brute")
        index.fit(k_proj)
        return _AnnIndex(backend="sklearn", obj=index, metric=sk_metric, head_dim=head_dim)

    if backend == "hnswlib":
        hnswlib = _ensure_ann_available("hnswlib")
        index = hnswlib.Index(space=space, dim=head_dim)
        index.init_index(max_elements=n_train, ef_construction=ef_construction, M=M)
        index.set_num_threads(n_threads)
        ids = np.arange(n_train, dtype=np.int64)
        if progress_log_every <= 0:
            index.add_items(k_proj, ids)
        else:
            for start in range(0, n_train, progress_log_every):
                end = min(start + progress_log_every, n_train)
                index.add_items(k_proj[start:end], ids[start:end])
                logger.info("hnswlib build progress: %d / %d points (%.1f%%)", end, n_train, 100.0 * end / n_train)
        return _AnnIndex(backend="hnswlib", obj=index, metric=space, head_dim=head_dim)

    pynndescent = _ensure_ann_available("pynndescent")
    # Map "space" parameter to pynndescent's metric naming. pynndescent uses scikit-learn names.
    metric_map = {"cosine": "cosine", "l2": "euclidean", "ip": "dot"}
    pynnd_metric = metric_map.get(space, space)
    # n_neighbors is the graph degree at build time; we use M+5 to give comparable recall to hnswlib at the typical query k=32.
    # low_memory=True: required to avoid OOM at moderate n with many CPU workers. pynndescent's dense (low_memory=False) path allocates per-thread copies of
    # intermediate neighbour graphs, which blows up at 16 workers x ~50 MB each = 800 MB just for scratch; low_memory=True uses the iterative algorithm with
    # ~1/4 the peak memory at ~30% lower build throughput. Acceptable trade for stability.
    # Cap n_jobs at 4 for the same reason; pynndescent threading speedup tops out around 4 workers anyway.
    capped_n_jobs = min(n_threads, 4)
    t0 = time.perf_counter()
    index = pynndescent.NNDescent(
        k_proj, metric=pynnd_metric, n_neighbors=max(M + 5, 16),
        random_state=random_state, n_jobs=capped_n_jobs, low_memory=True,
    )
    # Force graph construction now (NNDescent is lazy by default) so subsequent query calls hit the warm index.
    index.prepare()
    logger.info("pynndescent build: %d points x dim %d in %.2fs (n_jobs=%d, low_memory=True)", n_train, head_dim, time.perf_counter() - t0, capped_n_jobs)
    return _AnnIndex(backend="pynndescent", obj=index, metric=pynnd_metric, head_dim=head_dim)


def query_topk(
    index: _AnnIndex,
    q_proj: np.ndarray,
    k: int,
    *,
    ef_search: Optional[int] = None,
    num_threads: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Query the top-k neighbours for each row of ``q_proj``. Returns ``(labels, distances)`` both shape (n_queries, k).

    Backend-agnostic: works on whichever index ``build_hnsw_index`` produced.

    ``ef_search`` is hnswlib-only (search width); ignored for pynndescent.
    """
    if q_proj.dtype != np.float32 or not q_proj.flags["C_CONTIGUOUS"]:
        q_proj = np.ascontiguousarray(q_proj, dtype=np.float32)
    n_threads = _resolve_num_threads(num_threads)

    if index.backend == "sklearn":
        # sklearn returns (distances, indices) in that order; flip to match our (labels, distances) signature.
        distances, labels = index.obj.kneighbors(q_proj, n_neighbors=k)
    elif index.backend == "hnswlib":
        if ef_search is None:
            ef_search = max(k * 2, 64)
        index.obj.set_ef(ef_search)
        index.obj.set_num_threads(n_threads)
        labels, distances = index.obj.knn_query(q_proj, k=k)
    else:
        # pynndescent: query returns (indices, distances). Honour n_jobs via env / thread pool; the index already has n_jobs baked in.
        labels, distances = index.obj.query(q_proj, k=k)

    return labels.astype(np.int32, copy=False), distances.astype(np.float32, copy=False)


# ---- Legacy aliases / smoke tests ----

_SMOKE_DONE: bool = False


def reset_openmp_smoke() -> None:
    """Legacy alias kept so tests / users that imported the old hnswlib-only API don't break. No-op under pynndescent."""
    global _SMOKE_DONE
    _SMOKE_DONE = False
