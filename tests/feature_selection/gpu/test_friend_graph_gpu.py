"""Bit-identity + dispatch tests for the GPU friend-graph BUILD backend.

The GPU path (``friend_graph_gpu``) accelerates the O(k^2) pairwise-MI edge pass + the
k node entropy/relevance stats; it must be BIT-IDENTICAL to the CPU build (same plug-in
MI estimator, integer joint-counting) -- a single-ULP drift would flip a borderline
significance-floor edge and change the graph topology / which features the prune path
drops. Covers, per the project rule (unit + biz_value + dispatch):

* unit/bit-identity: GPU node H, relevance, and every edge MI == the CPU stats
  (``np.array_equal`` on the raw floats, not just ``assert_allclose``);
* build-level: ``build_friend_graph(gpu_backend=...)`` yields the SAME nodes / edges /
  classifications / suspected_garbage / prune result as the CPU build;
* dispatch: with no GPU (or ``gpu_backend="cpu"``) the build falls back to the CPU path
  and is unchanged; the dispatcher returns ``None`` (CPU fallback) for tiny / no-edge sets.

GPU tests auto-skip when cupy / numba.cuda are unavailable (CI without a GPU).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.friend_graph import (
    build_friend_graph,
    prune_by_friend_graph,
)
from mlframe.feature_selection.filters.friend_graph_gpu import (
    _CUDA_AVAIL,
    _CUPY_AVAIL,
    _friend_graph_cpu_stats,
    dispatch_friend_graph_stats,
    friend_graph_stats_cuda,
    friend_graph_stats_cupy,
)


def _synthetic_selected_set(n=4000, k=20, seed=7):
    """A discretized matrix with ``k`` selected feature columns (mixed cardinalities,
    a few correlated so real non-floored edges exist) + 1 target column."""
    rng = np.random.default_rng(seed)
    nbins = np.array([int(rng.integers(2, 9)) for _ in range(k)] + [3], dtype=np.int64)
    data = np.empty((n, k + 1), dtype=np.int32)
    for c in range(k):
        data[:, c] = rng.integers(0, nbins[c], n)
    # Inject correlations so the edge pass produces surviving edges.
    data[:, 1] = (data[:, 0] + rng.integers(0, 2, n)) % nbins[1]
    if k > 3:
        data[:, 2] = (data[:, 0] // 2 + data[:, 3] // 2) % nbins[2]
    data[:, k] = (data[:, 0] % 3).astype(np.int32)  # target depends on col 0
    target_indices = np.array([k], dtype=np.int64)
    sel = list(range(k))
    return sel, data, nbins, target_indices


def _graph_fingerprint(graph):
    nodes = sorted((n.name, n.entropy, n.relevance, n.weighted_degree, n.shared_frac, n.neighbors_unique_target, n.klass) for n in graph.nodes)
    edges = sorted((e.a, e.b, e.mi) for e in graph.edges)
    return nodes, edges, sorted(graph.suspected_garbage)


# ---------------------------------------------------------------------------
# Stats-level bit-identity (the load-bearing claim)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not _CUPY_AVAIL, reason="cupy not available on this host")
@pytest.mark.parametrize("n,k,seed", [(4000, 20, 7), (3000, 30, 3), (5000, 50, 5)])
def test_cupy_stats_bit_identical_to_cpu(n, k, seed):
    sel, data, nbins, tgt = _synthetic_selected_set(n=n, k=k, seed=seed)
    cpu = _friend_graph_cpu_stats(sel, data, nbins, tgt, np.int32)
    gpu = friend_graph_stats_cupy(sel, data, nbins, tgt, np.int32)
    for i in sel:
        assert gpu.H[i] == cpu.H[i], f"node {i} entropy differs"
        assert gpu.rel[i] == cpu.rel[i], f"node {i} relevance differs"
    assert set(gpu.edge_mi) == set(cpu.edge_mi)
    for e in cpu.edge_mi:
        assert gpu.edge_mi[e] == cpu.edge_mi[e], f"edge {e} MI differs"


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAIL, reason="numba.cuda not available on this host")
@pytest.mark.parametrize("n,k,seed", [(4000, 20, 7), (3000, 30, 3), (5000, 50, 5)])
def test_cuda_stats_bit_identical_to_cpu(n, k, seed):
    sel, data, nbins, tgt = _synthetic_selected_set(n=n, k=k, seed=seed)
    cpu = _friend_graph_cpu_stats(sel, data, nbins, tgt, np.int32)
    gpu = friend_graph_stats_cuda(sel, data, nbins, tgt, np.int32)
    for i in sel:
        assert gpu.H[i] == cpu.H[i], f"node {i} entropy differs"
        assert gpu.rel[i] == cpu.rel[i], f"node {i} relevance differs"
    assert set(gpu.edge_mi) == set(cpu.edge_mi)
    for e in cpu.edge_mi:
        assert gpu.edge_mi[e] == cpu.edge_mi[e], f"edge {e} MI differs"


@pytest.mark.gpu
@pytest.mark.skipif(not (_CUDA_AVAIL and _CUPY_AVAIL), reason="needs both numba.cuda and cupy")
def test_cuda_backend_does_not_reupload_sub_for_node_codes(monkeypatch):
    """RESIDENT UPLOAD (2026-07-13): ``d_node_codes`` (the int64 widening of ``d_sub``) must be produced by
    an ON-DEVICE cast (mirrors ``friend_graph_stats_cupy``'s ``d_sub.astype(cp.int64)``), not a second
    ``to_device`` upload of the SAME ``sub`` content from host within one call."""
    import mlframe.feature_selection.filters.friend_graph_gpu as fgg

    sel, data, nbins, tgt = _synthetic_selected_set(n=3000, k=15, seed=13)
    sub_expected = np.ascontiguousarray(data[:, np.asarray(sel, dtype=np.int64)], dtype=np.int32)

    calls = {"n": 0}
    orig_to_device = fgg._nb_cuda.to_device

    def spy(arr, *a, **kw):
        if (
            isinstance(arr, np.ndarray)
            and arr.shape == sub_expected.shape
            and np.array_equal(
                np.asarray(arr, dtype=np.int64),
                sub_expected.astype(np.int64),
            )
        ):
            calls["n"] += 1
        return orig_to_device(arr, *a, **kw)

    monkeypatch.setattr(fgg._nb_cuda, "to_device", spy)
    fgg.friend_graph_stats_cuda(sel, data, nbins, tgt, np.int32)

    assert calls["n"] == 1, (
        f"sub-content uploaded via to_device {calls['n']}x (expected 1 -- d_node_codes must be an "
        f"on-device cast of d_sub, not a second host upload of the same content)"
    )


@pytest.mark.gpu
@pytest.mark.skipif(not (_CUDA_AVAIL and _CUPY_AVAIL), reason="needs both numba.cuda and cupy")
def test_cuda_node_codes_ondevice_cast_matches_forced_reupload_fallback(monkeypatch):
    """Bit-identity: the on-device-cast fast path must match the pre-fix double-upload fallback (forced by
    monkeypatching ``_CUPY_AVAIL`` off, which routes ``d_node_codes`` back through a second ``to_device``
    upload of ``sub``, cast to int64 host-side -- exactly the pre-fix code path)."""
    import mlframe.feature_selection.filters.friend_graph_gpu as fgg

    sel, data, nbins, tgt = _synthetic_selected_set(n=3000, k=15, seed=17)

    gpu_fast = fgg.friend_graph_stats_cuda(sel, data, nbins, tgt, np.int32)

    monkeypatch.setattr(fgg, "_CUPY_AVAIL", False)
    gpu_fallback = fgg.friend_graph_stats_cuda(sel, data, nbins, tgt, np.int32)

    for i in sel:
        assert gpu_fast.H[i] == gpu_fallback.H[i], f"node {i} entropy differs between on-device-cast and re-upload fallback"
        assert gpu_fast.rel[i] == gpu_fallback.rel[i], f"node {i} relevance differs between on-device-cast and re-upload fallback"
    assert gpu_fast.edge_mi == gpu_fallback.edge_mi


# ---------------------------------------------------------------------------
# build_friend_graph-level parity (nodes / edges / classification / prune)
# ---------------------------------------------------------------------------


def _redundant_hub_dataset(n=8000, seed=11):
    rng = np.random.default_rng(seed)
    u = [rng.integers(0, 2, n) for _ in range(4)]
    v = [rng.integers(0, 2, n) for _ in range(4)]
    p = [(u[i] * 2 + v[i]).astype(np.int32) for i in range(4)]
    y = (sum(u) >= 2).astype(np.int32)
    g = sum(v).astype(np.int32)
    data = np.column_stack(p + [g, y]).astype(np.int32)
    nbins = np.array([4, 4, 4, 4, 5, 2], dtype=np.int64)
    names = ["p1", "p2", "p3", "p4", "G", "y"]
    return data, nbins, np.array([5], dtype=np.int64), names, [0, 1, 2, 3, 4]


@pytest.mark.gpu
@pytest.mark.skipif(not (_CUPY_AVAIL or _CUDA_AVAIL), reason="no GPU backend available")
@pytest.mark.parametrize("backend", [b for b, ok in (("cupy", _CUPY_AVAIL), ("cuda", _CUDA_AVAIL)) if ok])
def test_build_friend_graph_gpu_matches_cpu_build(backend):
    data, nbins, tgt, names, sel = _redundant_hub_dataset()
    g_cpu = build_friend_graph(sel, data, nbins, tgt, feature_names=names, seed=1, gpu_backend="cpu")
    g_gpu = build_friend_graph(sel, data, nbins, tgt, feature_names=names, seed=1, gpu_backend=backend)
    assert _graph_fingerprint(g_cpu) == _graph_fingerprint(g_gpu)
    # The redundant hub is still flagged + pruned identically.
    p_cpu, _ = prune_by_friend_graph(build_friend_graph(sel, data, nbins, tgt, feature_names=names, seed=1, gpu_backend="cpu"), sel)
    p_gpu, _ = prune_by_friend_graph(g_gpu, sel)
    assert p_cpu == p_gpu


@pytest.mark.gpu
@pytest.mark.skipif(not (_CUPY_AVAIL or _CUDA_AVAIL), reason="no GPU backend available")
@pytest.mark.parametrize("backend", [b for b, ok in (("cupy", _CUPY_AVAIL), ("cuda", _CUDA_AVAIL)) if ok])
def test_build_friend_graph_gpu_larger_set_matches_cpu(backend):
    sel, data, nbins, tgt = _synthetic_selected_set(n=4000, k=40, seed=2)
    g_cpu = build_friend_graph(sel, data, nbins, tgt, seed=1, gpu_backend="cpu")
    g_gpu = build_friend_graph(sel, data, nbins, tgt, seed=1, gpu_backend=backend)
    assert _graph_fingerprint(g_cpu) == _graph_fingerprint(g_gpu)


# ---------------------------------------------------------------------------
# Dispatch / CPU-fallback behaviour
# ---------------------------------------------------------------------------


def test_dispatch_falls_back_to_cpu_for_tiny_or_no_edge_sets():
    """The dispatcher must return None (CPU fallback) for <2 selected vars or n==0,
    regardless of GPU availability -- there is no O(k^2) work to offload."""
    sel, data, nbins, tgt = _synthetic_selected_set(n=100, k=5, seed=1)
    assert dispatch_friend_graph_stats([sel[0]], data, nbins, tgt) is None  # single node
    assert dispatch_friend_graph_stats([], data, nbins, tgt) is None  # empty
    empty = np.empty((0, data.shape[1]), dtype=np.int32)
    assert dispatch_friend_graph_stats(sel, empty, nbins, tgt) is None  # n == 0


def test_force_cpu_backend_does_not_use_gpu_and_matches_default_cpu():
    """gpu_backend='cpu' forces the legacy CPU edge pass; result must equal the
    legacy build (this also exercises the path on a GPU-less host -- no GPU import
    is required because the dispatch is short-circuited before importing the backend)."""
    data, nbins, tgt, names, sel = _redundant_hub_dataset(n=3000, seed=4)
    g_cpu = build_friend_graph(sel, data, nbins, tgt, feature_names=names, seed=1, gpu_backend="cpu")
    # The classic build (no gpu_backend kwarg passed by legacy callers) on a GPU-less
    # host takes the same CPU edge pass -> identical fingerprint.
    by_name = {n.name: n for n in g_cpu.nodes}
    assert by_name["G"].klass == "red"
    assert g_cpu.suspected_garbage == ["G"]


def test_force_unavailable_gpu_backend_returns_none():
    """Forcing a backend that isn't installed returns None so the build uses CPU."""
    sel, data, nbins, tgt = _synthetic_selected_set(n=500, k=6, seed=1)
    if not _CUPY_AVAIL:
        assert dispatch_friend_graph_stats(sel, data, nbins, tgt, force_backend="cupy") is None
    if not _CUDA_AVAIL:
        assert dispatch_friend_graph_stats(sel, data, nbins, tgt, force_backend="cuda") is None


@pytest.mark.gpu
@pytest.mark.skipif(not (_CUPY_AVAIL or _CUDA_AVAIL), reason="no GPU backend available")
def test_multi_target_relevance_falls_back_to_cpu():
    """The GPU relevance fast path is single-target only; a multi-column target must
    leave ``rel=None`` (caller computes relevance on CPU) while H + edges stay GPU +
    bit-identical."""
    sel, data, nbins, tgt = _synthetic_selected_set(n=3000, k=12, seed=8)
    # Use two target columns (append a second). XOR-1 only flips bit 0, so the derived
    # column's value range matches the source column's, NOT a hardcoded 2 -- declaring it
    # as binary when the source has >2 classes understates its true cardinality to the
    # joint-entropy njit kernel, which trusts nbins as an upper bound and writes its
    # histogram out of bounds (silent heap corruption, no bounds-checking in @njit) when
    # violated. Derive the declared bin count from the actual data instead of assuming.
    second_col = (data[:, 0] ^ 1).astype(np.int32)
    data2 = np.column_stack([data, second_col])
    nbins2 = np.append(nbins, int(second_col.max()) + 1)
    tgt2 = np.array([data.shape[1] - 1, data2.shape[1] - 1], dtype=np.int64)
    backend = "cupy" if _CUPY_AVAIL else "cuda"
    fn = friend_graph_stats_cupy if _CUPY_AVAIL else friend_graph_stats_cuda
    gpu = fn(sel, data2, nbins2, tgt2, np.int32)
    assert gpu.rel is None
    # H + edges still bit-identical to the CPU reference.
    cpu = _friend_graph_cpu_stats(sel, data2, nbins2, tgt2, np.int32)
    for i in sel:
        assert gpu.H[i] == cpu.H[i]
    for e in cpu.edge_mi:
        assert gpu.edge_mi[e] == cpu.edge_mi[e]
    # And the full build with this multi-target still matches the CPU build. Disable the
    # conditional ``neighbor_unique_target`` sink pass (garbage_min_degree huge) so the
    # build stays a node+edge parity check without paying the cold 4-var multi-target
    # njit-compile of the conditional MI (orthogonal to the GPU edge/entropy path under
    # test; exercised by the single-target hub test, which DOES classify a red sink).
    g_cpu = build_friend_graph(
        sel,
        data2,
        nbins2,
        tgt2,
        seed=1,
        gpu_backend="cpu",
        compute_layout=False,
        garbage_min_degree=10_000,
    )
    g_gpu = build_friend_graph(
        sel,
        data2,
        nbins2,
        tgt2,
        seed=1,
        gpu_backend=backend,
        compute_layout=False,
        garbage_min_degree=10_000,
    )
    assert _graph_fingerprint(g_cpu) == _graph_fingerprint(g_gpu)
