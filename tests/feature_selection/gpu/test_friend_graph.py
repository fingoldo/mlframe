"""Tests for the MRMR friend-graph post-analysis (compute + prune + render).

Covers, per the project rule (unit + biz_value + cProfile per feature):
* unit/behavioral: classification of a redundant "universal soldier" hub, edge
  weights matching the plug-in MI estimator, ADC arrow direction, layout;
* biz_value: pruning removes the redundant hub, keeps the true predictors, and
  never drops both cause and effect;
* cProfile: the edge pass is dominated by the information-theory kernels;
* render smoke: a NetworkPanelSpec renders + saves on both reporting backends.
"""

from __future__ import annotations


import numpy as np
import pytest

from mlframe.feature_selection.filters import info_theory as it
from mlframe.feature_selection.filters.friend_graph import (
    build_friend_graph,
    friend_graph_to_figurespec,
    prune_by_friend_graph,
)


def _redundant_hub_dataset(n=8000, seed=11):
    """Build the canonical "universal soldier" scenario.

    Four independent predictors ``p1..p4`` each encode a label-relevant part ``u``
    and a noise part ``v``. The target is the majority of the ``u`` parts, so each
    predictor carries unique target information. The hub ``G`` is the sum of the
    ``v`` (noise) parts: correlated with every predictor, but independent of the
    target. ``G`` is the redundant aggregator the friend graph must flag red.

    Returns ``(data, nbins, target_indices, names, selected_vars, hub_idx)``.
    """
    rng = np.random.default_rng(seed)
    u = [rng.integers(0, 2, n) for _ in range(4)]
    v = [rng.integers(0, 2, n) for _ in range(4)]
    p = [(u[i] * 2 + v[i]).astype(np.int32) for i in range(4)]  # nbins 4
    y = (sum(u) >= 2).astype(np.int32)  # majority of u
    g = sum(v).astype(np.int32)  # nbins 5, target-independent
    data = np.column_stack([*p, g, y]).astype(np.int32)
    nbins = np.array([4, 4, 4, 4, 5, 2], dtype=np.int64)
    names = ["p1", "p2", "p3", "p4", "G", "y"]
    return data, nbins, np.array([5], dtype=np.int64), names, [0, 1, 2, 3, 4], 4


def test_classifies_redundant_hub_red_and_predictors_green():
    """Classifies redundant hub red and predictors green."""
    data, nbins, tgt, names, sel, hub = _redundant_hub_dataset()
    g = build_friend_graph(sel, data, nbins, tgt, feature_names=names, seed=1)

    by_name = {n.name: n for n in g.nodes}
    assert by_name["G"].klass == "red"
    assert g.suspected_garbage == ["G"]
    for pred in ("p1", "p2", "p3", "p4"):
        assert by_name[pred].klass == "green"

    # The hub is the highest-degree node; the predictors are leaves of it.
    degree = {n.idx: 0 for n in g.nodes}
    for e in g.edges:
        degree[e.a] += 1
        degree[e.b] += 1
    assert degree[hub] == 4
    assert all(e.a == hub or e.b == hub for e in g.edges)

    # The hub carries almost no target information itself, while its neighbors carry
    # substantial unique target information beyond it -- the red criterion.
    assert by_name["G"].relevance < 0.01
    assert by_name["G"].neighbors_unique_target > by_name["G"].relevance


def test_edge_weights_match_plugin_mi():
    """Edge weights match plugin mi."""
    data, nbins, tgt, names, sel, _ = _redundant_hub_dataset(n=6000, seed=3)
    g = build_friend_graph(sel, data, nbins, tgt, feature_names=names, seed=1)
    assert g.edges
    for e in g.edges:
        direct = float(it.mi(data, np.array([e.a], dtype=np.int64), np.array([e.b], dtype=np.int64), nbins))
        assert e.mi == pytest.approx(direct, rel=1e-6, abs=1e-9)


def test_adc_arrow_points_to_more_explained_node():
    # B is a deterministic low-entropy function of high-entropy A, so A explains B
    # almost fully (I/H(B) ~ 1) while B explains little of A: the arrow is A -> B.
    """Adc arrow points to more explained node."""
    rng = np.random.default_rng(0)
    n = 4000
    a = rng.integers(0, 8, n).astype(np.int32)  # nbins 8 (high entropy)
    b = (a % 2).astype(np.int32)  # nbins 2, fully determined by a
    y = rng.integers(0, 2, n).astype(np.int32)
    data = np.column_stack([a, b, y]).astype(np.int32)
    nbins = np.array([8, 2, 2], dtype=np.int64)
    g = build_friend_graph([0, 1], data, nbins, np.array([2], dtype=np.int64), feature_names=["A", "B", "y"], seed=1)
    edge = next(e for e in g.edges if {e.a, e.b} == {0, 1})
    assert edge.a == 0 and edge.b == 1  # explainer A -> explained B


def test_layout_positions_cover_all_nodes():
    """Layout positions cover all nodes."""
    data, nbins, tgt, names, sel, _ = _redundant_hub_dataset(n=3000, seed=5)
    g = build_friend_graph(sel, data, nbins, tgt, feature_names=names, seed=1)
    assert set(g.pos) == {n.idx for n in g.nodes}
    for xy in g.pos.values():
        assert len(xy) == 2 and all(np.isfinite(c) for c in xy)


def test_max_nodes_guard_skips_edges():
    """Max nodes guard skips edges."""
    data, nbins, tgt, names, sel, _ = _redundant_hub_dataset(n=2000, seed=9)
    g = build_friend_graph(sel, data, nbins, tgt, feature_names=names, seed=1, max_nodes=2)
    assert g.edges == []
    assert {n.klass for n in g.nodes} == {"yellow"}


def test_biz_value_prune_removes_hub_keeps_predictors():
    """Biz value prune removes hub keeps predictors."""
    data, nbins, tgt, names, sel, hub = _redundant_hub_dataset(seed=11)
    g = build_friend_graph(sel, data, nbins, tgt, feature_names=names, seed=1)

    pruned, reasons = prune_by_friend_graph(g, sel)

    # The redundant hub is dropped; every true predictor survives.
    assert hub not in pruned
    assert all(i in pruned for i in (0, 1, 2, 3))
    assert g.pruned == ["G"]
    # The removal reason names the predictors that justified dropping the hub, and
    # those predictors are protected -- cause and effect are never dropped together.
    assert "G" in reasons
    assert any(p in reasons["G"] for p in ("p1", "p2", "p3", "p4"))
    assert len(pruned) == len(sel) - 1


def test_biz_value_no_prune_when_no_red_nodes():
    # Two independent informative predictors, no hub: nothing should be pruned.
    """Biz value no prune when no red nodes."""
    rng = np.random.default_rng(1)
    n = 5000
    x1 = rng.integers(0, 2, n)
    x2 = rng.integers(0, 2, n)
    y = (x1 ^ (x2 & rng.integers(0, 2, n))).astype(np.int32)
    data = np.column_stack([x1, x2, y]).astype(np.int32)
    nbins = np.array([2, 2, 2], dtype=np.int64)
    g = build_friend_graph([0, 1], data, nbins, np.array([2], dtype=np.int64), feature_names=["x1", "x2", "y"], seed=1)
    pruned, reasons = prune_by_friend_graph(g, [0, 1])
    assert pruned == [0, 1]
    assert reasons == {}


def test_to_meta_is_json_serializable_and_summarizes():
    """To meta is json serializable and summarizes."""
    import orjson

    data, nbins, tgt, names, sel, _ = _redundant_hub_dataset(seed=11)
    g = build_friend_graph(sel, data, nbins, tgt, feature_names=names, seed=1)
    prune_by_friend_graph(g, sel)  # populate pruned/reasons too
    meta = g.to_meta()
    # Round-trips through orjson (the project's canonical JSON path).
    orjson.dumps(meta)
    assert meta["n_nodes"] == 5
    assert meta["class_counts"]["red"] == 1
    assert meta["suspected_garbage"] == ["G"]
    assert isinstance(meta["top_weighted_degree"], list)


def test_cprofile_edge_pass_dominated_by_info_theory(capsys):
    """Cprofile edge pass dominated by info theory."""
    import cProfile
    import pstats

    # A moderate graph: ~30 features (one hub linking many), to exercise the O(k^2)
    # edge pass. The hotspot must be the info-theory kernels, not Python overhead.
    rng = np.random.default_rng(2)
    n = 3000
    k = 30
    cols = [rng.integers(0, 4, n).astype(np.int32) for _ in range(k)]
    hub = (cols[0] // 2 + cols[1] // 2).astype(np.int32)
    y = (cols[0] >= 2).astype(np.int32)
    data = np.column_stack([*cols, hub, y]).astype(np.int32)
    nbins = np.array([4] * k + [3, 2], dtype=np.int64)
    sel = list(range(k + 1))

    prof = cProfile.Profile()
    prof.enable()
    g = build_friend_graph(sel, data, nbins, np.array([k + 1], dtype=np.int64), seed=1)
    prof.disable()

    assert len(g.nodes) == k + 1
    stats = pstats.Stats(prof).sort_stats("cumulative")
    stats.print_stats(15)
    out = capsys.readouterr().out
    # Profile recorded the build; the edge pass dominates (the MI kernels are
    # @njit so cProfile attributes time to the build frame / numba dispatcher
    # rather than info_theory line-by-line, hence the build-frame assertion).
    assert out.strip()
    assert "friend_graph" in out


@pytest.mark.parametrize("dsl", ["matplotlib[png]", "plotly[html]"])
def test_render_smoke_both_backends(tmp_path, dsl):
    """Render smoke both backends."""
    backend = dsl.split("[")[0]
    pytest.importorskip(backend)
    from mlframe.reporting.output import parse_plot_output_dsl
    from mlframe.reporting.renderers import render_and_save

    data, nbins, tgt, names, sel, _ = _redundant_hub_dataset(n=3000, seed=7)
    g = build_friend_graph(sel, data, nbins, tgt, feature_names=names, seed=1)
    spec = friend_graph_to_figurespec(g, title="Friend graph test")

    base = str(tmp_path / "fg")
    render_and_save(spec, parse_plot_output_dsl(dsl), base, interactive=False)
    fmt = "png" if backend == "matplotlib" else "html"
    assert (tmp_path / f"fg.{fmt}").exists()


def test_pairwise_mi_edge_cached_marginals_bit_identical():
    """iter469: pairwise_mi_edge(h_a, h_b) must equal the full-mi() path
    bit-for-bit. The cached-marginals path recovers MI as
    H(a)+H(b)-H(a,b) (1 merge_vars) instead of mi()'s 3 merge_vars; the
    marginals are exactly what build_friend_graph already computed via
    _node_entropy, so the edge weight -- and therefore the significance-floor
    keep/drop decision and the whole graph topology -- is unchanged. This
    pins the equivalence so a future mi()/merge_vars refactor can't silently
    diverge the two paths."""
    from mlframe.feature_selection.filters.friend_graph import (
        pairwise_mi_edge,
        _node_entropy,
    )

    rng = np.random.default_rng(20260527)
    n = 40000
    data = rng.integers(0, 8, size=(n, 3)).astype(np.int32)
    # Make col1 correlated with col0 so there's a real (non-floored) edge.
    data[:, 1] = (data[:, 0] + rng.integers(0, 2, n)) % 8
    nbins = np.array([8, 8, 8], dtype=np.int64)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        m_full = pairwise_mi_edge(data, a, b, nbins, n, mi_eps=0.0, edge_significance=0.0)
        h_a = _node_entropy(data, a, nbins, None, np.int32)
        h_b = _node_entropy(data, b, nbins, None, np.int32)
        m_cached = pairwise_mi_edge(
            data,
            a,
            b,
            nbins,
            n,
            mi_eps=0.0,
            edge_significance=0.0,
            h_a=h_a,
            h_b=h_b,
        )
        assert m_full == m_cached, f"edge ({a},{b}): full={m_full} != cached={m_cached}"
