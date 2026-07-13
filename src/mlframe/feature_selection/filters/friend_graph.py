"""Friend-graph post-analysis of mRMR-selected predictors.

mRMR's greedy front-loading can let a "universal soldier" feature win an early
slot: it carries no unique information about the target, but it is correlated
with so many genuine predictors that it out-scores them while the selected set
is still small (every conditioning variable lowers its conditional MI, yet it
stays ahead because it overlaps with everything). Such a feature is handy for
cheap approximate models but distracts the strongest models with noise.

This module renders the selected set as a "friend graph" for diagnosis:

* node = feature; node size = its entropy ``H(X)``;
* edge = pairwise mutual information ``I(X_a; X_b)`` (width + color);
* arrow direction = asymmetric dependency (the uncertainty coefficient
  ``I/H``): the arrow points from the explaining feature to the one whose
  entropy is more explained;
* node color: green = unique knowledge, red = suspected redundant sink /
  aggregator, yellow = middling.

A feature is flagged red when it shares a large fraction of its own entropy
with neighbors AND those neighbors collectively carry more *unique* target
information than the feature itself adds -- i.e. the friends know more about the
target than the node does, so it is mostly an aggregator. The conditional
quantity ``I(Y; X_j | X_i)`` is computed via the chain rule
``I((X_i, X_j); Y) - I(X_i; Y)`` so the whole module needs only the plug-in
``mi`` estimator (no extra kernels).

The compute layer reuses the information-theory kernels in :mod:`info_theory`
and the discretized factor matrix already built during ``MRMR.fit``. Rendering
goes through the declarative ``mlframe.reporting`` system (a single
``NetworkPanelSpec``), so the same graph yields interactive plotly HTML and a
static image with no bespoke plotting code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .info_theory import entropy, merge_vars, mi, joint_entropy_2var

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------


@dataclass
class FriendGraphNode:
    """One feature's summary stats within the friend graph: its relevance, entropy, and how much MI it shares with its neighbors (used to classify it green/red/yellow)."""

    idx: int  # column index into the discretized factor matrix
    name: str
    entropy: float  # H(X) in nats
    relevance: float  # I(X; Y)
    weighted_degree: float  # sum of edge MIs to neighbors
    shared_frac: float  # weighted_degree / H(X) (capped at compute time)
    neighbors_unique_target: float  # sum_j I(Y; X_j | X) over neighbors (0 unless suspect)
    klass: str  # "green" | "red" | "yellow"


@dataclass
class FriendGraphEdge:
    """One directed pairwise-MI link between two features in the friend graph, pointing from the explaining feature to the explained one."""

    a: int  # explaining feature (arrow source)
    b: int  # explained feature (arrow target)
    mi: float  # I(X_a; X_b)


@dataclass
class FriendGraph:
    """The full computed friend graph for a feature set: nodes, edges, suspected-garbage/pruned features, and the layout needed to render or summarize it."""

    nodes: List[FriendGraphNode] = field(default_factory=list)
    edges: List[FriendGraphEdge] = field(default_factory=list)
    suspected_garbage: List[str] = field(default_factory=list)
    pruned: List[str] = field(default_factory=list)
    prune_reasons: Dict[str, str] = field(default_factory=dict)
    # node idx -> (x, y) layout position
    pos: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    # Per-suspect detail used by pruning: idx -> [(neighbor_idx, I(Y; neighbor | idx)), ...]
    _neighbor_unique_detail: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)

    def to_meta(self, *, max_nodes_in_meta: int = 200) -> Dict[str, Any]:
        """Compact, JSON-serializable summary for ``meta_info``.

        Rounds floats and keeps only the fields a reviewer needs; the full graph
        (positions, every edge) is reconstructable on demand from a re-fit, so it
        is intentionally not persisted here.
        """
        counts = {"green": 0, "red": 0, "yellow": 0}
        for n in self.nodes:
            counts[n.klass] = counts.get(n.klass, 0) + 1
        top_wd = sorted(self.nodes, key=lambda n: n.weighted_degree, reverse=True)[:10]
        nodes_meta = [
            {
                "name": n.name,
                "entropy": round(n.entropy, 5),
                "relevance": round(n.relevance, 5),
                "weighted_degree": round(n.weighted_degree, 5),
                "shared_frac": round(n.shared_frac, 5),
                "neighbors_unique_target": round(n.neighbors_unique_target, 5),
                "klass": n.klass,
            }
            for n in self.nodes[:max_nodes_in_meta]
        ]
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "class_counts": counts,
            "suspected_garbage": list(self.suspected_garbage),
            "pruned": list(self.pruned),
            "prune_reasons": dict(self.prune_reasons),
            "top_weighted_degree": [(n.name, round(n.weighted_degree, 5)) for n in top_wd],
            "nodes": nodes_meta,
        }


# ----------------------------------------------------------------------------
# Compute
# ----------------------------------------------------------------------------


def _node_entropy(factors_data, idx, factors_nbins, entropy_cache, dtype) -> float:
    """H(X_idx) in nats, reusing ``entropy_cache`` (keyed by sorted index tuple) when present."""
    key = (int(idx),)
    if entropy_cache is not None and key in entropy_cache:
        return float(entropy_cache[key])
    _, freqs, _ = merge_vars(
        factors_data=factors_data, vars_indices=np.array([idx], dtype=np.int64),
        var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
    )
    h = float(entropy(freqs=freqs))
    if entropy_cache is not None:
        entropy_cache[key] = h
    return h


# ----------------------------------------------------------------------------
# Reusable MI primitives (shared with the cluster-aggregate FE step, which needs
# the SAME feature-feature edge metric + significance floor + redundancy criterion
# to discover correlated reflection clusters). Factored out of build_friend_graph
# so both call one implementation.
# ----------------------------------------------------------------------------


def node_relevance(factors_data, idx, target, factors_nbins, dtype=np.int32) -> float:
    """I(X_idx; Y) in nats (the plug-in MI of one feature with the target)."""
    return float(mi(factors_data, np.array([idx], dtype=np.int64), target, factors_nbins, dtype=dtype))


def _joint_entropy_2vars(factors_data, a, b, factors_nbins, dtype=np.int32) -> float:
    """H(X_a, X_b) in nats -- the joint half of I(X_a; X_b). Split out so the
    O(k^2) edge pass can reuse already-computed per-node marginals H(X_a),
    H(X_b) (mi() otherwise recomputes both marginals on EVERY edge -- 2 of its
    3 merge_vars passes are redundant when the caller already has them).

    iter (2026-06-08 wasted-work sweep): the single per-edge ``merge_vars`` this
    used to call builds a length-n ``final_classes`` array + a lookup-table remap
    pass, then the result feeds STRAIGHT into ``entropy`` -- the array, the remap,
    and entropy's ``freqs[freqs > 0]`` / ``log(freqs) * freqs`` temporaries are all
    allocated-then-discarded for a single scalar. ``joint_entropy_2var`` fuses the
    histogram->entropy reduction with NONE of that per-edge waste, at BIT-IDENTICAL
    numerics (verified max-abs-diff 0.0 vs ``entropy(merge_vars(...)[1])`` over 960
    cases; test_joint_entropy_2var.py). The GPU friend-graph twin
    (``friend_graph_gpu._friend_graph_cpu_stats``) consumes this SAME helper, so its
    CPU-reference bit-identity contract is preserved (the value is unchanged).
    """
    return float(joint_entropy_2var(factors_data, int(a), int(b), int(factors_nbins[a]), int(factors_nbins[b])))


def pairwise_mi_edge(factors_data, a, b, factors_nbins, n_samples, mi_eps=1e-6, edge_significance=3.0, dtype=np.int32, h_a=None, h_b=None):
    """Feature-feature ``I(X_a; X_b)`` if it clears the finite-sample significance floor, else ``None``.

    The floor ``max(mi_eps, edge_significance*(na-1)(nb-1)/(2n))`` suppresses spurious edges between
    independent features (the plug-in MI is biased upward by ~the G-test chi-square expectation).

    iter469: when the caller passes the pre-computed marginal entropies
    ``h_a = H(X_a)`` and ``h_b = H(X_b)``, MI is recovered as
    ``H(X_a) + H(X_b) - H(X_a, X_b)`` and only the joint entropy is computed --
    1 merge_vars pass instead of mi()'s 3. Bit-identical to the mi() path
    (same merge_vars + entropy plug-in estimator; the marginals ARE what
    build_friend_graph already computed via _node_entropy). Falls back to the
    full mi() when marginals aren't supplied (cluster-aggregate callers).
    """
    if h_a is not None and h_b is not None:
        h_ab = _joint_entropy_2vars(factors_data, a, b, factors_nbins, dtype=dtype)
        m = h_a + h_b - h_ab
        if m < 0.0:
            m = 0.0
    else:
        m = float(mi(factors_data, np.array([a], dtype=np.int64), np.array([b], dtype=np.int64), factors_nbins, dtype=dtype))
    na, nb = int(factors_nbins[a]), int(factors_nbins[b])
    # INVARIANT (audit friend-graph-7): ``n_samples`` here MUST equal the row
    # count that ``merge_vars``/``mi`` normalised the entropies/MI by (i.e.
    # ``len(factors_data)``). The G-test significance floor below mixes the
    # MI value (a per-row-normalised nats quantity) with this n; if a caller
    # ever passes a different n the floor silently mis-scales. The sole caller
    # passes ``factors_data.shape[0]``, so they coincide -- keep it that way.
    floor = max(mi_eps, edge_significance * (na - 1) * (nb - 1) / (2.0 * max(1, int(n_samples))))
    return m if m > floor else None


def _apply_edge_floor(m, a, b, factors_nbins, n_samples, mi_eps=1e-6, edge_significance=3.0):
    """Apply ``pairwise_mi_edge``'s finite-sample significance floor to a PRE-COMPUTED
    raw edge MI ``m`` (e.g. the bit-identical GPU value ``H_a + H_b - H_ab`` clamped >=0).

    Returns ``m`` if it clears the floor, else ``None`` -- the IDENTICAL keep/drop rule as
    ``pairwise_mi_edge``, just without recomputing the MI (the GPU already produced it).
    ``m is None`` (pair absent from the GPU result -- should not happen for an in-``sel``
    pair) is treated as "no edge". The floor formula is duplicated verbatim from
    ``pairwise_mi_edge`` so the two paths can never diverge in their gating threshold.
    """
    if m is None:
        return None
    na, nb = int(factors_nbins[a]), int(factors_nbins[b])
    floor = max(mi_eps, edge_significance * (na - 1) * (nb - 1) / (2.0 * max(1, int(n_samples))))
    return m if m > floor else None


def neighbor_unique_target(factors_data, i, neighbor_indices, target, rel_i, factors_nbins, dtype=np.int32, cached_MIs: Optional[dict] = None):
    """Sum_j I(Y; X_j | X_i) over neighbors via the chain rule ``I((X_i,X_j);Y) - I(X_i;Y)``.

    Returns ``(total_unique, [(j, cmi_raw), ...])``. Large total => neighbors carry target info beyond X_i
    (friend-graph "sink"); small total => neighbors are noisy copies of the same signal (a reflection
    cluster the aggregate step wants).

    The per-neighbor value carried in ``detail`` is the RAW chain-rule CMI (it may be slightly negative from plug-in finite-sample noise -- the joint estimate over the
    (X_i, X_j) cells carries more positive bias than the marginal I(X_i;Y), so a true-zero CMI fluctuates around 0). Only the ``total_unique`` AGGREGATE is clamped at 0
    for the red-flag comparison, which must be a non-negative "how much extra do the friends know" quantity. Previously the per-neighbor value was clamped too, which
    silently zeroed every noisy-but-real positive neighbor and could leave a red node flagged-but-unprunable (no justifier survived the clamp); keeping the raw sign in
    ``detail`` lets ``prune_by_friend_graph`` recognise a weakly-positive justifier instead of discarding it.

    ``cached_MIs`` (optional, keyed by the order-independent ``(min(i,j), max(i,j))`` pair): the joint
    ``I((X_i,X_j);Y)`` is symmetric in ``i``/``j``, so when the caller iterates suspects and both ``i``
    and ``j`` are suspects that are each other's neighbors, this value is computed twice (once per side)
    without the cache. Threading a dict shared across the caller's suspect loop makes the second side a
    lookup instead of a recompute -- bit-identical (same ``mi()`` value, just reused).
    """
    detail: List[Tuple[int, float]] = []
    total_unique = 0.0
    for j in neighbor_indices:
        key = (i, j) if i <= j else (j, i)
        if cached_MIs is not None and key in cached_MIs:
            joint = cached_MIs[key]
        else:
            joint = float(mi(factors_data, np.array([i, j], dtype=np.int64), target, factors_nbins, dtype=dtype))
            if cached_MIs is not None:
                cached_MIs[key] = joint
        cmi = joint - rel_i
        detail.append((int(j), cmi))
        total_unique += max(0.0, cmi)
    return total_unique, detail


def _layout(sel: List[int], edges: List[FriendGraphEdge], seed) -> Dict[int, Tuple[float, float]]:
    """Node positions via networkx ``spring_layout`` (force-directed); deterministic
    circular fallback when networkx is absent so the build never blocks on an
    optional dependency."""
    if not sel:
        return {}
    if len(sel) == 1:
        return {sel[0]: (0.0, 0.0)}
    try:
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(sel)
        for e in edges:
            g.add_edge(e.a, e.b, weight=float(e.mi))
        raw = nx.spring_layout(g, weight="weight", seed=int(seed) if seed is not None else None)
        return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}
    except (ImportError, ModuleNotFoundError) as _exc:
        logger.debug("networkx unavailable (%s); using deterministic circular layout", _exc)
    except Exception as _exc:
        # Layout is purely cosmetic (node positions for the plot), so a degenerate-graph failure inside spring_layout must not abort graph construction -- but WARN
        # so a real networkx bug is distinguishable from the expected "networkx absent" path above, then fall through to the deterministic circular layout.
        logger.warning("friend_graph spring_layout raised %s: %s; using circular layout", type(_exc).__name__, _exc, exc_info=True)
    n = len(sel)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return {int(v): (float(np.cos(a)), float(np.sin(a))) for v, a in zip(sel, angles)}


def build_friend_graph(
    selected_vars: Sequence[int],
    factors_data: np.ndarray,
    factors_nbins: np.ndarray,
    target_indices: np.ndarray,
    *,
    feature_names: Optional[Sequence[str]] = None,
    entropy_cache: Optional[dict] = None,
    cached_MIs: Optional[dict] = None,  # accepted for forward reuse; computation falls back to mi()
    mi_eps: float = 1e-6,
    edge_significance: float = 3.0,
    garbage_min_degree: int = 3,
    garbage_unique_ratio: float = 1.0,
    unique_max_degree: int = 1,
    max_nodes: int = 200,
    compute_layout: bool = True,
    dtype=np.int32,
    seed=None,
    gpu_backend: Optional[str] = None,
) -> FriendGraph:
    """Build the friend graph for ``selected_vars`` over the discretized matrix.

    Parameters mirror ``MRMR.fit`` internals: ``factors_data`` is the binned
    integer matrix from ``categorize_dataset``; ``target_indices`` are the
    target column(s). Edges and the conditional garbage criterion use the
    plug-in ``mi`` estimator over that matrix.

    ``max_nodes`` guards the O(k^2) edge pass: above it, only node-level stats
    are computed (no edges, no classification beyond entropy/relevance) and a
    warning is logged. True higher-order "unique information" is intractable at
    hundreds-to-thousands of features / 100+ lags; the neighbor-aggregated
    ``sum_j I(Y; X_j | X)`` is the practical proxy this graph reports.

    ``gpu_backend`` controls the GPU acceleration of the O(k^2) pairwise-MI edge
    pass + the k node entropy/relevance stats (``friend_graph_gpu``): ``None``
    (default) dispatches via the per-host kernel_tuning cache keyed by ``(k, n)``
    and falls back to CPU when no GPU / not chosen; ``"cpu"`` forces the legacy
    CPU edge pass; ``"cupy"`` / ``"cuda"`` force a GPU backend. The GPU path is
    BIT-IDENTICAL: the GPU does only the integer joint/marginal counting, the
    entropy + every keep/drop decision (significance floor, ADC direction,
    garbage classification) stay on the bit-exact CPU path.
    """
    sel = [int(v) for v in selected_vars]
    target = np.asarray(target_indices, dtype=np.int64)
    if feature_names is not None:
        names = {i: str(feature_names[i]) if i < len(feature_names) else f"col_{i}" for i in sel}
    else:
        names = {i: f"col_{i}" for i in sel}

    graph = FriendGraph()
    if not sel:
        return graph

    edges_skipped = len(sel) > max_nodes
    if edges_skipped:
        logger.warning(
            "friend_graph: %d selected features exceeds max_nodes=%d; computing node "
            "stats only and skipping the O(k^2) edge pass. Raise max_nodes to force it.",
            len(sel), max_nodes,
        )

    # GPU dispatch for the O(k^2) edge pass + the k node entropy/relevance stats.
    # BIT-IDENTICAL (GPU does only integer counting; entropy + every keep/drop decision
    # stay on the bit-exact CPU path). ``None`` -> per-host kernel_tuning dispatch (CPU
    # fallback default); "cpu" -> force the legacy CPU pass; "cupy"/"cuda" -> force GPU.
    # Skipped when the edge pass is skipped (max_nodes guard) -- no O(k^2) cost to offload.
    gpu_stats = None
    if not edges_skipped and gpu_backend != "cpu":
        try:
            from .friend_graph_gpu import dispatch_friend_graph_stats

            gpu_stats = dispatch_friend_graph_stats(
                sel, factors_data, factors_nbins, target,
                dtype=dtype, force_backend=gpu_backend,
            )
        except (ImportError, ModuleNotFoundError) as _exc:
            # The expected "no GPU here" outcome: cupy / the GPU twin module is absent. Quietly fall back to the bit-identical CPU edge pass.
            logger.debug("friend_graph GPU dispatch unavailable (%s); using CPU edge pass", _exc)
            gpu_stats = None
        except Exception as _exc:
            # A real error from the GPU path (shape mismatch, bad dtype, CUDA OOM) -- distinct from "GPU absent". WARN so a genuine kernel bug is not silently
            # indistinguishable from a missing device, then still fall back to CPU so the diagnostic graph is produced.
            logger.warning("friend_graph GPU dispatch raised %s: %s; falling back to CPU edge pass", type(_exc).__name__, _exc, exc_info=True)
            gpu_stats = None

    # Per-node entropy + target relevance (from GPU stats when present, else CPU
    # primitives). GPU H/rel are bit-identical to ``_node_entropy`` / ``node_relevance``.
    H: Dict[int, float] = {}
    rel: Dict[int, float] = {}
    gpu_rel = gpu_stats.rel if (gpu_stats is not None) else None
    for i in sel:
        if gpu_stats is not None and i in gpu_stats.H:
            H[i] = gpu_stats.H[i]
            if entropy_cache is not None:
                entropy_cache.setdefault((int(i),), H[i])
        else:
            H[i] = _node_entropy(factors_data, i, factors_nbins, entropy_cache, dtype)
        if gpu_rel is not None and i in gpu_rel:
            rel[i] = gpu_rel[i]
        else:
            rel[i] = node_relevance(factors_data, i, target, factors_nbins, dtype=dtype)

    edges: List[FriendGraphEdge] = []
    neighbors: Dict[int, List[Tuple[int, float]]] = {i: [] for i in sel}
    weighted_degree: Dict[int, float] = {i: 0.0 for i in sel}

    if not edges_skipped:
        n_samples = max(1, int(factors_data.shape[0]))
        gpu_edges = gpu_stats.edge_mi if (gpu_stats is not None) else None
        for a, b in combinations(sel, 2):
            # Pairwise feature-feature MI gated by the finite-sample significance floor (see
            # ``pairwise_mi_edge``): suppresses spurious links between independent features.
            # iter469: pass the cached per-node marginals so the edge MI only
            # computes the joint entropy (1 merge_vars vs mi()'s 3). On
            # k selected features this drops the O(k^2) edge pass from
            # 3*C(k,2) marginal+joint passes to C(k,2)+k.
            #
            # GPU path: the raw I(X_a; X_b) is the bit-identical GPU value
            # (H_a + H_b - H_ab, clamped >=0); the SAME significance floor is then
            # applied here so the keep/drop decision + topology are unchanged.
            if gpu_edges is not None:
                m = _apply_edge_floor(
                    gpu_edges.get((a, b)), a, b, factors_nbins, n_samples,
                    mi_eps=mi_eps, edge_significance=edge_significance,
                )
            else:
                m = pairwise_mi_edge(
                    factors_data, a, b, factors_nbins, n_samples,
                    mi_eps=mi_eps, edge_significance=edge_significance, dtype=dtype,
                    h_a=H[a], h_b=H[b],
                )
            if m is None:
                continue
            # ADC / uncertainty-coefficient direction: arrow points to the more-explained
            # node (larger I/H). The explainer is the lower-coefficient (higher-entropy) end.
            u_a = m / H[a] if H[a] > 0 else 0.0
            u_b = m / H[b] if H[b] > 0 else 0.0
            if u_a >= u_b:
                src, dst = b, a  # b explains a more -> arrow b -> a
            else:
                src, dst = a, b
            edges.append(FriendGraphEdge(a=src, b=dst, mi=m))
            neighbors[a].append((b, m))
            neighbors[b].append((a, m))
            weighted_degree[a] += m
            weighted_degree[b] += m

    shared_frac = {i: (weighted_degree[i] / H[i] if H[i] > 0 else 0.0) for i in sel}

    # Garbage classification. A node is a sink suspect when it is connected to many
    # others (degree >= garbage_min_degree -- "связан с множеством других"). Only
    # suspects pay the conditional-MI pass, which measures how much UNIQUE target
    # information their neighbors carry beyond them: sum_j I(Y; X_j | X_i). A suspect
    # is flagged red when that exceeds its own relevance (its friends know more about
    # the target than it adds -- an aggregator). A node connected to at most
    # unique_max_degree others carries non-shared knowledge and is green.
    neighbors_unique: Dict[int, float] = {i: 0.0 for i in sel}
    klass: Dict[int, str] = {}
    # Bias-debias constants for the F3 garbage threshold: n (row count) and the target cardinality (product of the target columns' bin counts).
    n_samples_garbage = float(max(1, int(factors_data.shape[0])))
    n_y_card = 1
    for _t in np.asarray(target, dtype=np.int64).ravel():
        n_y_card *= int(factors_nbins[int(_t)])
    n_y_card = max(2, n_y_card)
    # Shared across the whole suspect loop: I((X_i,X_j);Y) is symmetric, so when both i and j are
    # suspects and neighbors of each other this cache turns the second side's computation into a
    # lookup instead of a full mi() recompute (see neighbor_unique_target docstring).
    _joint_mi_cache: dict = {}
    for i in sel:
        if edges_skipped:
            klass[i] = "yellow"
            continue
        # Per-suspect neighbor scan below runs only when the edge pass was not skipped, i.e. len(sel) <= max_nodes (200);
        # above that, edges_skipped short-circuits every node here, so this scan is already bounded to <=200 suspects.
        degree = len(neighbors[i])
        if degree >= garbage_min_degree:
            # sum_j I(Y; X_j | X_i) via the chain rule (shared helper).
            total_unique, detail = neighbor_unique_target(
                factors_data, i, [j for j, _m in neighbors[i]], target, rel[i], factors_nbins, dtype=dtype, cached_MIs=_joint_mi_cache,
            )
            neighbors_unique[i] = total_unique
            graph._neighbor_unique_detail[i] = detail
            # detail is built in the same order as neighbors[i] two lines above; index it once here
            # instead of a linear next(...) scan per neighbor below (O(degree) instead of O(degree^2)).
            detail_by_j = {jj: c for jj, c in detail}
            # F3 finite-sample debias: the red-flag compares ``total_unique`` (a SUM of ``degree`` chain-rule CMIs, each carrying the upward plug-in bias of a 2-variable
            # joint MI ~ (n_i*n_j-1)(n_y-1)/(2n)) against ``rel[i]`` (a single 1-variable MI, bias ~ (n_i-1)(n_y-1)/(2n)). Comparing the inflated multi-term sum against the
            # less-inflated single term tilts the decision toward flagging high-cardinality nodes red purely from bias, flipping which features ``prune_by_friend_graph``
            # removes on small n. Subtract the Miller-Madow bias from BOTH sides (each chain-rule term debiased by [(n_i*n_j-1)-(n_i-1)](n_y-1)/(2n); rel[i] by its own
            # (n_i-1)(n_y-1)/(2n)) so the threshold compares bias-matched quantities. At large n the corrections vanish and the decision is unchanged.
            n_i = int(factors_nbins[i])
            total_unique_db = 0.0
            for j, _m in neighbors[i]:
                n_j = int(factors_nbins[j])
                cmi_bias = (n_i * n_j - 1 - (n_i - 1)) * (n_y_card - 1) / (2.0 * n_samples_garbage)
                idx_cmi = detail_by_j.get(int(j), 0.0)
                total_unique_db += max(0.0, idx_cmi - cmi_bias)
            rel_i_db = max(0.0, rel[i] - (n_i - 1) * (n_y_card - 1) / (2.0 * n_samples_garbage))
            if total_unique_db > max(mi_eps, garbage_unique_ratio * rel_i_db):
                klass[i] = "red"
            else:
                klass[i] = "yellow"
        elif degree <= unique_max_degree:
            klass[i] = "green"
        else:
            klass[i] = "yellow"

    for i in sel:
        graph.nodes.append(FriendGraphNode(
            idx=i, name=names[i], entropy=H[i], relevance=rel[i],
            weighted_degree=weighted_degree[i], shared_frac=shared_frac[i],
            neighbors_unique_target=neighbors_unique[i], klass=klass[i],
        ))
    graph.edges = edges
    graph.suspected_garbage = [names[i] for i in sel if klass[i] == "red"]
    if compute_layout:
        graph.pos = _layout(sel, edges, seed)
    return graph


def prune_by_friend_graph(graph: FriendGraph, selected_vars: Sequence[int], protect_indices: Sequence[int] = ()) -> Tuple[List[int], Dict[str, str]]:
    """Drop suspected-garbage features while protecting the cause of each removal.

    Worst-first over red nodes: a red feature ``X`` is removed only when at least
    one neighbor ``Z`` carries unique target information beyond it
    (``I(Y; Z | X) > 0``), meaning ``X``'s contribution is recoverable through its
    friends. Those justifying neighbors are then marked protected so a later
    iteration never removes both the cause (``Z``) and the effect (``X``).

    ``protect_indices``: cols-space indices never to prune (e.g. cluster-aggregate columns, which are
    correlated with all their members by construction and would otherwise be mis-flagged as sinks).
    """
    by_idx = {n.idx: n for n in graph.nodes}
    reds = [n for n in graph.nodes if n.klass == "red"]
    reds.sort(key=lambda n: (n.shared_frac, n.neighbors_unique_target), reverse=True)

    protected: set = set(int(i) for i in protect_indices)
    removed: List[int] = []
    reasons: Dict[str, str] = {}
    for node in reds:
        if node.idx in protected:
            continue
        detail = graph._neighbor_unique_detail.get(node.idx, [])
        justifiers = [j for j, cmi in detail if cmi > 0.0 and j not in removed]
        if not justifiers:
            continue
        removed.append(node.idx)
        just_names = [by_idx[j].name for j in justifiers if j in by_idx]
        reasons[node.name] = (
            "redundant sink: shares "
            f"{node.shared_frac:.0%} of its entropy with neighbors that carry "
            f"more unique target info; recoverable via {', '.join(just_names[:8])}"
        )
        protected.update(justifiers)

    removed_set = set(removed)
    pruned_vars = [int(v) for v in selected_vars if int(v) not in removed_set]
    graph.pruned = [by_idx[i].name for i in removed if i in by_idx]
    graph.prune_reasons = reasons
    return pruned_vars, reasons


# ----------------------------------------------------------------------------
# Rendering (via the declarative reporting system)
# ----------------------------------------------------------------------------


def friend_graph_to_figurespec(graph: FriendGraph, *, title: str = "Feature friend graph", edge_cmap: Optional[str] = None):
    """Wrap a ``FriendGraph`` in a single-panel ``FigureSpec`` (``NetworkPanelSpec``)."""
    from mlframe.reporting.colors import FRIEND_GRAPH_EDGE_CMAP, friend_graph_node_color
    from mlframe.reporting.spec import FigureSpec, NetworkPanelSpec

    nodes = graph.nodes
    pos_of = {n.idx: idx for idx, n in enumerate(nodes)}
    node_x = np.array([graph.pos.get(n.idx, (0.0, 0.0))[0] for n in nodes], dtype=float)
    node_y = np.array([graph.pos.get(n.idx, (0.0, 0.0))[1] for n in nodes], dtype=float)

    h_max = max((n.entropy for n in nodes), default=1.0) or 1.0
    node_size = np.array([250.0 + 1000.0 * (n.entropy / h_max) for n in nodes], dtype=float)
    node_color = tuple(friend_graph_node_color(n.klass) for n in nodes)
    node_label = tuple(n.name for n in nodes)
    node_hovertext = tuple(
        f"{n.name}<br>class={n.klass}<br>H(X)={n.entropy:.4f}<br>I(X;Y)={n.relevance:.4f}"
        f"<br>weighted degree={n.weighted_degree:.4f}<br>shared={n.shared_frac:.0%}"
        f"<br>neighbors' unique target info={n.neighbors_unique_target:.4f}"
        for n in nodes
    )

    edge_src = np.array([pos_of[e.a] for e in graph.edges], dtype=np.int64)
    edge_dst = np.array([pos_of[e.b] for e in graph.edges], dtype=np.int64)
    edge_weight = np.array([e.mi for e in graph.edges], dtype=float)

    panel = NetworkPanelSpec(
        node_x=node_x, node_y=node_y, node_size=node_size, node_color=node_color,
        node_label=node_label, node_hovertext=node_hovertext,
        edge_src=edge_src, edge_dst=edge_dst, edge_weight=edge_weight,
        edge_directed=True, colormap=edge_cmap or FRIEND_GRAPH_EDGE_CMAP,
        colorbar_label="Mutual information (nats)",
        node_legend=(
            ("unique", friend_graph_node_color("green")),
            ("suspected sink", friend_graph_node_color("red")),
            ("middling", friend_graph_node_color("yellow")),
        ),
        title="",
    )
    return FigureSpec(suptitle=title, panels=((panel,),), figsize=(11.0, 8.0))


def plot_friend_graph(graph: FriendGraph, *, plot_outputs: str, base_path: str, title: str = "Feature friend graph") -> None:
    """Render + save the friend graph via the reporting DSL (mirrors other domain plots)."""
    from mlframe.reporting.output import parse_plot_output_dsl
    from mlframe.reporting.renderers import render_and_save

    spec = friend_graph_to_figurespec(graph, title=title)
    render_and_save(spec, parse_plot_output_dsl(plot_outputs), base_path)


__all__ = [
    "FriendGraphNode",
    "FriendGraphEdge",
    "FriendGraph",
    "build_friend_graph",
    "prune_by_friend_graph",
    "friend_graph_to_figurespec",
    "plot_friend_graph",
    "node_relevance",
    "pairwise_mi_edge",
    "neighbor_unique_target",
]
