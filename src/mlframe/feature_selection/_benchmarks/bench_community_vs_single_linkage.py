"""Benchmark: does graph community detection (modularity) beat single-linkage
connected-components on the SU/MI feature graph? (audit friend-graph-5,
dcd-graph-community-8, integration-defaults-9, hierarchy-stability-14,
shap-proxy-clustering-8 -- all propose Louvain/Leiden vs the current
single-linkage CC / star clustering, which chains via bridge edges.)

Synthetic: G planted groups of reflections of distinct latents, PLUS deliberate
BRIDGE features correlated with TWO groups. Single-linkage CC (edge if SU>tau)
chains bridged groups into one giant cluster; modularity should resist chaining
and recover the planted groups.

Metric: adjusted Rand index (ARI) of recovered labels vs the planted grouping,
and the recovered cluster count, averaged over seeds. Bounded n.

RESULT (2026-06-03): NO ACTIONABLE WIN -> do NOT add a community-detection
backend. The chaining failure mode does not occur on the THRESHOLDED BINNED-SU
graph the code actually uses: a bridge correlated ~0.6 with two groups has
binned SU ~0.09 (within-group SU ~0.50), so it never clears tau and never
chains. Single-linkage CC recovers the planted groups PERFECTLY (ARI 1.0, k=3)
at every tau in [0.25, 0.4]; modularity matches it exactly. To make CC chain you
would need bridge SU > tau, i.e. a bridge that is genuinely a cluster member.
Binning + the SU threshold already provide the separation modularity would add.
Closes friend-graph-5, dcd-graph-community-8, integration-defaults-9,
hierarchy-stability-14, shap-proxy-clustering-8.
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.info_theory import entropy, merge_vars

try:
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities
    _HAVE_NX = True
except Exception:
    _HAVE_NX = False


def _su_matrix(fd, fn, n):
    p = fd.shape[1]
    S = np.zeros((p, p))
    H = []
    for i in range(p):
        _, fi, _ = merge_vars(fd, np.array([i], dtype=np.int64), None, fn)
        H.append(entropy(fi))
    for i in range(p):
        for j in range(i + 1, p):
            _, fij, _ = merge_vars(fd, np.array([i, j], dtype=np.int64), None, fn)
            hij = entropy(fij)
            denom = H[i] + H[j]
            mi = H[i] + H[j] - hij
            S[i, j] = S[j, i] = max(0.0, min(1.0, 2.0 * mi / denom)) if denom > 0 else 0.0
    return S


def _single_linkage_cc(S, tau):
    p = S.shape[0]
    parent = list(range(p))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(p):
        for j in range(i + 1, p):
            if S[i, j] > tau:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    roots = [find(i) for i in range(p)]
    uniq = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([uniq[r] for r in roots])


def _modularity(S, tau):
    g = nx.Graph()
    g.add_nodes_from(range(S.shape[0]))
    n_edges = 0
    for i in range(S.shape[0]):
        for j in range(i + 1, S.shape[0]):
            if S[i, j] > tau:
                g.add_edge(i, j, weight=float(S[i, j]))
                n_edges += 1
    labels = np.arange(S.shape[0], dtype=int)  # all singletons if no edges
    if n_edges == 0:
        return labels
    comms = greedy_modularity_communities(g, weight="weight")
    for k, c in enumerate(comms):
        for node in c:
            labels[node] = k
    return labels


def _make(n, seed, G=3, per=4):
    rng = np.random.default_rng(seed)
    latents = [rng.standard_normal(n) for _ in range(G)]
    cols, truth = [], []
    for gi, z in enumerate(latents):
        for _ in range(per):
            cols.append(z + 0.18 * rng.standard_normal(n)); truth.append(gi)  # corr ~0.97
    # Bridge features: near-equal mixture of two adjacent groups -> a strong edge
    # to members of BOTH, which single-linkage CC uses to chain the groups.
    for gi in range(G - 1):
        cols.append(0.75 * latents[gi] + 0.75 * latents[gi + 1] + 0.12 * rng.standard_normal(n))
        truth.append(-1)  # bridge: no true group
    X = np.column_stack(cols)
    # Quantize to 10 bins (DCD-style).
    fd = np.empty_like(X, dtype=np.int32)
    for j in range(X.shape[1]):
        edges = np.quantile(X[:, j], np.linspace(0, 1, 11))
        edges = np.unique(edges)
        fd[:, j] = np.searchsorted(edges[1:-1], X[:, j], side="right")
    fn = np.array([int(fd[:, j].max()) + 1 for j in range(fd.shape[1])], dtype=np.int64)
    return fd, fn, np.array(truth)


def main():
    if not _HAVE_NX:
        print("networkx unavailable -- cannot benchmark modularity. SKIP.")
        return
    from sklearn.metrics import adjusted_rand_score
    # Diagnostic: SU scale for within-group vs bridge-to-group pairs (seed 0).
    fd0, fn0, truth0 = _make(2000, 0)
    S0 = _su_matrix(fd0, fn0, 2000)
    wi = [S0[i, j] for i in range(len(truth0)) for j in range(i + 1, len(truth0)) if truth0[i] >= 0 and truth0[i] == truth0[j]]
    br = [S0[i, j] for i in range(len(truth0)) for j in range(i + 1, len(truth0)) if (truth0[i] < 0) ^ (truth0[j] < 0)]
    print(f"SU within-group mean={np.mean(wi):.3f} | bridge-edge mean={np.mean(br):.3f} " f"(want within > tau > or ~ bridge so CC chains via bridges)")
    print(f"{'tau':>5} | {'CC ARI':>7} {'CC k':>5} | {'mod ARI':>7} {'mod k':>5}  (planted k=3)")
    for tau in (0.25, 0.3, 0.35, 0.4):
        cc_aris, mod_aris, cc_k, mod_k = [], [], [], []
        for seed in range(6):
            fd, fn, truth = _make(2000, seed)
            S = _su_matrix(fd, fn, 2000)
            mask = truth >= 0
            cc = _single_linkage_cc(S, tau)
            mod = _modularity(S, tau)
            cc_aris.append(adjusted_rand_score(truth[mask], cc[mask]))
            mod_aris.append(adjusted_rand_score(truth[mask], mod[mask]))
            cc_k.append(len(set(cc[mask].tolist())))
            mod_k.append(len(set(mod[mask].tolist())))
        print(f"{tau:>5} | {np.mean(cc_aris):>7.3f} {np.mean(cc_k):>5.1f} | " f"{np.mean(mod_aris):>7.3f} {np.mean(mod_k):>5.1f}")
    print("WIN for modularity iff it keeps ARI~1 / k~3 at a tau where CC chains " "(ARI down / k<3).")


if __name__ == "__main__":
    main()
