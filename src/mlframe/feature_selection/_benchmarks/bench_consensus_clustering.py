"""Benchmark: does bootstrap consensus (co-association) clustering recover
planted feature groups better than single-shot single-linkage? (audit
hierarchy-stability-6 / friend-graph-6).

The clean case already gives single-linkage ARI 1.0 (see
bench_community_vs_single_linkage), so consensus can only help in a BORDERLINE
regime where within-group SU sits near tau and per-sample fluctuations flip
edges -> unstable single-shot membership. We tune noise to that regime and
compare single-shot ARI vs consensus (B bootstraps -> KxK co-association ->
cluster at 0.5) against the planted truth.

RESULT (2026-06-03): MARGINAL narrow-band win -> NOT worth the B x cost. At
noise 0.25 (corr ~0.94) single-shot is already perfect (ARI 1.0); at the
borderline noise 0.35 consensus edges it (1.000 vs 0.963, +0.037 ARI); at 0.45
binned SU drops below tau and BOTH fail (ARI 0). So consensus improves cluster-
LABEL stability only in a narrow band, costs B x the clustering work per fit, and
the hierarchy it would stabilise is analyst metadata that does not gate
selection. Not shipped.
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.info_theory import entropy, merge_vars


def _quantize(X):
    fd = np.empty_like(X, dtype=np.int32)
    for j in range(X.shape[1]):
        e = np.unique(np.quantile(X[:, j], np.linspace(0, 1, 11)))
        fd[:, j] = np.searchsorted(e[1:-1], X[:, j], side="right")
    fn = np.array([int(fd[:, j].max()) + 1 for j in range(fd.shape[1])], dtype=np.int64)
    return fd, fn


def _su_matrix(fd, fn):
    p = fd.shape[1]
    H = [entropy(merge_vars(fd, np.array([i], dtype=np.int64), None, fn)[1]) for i in range(p)]
    S = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            hij = entropy(merge_vars(fd, np.array([i, j], dtype=np.int64), None, fn)[1])
            d = H[i] + H[j]
            S[i, j] = S[j, i] = max(0.0, min(1.0, 2.0 * (H[i] + H[j] - hij) / d)) if d > 0 else 0.0
    return S


def _cc(S, tau):
    p = S.shape[0]; parent = list(range(p))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(p):
        for j in range(i + 1, p):
            if S[i, j] > tau:
                parent[find(j)] = find(i)
    roots = [find(i) for i in range(p)]
    u = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([u[r] for r in roots])


def _make(n, seed, G=4, per=4, noise=0.7):
    rng = np.random.default_rng(seed)
    Z = [rng.standard_normal(n) for _ in range(G)]
    cols, truth = [], []
    for gi, z in enumerate(Z):
        for _ in range(per):
            cols.append(z + noise * rng.standard_normal(n)); truth.append(gi)
    return np.column_stack(cols), np.array(truth)


def main():
    from sklearn.metrics import adjusted_rand_score
    tau, B = 0.3, 25
    print(f"tau={tau} B={B} bootstraps; planted G=4 x4 members; borderline noise")
    for noise in (0.25, 0.35, 0.45):
        ss, cons = [], []
        for seed in range(6):
            X, truth = _make(2000, seed, noise=noise)
            p = X.shape[1]
            fd, fn = _quantize(X)
            ss_labels = _cc(_su_matrix(fd, fn), tau)
            ss.append(adjusted_rand_score(truth, ss_labels))
            # consensus: co-association over bootstraps
            co = np.zeros((p, p))
            rng = np.random.default_rng(1000 + seed)
            for _ in range(B):
                idx = rng.integers(0, X.shape[0], X.shape[0])
                fdb, fnb = _quantize(X[idx])
                lab = _cc(_su_matrix(fdb, fnb), tau)
                same = lab[:, None] == lab[None, :]
                co += same
            co /= B
            cons_labels = _cc(co, 0.5)  # co-cluster >= 50% of bootstraps
            cons.append(adjusted_rand_score(truth, cons_labels))
        print(f"noise={noise}: single-shot ARI={np.mean(ss):.3f}  consensus ARI={np.mean(cons):.3f}  " f"delta={np.mean(cons)-np.mean(ss):+.3f}")
    print("WIN for consensus iff ARI materially higher where single-shot is unstable.")


if __name__ == "__main__":
    main()
