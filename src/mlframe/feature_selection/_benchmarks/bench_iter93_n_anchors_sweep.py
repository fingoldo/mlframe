"""Iter93 n_anchors sweep at C3.

Sweep of ``ShapProxiedFS(n_anchors=...)`` over {30, 24, 16, 12, 8} at C3
(width=10000, n_rows=10000, n_informative=20, n_redundant=20, snr=8.0, seed=0) to test
whether the iter40 docs target (24) or a smaller anchor budget preserves the trust-guard
signal. Result: chosen subset is BIT-IDENTICAL across the whole sweep, ``trustworthy=True``
everywhere, but the composite proxy_fidelity_score and recall@k degrade sharply below
n_anchors=30 (recall@k 1.0 -> 0.0 between n=30 and n=8). trust_guard wall is parallelism-
overhead bound until cardinality drops to n=8. Default n_anchors=30 retained.

Run::

    $env:PYTHONPATH = '<worktree>/src'
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_iter93_n_anchors_sweep
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time
import warnings

warnings.filterwarnings("ignore")


C3 = dict(width=10000, n_rows=10000, n_informative=20, n_redundant=20, redundancy_rho=0.8, snr=8.0, seed=0)


def _make_dataset(cfg):
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    n_noise = max(0, cfg["width"] - cfg["n_informative"] - cfg["n_redundant"])
    X, y, roles = make_regime_dataset(
        n_samples=cfg["n_rows"], n_informative=cfg["n_informative"],
        n_redundant=cfg["n_redundant"], redundancy_rho=cfg["redundancy_rho"],
        n_noise=n_noise, snr=cfg["snr"], task="binary", seed=cfg["seed"])
    return X, y, roles


def _build_selector(seed, *, n_anchors):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    return ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=500, cluster_features=True, cluster_corr_threshold=0.7,
        top_n=20, n_splits=4, n_revalidation_models=3, trust_guard=True,
        n_anchors=n_anchors,
        run_importance_ablation=True, within_cluster_refine=True,
        revalidation_ucb_enabled=True,
        revalidation_adaptive_n_models=True,
        random_state=seed, verbose=False)


def run_one(n_anchors, cfg, X, y, roles):
    print(f"\n[n_anchors={n_anchors}] starting", flush=True)
    sel = _build_selector(cfg["seed"], n_anchors=n_anchors)
    sel._stage_timings = {}
    t0 = time.perf_counter()
    sel.fit(X, y)
    total = time.perf_counter() - t0
    trust = sel.shap_proxy_report_.get("trust", {}) or {}
    chosen = tuple(sorted(sel.selected_features_))
    return dict(
        n_anchors=n_anchors,
        total=total,
        trust_guard_wall=sel._stage_timings.get("trust_guard"),
        chosen=chosen,
        n_selected=len(chosen),
        trustworthy=trust.get("trustworthy"),
        spearman=trust.get("spearman"),
        kendall=trust.get("kendall"),
        recall_at_k=trust.get("recall_at_k"),
        proxy_fidelity_score=trust.get("proxy_fidelity_score"),
        report_n_anchors=trust.get("n_anchors"),
    )


def main():
    print(f"[iter93] cfg={C3}", flush=True)
    t0 = time.perf_counter()
    X, y, roles = _make_dataset(C3)
    print(f"[iter93] dataset shape={X.shape} in {time.perf_counter()-t0:.1f}s", flush=True)

    results = []
    for n_anchors in (30, 24, 16, 12, 8):
        r = run_one(n_anchors, C3, X, y, roles)
        print(f"[n_anchors={n_anchors}] total={r['total']:.2f}s "
              f"tg_wall={r['trust_guard_wall']:.3f}s "
              f"trustworthy={r['trustworthy']} "
              f"sp={r['spearman']:.4f} "
              f"recall@k={r['recall_at_k']:.4f} "
              f"fidelity={r['proxy_fidelity_score']:.4f} "
              f"n_sel={r['n_selected']} "
              f"report_n_anchors={r['report_n_anchors']}", flush=True)
        results.append(r)

    print("\n=== iter93 n_anchors sweep ===")
    print(f"{'n_anchors':>10} {'tg_wall':>10} {'e2e':>8} {'trust':>7} {'spearman':>10} {'recall@k':>10} {'fidelity':>10} {'n_sel':>6}")
    for r in results:
        print(f"{r['n_anchors']:>10} {r['trust_guard_wall']:>10.3f} {r['total']:>8.2f} "
              f"{str(r['trustworthy']):>7} {r['spearman']:>10.4f} "
              f"{r['recall_at_k']:>10.4f} {r['proxy_fidelity_score']:>10.4f} {r['n_selected']:>6}")

    base = results[0]
    print("\n=== chosen-subset comparison vs baseline n_anchors=30 ===")
    for r in results:
        ident = "IDENTICAL" if r["chosen"] == base["chosen"] else "DIFFER"
        jac = len(set(r["chosen"]) & set(base["chosen"])) / max(1, len(set(r["chosen"]) | set(base["chosen"])))
        symdiff = set(base["chosen"]) ^ set(r["chosen"])
        print(f"  n_anchors={r['n_anchors']:>3}: {ident}  jaccard={jac:.3f}  symdiff={sorted(symdiff)}")
    print("\nbaseline (n_anchors=30) chosen:", sorted(base["chosen"]))


if __name__ == "__main__":
    main()
