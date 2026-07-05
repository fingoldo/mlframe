"""Iter94 trust_guard_n_estimators sweep at C3.

Sweep of ``ShapProxiedFS(trust_guard_n_estimators=...)`` over {100, 50, 25} at C3
(width=10000, n_rows=10000, n_informative=20, n_redundant=20, snr=8.0, seed=0).

iter93 fixed n_anchors=30. PER-ANCHOR booster size is the actual trust_guard wall driver:
30 anchors x N trees. The trust report only consumes RANKS of anchor losses
(Spearman / Kendall / recall@k); a capped booster should yield faithful rank fidelity at
lower cost. Gate: chosen subset BIT-IDENTICAL across all three, trustworthy=True everywhere,
composite proxy_fidelity_score within +-5% of baseline at the smallest tested value.

Run::

    $env:PYTHONPATH = '<worktree>/src'
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_iter94_trust_guard_n_estimators_sweep
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


def _build_selector(seed, *, trust_guard_n_estimators):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    return ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=500, cluster_features=True, cluster_corr_threshold=0.7,
        top_n=20, n_splits=4, n_revalidation_models=3, trust_guard=True,
        n_anchors=30,
        trust_guard_n_estimators=trust_guard_n_estimators,
        run_importance_ablation=True, within_cluster_refine=True,
        revalidation_ucb_enabled=True,
        revalidation_adaptive_n_models=True,
        random_state=seed, verbose=False)


def run_one(value, cfg, X, y, roles):
    print(f"\n[trust_guard_n_estimators={value}] starting", flush=True)
    sel = _build_selector(cfg["seed"], trust_guard_n_estimators=value)
    sel._stage_timings = {}
    t0 = time.perf_counter()
    sel.fit(X, y)
    total = time.perf_counter() - t0
    trust = sel.shap_proxy_report_.get("trust", {}) or {}
    chosen = tuple(sorted(sel.selected_features_))
    return dict(
        value=value,
        total=total,
        trust_guard_wall=sel._stage_timings.get("trust_guard"),
        chosen=chosen,
        n_selected=len(chosen),
        trustworthy=trust.get("trustworthy"),
        spearman=trust.get("spearman"),
        kendall=trust.get("kendall"),
        recall_at_k=trust.get("recall_at_k"),
        proxy_fidelity_score=trust.get("proxy_fidelity_score"),
    )


def main():
    print(f"[iter94] cfg={C3}", flush=True)
    t0 = time.perf_counter()
    X, y, roles = _make_dataset(C3)
    print(f"[iter94] dataset shape={X.shape} in {time.perf_counter()-t0:.1f}s", flush=True)

    results = []
    for value in (100, 50, 25):
        r = run_one(value, C3, X, y, roles)
        print(f"[trust_guard_n_estimators={value}] total={r['total']:.2f}s "
              f"tg_wall={r['trust_guard_wall']:.3f}s "
              f"trustworthy={r['trustworthy']} "
              f"sp={r['spearman']:.4f} "
              f"recall@k={r['recall_at_k']:.4f} "
              f"fidelity={r['proxy_fidelity_score']:.4f} "
              f"n_sel={r['n_selected']}", flush=True)
        results.append(r)

    print("\n=== iter94 trust_guard_n_estimators sweep ===")
    print(f"{'value':>6} {'tg_wall':>10} {'e2e':>8} {'trust':>7} {'spearman':>10} {'kendall':>10} {'recall@k':>10} {'fidelity':>10} {'n_sel':>6}")
    for r in results:
        print(f"{r['value']:>6} {r['trust_guard_wall']:>10.3f} {r['total']:>8.2f} "
              f"{str(r['trustworthy']):>7} {r['spearman']:>10.4f} {r['kendall']:>10.4f} "
              f"{r['recall_at_k']:>10.4f} {r['proxy_fidelity_score']:>10.4f} {r['n_selected']:>6}")

    base = results[0]
    print("\n=== chosen-subset comparison vs baseline value=100 ===")
    for r in results:
        ident = "IDENTICAL" if r["chosen"] == base["chosen"] else "DIFFER"
        jac = len(set(r["chosen"]) & set(base["chosen"])) / max(1, len(set(r["chosen"]) | set(base["chosen"])))
        symdiff = set(base["chosen"]) ^ set(r["chosen"])
        print(f"  value={r['value']:>3}: {ident}  jaccard={jac:.3f}  symdiff={sorted(symdiff)}")

    print("\n=== composite fidelity delta vs baseline ===")
    base_fid = base["proxy_fidelity_score"] or 0.0
    for r in results:
        fid = r["proxy_fidelity_score"] or 0.0
        delta = (fid - base_fid) / abs(base_fid) if base_fid else 0.0
        print(f"  value={r['value']:>3}: fidelity={fid:.4f}  delta={delta*100:+.2f}%")

    print("\nbaseline (value=100) chosen:", sorted(base["chosen"]))


if __name__ == "__main__":
    main()
