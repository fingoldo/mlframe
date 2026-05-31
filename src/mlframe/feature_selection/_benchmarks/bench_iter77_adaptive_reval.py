"""Iter77 adaptive ``n_revalidation_models`` A/B at C3.

Baseline (``revalidation_adaptive_n_models=False``, runs all 3 seed-rounds in one combined batch)
vs lever (``revalidation_adaptive_n_models=True``, model-round early-stop on winner stability).

Run::

    $env:PYTHONPATH = '<worktree>/src'
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_iter77_adaptive_reval
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


CONFIGS = {
    "C3": dict(width=10000, n_rows=10000, n_informative=20, n_redundant=20,
               redundancy_rho=0.8, snr=8.0, seed=0),
    # iter92 lever: comparison criterion changes only inside the adaptive loop; at higher
    # redundancy_rho cluster aggregation collapses more units to identical member sets, so the
    # member-equiv early-stop is more likely to fire one round before the unit-equiv would.
    "C3_high_redundancy": dict(width=10000, n_rows=10000, n_informative=20, n_redundant=20,
                               redundancy_rho=0.9, snr=8.0, seed=0),
}


def _make_dataset(cfg):
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    n_noise = max(0, cfg["width"] - cfg["n_informative"] - cfg["n_redundant"])
    X, y, roles = make_regime_dataset(
        n_samples=cfg["n_rows"], n_informative=cfg["n_informative"],
        n_redundant=cfg["n_redundant"], redundancy_rho=cfg["redundancy_rho"],
        n_noise=n_noise, snr=cfg["snr"], task="binary", seed=cfg["seed"])
    return X, y, roles


def _build_selector(seed, *, adaptive):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    return ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=500, cluster_features=True, cluster_corr_threshold=0.7,
        top_n=20, n_splits=4, n_revalidation_models=3, trust_guard=True, n_anchors=24,
        run_importance_ablation=True, within_cluster_refine=True,
        revalidation_ucb_enabled=True,
        revalidation_adaptive_n_models=adaptive,
        random_state=seed, verbose=False)


def _recovered(sel, roles):
    inf = {n for n, r in roles.items() if r == "informative"}
    return len(inf & set(sel.selected_features_))


def run_one(name, cfg, *, adaptive):
    label = "ADAPTIVE" if adaptive else "BASELINE"
    print(f"\n[{name} {label}] cfg={cfg}", flush=True)
    t0 = time.perf_counter()
    X, y, roles = _make_dataset(cfg)
    print(f"[{name} {label}] dataset shape={X.shape} in {time.perf_counter()-t0:.1f}s", flush=True)
    sel = _build_selector(cfg["seed"], adaptive=adaptive)
    sel._stage_timings = {}
    t0 = time.perf_counter()
    sel.fit(X, y)
    total = time.perf_counter() - t0
    print(f"[{name} {label}] fit done in {total:.2f}s", flush=True)

    rec = _recovered(sel, roles)
    baseline = sel.shap_proxy_report_.get("revalidation", {}).get("random_baseline", {}) or {}
    ucb = baseline.get("ucb", {}) if isinstance(baseline, dict) else {}
    chosen = tuple(sorted(sel.selected_features_))
    return dict(
        name=name, label=label, total=total,
        stage_timings=dict(sel._stage_timings),
        recall=rec, n_selected=len(sel.selected_features_),
        chosen=chosen,
        adaptive_n_models=ucb.get("adaptive_n_models"),
        n_models_configured=ucb.get("n_models_configured"),
        n_models_run=ucb.get("n_models_run"),
        n_candidates_total=ucb.get("n_candidates_total"),
        n_candidates_evaluated=ucb.get("n_candidates_evaluated"),
        # iter92 diagnostic: True iff member-equiv early-stop fired where the unit-tuple
        # comparison alone would NOT have fired (cluster aggregation collapsed two unit tuples
        # to the same deployed feature set).
        member_equiv_fired=ucb.get("n_reval_models_run_via_member_equiv"),
    )


def print_stage_table(timings, total):
    order = ("prefilter", "clustering", "oof_shap", "prescreen", "search",
             "trust_guard", "revalidation", "importance_ablation", "within_cluster_refine")
    print(f"  total={total:.2f}s  stages:")
    for k in order:
        v = timings.get(k)
        if v is not None:
            print(f"    {k:24s} {v:8.3f}s ({100*v/total:5.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="C3", choices=list(CONFIGS.keys()))
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    results = []
    for adaptive in (False, True):
        r = run_one(args.config, cfg, adaptive=adaptive)
        print_stage_table(r["stage_timings"], r["total"])
        print(f"  recall={r['recall']}/{cfg['n_informative']}, n_selected={r['n_selected']}, "
              f"n_models_run={r['n_models_run']} (configured={r['n_models_configured']})",
              flush=True)
        results.append(r)

    base = results[0]
    lev = results[1]
    rev_base = base["stage_timings"].get("revalidation", 0.0)
    rev_lev = lev["stage_timings"].get("revalidation", 0.0)
    delta_rev = rev_lev - rev_base
    delta_e2e = lev["total"] - base["total"]
    print("\n=== iter77 summary ===")
    print(f"  reval baseline={rev_base:.3f}s   adaptive={rev_lev:.3f}s   delta={delta_rev:+.3f}s "
          f"({100*delta_rev/rev_base:+.1f}%)")
    print(f"  e2e   baseline={base['total']:.3f}s   adaptive={lev['total']:.3f}s   delta={delta_e2e:+.3f}s "
          f"({100*delta_e2e/base['total']:+.1f}%)")
    print(f"  recall baseline={base['recall']}, adaptive={lev['recall']}")
    print(f"  n_models_run baseline={base['n_models_run']}, adaptive={lev['n_models_run']}")
    print(f"  chosen-subset Jaccard: "
          f"{len(set(base['chosen']) & set(lev['chosen'])) / max(1, len(set(base['chosen']) | set(lev['chosen']))):.3f}")
    print(f"  iter92 member_equiv_fired (adaptive): {lev.get('member_equiv_fired')}")


if __name__ == "__main__":
    main()
