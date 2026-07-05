"""Iter34 UCB revalidation A/B: baseline (UCB off) vs lever (UCB on) at C2, C3, C4.

Phase 0 measurement (Phase 0 of the iter34 task spec) confirmed the joblib threading pool BATCHES
at C3 with 60 honest fits over 8 workers: wall 4.86s vs per-fit 337 ms = 14.4x ratio (NOT
saturation). Iter32's cull-gate didn't pay because at the live regime (width=1000, top_n=20) the
per-fit cost is small and the pool absorbed everything in roughly one productive batch. iter34
keeps all candidates eligible but stops DISPATCHING new batches once the running best stable_score
is provably better than every remaining candidate's UCB lower bound.

Run::

    $env:PYTHONPATH = '<worktree>/src'
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_iter34_ucb_reval
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


CONFIGS = {
    "C2": dict(width=10000, n_rows=5000, n_informative=20, n_redundant=0, redundancy_rho=0.8, snr=8.0, seed=0),
    "C3": dict(width=10000, n_rows=10000, n_informative=20, n_redundant=20, redundancy_rho=0.8, snr=8.0, seed=0),
    "C4": dict(width=20000, n_rows=10000, n_informative=20, n_redundant=20, redundancy_rho=0.8, snr=8.0, seed=0),
    "C4_small": dict(width=20000, n_rows=5000, n_informative=20, n_redundant=20, redundancy_rho=0.8, snr=8.0, seed=0),
}


def _make_dataset(cfg):
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    n_noise = max(0, cfg["width"] - cfg["n_informative"] - cfg["n_redundant"])
    X, y, roles = make_regime_dataset(
        n_samples=cfg["n_rows"], n_informative=cfg["n_informative"],
        n_redundant=cfg["n_redundant"], redundancy_rho=cfg["redundancy_rho"],
        n_noise=n_noise, snr=cfg["snr"], task="binary", seed=cfg["seed"])
    return X, y, roles


def _build_selector(seed: int, *, ucb_enabled: bool):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    return ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=500, cluster_features=True, cluster_corr_threshold=0.7,
        top_n=20, n_splits=4, n_revalidation_models=3, trust_guard=True, n_anchors=24,
        run_importance_ablation=True, within_cluster_refine=True,
        revalidation_ucb_enabled=ucb_enabled,
        random_state=seed, verbose=False)


def _recovered(sel, roles):
    inf = {n for n, r in roles.items() if r == "informative"}
    return len(inf & set(sel.selected_features_))


def _random_brier(y):
    p = float(np.asarray(y).mean())
    return float(np.mean((np.asarray(y, dtype=np.float64) - p) ** 2))


def run_one(name, cfg, *, ucb_enabled, per_config_cap_s=120.0):
    label = "UCB" if ucb_enabled else "BASELINE"
    print(f"\n[{name} {label}] cfg={cfg}", flush=True)
    print(f"[{name} {label}] making dataset...", flush=True)
    t0 = time.perf_counter()
    X, y, roles = _make_dataset(cfg)
    print(f"[{name} {label}] dataset shape={X.shape} in {time.perf_counter()-t0:.1f}s", flush=True)
    sel = _build_selector(cfg["seed"], ucb_enabled=ucb_enabled)
    sel._stage_timings = {}
    t0 = time.perf_counter()
    sel.fit(X, y)
    total = time.perf_counter() - t0
    print(f"[{name} {label}] fit done in {total:.2f}s", flush=True)

    rec = _recovered(sel, roles)
    rb = _random_brier(y)
    chosen_loss = None
    ranked = sel.shap_proxy_report_.get("revalidation", {}).get("ranked", [])
    if ranked:
        chosen_loss = ranked[0].get("honest_loss", ranked[0].get("honest_loss_capped"))
    trust = sel.shap_proxy_report_.get("trust", {})
    trustworthy = trust.get("trustworthy")
    baseline = sel.shap_proxy_report_.get("revalidation", {}).get("random_baseline", {}) or {}
    ucb = baseline.get("ucb", {}) if isinstance(baseline, dict) else {}
    return dict(
        name=name, label=label,
        total=total, stage_timings=dict(sel._stage_timings),
        recall=rec,
        n_selected=len(sel.selected_features_),
        chosen_loss=chosen_loss,
        random_brier=rb,
        trustworthy=trustworthy,
        ucb_enabled=ucb.get("enabled"),
        n_total=ucb.get("n_candidates_total"),
        n_evaluated=ucb.get("n_candidates_evaluated"),
        ucb_slack=ucb.get("slack"),
    )


def print_stage_table(timings, total):
    order = ("prefilter", "clustering", "oof_shap", "prescreen", "search", "trust_guard", "revalidation", "importance_ablation", "within_cluster_refine")
    print(f"  total={total:.2f}s  stages:")
    for k in order:
        v = timings.get(k)
        if v is not None:
            print(f"    {k:24s} {v:8.3f}s ({100*v/total:5.1f}%)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="C2,C3", help="Comma-separated config names (subset of C2,C3,C4,C4_small).")
    ap.add_argument("--per_config_cap_s", type=float, default=120.0)
    args = ap.parse_args(argv)

    requested = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in requested:
        if c not in CONFIGS:
            raise SystemExit(f"unknown config {c}; known: {sorted(CONFIGS)}")

    print(f"iter34 UCB revalidation A/B: configs={requested}", flush=True)
    t_bench0 = time.perf_counter()
    results = {}
    for name in requested:
        cfg = CONFIGS[name]
        r_base = run_one(name, cfg, ucb_enabled=False, per_config_cap_s=args.per_config_cap_s)
        print_stage_table(r_base["stage_timings"], r_base["total"])
        r_ucb = run_one(name, cfg, ucb_enabled=True, per_config_cap_s=args.per_config_cap_s)
        print_stage_table(r_ucb["stage_timings"], r_ucb["total"])
        results[name] = (r_base, r_ucb)

    # Summary
    print("\n" + "=" * 90)
    print("ITER34 SUMMARY")
    print("=" * 90)
    print(f"{'cfg':<8} {'label':<10} {'e2e':>8} {'reval':>8} {'recall':>8} " f"{'subset':>7} {'trust':>6} {'n_eval/n_tot':>14} {'slack':>10}")
    for name in requested:
        for r in results[name]:
            t = r["stage_timings"].get("revalidation", 0.0)
            ne = r.get("n_evaluated"); nt = r.get("n_total")
            slack = r.get("ucb_slack")
            sl = f"{slack:.4f}" if slack is not None else "-"
            net = f"{ne}/{nt}" if ne is not None else "-"
            print(f"{name:<8} {r['label']:<10} {r['total']:>7.2f}s {t:>7.2f}s "
                  f"{r['recall']:>4}/20 {r['n_selected']:>7} "
                  f"{str(r['trustworthy']):>6} {net:>14} {sl:>10}")
    print("\nSpeed comparison (UCB vs BASELINE):")
    for name in requested:
        rb, ru = results[name]
        e2e_speedup = rb["total"] / max(1e-9, ru["total"])
        reval_speedup = (rb["stage_timings"].get("revalidation", 0.0) /
                         max(1e-9, ru["stage_timings"].get("revalidation", 1e-9)))
        print(f"  {name}: e2e {rb['total']:.2f}s -> {ru['total']:.2f}s "
              f"({e2e_speedup:.2f}x); reval "
              f"{rb['stage_timings'].get('revalidation', 0.0):.2f}s -> "
              f"{ru['stage_timings'].get('revalidation', 0.0):.2f}s ({reval_speedup:.2f}x); "
              f"recall {rb['recall']}/20 -> {ru['recall']}/20")
    print(f"\nbench-wall total: {time.perf_counter()-t_bench0:.1f}s")
    return results


if __name__ == "__main__":
    main()
