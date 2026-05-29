"""Iter35 within_cluster_refine UCB A/B: baseline (refine UCB off) vs lever (refine UCB on) at C2, C3.

Mirrors iter34's bench harness; iter34 reval UCB stays ON in both arms so the comparison isolates the
new refine-stage-2b UCB lever. Phase-0 cProfile (D:/Temp/iter35_c3_baseline.profile) attributed 6.14s
of the 34.9s C3 fit to within_cluster_refine, of which ~5s is stage-2b's per-round single-drop trial
dispatch -- the target this lever short-circuits.

Run::

    $env:PYTHONPATH = '<worktree>/src'
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_iter35_refine_ucb
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


CONFIGS = {
    "C2": dict(width=10000, n_rows=5000, n_informative=20, n_redundant=0,
               redundancy_rho=0.8, snr=8.0, seed=0),
    "C3": dict(width=10000, n_rows=10000, n_informative=20, n_redundant=20,
               redundancy_rho=0.8, snr=8.0, seed=0),
}


def _make_dataset(cfg):
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    n_noise = max(0, cfg["width"] - cfg["n_informative"] - cfg["n_redundant"])
    X, y, roles = make_regime_dataset(
        n_samples=cfg["n_rows"], n_informative=cfg["n_informative"],
        n_redundant=cfg["n_redundant"], redundancy_rho=cfg["redundancy_rho"],
        n_noise=n_noise, snr=cfg["snr"], task="binary", seed=cfg["seed"])
    return X, y, roles


def _build_selector(seed: int, *, refine_ucb_enabled: bool):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    return ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=500, cluster_features=True, cluster_corr_threshold=0.7,
        top_n=20, n_splits=4, n_revalidation_models=3, trust_guard=True, n_anchors=24,
        run_importance_ablation=True, within_cluster_refine=True,
        revalidation_ucb_enabled=True,  # iter34 stays on in both arms
        refine_ucb_enabled=refine_ucb_enabled,
        random_state=seed, verbose=False)


def _recovered(sel, roles):
    inf = {n for n, r in roles.items() if r == "informative"}
    return len(inf & set(sel.selected_features_))


def _random_brier(y):
    p = float(np.asarray(y).mean())
    return float(np.mean((np.asarray(y, dtype=np.float64) - p) ** 2))


def run_one(name, cfg, *, refine_ucb_enabled, per_config_cap_s=120.0):
    label = "REFINE-UCB" if refine_ucb_enabled else "BASELINE"
    print(f"\n[{name} {label}] cfg={cfg}", flush=True)
    print(f"[{name} {label}] making dataset...", flush=True)
    t0 = time.perf_counter()
    X, y, roles = _make_dataset(cfg)
    print(f"[{name} {label}] dataset shape={X.shape} in {time.perf_counter()-t0:.1f}s", flush=True)
    sel = _build_selector(cfg["seed"], refine_ucb_enabled=refine_ucb_enabled)
    sel._stage_timings = {}
    t0 = time.perf_counter()
    sel.fit(X, y)
    total = time.perf_counter() - t0
    print(f"[{name} {label}] fit done in {total:.2f}s", flush=True)
    if total > per_config_cap_s:
        print(f"[{name} {label}] WARNING: exceeded per-config cap {per_config_cap_s:.0f}s", flush=True)

    rec = _recovered(sel, roles)
    rb = _random_brier(y)
    chosen_loss = None
    ranked = sel.shap_proxy_report_.get("revalidation", {}).get("ranked", [])
    if ranked:
        chosen_loss = ranked[0].get("honest_loss", ranked[0].get("honest_loss_capped"))
    refine_block = sel.shap_proxy_report_.get("within_cluster_refine", {})
    return dict(
        name=name, label=label, total=total,
        stage_timings=dict(sel._stage_timings),
        recall=rec, n_selected=len(sel.selected_features_),
        chosen_loss=chosen_loss, random_brier=rb,
        refine_before=refine_block.get("before"),
        refine_after=refine_block.get("after"),
        refine_honest_loss_full=refine_block.get("honest_loss_full"),
    )


def print_stage_table(timings, total):
    order = ("prefilter", "clustering", "oof_shap", "prescreen", "search",
             "trust_guard", "revalidation", "importance_ablation", "within_cluster_refine")
    print(f"  total={total:.2f}s  stages:")
    for k in order:
        v = timings.get(k)
        if v is not None:
            print(f"    {k:24s} {v:8.3f}s ({100*v/total:5.1f}%)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="C2,C3",
                    help="Comma-separated config names (subset of C2,C3).")
    ap.add_argument("--per_config_cap_s", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=None,
                    help="Override the per-config seed (lets us re-run for variance characterisation).")
    args = ap.parse_args(argv)

    requested = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in requested:
        if c not in CONFIGS:
            raise SystemExit(f"unknown config {c}; known: {sorted(CONFIGS)}")

    print(f"iter35 refine UCB A/B: configs={requested}", flush=True)
    t_bench0 = time.perf_counter()
    results = {}
    for name in requested:
        cfg = dict(CONFIGS[name])
        if args.seed is not None:
            cfg["seed"] = int(args.seed)
        r_base = run_one(name, cfg, refine_ucb_enabled=False, per_config_cap_s=args.per_config_cap_s)
        print_stage_table(r_base["stage_timings"], r_base["total"])
        r_ucb = run_one(name, cfg, refine_ucb_enabled=True, per_config_cap_s=args.per_config_cap_s)
        print_stage_table(r_ucb["stage_timings"], r_ucb["total"])
        results[name] = (r_base, r_ucb)

    print("\n" + "=" * 90)
    print("ITER35 SUMMARY")
    print("=" * 90)
    print(f"{'cfg':<8} {'label':<12} {'e2e':>8} {'refine':>8} {'reval':>8} {'recall':>8} "
          f"{'subset':>7} {'rfn_b/a':>9} {'loss':>8} {'random':>8}")
    for name in requested:
        for r in results[name]:
            tref = r["stage_timings"].get("within_cluster_refine", 0.0)
            trev = r["stage_timings"].get("revalidation", 0.0)
            cl = f"{r['chosen_loss']:.4f}" if r["chosen_loss"] is not None else "-"
            rb = f"{r['random_brier']:.4f}"
            rfn_ba = f"{r.get('refine_before')}/{r.get('refine_after')}"
            print(f"{name:<8} {r['label']:<12} {r['total']:>7.2f}s {tref:>7.2f}s {trev:>7.2f}s "
                  f"{r['recall']:>4}/20 {r['n_selected']:>7} {rfn_ba:>9} {cl:>8} {rb:>8}")
    print("\nSpeed comparison (REFINE-UCB vs BASELINE):")
    for name in requested:
        rb, ru = results[name]
        e2e_speedup = rb["total"] / max(1e-9, ru["total"])
        refine_speedup = (rb["stage_timings"].get("within_cluster_refine", 0.0) /
                         max(1e-9, ru["stage_timings"].get("within_cluster_refine", 1e-9)))
        print(f"  {name}: e2e {rb['total']:.2f}s -> {ru['total']:.2f}s "
              f"({e2e_speedup:.2f}x); refine "
              f"{rb['stage_timings'].get('within_cluster_refine', 0.0):.2f}s -> "
              f"{ru['stage_timings'].get('within_cluster_refine', 0.0):.2f}s ({refine_speedup:.2f}x); "
              f"recall {rb['recall']}/20 -> {ru['recall']}/20")
    print(f"\nbench-wall total: {time.perf_counter()-t_bench0:.1f}s")
    return results


if __name__ == "__main__":
    main()
