"""Iter58 beam-width sweep: ``brute_force_max_features`` in {22, 28, 32, 40} at C3 / C3_hard.

Iter57's audit established that at default ``max_features=None`` the brute-force kernel enumerates
2^n - 1 subsets, so the dispatcher's 80M n_sub gate caps the actual brute-force path at n<=26.
n in {27, 28} (and any higher value the cap permits) runs BEAM over the wider prescreen pool. So
``brute_force_max_features`` at default config is really a prescreen-pool-width knob for beam.

Beam scales as O(beam_width * max_card * n_samples * n_features_pool) - linear in the pool width -
so widening the pool is much cheaper than widening brute force, and may keep recovering missing
informatives well past n=28. This bench measures the recall/wall trade-off across the sweep so we
can pick the smallest cap where recall plateaus (or, if wall blows up, the last fast point).

Run::

    $env:PYTHONPATH = '<worktree>/src'
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_iter58_beam_width_sweep --configs C3,C3_hard
"""

from __future__ import annotations

import argparse
import time
import warnings

warnings.filterwarnings("ignore")


CONFIGS = {
    "C3": dict(width=10000, n_rows=10000, n_informative=20, n_redundant=20, redundancy_rho=0.8, snr=8.0, seed=0),
    # Hard regime: low SNR + no redundants. Iter56 saw the larger absolute recall gain here.
    "C3_hard": dict(width=10000, n_rows=10000, n_informative=20, n_redundant=0, redundancy_rho=0.8, snr=2.0, seed=0),
}

WIDTHS = (22, 28, 32, 40)


def _make_dataset(cfg):
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    n_noise = max(0, cfg["width"] - cfg["n_informative"] - cfg["n_redundant"])
    X, y, roles = make_regime_dataset(
        n_samples=cfg["n_rows"], n_informative=cfg["n_informative"],
        n_redundant=cfg["n_redundant"], redundancy_rho=cfg["redundancy_rho"],
        n_noise=n_noise, snr=cfg["snr"], task="binary", seed=cfg["seed"])
    return X, y, roles


def _build_selector(seed: int, brute_force_max_features: int):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    return ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=500, cluster_features=True, cluster_corr_threshold=0.7,
        top_n=20, n_splits=4, n_revalidation_models=3, trust_guard=True, n_anchors=24,
        run_importance_ablation=True, within_cluster_refine=True,
        brute_force_max_features=brute_force_max_features,
        random_state=seed, verbose=False)


def _recovered(sel, roles):
    inf = {n for n, r in roles.items() if r == "informative"}
    return len(inf & set(sel.selected_features_)), len(inf)


def run_one(name: str, cfg: dict, brute_force_max_features: int, X, y, roles):
    label = f"cap{brute_force_max_features}"
    print(f"\n[{name} / {label}] fitting...", flush=True)
    sel = _build_selector(cfg["seed"], brute_force_max_features=brute_force_max_features)
    sel._stage_timings = {}
    t0 = time.perf_counter()
    sel.fit(X, y)
    total = time.perf_counter() - t0
    rec_hit, rec_total = _recovered(sel, roles)
    stage_timings = dict(sel._stage_timings)
    report = dict(sel.shap_proxy_report_)

    ranked = report.get("revalidation", {}).get("ranked", []) if isinstance(report.get("revalidation"), dict) else []
    chosen_loss = None
    if ranked:
        chosen_loss = ranked[0].get("honest_loss", ranked[0].get("honest_loss_capped"))

    brier_vs_random = report.get("brier_vs_random")
    chosen_subset = tuple(sel.selected_features_)
    return dict(
        label=label, cap=brute_force_max_features, total=total,
        stage_timings=stage_timings,
        recall=(rec_hit, rec_total), n_selected=len(sel.selected_features_),
        chosen_subset=chosen_subset, chosen_honest_loss=chosen_loss,
        brier_vs_random=brier_vs_random,
        search_wall=stage_timings.get("search"),
    )


def _print_summary(name: str, results: list[dict]):
    print(f"\n=== [{name}] sweep summary ===", flush=True)
    header = f"  {'cap':>4} {'e2e_s':>8} {'search_s':>9} {'recall':>9} " f"{'n_sel':>5} {'honest_loss':>12} {'brier_vs_rand':>14}"
    print(header, flush=True)
    base = results[0]
    for r in results:
        rec_str = f"{r['recall'][0]}/{r['recall'][1]}"
        hl = "-" if r["chosen_honest_loss"] is None else f"{r['chosen_honest_loss']:.6f}"
        bv = "-" if r["brier_vs_random"] is None else f"{r['brier_vs_random']:.4f}"
        sw = "-" if r["search_wall"] is None else f"{r['search_wall']:.2f}"
        print(f"  {r['cap']:>4} {r['total']:>8.2f} {sw:>9} {rec_str:>9} " f"{r['n_selected']:>5} {hl:>12} {bv:>14}", flush=True)
    print(f"  (deltas vs cap{base['cap']} baseline)", flush=True)
    for r in results[1:]:
        d_e2e = (r["total"] - base["total"]) / base["total"] * 100
        if base["search_wall"] and r["search_wall"]:
            d_search = (r["search_wall"] - base["search_wall"]) / base["search_wall"] * 100
            d_search_str = f"{d_search:+.1f}%"
        else:
            d_search_str = "-"
        d_recall = r["recall"][0] - base["recall"][0]
        print(f"  cap{r['cap']}: e2e {d_e2e:+.1f}% search {d_search_str} " f"recall {d_recall:+d}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="C3,C3_hard", help="Comma-separated config names (subset of C3, C3_hard).")
    ap.add_argument("--widths", default=",".join(str(w) for w in WIDTHS), help="Comma-separated brute_force_max_features values to sweep.")
    ap.add_argument("--out_file", default=None)
    args = ap.parse_args(argv)

    if args.out_file:
        import atexit
        import builtins
        _fp = open(args.out_file, "w", buffering=1, encoding="utf-8")
        atexit.register(_fp.close)
        _orig = builtins.print

        def _tee_print(*a, **kw):
            kw["flush"] = True
            _orig(*a, **kw)
            try:
                kw2 = dict(kw); kw2["file"] = _fp; kw2["flush"] = True
                _orig(*a, **kw2)
            except (OSError, ValueError):
                pass
        builtins.print = _tee_print

    requested = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in requested:
        if c not in CONFIGS:
            raise SystemExit(f"unknown config {c}; known: {sorted(CONFIGS)}")
    widths = tuple(int(w.strip()) for w in args.widths.split(",") if w.strip())

    print(f"iter58 beam-width sweep: configs={requested} widths={widths}", flush=True)
    overall = {}
    # Single-process JIT warmup on a tiny synthetic so the cap-sweep wall is steady-state, not
    # dominated by first-call compile. Without this the first measured config absorbs ~3 min of
    # numba compile that the later configs see cached, badly skewing the cross-cap delta.
    print("\n[warmup] tiny fit to amortise JIT compile across the sweep...", flush=True)
    t_warm = time.perf_counter()
    _warm_X, _warm_y, _warm_roles = _make_dataset(dict(width=200, n_rows=400, n_informative=10, n_redundant=5, redundancy_rho=0.5, snr=4.0, seed=0))
    _warm_sel = _build_selector(seed=0, brute_force_max_features=22)
    _warm_sel.fit(_warm_X, _warm_y)
    print(f"[warmup] done in {time.perf_counter()-t_warm:.1f}s", flush=True)

    for name in requested:
        cfg = CONFIGS[name]
        print(f"\n[{name}] cfg={cfg}", flush=True)
        t_data = time.perf_counter()
        X, y, roles = _make_dataset(cfg)
        print(f"[{name}] dataset shape={X.shape} in {time.perf_counter()-t_data:.1f}s", flush=True)
        results = []
        for cap in widths:
            t_one = time.perf_counter()
            r = run_one(name, cfg, brute_force_max_features=cap, X=X, y=y, roles=roles)
            print(f"  [{name}/cap{cap}] done in {time.perf_counter()-t_one:.1f}s " f"(e2e={r['total']:.2f}s, recall={r['recall']})", flush=True)
            results.append(r)
        overall[name] = results
        _print_summary(name, results)

    return overall


if __name__ == "__main__":
    main()
