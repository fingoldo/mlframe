"""Iter59 adaptive-prescreen A/B: ``adaptive_prescreen_by_stability`` False vs True at C3 / C3_hard.

The lever measures per-fold SHAP rank stability and narrows the prescreen pool when stability is
low (where the rank tail past the strongly-informative core is noise). Iter58 found cap=28 beats
cap>=32 at C3_hard (snr=2); the hypothesis is that mid-rank features 29..32 are noise at low SNR,
hurting beam's choice. This bench measures whether narrowing the cap below the default at the
low-SNR regime (where stability drops) recovers / improves recall.

Two scales::

    # full scale (matches iter58 sweep; ~50-100s per run, 4 runs total):
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_iter59_adaptive_prescreen --scale full --configs C3,C3_hard

    # compact (n_rows=4000, width=4000; ~15-30s per run -- preferred when the box has heavy concurrent load):
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_iter59_adaptive_prescreen --scale compact --configs C3,C3_hard
"""

from __future__ import annotations

import argparse
import time
import warnings

warnings.filterwarnings("ignore")


SCALES = {
    "full": {"width": 10000, "n_rows": 10000},
    "compact": {"width": 4000, "n_rows": 4000},
}


def _config_for(scale: str, name: str) -> dict:
    s = SCALES[scale]
    if name == "C3":
        return dict(width=s["width"], n_rows=s["n_rows"], n_informative=20, n_redundant=20, redundancy_rho=0.8, snr=8.0, seed=0)
    if name == "C3_hard":
        return dict(width=s["width"], n_rows=s["n_rows"], n_informative=20, n_redundant=0, redundancy_rho=0.8, snr=2.0, seed=0)
    raise SystemExit(f"unknown config {name!r}; known: C3, C3_hard")


def _make_dataset(cfg):
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    n_noise = max(0, cfg["width"] - cfg["n_informative"] - cfg["n_redundant"])
    X, y, roles = make_regime_dataset(
        n_samples=cfg["n_rows"], n_informative=cfg["n_informative"],
        n_redundant=cfg["n_redundant"], redundancy_rho=cfg["redundancy_rho"],
        n_noise=n_noise, snr=cfg["snr"], task="binary", seed=cfg["seed"])
    return X, y, roles


def _build_selector(seed: int, adaptive: bool, n_jobs: int):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    return ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=500, cluster_features=True, cluster_corr_threshold=0.7,
        top_n=20, n_splits=4, n_revalidation_models=3, trust_guard=True, n_anchors=24,
        run_importance_ablation=True, within_cluster_refine=True,
        brute_force_max_features=28,
        adaptive_prescreen_by_stability=adaptive,
        n_jobs=n_jobs,
        random_state=seed, verbose=False)


def _recovered(sel, roles):
    inf = {n for n, r in roles.items() if r == "informative"}
    return len(inf & set(sel.selected_features_)), len(inf)


def run_one(name: str, cfg: dict, adaptive: bool, X, y, roles, n_jobs: int):
    label = "adaptive" if adaptive else "static"
    print(f"\n[{name} / {label}] fitting...", flush=True)
    sel = _build_selector(cfg["seed"], adaptive=adaptive, n_jobs=n_jobs)
    sel._stage_timings = {}
    t0 = time.perf_counter()
    sel.fit(X, y)
    total = time.perf_counter() - t0
    rec_hit, rec_total = _recovered(sel, roles)
    stage_timings = dict(sel._stage_timings)
    report = dict(sel.shap_proxy_report_)
    adapt_info = report.get("adaptive_prescreen") or {}
    ranked = report.get("revalidation", {}).get("ranked", []) if isinstance(report.get("revalidation"), dict) else []
    chosen_loss = ranked[0].get("honest_loss", ranked[0].get("honest_loss_capped")) if ranked else None
    return dict(
        label=label, adaptive=adaptive, total=total,
        stage_timings=stage_timings,
        recall=(rec_hit, rec_total), n_selected=len(sel.selected_features_),
        chosen_subset=tuple(sel.selected_features_),
        chosen_honest_loss=chosen_loss,
        search_wall=stage_timings.get("search"),
        stability=adapt_info.get("stability"),
        effective_cap=adapt_info.get("effective_cap"),
        default_cap=adapt_info.get("default_cap"),
    )


def _print_summary(name: str, results: list[dict]):
    print(f"\n=== [{name}] adaptive A/B summary ===", flush=True)
    header = f"  {'label':>9} {'e2e_s':>8} {'search_s':>9} {'recall':>9} " f"{'n_sel':>5} {'honest_loss':>12} {'stab':>6} {'cap':>4}"
    print(header, flush=True)
    for r in results:
        rec_str = f"{r['recall'][0]}/{r['recall'][1]}"
        hl = "-" if r["chosen_honest_loss"] is None else f"{r['chosen_honest_loss']:.6f}"
        sw = "-" if r["search_wall"] is None else f"{r['search_wall']:.2f}"
        stab = "-" if r["stability"] is None else f"{r['stability']:.3f}"
        cap = "-" if r["effective_cap"] is None else str(r["effective_cap"])
        print(f"  {r['label']:>9} {r['total']:>8.2f} {sw:>9} {rec_str:>9} " f"{r['n_selected']:>5} {hl:>12} {stab:>6} {cap:>4}", flush=True)
    base = next((r for r in results if not r["adaptive"]), results[0])
    for r in results:
        if r is base:
            continue
        d_e2e = (r["total"] - base["total"]) / base["total"] * 100
        d_recall = r["recall"][0] - base["recall"][0]
        print(f"  delta vs static: e2e {d_e2e:+.1f}%, recall {d_recall:+d}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="C3,C3_hard")
    ap.add_argument("--scale", default="full", choices=list(SCALES))
    ap.add_argument("--n_jobs", type=int, default=-1, help="Selector n_jobs; set to 1 for serial when CPU is busy with other work.")
    ap.add_argument("--out_file", default=None)
    args = ap.parse_args(argv)

    if args.out_file:
        import builtins
        _fp = open(args.out_file, "w", buffering=1, encoding="utf-8")
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
    print(f"iter59 adaptive-prescreen A/B: scale={args.scale} configs={requested} n_jobs={args.n_jobs}", flush=True)
    print("\n[warmup] tiny fit to amortise JIT compile across the A/B...", flush=True)
    t_warm = time.perf_counter()
    _warm_X, _warm_y, _ = _make_dataset(dict(width=200, n_rows=400, n_informative=10, n_redundant=5, redundancy_rho=0.5, snr=4.0, seed=0))
    _warm_sel = _build_selector(seed=0, adaptive=True, n_jobs=args.n_jobs)
    _warm_sel.fit(_warm_X, _warm_y)
    print(f"[warmup] done in {time.perf_counter()-t_warm:.1f}s", flush=True)

    overall = {}
    for name in requested:
        cfg = _config_for(args.scale, name)
        print(f"\n[{name}] cfg={cfg}", flush=True)
        t_data = time.perf_counter()
        X, y, roles = _make_dataset(cfg)
        print(f"[{name}] dataset shape={X.shape} in {time.perf_counter()-t_data:.1f}s", flush=True)
        results = []
        for adaptive in (False, True):
            t_one = time.perf_counter()
            r = run_one(name, cfg, adaptive=adaptive, X=X, y=y, roles=roles, n_jobs=args.n_jobs)
            print(f"  [{name}/{r['label']}] done in {time.perf_counter()-t_one:.1f}s "
                  f"(e2e={r['total']:.2f}s, recall={r['recall']}, stab={r['stability']}, "
                  f"cap={r['effective_cap']})", flush=True)
            results.append(r)
        overall[name] = results
        _print_summary(name, results)

    return overall


if __name__ == "__main__":
    main()
