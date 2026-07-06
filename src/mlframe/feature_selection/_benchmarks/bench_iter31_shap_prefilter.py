"""Iter31 benchmark: SHAP-pre-prefilter (cheap importance pass) BEFORE OOF-SHAP.

Compares baseline vs new lever at the live wide regime (n_features=1000, n_rows=5000, n_inf=12,
snr=8.0) across seeds 0 and 1. Reports stage timings, cProfile attribution, e2e wall, and recall.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def _make_dataset(*, n_features=1000, n_rows=5000, n_inf=12, snr=8.0, seed=0):
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    n_noise = max(0, n_features - n_inf)
    X, y, roles = make_regime_dataset(
        n_samples=n_rows, n_informative=n_inf, n_redundant=0, redundancy_rho=0.9, n_noise=n_noise, snr=snr, task="binary", seed=seed
    )
    return X, y, roles


def _build_selector(seed: int, *, shap_prefilter_enabled: bool):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    kwargs = dict(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=500, cluster_features=True, cluster_corr_threshold=0.7,
        top_n=20, n_splits=4, n_revalidation_models=3, trust_guard=True, n_anchors=24,
        run_importance_ablation=True, within_cluster_refine=True,
        random_state=seed, verbose=False)
    if shap_prefilter_enabled:
        kwargs["shap_prefilter_enabled"] = True
    else:
        # If the constructor accepts the kwarg, explicitly disable; else ignore.
        try:
            from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS as _S
            import inspect
            sig = inspect.signature(_S.__init__)
            if "shap_prefilter_enabled" in sig.parameters:
                kwargs["shap_prefilter_enabled"] = False
        except Exception:  # nosec B110 - optional dependency import guard
            pass
    return ShapProxiedFS(**kwargs)


def _recovered(sel, roles):
    inf = {n for n, r in roles.items() if r == "informative"}
    return len(inf & set(sel.selected_features_)), len(inf)


def run_one(
    seed: int, *, shap_prefilter_enabled: bool, n_features: int = 1000, n_rows: int = 5000, n_inf: int = 12, snr: float = 8.0, do_cprofile: bool = False
):
    print(f"  [seed={seed} shap_prefilter_enabled={shap_prefilter_enabled}] building dataset...", flush=True)
    X, y, roles = _make_dataset(n_features=n_features, n_rows=n_rows, n_inf=n_inf, snr=snr, seed=seed)
    sel = _build_selector(seed, shap_prefilter_enabled=shap_prefilter_enabled)
    sel._stage_timings = {}
    print(f"  [seed={seed}] fitting...", flush=True)
    t0 = time.perf_counter()
    if do_cprofile:
        pr = cProfile.Profile()
        pr.enable()
        sel.fit(X, y)
        pr.disable()
    else:
        pr = None
        sel.fit(X, y)
    total = time.perf_counter() - t0
    rec_hit, rec_total = _recovered(sel, roles)
    stage_timings = dict(sel._stage_timings)
    return dict(seed=seed, total=total, stage_timings=stage_timings, recall=(rec_hit, rec_total), profile=pr, selected=list(sel.selected_features_))


def print_stage_table(timings: dict, total: float):
    order = ("prefilter", "shap_prefilter", "clustering", "oof_shap", "prescreen",
             "search", "trust_guard", "revalidation", "importance_ablation",
             "within_cluster_refine")
    print(f"  total={total:.2f}s  stages:")
    for k in order:
        v = timings.get(k)
        if v is not None:
            print(f"    {k:24s} {v:8.3f}s ({100*v/total:5.1f}%)")
    # Any unrecognized
    for k, v in timings.items():
        if k not in order:
            print(f"    {k:24s} {v:8.3f}s ({100*v/total:5.1f}%) [extra]")


def print_top_cprofile(pr: cProfile.Profile, n: int = 10):
    sio = io.StringIO()
    ps = pstats.Stats(pr, stream=sio).sort_stats("tottime")
    ps.print_stats(n)
    print(sio.getvalue())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--mode", default="both", choices=["both", "baseline", "new"])
    ap.add_argument("--cprofile_seed", default=0, type=int)
    ap.add_argument("--n_features", default=1000, type=int)
    ap.add_argument("--n_rows", default=5000, type=int)
    ap.add_argument("--n_inf", default=12, type=int)
    ap.add_argument("--snr", default=8.0, type=float)
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s]
    modes = ["baseline", "new"] if args.mode == "both" else [args.mode]

    common = dict(n_features=args.n_features, n_rows=args.n_rows, n_inf=args.n_inf, snr=args.snr)
    print(f"Live regime: width={args.n_features} rows={args.n_rows} n_inf={args.n_inf} " f"snr={args.snr}")
    print(f"Modes: {modes}  Seeds: {seeds}", flush=True)

    results = {}
    for mode in modes:
        results[mode] = []
        enabled = mode == "new"
        for seed in seeds:
            do_p = seed == args.cprofile_seed
            print(f"\n[{mode}] seed={seed} cprofile={do_p}", flush=True)
            r = run_one(seed, shap_prefilter_enabled=enabled, do_cprofile=do_p, **common)
            results[mode].append(r)
            print_stage_table(r["stage_timings"], r["total"])
            print(f"  recall: {r['recall'][0]}/{r['recall'][1]}", flush=True)
            if do_p:
                print(f"  cProfile top 10 by tottime:")
                print_top_cprofile(r["profile"], 10)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (e2e wall + recall per seed):")
    print("=" * 60)
    for mode in modes:
        print(f"\n[{mode}]")
        for r in results[mode]:
            print(f"  seed={r['seed']} total={r['total']:.2f}s recall={r['recall'][0]}/{r['recall'][1]}")
    if len(modes) == 2:
        print("\n[delta]")
        for s_idx, seed in enumerate(seeds):
            b = results["baseline"][s_idx]["total"]
            n = results["new"][s_idx]["total"]
            print(f"  seed={seed} baseline={b:.2f}s new={n:.2f}s " f"speedup={b/n:.2f}x ({100*(b-n)/b:+.1f}%)")

    return results


if __name__ == "__main__":
    main()
