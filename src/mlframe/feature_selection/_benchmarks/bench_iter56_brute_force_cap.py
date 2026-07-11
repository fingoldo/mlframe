"""Iter56 A/B: raised brute_force_max_features 22 -> 28 + n_sub gate 2M -> 80M.

Quality + speed bench at C3 and a low-SNR hard regime. At default ``max_features=None`` (the bench
config) the n_sub gate routes n<=26 to brute force and n in {27, 28} to beam (2^27 = 134M > 80M
gate, 2^28 = 268M > 80M gate). So the cap28 arm here measures BEAM over a 28-column prescreen
pool, not brute force at n=28; the wall-clock and recall gains come from beam having a wider
candidate pool, not from exhaustive search at n=28. Brute force at n=28 only fires when the
caller ALSO pins ``max_features<=12`` (see ``_resolve_optimizer`` docstring + module-level comment
on ``_DEFAULT_BRUTE_FORCE_MAX_FEATURES`` in shap_proxied_fs.py).

The bench is still informative: it measures the user-visible effect of raising
``brute_force_max_features`` from 22 to 28 on the default ``max_features=None`` workload
(prescreen-pool widening, beam over wider pool at n=27,28). Pin ``max_features=12`` and re-run if
you want to measure the actual brute-force-at-n=28 path.

Run::

    $env:PYTHONPATH = '<worktree>/src'
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_iter56_brute_force_cap --configs C3,C3_hard
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


CONFIGS = {
    # Default regime: brute force is the dispatched optimizer once prefilter narrows to <=28 cols.
    "C3": dict(width=10000, n_rows=10000, n_informative=20, n_redundant=20, redundancy_rho=0.8, snr=8.0, seed=0),
    # Hard regime: noisy SHAP ranking, no redundants. Weak informatives may rank in 23..28.
    "C3_hard": dict(width=10000, n_rows=5000, n_informative=20, n_redundant=0, redundancy_rho=0.8, snr=2.0, seed=0),
}


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

    chosen_subset = tuple(sel.selected_features_)
    return dict(
        label=label, total=total, stage_timings=stage_timings,
        recall=(rec_hit, rec_total), n_selected=len(sel.selected_features_),
        chosen_subset=chosen_subset, chosen_honest_loss=chosen_loss,
        search_wall=stage_timings.get("search"),
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="C3,C3_hard", help="Comma-separated config names (subset of C3, C3_hard).")
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

    print(f"iter56 brute_force_max_features A/B: configs={requested}", flush=True)
    overall = {}
    for name in requested:
        cfg = CONFIGS[name]
        print(f"\n[{name}] cfg={cfg}", flush=True)
        t_data = time.perf_counter()
        X, y, roles = _make_dataset(cfg)
        print(f"[{name}] dataset shape={X.shape} in {time.perf_counter()-t_data:.1f}s", flush=True)
        # Run cap22 (legacy) first so JIT warmup is amortised against the slower path. The lever
        # only changes the search stage; all other stages share warmed kernels across runs.
        r22 = run_one(name, cfg, brute_force_max_features=22, X=X, y=y, roles=roles)
        r28 = run_one(name, cfg, brute_force_max_features=28, X=X, y=y, roles=roles)
        overall[name] = dict(cap22=r22, cap28=r28)

        d_e2e = (r28["total"] - r22["total"]) / r22["total"]
        d_search = None
        if r22["search_wall"] and r28["search_wall"]:
            d_search = (r28["search_wall"] - r22["search_wall"]) / r22["search_wall"]
        print(f"\n  [{name}] e2e: cap22={r22['total']:.2f}s  cap28={r28['total']:.2f}s  " f"delta={d_e2e*100:+.1f}% (positive = lever is slower)", flush=True)
        if d_search is not None:
            print(f"  [{name}] search wall: cap22={r22['search_wall']:.2f}s  " f"cap28={r28['search_wall']:.2f}s  delta={d_search*100:+.1f}%", flush=True)
        print(f"  [{name}] recall: cap22={r22['recall']}  cap28={r28['recall']}", flush=True)
        print(f"  [{name}] n_selected: cap22={r22['n_selected']}  cap28={r28['n_selected']}", flush=True)
        subset_equal = r22["chosen_subset"] == r28["chosen_subset"]
        print(f"  [{name}] chosen_subset_equal={subset_equal}", flush=True)
        if r22["chosen_honest_loss"] is not None and r28["chosen_honest_loss"] is not None:
            print(f"  [{name}] chosen_honest_loss: cap22={r22['chosen_honest_loss']:.6f}  " f"cap28={r28['chosen_honest_loss']:.6f}", flush=True)

    return overall


if __name__ == "__main__":
    main()
