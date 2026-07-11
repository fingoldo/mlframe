"""Iter54 A/B: ``inner_n_jobs_cap=False`` (new default, xgboost decides) vs ``True`` (legacy
``n_cores // outer`` cap inside OOF-SHAP / reval / refine / trust-guard parallel pools).

iter53 verified the legacy cap costs 8-9% e2e at width 4000+10000 on 8-core boxes. iter54 ships the
flip + re-measures at C3 / C4. The two runs MUST produce the same chosen subset (and the same honest
loss given the same seed) -- the cap only affects xgboost's internal thread pool sizing.

Run::

    $env:PYTHONPATH = '<worktree>/src'
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_iter54_inner_n_jobs_cap --configs C3
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


CONFIGS = {
    "C3": dict(width=10000, n_rows=10000, n_informative=20, n_redundant=20, redundancy_rho=0.8, snr=8.0, seed=0),
    "C3_wide": dict(width=15000, n_rows=8000, n_informative=20, n_redundant=20, redundancy_rho=0.8, snr=8.0, seed=0),
    "C4": dict(width=20000, n_rows=10000, n_informative=20, n_redundant=20, redundancy_rho=0.8, snr=8.0, seed=0),
}

_STAGE_ORDER = (
    "prefilter", "clustering", "oof_shap", "prescreen", "search",
    "trust_guard", "revalidation", "importance_ablation", "within_cluster_refine",
)


def _make_dataset(cfg):
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    n_noise = max(0, cfg["width"] - cfg["n_informative"] - cfg["n_redundant"])
    X, y, roles = make_regime_dataset(
        n_samples=cfg["n_rows"], n_informative=cfg["n_informative"],
        n_redundant=cfg["n_redundant"], redundancy_rho=cfg["redundancy_rho"],
        n_noise=n_noise, snr=cfg["snr"], task="binary", seed=cfg["seed"])
    return X, y, roles


def _build_selector(seed: int, inner_n_jobs_cap: bool):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    return ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=500, cluster_features=True, cluster_corr_threshold=0.7,
        top_n=20, n_splits=4, n_revalidation_models=3, trust_guard=True, n_anchors=24,
        run_importance_ablation=True, within_cluster_refine=True,
        inner_n_jobs_cap=inner_n_jobs_cap,
        random_state=seed, verbose=False)


def _recovered(sel, roles):
    inf = {n for n, r in roles.items() if r == "informative"}
    return len(inf & set(sel.selected_features_)), len(inf)


def run_one(name: str, cfg: dict, inner_n_jobs_cap: bool, X, y, roles):
    label = "cap_on" if inner_n_jobs_cap else "cap_off"
    print(f"\n[{name} / {label}] fitting...", flush=True)
    sel = _build_selector(cfg["seed"], inner_n_jobs_cap=inner_n_jobs_cap)
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
    )


def print_stage_table(name: str, label: str, timings: dict, total: float):
    print(f"  [{name} / {label}] total={total:.2f}s  stages:")
    for k in _STAGE_ORDER:
        v = timings.get(k)
        if v is not None:
            print(f"    {k:24s} {v:8.3f}s ({100*v/total:5.1f}%)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="C3", help="Comma-separated config names (subset of C3, C3_wide, C4).")
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

    print(f"iter54 inner_n_jobs_cap A/B: configs={requested}", flush=True)
    overall = {}
    for name in requested:
        cfg = CONFIGS[name]
        print(f"\n[{name}] cfg={cfg}", flush=True)
        t_data = time.perf_counter()
        X, y, roles = _make_dataset(cfg)
        print(f"[{name}] dataset shape={X.shape} in {time.perf_counter()-t_data:.1f}s", flush=True)
        # Run cap_off (NEW default) first so any JIT warmup is amortised against the slower legacy
        # path. Then cap_on. iter53 used the same warm-then-warm ordering so reported deltas are
        # the steady-state perf gap, not first-fit JIT cost.
        r_off = run_one(name, cfg, inner_n_jobs_cap=False, X=X, y=y, roles=roles)
        r_on = run_one(name, cfg, inner_n_jobs_cap=True, X=X, y=y, roles=roles)
        overall[name] = dict(cap_off=r_off, cap_on=r_on)

        print_stage_table(name, "cap_off", r_off["stage_timings"], r_off["total"])
        print_stage_table(name, "cap_on", r_on["stage_timings"], r_on["total"])
        d_e2e = (r_on["total"] - r_off["total"]) / r_off["total"]
        print(f"\n  [{name}] e2e: cap_off={r_off['total']:.2f}s  cap_on={r_on['total']:.2f}s  "
              f"delta={d_e2e*100:+.1f}% (positive = cap is slower)", flush=True)
        print(f"  [{name}] recall: cap_off={r_off['recall']}  cap_on={r_on['recall']}", flush=True)
        print(f"  [{name}] n_selected: cap_off={r_off['n_selected']}  cap_on={r_on['n_selected']}", flush=True)
        subset_equal = r_off["chosen_subset"] == r_on["chosen_subset"]
        print(f"  [{name}] chosen_subset_equal={subset_equal}", flush=True)
        if r_off["chosen_honest_loss"] is not None and r_on["chosen_honest_loss"] is not None:
            print(f"  [{name}] chosen_honest_loss: cap_off={r_off['chosen_honest_loss']:.6f}  " f"cap_on={r_on['chosen_honest_loss']:.6f}", flush=True)
        # Per-stage deltas (positive = cap is slower)
        print(f"  [{name}] per-stage walls (delta = cap_on/cap_off - 1):")
        for k in _STAGE_ORDER:
            a = r_off["stage_timings"].get(k)
            b = r_on["stage_timings"].get(k)
            if a and b:
                print(f"    {k:24s} cap_off={a:7.3f}s  cap_on={b:7.3f}s  " f"delta={(b/a - 1)*100:+.1f}%")

    return overall


if __name__ == "__main__":
    main()
