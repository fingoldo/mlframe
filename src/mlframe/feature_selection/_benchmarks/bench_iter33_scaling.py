"""Iter33 scaling sweep: ShapProxiedFS at the user's actual target regime (tens of thousands).

iter1-31 squeezed the live 1000-feature regime to ~8-10s e2e warm; iter33 pivots to the user's stated
target (tens of thousands of features) and asks: which stage dominates wall-clock at width >= 5000?
The bench runs ``ShapProxiedFS().fit(X, y)`` once per configuration with per-stage timings + cProfile
attribution on the dominant stage; it does NOT modify pipeline behaviour. C1-C3 are the canonical
configs; C4 (width=20000) is run only if budget permits (config-level <=120s cap).

Findings (iter33, baseline = pre-lever 2dbfe1cb):

  - C2 (width=10000, n_rows=5000): prefilter is 57.6% of fit (15.3s of 26.65s). cProfile pins 14.8s
    of that to ``xgboost.core.update`` on a 2000-column matrix (two_stage stage B). The downstream
    SHAP-aware cap shrinks the eventual prefilter output to ~88 columns, so stage B's booster fits on
    1900+ columns it will then discard. The iter33 lever ``shap_aware_stage1_keep`` tightens stage A
    survivors to ``max(floor, effective_prefilter_top * cushion)`` (default ``max(200, 88*8) = 704``).
    Measured: prefilter 15.3s -> 5.0s (3.07x); e2e 26.65s -> 19.43s (1.37x / 27% faster); recall
    16/20 -> 17/20 (+1).

  - C3 (width=10000, n_rows=10000): prefilter 17.9s -> 7.2s (2.49x); e2e 39.56s -> 28.17s (1.40x /
    29% faster); recall 17/20 -> 17/20 (parity).

  - C1 (width=5000, n_rows=5000): warm e2e ~17s vs cold baseline 37.95s (cold included JIT warmup
    in ``_uf_labels`` + numba TreeSHAP); lever still applies because width>=1000 routes to two_stage.

Run::

    $env:PYTHONPATH = '<worktree>/src'
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_iter33_scaling

Optional args: ``--configs C1,C2,C3`` (default C1,C2,C3), ``--cprofile_configs C2,C3``
(default empty -- profile happens only when requested because cProfile adds 5-15% to walls),
``--no_preflight`` (skip the diagnostic; expensive at width=10000+ because it runs a 3-fold
CV of two boosters on all columns).
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


# Canonical configs from the iter33 task spec. Edit ONLY here; everything else reads from this dict.
CONFIGS = {
    "C1": dict(width=5000, n_rows=5000, n_informative=12, n_redundant=0, redundancy_rho=0.8, snr=8.0, seed=0),
    "C2": dict(width=10000, n_rows=5000, n_informative=20, n_redundant=0, redundancy_rho=0.8, snr=8.0, seed=0),
    "C3": dict(width=10000, n_rows=10000, n_informative=20, n_redundant=20, redundancy_rho=0.8, snr=8.0, seed=0),
    "C4": dict(width=20000, n_rows=10000, n_informative=20, n_redundant=20, redundancy_rho=0.8, snr=8.0, seed=0),
}

_STAGE_ORDER = (
    "prefilter", "clustering", "oof_shap", "prescreen", "search",
    "trust_guard", "revalidation", "importance_ablation", "within_cluster_refine",
)


def _make_dataset(cfg: dict):
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    n_noise = max(0, cfg["width"] - cfg["n_informative"] - cfg["n_redundant"])
    X, y, roles = make_regime_dataset(
        n_samples=cfg["n_rows"], n_informative=cfg["n_informative"],
        n_redundant=cfg["n_redundant"], redundancy_rho=cfg["redundancy_rho"],
        n_noise=n_noise, snr=cfg["snr"], task="binary", seed=cfg["seed"])
    return X, y, roles


def _build_selector(seed: int):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    # Wide-data config: same shape as bench_iter31 / bench_shap_proxy_scaling so iter33 numbers
    # compose cleanly with the prior live-regime measurements.
    return ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=500, cluster_features=True, cluster_corr_threshold=0.7,
        top_n=20, n_splits=4, n_revalidation_models=3, trust_guard=True, n_anchors=24,
        run_importance_ablation=True, within_cluster_refine=True,
        random_state=seed, verbose=False)


def _recovered(sel, roles):
    inf = {n for n, r in roles.items() if r == "informative"}
    return len(inf & set(sel.selected_features_)), len(inf)


def _random_baseline_brier(y) -> float:
    """Predict the prior on every row (constant probability == positive rate). Reference floor for
    the proxy: anything we ship MUST beat this. y is binary 0/1."""
    p = float(np.asarray(y).mean())
    return float(np.mean((np.asarray(y, dtype=np.float64) - p) ** 2))


def run_one(name: str, cfg: dict, *, do_cprofile: bool = False, run_preflight: bool = True):
    print(f"\n[{name}] cfg={cfg}", flush=True)
    print(f"[{name}] making dataset...", flush=True)
    t_data = time.perf_counter()
    X, y, roles = _make_dataset(cfg)
    print(f"[{name}] dataset shape={X.shape} in {time.perf_counter()-t_data:.1f}s", flush=True)
    sel = _build_selector(cfg["seed"])
    sel._stage_timings = {}

    print(f"[{name}] fitting (cprofile={do_cprofile})...", flush=True)
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
    print(f"[{name}] fit done in {total:.2f}s", flush=True)

    rec_hit, rec_total = _recovered(sel, roles)
    stage_timings = dict(sel._stage_timings)
    report = dict(sel.shap_proxy_report_)

    # Brier vs random baseline on the holdout. The selector already reports honest holdout numbers
    # inside report['revalidation']['ranked'][0] (top-1 by stable_score / parsimony). Pull the chosen
    # subset's honest_loss out and compare against the prior-only baseline computed inline.
    rb = _random_baseline_brier(y)
    chosen_loss = None
    ranked = report.get("revalidation", {}).get("ranked", []) if isinstance(report.get("revalidation"), dict) else []
    if ranked:
        chosen_loss = ranked[0].get("honest_loss", ranked[0].get("honest_loss_capped"))

    trust = report.get("trust", {}) if isinstance(report.get("trust"), dict) else {}
    trustworthy = trust.get("trustworthy")

    out = dict(
        name=name, total=total, stage_timings=stage_timings,
        recall=(rec_hit, rec_total),
        n_selected=len(sel.selected_features_),
        random_baseline_brier=rb,
        chosen_honest_loss=chosen_loss,
        trustworthy=trustworthy,
        preflight=None,  # filled below
        preflight_recommendation=None,
        profile=pr,
    )
    # Preflight: reuse the same static helper that lives on the class. NOT cheap at width=10000
    # (a 3-fold CV of two boosters on all columns); behind a flag so the scaling sweep can skip it
    # and report only the diagnostic FROM the fit (preflight is a what-the-user-sees-before-fit
    # check, not a stage that runs inside fit).
    if run_preflight:
        print(f"[{name}] running preflight diagnostic...", flush=True)
        t_pf = time.perf_counter()
        try:
            from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
            pf = ShapProxiedFS.preflight(X, y, classification=True)
            out["preflight"] = pf
            out["preflight_recommendation"] = pf.get("recommendation") if isinstance(pf, dict) else None
        except Exception as exc:
            out["preflight"] = dict(error=str(exc))
        print(f"[{name}] preflight done in {time.perf_counter()-t_pf:.1f}s", flush=True)

    return out


def print_stage_table(timings: dict, total: float):
    print(f"  total={total:.2f}s  stages:")
    for k in _STAGE_ORDER:
        v = timings.get(k)
        if v is not None:
            print(f"    {k:24s} {v:8.3f}s ({100*v/total:5.1f}%)")
    for k, v in timings.items():
        if k not in _STAGE_ORDER:
            print(f"    {k:24s} {v:8.3f}s ({100*v/total:5.1f}%) [extra]")


def print_cprofile(pr: cProfile.Profile, n: int = 10, sort: str = "tottime"):
    sio = io.StringIO()
    ps = pstats.Stats(pr, stream=sio).sort_stats(sort)
    ps.print_stats(n)
    print(sio.getvalue())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="C1,C2,C3",
                    help="Comma-separated config names (subset of C1,C2,C3,C4).")
    ap.add_argument("--cprofile_configs", default="",
                    help="Comma-separated config names to cProfile (default: empty = no cprofile).")
    ap.add_argument("--per_config_cap_s", type=float, default=120.0,
                    help="Soft cap per-config wall-clock; bench logs over-cap configs and continues.")
    ap.add_argument("--no_preflight", action="store_true",
                    help="Skip the preflight diagnostic (it's a what-the-user-sees-before-fit check, "
                         "not a stage inside fit; expensive at width=10000+).")
    ap.add_argument("--out_file", default=None,
                    help="Tee output to this file (with explicit per-line flush) so a background "
                         "task on Windows can be polled without waiting for the parent shell pipe "
                         "to flush.")
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
    profile_set = {c.strip() for c in args.cprofile_configs.split(",") if c.strip()}
    for c in requested:
        if c not in CONFIGS:
            raise SystemExit(f"unknown config {c}; known: {sorted(CONFIGS)}")

    print(f"iter33 scaling sweep: configs={requested}  cprofile={sorted(profile_set)}", flush=True)
    print(f"per-config cap = {args.per_config_cap_s:.0f}s (soft)", flush=True)
    results = {}
    t_bench0 = time.perf_counter()
    for name in requested:
        cfg = CONFIGS[name]
        r = run_one(name, cfg, do_cprofile=(name in profile_set), run_preflight=not args.no_preflight)
        results[name] = r
        print_stage_table(r["stage_timings"], r["total"])
        print(f"  recall: {r['recall'][0]}/{r['recall'][1]}  " f"n_selected={r['n_selected']}  trustworthy={r['trustworthy']}", flush=True)
        if r["chosen_honest_loss"] is not None:
            print(f"  chosen_honest_loss={r['chosen_honest_loss']:.4f}  "
                  f"random_baseline_brier={r['random_baseline_brier']:.4f}  "
                  f"lift_vs_random={r['random_baseline_brier']-r['chosen_honest_loss']:+.4f}",
                  flush=True)
        if r["preflight_recommendation"] is not None:
            print(f"  preflight_recommendation={r['preflight_recommendation']}")
        if r["profile"] is not None:
            print(f"  cProfile top 10 by tottime:")
            print_cprofile(r["profile"], n=10, sort="tottime")
            print(f"  cProfile top 10 by cumulative:")
            print_cprofile(r["profile"], n=10, sort="cumulative")
        if r["total"] > args.per_config_cap_s:
            print(f"  [WARN] {name} exceeded soft cap {args.per_config_cap_s:.0f}s " f"({r['total']:.1f}s)", flush=True)

    # Summary table
    print("\n" + "=" * 70)
    print("SCALING SUMMARY")
    print("=" * 70)
    header = f"{'config':<6} {'width':>6} {'rows':>6} {'inf':>4} {'red':>4} " f"{'total':>8} {'recall':>8} {'subset':>7} {'trust':>6}"
    print(header)
    for name in requested:
        c = CONFIGS[name]; r = results[name]
        rec = f"{r['recall'][0]}/{r['recall'][1]}"
        print(f"{name:<6} {c['width']:>6} {c['n_rows']:>6} {c['n_informative']:>4} "
              f"{c['n_redundant']:>4} {r['total']:>7.2f}s {rec:>8} {r['n_selected']:>7} "
              f"{str(r['trustworthy']):>6}")
    print(f"\nbench-wall total: {time.perf_counter()-t_bench0:.1f}s")
    return results


if __name__ == "__main__":
    main()
