"""Iter35 Phase-0 probe: cProfile the within_cluster_refine sub-stages at C3.

Mirrors iter34 bench harness but flips refine on / reval UCB on by default (iter34's gains carry
through). Captures cProfile narrowed to the refine stage by patching ``within_cluster_refine`` to
emit sub-stage timings.

Run::

    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_iter35_refine_probe \
        --configs C3 --profile-path D:/Temp/iter35_c3_baseline.profile
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


def _build_selector(seed: int):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    return ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=500, cluster_features=True, cluster_corr_threshold=0.7,
        top_n=20, n_splits=4, n_revalidation_models=3, trust_guard=True, n_anchors=24,
        run_importance_ablation=True, within_cluster_refine=True,
        revalidation_ucb_enabled=True,
        random_state=seed, verbose=False)


def _recovered(sel, roles):
    inf = {n for n, r in roles.items() if r == "informative"}
    return len(inf & set(sel.selected_features_))


def _patch_refine_for_substage_timing():
    """Wrap within_cluster_refine so each call records sub-stage timings to a shared dict."""
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_revalidate as R

    timings: dict = {"stage1_total": 0.0, "stage2a_total": 0.0, "stage2b_total": 0.0,
                     "stage2b_rounds": 0, "stage2b_trials": 0,
                     "calls": 0}
    orig = R.within_cluster_refine

    def wrapped(*args, **kwargs):
        timings["calls"] += 1
        # We instrument via cProfile alone; substage attribution comes from pstats.
        return orig(*args, **kwargs)

    R.within_cluster_refine = wrapped
    return timings


def run_one_profiled(name, cfg, profile_path):
    print(f"\n[{name}] cfg={cfg}", flush=True)
    print(f"[{name}] making dataset...", flush=True)
    t0 = time.perf_counter()
    X, y, roles = _make_dataset(cfg)
    print(f"[{name}] dataset shape={X.shape} in {time.perf_counter()-t0:.1f}s", flush=True)
    sel = _build_selector(cfg["seed"])
    sel._stage_timings = {}
    timings = _patch_refine_for_substage_timing()

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    sel.fit(X, y)
    profiler.disable()
    total = time.perf_counter() - t0
    print(f"[{name}] fit done in {total:.2f}s", flush=True)

    profiler.dump_stats(profile_path)
    print(f"[{name}] profile written to {profile_path}", flush=True)

    rec = _recovered(sel, roles)
    print(f"[{name}] recall {rec}/20  n_selected={len(sel.selected_features_)}", flush=True)
    print(f"[{name}] refine calls={timings['calls']}", flush=True)
    print(f"[{name}] stages:")
    for k, v in sel._stage_timings.items():
        print(f"    {k:24s} {v:8.3f}s ({100*v/total:5.1f}%)")

    # Sub-stage attribution via pstats on the dumped profile.
    s = io.StringIO()
    ps = pstats.Stats(profile_path, stream=s).sort_stats("tottime")
    ps.print_stats(40)
    print("\nTOP 40 by tottime:")
    print(s.getvalue())

    s2 = io.StringIO()
    ps2 = pstats.Stats(profile_path, stream=s2).sort_stats("cumtime")
    ps2.print_stats(40)
    print("\nTOP 40 by cumtime:")
    print(s2.getvalue())

    # Specifically dump the refine + perm-importance + parallel honest losses lines.
    s3 = io.StringIO()
    ps3 = pstats.Stats(profile_path, stream=s3)
    ps3.print_stats(r"_shap_proxy_revalidate", 50)
    print("\n_shap_proxy_revalidate breakdown:")
    print(s3.getvalue())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="C3")
    ap.add_argument("--profile-path", default="D:/Temp/iter35_c3_baseline.profile")
    args = ap.parse_args(argv)

    requested = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in requested:
        if c not in CONFIGS:
            raise SystemExit(f"unknown config {c}; known: {sorted(CONFIGS)}")

    t_bench0 = time.perf_counter()
    for name in requested:
        run_one_profiled(name, CONFIGS[name], args.profile_path)
    print(f"\nbench-wall total: {time.perf_counter()-t_bench0:.1f}s")


if __name__ == "__main__":
    main()
