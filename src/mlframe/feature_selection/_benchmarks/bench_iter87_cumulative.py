"""iter87 informational bench: cumulative iter66->iter86 speedup at C2/C3/C4.

Measures the full arc impact of the iter66-86 perf push (21 iters, ~17 perf landings) for ShapProxiedFS
against the pre-iter66 reference (commit 497f1076 = iter35's UCB stage-2b in within_cluster_refine,
the last published commit before iter66's MRMR.export_artifacts + precomputed pipeline landed).

This is INFORMATIONAL: the script and its JSON output capture the cumulative speedup but no selector
code changes. The baseline is run from a separate worktree (default ``D:/Temp/iter87_baseline_wt``);
the after-state is whichever worktree imports ``mlframe.feature_selection.shap_proxied_fs`` first on
``sys.path``. The two are compared with their respective DEFAULTS (baseline = Pearson clustering,
n_splits=5, no SHAP-aware tightening, no disk cache; after = SU clustering, n_splits=3, full arc).

Run:  python -m mlframe.feature_selection._benchmarks.bench_iter87_cumulative
"""
from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - module used safely in this file, see call sites below (no untrusted input reaches it)
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# C2 / C3 / C4 regime knobs. n_red=0 at C2 keeps the no-redundancy regime; C3/C4 add 20 redundant cols.
REGIMES = {
    "C2": dict(width=10000, n_rows=5000, n_inf=20, n_red=0, snr=8.0),
    "C3": dict(width=10000, n_rows=10000, n_inf=20, n_red=20, snr=8.0),
    "C4": dict(width=20000, n_rows=10000, n_inf=20, n_red=20, snr=8.0),
}

# Per-config wall-clock cap (the iter87 prompt's 240s hard cap).
PER_CONFIG_TIMEOUT_S = 240


WORKER = r"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

regime = json.loads(sys.argv[1])
cache_dir = sys.argv[2] or None
warm = (sys.argv[3] == "warm")

from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

X, y, roles = make_regime_dataset(
    n_samples=regime["n_rows"],
    n_informative=regime["n_inf"],
    n_redundant=regime["n_red"],
    n_noise=regime["width"] - regime["n_inf"] - regime["n_red"],
    snr=regime["snr"],
    task="binary",
    seed=0,
)

kwargs = dict(
    classification=True,
    metric="brier",
    n_models=1,
    max_features=regime["n_inf"],
    revalidate=True,
    trust_guard=True,
    random_state=0,
    verbose=False,
)
if cache_dir is not None:
    kwargs["cache_dir"] = cache_dir

sel = ShapProxiedFS(**kwargs)
sel._stage_timings = {}

t0 = time.perf_counter()
sel.fit(X, y)
t_total = time.perf_counter() - t0

informatives = {n for n, r in roles.items() if r == "informative"}
chosen = list(getattr(sel, "selected_features_", []) or [])
recall = sum(1 for c in chosen if c in informatives) / max(len(informatives), 1)

honest_loss = None
report = getattr(sel, "shap_proxy_report_", {}) or {}
ref_info = report.get("within_cluster_refine") or {}
if isinstance(ref_info, dict):
    honest_loss = ref_info.get("honest_loss_full")

print("__RESULT__" + json.dumps({
    "warm": warm,
    "total_s": t_total,
    "stage_timings": dict(sel._stage_timings),
    "n_chosen": len(chosen),
    "recall": recall,
    "chosen": chosen[:40],
    "honest_loss_refine": honest_loss,
}))
"""


def _run_subprocess(python_exe: str, pythonpath: str, regime: dict, cache_dir: str | None, warm: bool):
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["PYTHONWARNINGS"] = "ignore"

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(  # nosec B603 - fixed/trusted executable, args are not attacker-controlled
            [python_exe, "-c", WORKER, json.dumps(regime), cache_dir or "", "warm" if warm else "cold"],
            env=env,
            capture_output=True,
            text=True,
            timeout=PER_CONFIG_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {
            "error": f"timeout >{PER_CONFIG_TIMEOUT_S}s",
            "wall_s": time.perf_counter() - t0,
        }
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        return {
            "error": f"rc={proc.returncode}",
            "stderr_tail": (proc.stderr or "")[-800:],
            "stdout_tail": (proc.stdout or "")[-800:],
            "wall_s": wall,
        }
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            try:
                payload = json.loads(line[len("__RESULT__") :])
            except Exception:  # nosec B110 - best-effort path
                pass
    if payload is None:
        return {
            "error": "no_result_marker",
            "stderr_tail": (proc.stderr or "")[-800:],
            "stdout_tail": (proc.stdout or "")[-800:],
            "wall_s": wall,
        }
    payload["wall_s"] = wall
    return payload


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    python_exe = sys.executable
    after_src = str(Path(__file__).resolve().parents[3])
    baseline_root = os.environ.get("ITER87_BASELINE_WT", r"D:/Temp/iter87_baseline_wt")
    baseline_src = str(Path(baseline_root) / "src")

    results: dict = {
        "python": python_exe,
        "after_src": after_src,
        "baseline_src": baseline_src,
        "per_regime": {},
    }

    print(f"# baseline_src = {baseline_src}")
    print(f"# after_src    = {after_src}")
    print()

    with tempfile.TemporaryDirectory(prefix="iter87_cache_") as tmpdir:
        cache_dir = os.path.join(tmpdir, "shap_disk_cache")
        for name, regime in REGIMES.items():
            print(f"=== {name} regime={regime} ===")
            sub = {"regime": regime}

            # Baseline (pre-iter66): no disk cache, cold only.
            print(f"  baseline cold ...", flush=True)
            sub["baseline_cold"] = _run_subprocess(python_exe, baseline_src, regime, cache_dir=None, warm=False)
            print(f"    -> {sub['baseline_cold'].get('total_s', sub['baseline_cold'].get('error'))!s}")

            # After-state (iter86): cold, no cache.
            print(f"  after cold (no cache) ...", flush=True)
            sub["after_cold"] = _run_subprocess(python_exe, after_src, regime, cache_dir=None, warm=False)
            print(f"    -> {sub['after_cold'].get('total_s', sub['after_cold'].get('error'))!s}")

            # After-state: warm cache pair (miss then hit) -- isolates cache-hit warm path.
            print(f"  after warm-cache miss ...", flush=True)
            sub["after_warm_miss"] = _run_subprocess(python_exe, after_src, regime, cache_dir=cache_dir, warm=False)
            print(f"    -> {sub['after_warm_miss'].get('total_s', sub['after_warm_miss'].get('error'))!s}")

            print(f"  after warm-cache hit ...", flush=True)
            sub["after_warm_hit"] = _run_subprocess(python_exe, after_src, regime, cache_dir=cache_dir, warm=True)
            print(f"    -> {sub['after_warm_hit'].get('total_s', sub['after_warm_hit'].get('error'))!s}")

            results["per_regime"][name] = sub
            # Cumulative speedup summary line.
            b = sub["baseline_cold"].get("total_s")
            a = sub["after_cold"].get("total_s")
            h = sub["after_warm_hit"].get("total_s")
            if b is not None and a is not None:
                print(f"  -> cumulative cold:  baseline={b:.2f}s after={a:.2f}s speedup={b/max(a,1e-9):.2f}x")
            if b is not None and h is not None:
                print(f"  -> cumulative warm:  baseline={b:.2f}s after_hit={h:.2f}s speedup={b/max(h,1e-9):.2f}x")
            print()

    out_path = os.environ.get("ITER87_RESULTS", r"D:/Temp/iter87_results.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True, default=str)
    print(f"# wrote {out_path}")


if __name__ == "__main__":
    main()
