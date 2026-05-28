"""Width-scaling benchmark for the full ``ShapProxiedFS.fit`` wide-data pipeline.

Runs the FULL fit (``cluster_features=True``, prefilter on) on synthetic data of increasing column
count (1k / 5k / 10k features) built from ``make_regime_dataset`` (a handful of informatives, some
correlated redundant copies, and lots of independent noise -- the user's real wide-data regime), and
times EACH PIPELINE STAGE separately:

    prefilter (one model fit on all columns) -> clustering -> OOF-SHAP -> importance pre-screen ->
    exhaustive-approx search -> trust guard -> honest re-validation -> importance ablation ->
    within-cluster refine.

It reads the per-stage wall-clock via the ``ShapProxiedFS._stage_timings`` instrumentation hook (set a
dict on the selector before ``fit`` and each stage's seconds land in it), then prints a stage-breakdown
table per width so we can see WHERE the wall-clock goes at scale (per earlier profiling: the honest
model retrains -- trust guard + re-validation + ablation + within-cluster refine -- and the prefilter /
OOF model fits dominate, NOT the proxy scan).

Run::

    $env:PYTHONPATH = '<worktree>\\src'  # PowerShell
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_shap_proxy_scaling

Optional args: ``--widths 1000,5000`` to override the swept widths, ``--rows 4000``, ``--profile`` to
also cProfile the widest fit and dump the top cumulative hotspots.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Stage order for the breakdown table (matches the pipeline order in ShapProxiedFS.fit).
_STAGE_ORDER = (
    "prefilter", "clustering", "oof_shap", "prescreen", "search",
    "trust_guard", "revalidation", "importance_ablation", "within_cluster_refine",
)


def make_wide(n_features: int, *, n_rows: int = 4000, n_informative: int = 8, n_redundant: int = 12,
              seed: int = 0):
    """Wide regime dataset: a few informatives + correlated redundant copies + the rest noise."""
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    n_noise = max(0, n_features - n_informative - n_redundant)
    X, y, roles = make_regime_dataset(
        n_samples=n_rows, n_informative=n_informative, n_redundant=n_redundant,
        redundancy_rho=0.9, n_noise=n_noise, snr=5.0, task="binary", seed=seed)
    return X, y, roles


def _build_selector(seed: int = 0):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    # Wide-data config: prefilter on, clustering on, exhaustive-approx search, honest re-validation.
    return ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=500, cluster_features=True, cluster_corr_threshold=0.7,
        top_n=20, n_splits=4, n_revalidation_models=3, trust_guard=True, n_anchors=24,
        run_importance_ablation=True, within_cluster_refine=True,
        random_state=seed, verbose=False)


def bench_width(n_features: int, *, n_rows: int) -> tuple[float, dict, object]:
    """Run one full fit at a given width; return (total_seconds, stage_timings, fitted_selector)."""
    X, y, roles = make_wide(n_features, n_rows=n_rows)
    sel = _build_selector()
    sel._stage_timings = {}
    t0 = time.perf_counter()
    sel.fit(X, y)
    total = time.perf_counter() - t0
    sel._roles = roles
    return total, dict(sel._stage_timings), sel


def _recovered_informatives(sel, roles) -> tuple[int, int]:
    informative = {name for name, r in roles.items() if r == "informative"}
    selected = set(sel.selected_features_)
    return len(informative & selected), len(informative)


def print_breakdown(results: dict[int, tuple[float, dict, object]]) -> None:
    widths = sorted(results)
    print("\n=== ShapProxiedFS stage-breakdown (seconds) ===")
    header = f"{'stage':<22}" + "".join(f"{w:>12}" for w in widths)
    print(header)
    print("-" * len(header))
    for stage in _STAGE_ORDER:
        row = f"{stage:<22}"
        for w in widths:
            t = results[w][1].get(stage, 0.0)
            row += f"{t:>12.3f}" if t else f"{'-':>12}"
        print(row)
    print("-" * len(header))
    # measured-stage sum and total (gap = un-instrumented glue: splits, re-ranking, bookkeeping).
    row_sum = f"{'measured sum':<22}"
    row_tot = f"{'TOTAL fit':<22}"
    for w in widths:
        total, timings, _ = results[w]
        row_sum += f"{sum(timings.values()):>12.3f}"
        row_tot += f"{total:>12.3f}"
    print(row_sum)
    print(row_tot)
    print("-" * len(header))
    # dominant stage + recovery sanity per width.
    for w in widths:
        total, timings, sel = results[w]
        if timings:
            dom = max(timings, key=timings.get)
            share = 100.0 * timings[dom] / total if total else 0.0
            rec, n_inf = _recovered_informatives(sel, sel._roles)
            print(f"  width={w:>6}: dominant stage = {dom} ({timings[dom]:.2f}s, {share:.0f}% of fit); "
                  f"selected {len(sel.selected_features_)} feats, recovered {rec}/{n_inf} informatives")


def profile_widest(n_features: int, *, n_rows: int) -> None:
    import cProfile
    import io
    import pstats

    X, y, _ = make_wide(n_features, n_rows=n_rows)
    sel = _build_selector()
    pr = cProfile.Profile()
    pr.enable()
    sel.fit(X, y)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
    print(f"\n=== cProfile (width={n_features}, top 30 by cumulative) ===")
    print(s.getvalue())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="1000,5000,10000")
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    widths = [int(w) for w in args.widths.split(",") if w.strip()]

    print(f"=== ShapProxiedFS width-scaling bench (n_rows={args.rows}) ===")
    results: dict[int, tuple[float, dict, object]] = {}
    for w in widths:
        print(f"\n--- fitting width={w} ...", flush=True)
        total, timings, sel = bench_width(w, n_rows=args.rows)
        results[w] = (total, timings, sel)
        print(f"    done in {total:.2f}s", flush=True)
    print_breakdown(results)

    if args.profile:
        profile_widest(max(widths), n_rows=args.rows)


if __name__ == "__main__":
    main()
