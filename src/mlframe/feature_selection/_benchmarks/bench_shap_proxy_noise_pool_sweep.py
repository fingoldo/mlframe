"""Diagnostic sweep that confirms the iter23 noise-pool diagnosis.

Hypothesis (from iter23): at high width (e.g. 7000) and low ``n_rows`` (~2000), recall sometimes
sits at 7/8 because the weakest planted informatives (linspace 1.0->0.4 coefs) become indistinguishable
from spurious correlations in the large independent-noise pool. The signal/noise floor of
``make_regime_dataset`` (NOT a ShapProxiedFS pipeline bias) drives the apparent recall ceiling.

Prediction: the effect should DISSOLVE as we either
  - raise ``n_rows`` (random correlations average out as ~1/sqrt(n_rows)), or
  - raise ``snr`` (the planted informatives become well-separated from random noise correlations).

This bench picks ONE axis at a time, holds width fixed at 7000, sweeps the axis at 3 levels with
``n_seeds`` per cell, and prints mean +- std recall per cell so the trend is unambiguous.

Run (PowerShell)::

    $env:PYTHONPATH = '<worktree>\\src'
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_shap_proxy_noise_pool_sweep --axis n_rows

The default axis is ``n_rows`` because it's the most diagnostic for a *finite-sample* artifact;
``--axis snr`` swaps in the higher-SNR sweep instead.

WATCHDOG: each fit is ~30-60s, default sweep is 3 levels x ``n_seeds`` (default 3) = 9 fits, so
budget ~10min. Per-cell summary lands on stdout immediately for liveness.
"""

from __future__ import annotations

import argparse
import statistics
import time
import warnings

warnings.filterwarnings("ignore")


def _build_selector(seed: int = 0):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    return ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=500, cluster_features=True, cluster_corr_threshold=0.7,
        top_n=20, n_splits=4, n_revalidation_models=3, trust_guard=True, n_anchors=24,
        run_importance_ablation=True, within_cluster_refine=True,
        random_state=seed, verbose=False)


def _make(*, n_features: int, n_rows: int, snr: float, seed: int, n_informative: int = 8, n_redundant: int = 12):
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    n_noise = max(0, n_features - n_informative - n_redundant)
    X, y, roles = make_regime_dataset(
        n_samples=n_rows, n_informative=n_informative, n_redundant=n_redundant, redundancy_rho=0.9, n_noise=n_noise, snr=snr, task="binary", seed=seed
    )
    return X, y, roles


def _agg(values):
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"mean": sum(values) / n, "std": statistics.pstdev(values) if n >= 2 else 0.0, "min": min(values), "max": max(values)}


def run_cell(*, n_features: int, n_rows: int, snr: float, n_seeds: int):
    """Fit ``n_seeds`` times at the given regime; return aggregate recall + wall."""
    recalls, walls, per_seed = [], [], []
    for seed in range(n_seeds):
        print(f"[width={n_features} n_rows={n_rows} snr={snr} seed={seed}] starting fit", flush=True)
        X, y, roles = _make(n_features=n_features, n_rows=n_rows, snr=snr, seed=seed)
        informative = {n for n, r in roles.items() if r == "informative"}
        sel = _build_selector(seed=seed)
        t0 = time.perf_counter()
        sel.fit(X, y)
        wall = time.perf_counter() - t0
        rec = len(informative & set(sel.selected_features_))
        n_inf = len(informative)
        recall = rec / n_inf if n_inf else float("nan")
        recalls.append(recall)
        walls.append(wall)
        per_seed.append({"seed": seed, "recall": recall, "recovered": rec, "n_informative": n_inf, "wall": wall, "n_selected": len(sel.selected_features_)})
        print(f"[width={n_features} n_rows={n_rows} snr={snr} seed={seed}] " f"done in {wall:.2f}s, recall={recall:.3f} ({rec}/{n_inf})", flush=True)
    return {"recall": _agg(recalls), "wall": _agg(walls), "per_seed": per_seed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=("n_rows", "snr"), default="n_rows")
    ap.add_argument("--width", type=int, default=7000)
    ap.add_argument("--n_seeds", type=int, default=3)
    # Sweep levels per axis (3 levels each, chosen to bracket the iter23 problematic regime).
    ap.add_argument("--n_rows_levels", default="2000,5000,10000")
    ap.add_argument("--snr_levels", default="5,8,12")
    # Held-fixed values on the non-swept axis.
    ap.add_argument("--fixed_n_rows", type=int, default=2000, help="n_rows held fixed when sweeping snr (matches the iter23 regime)")
    ap.add_argument("--fixed_snr", type=float, default=5.0, help="snr held fixed when sweeping n_rows (matches the iter23 regime)")
    args = ap.parse_args()

    if args.axis == "n_rows":
        levels = [int(x) for x in args.n_rows_levels.split(",") if x.strip()]
        cells = [{"n_features": args.width, "n_rows": lv, "snr": args.fixed_snr} for lv in levels]
        axis_label = "n_rows"
    else:
        levels = [float(x) for x in args.snr_levels.split(",") if x.strip()]
        cells = [{"n_features": args.width, "n_rows": args.fixed_n_rows, "snr": lv} for lv in levels]
        axis_label = "snr"

    print(f"=== noise-pool diagnosis sweep: axis={axis_label}, width={args.width}, " f"n_seeds={args.n_seeds} ===", flush=True)

    results = []
    for cell in cells:
        print(f"\n--- cell: {cell} ---", flush=True)
        r = run_cell(n_features=cell["n_features"], n_rows=cell["n_rows"], snr=cell["snr"], n_seeds=args.n_seeds)
        results.append({"cell": cell, **r})
        print(f"--- cell done: recall mean={r['recall']['mean']:.3f} "
              f"std={r['recall']['std']:.3f} (min={r['recall']['min']:.3f}, "
              f"max={r['recall']['max']:.3f}) ---", flush=True)

    # Summary table.
    print(f"\n=== summary: width={args.width}, axis={axis_label} ===")
    header = f"{axis_label:>10}{'recall mean':>14}{'recall std':>12}{'recall min':>12}{'recall max':>12}{'wall mean(s)':>14}"
    print(header)
    print("-" * len(header))
    for entry in results:
        lv = entry["cell"][axis_label]
        rec = entry["recall"]
        wall = entry["wall"]
        print(f"{lv:>10}{rec['mean']:>14.3f}{rec['std']:>12.3f}" f"{rec['min']:>12.3f}{rec['max']:>12.3f}{wall['mean']:>14.2f}", flush=True)
    print("-" * len(header))

    # Diagnosis verdict: monotone increase in mean recall toward 1.0 confirms the noise-pool effect.
    means = [e["recall"]["mean"] for e in results]
    if all(b >= a - 1e-9 for a, b in zip(means, means[1:])) and means[-1] >= 0.99:
        print(f"\nDIAGNOSIS CONFIRMED: mean recall rises {means[0]:.3f} -> {means[-1]:.3f} as "
              f"{axis_label} increases; high-{axis_label} cell hits ceiling. "
              f"The 7/8 recall at width=7000 / low-{axis_label} is a noise-pool artifact, "
              f"NOT a ShapProxiedFS pipeline bias.")
    else:
        print(f"\nDIAGNOSIS NOT CONFIRMED on this run (mean recalls: {means}). " f"Recheck levels or n_seeds, or revisit the hypothesis.")


if __name__ == "__main__":
    main()
