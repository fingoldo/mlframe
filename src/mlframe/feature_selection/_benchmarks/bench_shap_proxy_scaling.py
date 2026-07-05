"""Width-scaling benchmark for the full ``ShapProxiedFS.fit`` wide-data pipeline.

Runs the FULL fit (``cluster_features=True``, prefilter on) on synthetic data of increasing column
count (default 3k + 5k for a watchdog-safe quick run; pass ``--widths 1000,5000,10000`` for the full
sweep) built from ``make_regime_dataset`` (a handful of informatives, some correlated redundant
copies, and lots of independent noise -- the user's real wide-data regime), and times EACH PIPELINE
STAGE separately:

    prefilter (one model fit on all columns) -> clustering -> OOF-SHAP -> importance pre-screen ->
    exhaustive-approx search -> trust guard -> honest re-validation -> importance ablation ->
    within-cluster refine.

Multi-seed aggregation (default ``--n_seeds 3``): at each (width) cell the bench fits ``n_seeds``
times with seeds ``0..n_seeds-1`` and reports mean / std / min / max of recall (informatives kept /
total) AND end-to-end wall-clock. Stage timings are reported as cross-seed means. This guards against
the single-seed comparison trap that drove the iter21-23 rabbit hole (a single-seed "recall floor"
was actually noise-pool variance from ``make_regime_dataset`` -- mean-over-seeds surfaces it as
variance immediately).

Reads the per-stage wall-clock via the ``ShapProxiedFS._stage_timings`` instrumentation hook (set a
dict on the selector before ``fit`` and each stage's seconds land in it).

Run::

    $env:PYTHONPATH = '<worktree>\\src'  # PowerShell
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_shap_proxy_scaling

Optional args: ``--widths 1000,5000`` to override the swept widths (default ``3000,5000``),
``--rows 4000``, ``--n_seeds 3``, ``--profile`` to also cProfile the widest fit and dump the top
cumulative hotspots.
"""

from __future__ import annotations

import argparse
import statistics
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Stage order for the breakdown table (matches the pipeline order in ShapProxiedFS.fit).
_STAGE_ORDER = (
    "prefilter", "clustering", "oof_shap", "prescreen", "search",
    "trust_guard", "revalidation", "importance_ablation", "within_cluster_refine",
)


def make_wide(n_features: int, *, n_rows: int = 4000, n_informative: int = 8, n_redundant: int = 12, snr: float = 5.0, seed: int = 0):
    """Wide regime dataset: a few informatives + correlated redundant copies + the rest noise.

    Recall caveat at width>=5000 / n_rows<=2000 (iter23): the dropped-informative set is
    NON-DETERMINISTIC across seeds and NOT coef-monotone (seed sweep at width=7000, n_rows=2000
    showed dropped inf indices varying by seed, with strong-coef informatives sometimes dropping
    while weaker ones survive). This is a finite-sample noise-pool artifact from
    ``make_regime_dataset`` (linspace 1.0->0.4 coefs + Gaussian noise) -- raise ``n_rows`` to
    >=5000 or ``snr`` to >=8 before reading recall numbers at the high-width end as algorithmic.
    """
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    n_noise = max(0, n_features - n_informative - n_redundant)
    X, y, roles = make_regime_dataset(
        n_samples=n_rows, n_informative=n_informative, n_redundant=n_redundant, redundancy_rho=0.9, n_noise=n_noise, snr=snr, task="binary", seed=seed
    )
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


def bench_width_single(n_features: int, *, n_rows: int, seed: int, snr: float = 5.0) -> tuple[float, dict, object, dict]:
    """Run one full fit at a given (width, seed); return (total_seconds, stage_timings,
    fitted_selector, roles)."""
    X, y, roles = make_wide(n_features, n_rows=n_rows, snr=snr, seed=seed)
    sel = _build_selector(seed=seed)
    sel._stage_timings = {}
    t0 = time.perf_counter()
    sel.fit(X, y)
    total = time.perf_counter() - t0
    return total, dict(sel._stage_timings), sel, roles


def _recovered_informatives(sel, roles) -> tuple[int, int]:
    informative = {name for name, r in roles.items() if r == "informative"}
    selected = set(sel.selected_features_)
    return len(informative & selected), len(informative)


def _agg(values: list[float]) -> dict[str, float]:
    """mean/std/min/max for >=1 sample (std=0.0 on a single sample, by convention)."""
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    mean = sum(values) / n
    std = statistics.pstdev(values) if n >= 2 else 0.0
    return {"mean": mean, "std": std, "min": min(values), "max": max(values)}


def bench_width_multi_seed(n_features: int, *, n_rows: int, n_seeds: int, snr: float = 5.0) -> dict:
    """Run ``n_seeds`` full fits at the given width; return aggregate dict with mean/std/min/max
    of recall + total wall, plus per-seed timings + the cross-seed mean of each stage timing."""
    recalls: list[float] = []
    walls: list[float] = []
    per_stage: dict[str, list[float]] = {s: [] for s in _STAGE_ORDER}
    per_seed: list[dict] = []
    for seed in range(n_seeds):
        print(f"[width={n_features} seed={seed}] starting fit", flush=True)
        t0 = time.perf_counter()
        total, timings, sel, roles = bench_width_single(n_features, n_rows=n_rows, seed=seed, snr=snr)
        rec, n_inf = _recovered_informatives(sel, roles)
        recall = rec / n_inf if n_inf else float("nan")
        recalls.append(recall)
        walls.append(total)
        for s in _STAGE_ORDER:
            if s in timings:
                per_stage[s].append(timings[s])
        per_seed.append({"seed": seed, "wall": total, "recall": recall, "recovered": rec, "n_informative": n_inf, "n_selected": len(sel.selected_features_)})
        print(
            f"[width={n_features} seed={seed}] done in {total:.2f}s, " f"recall={recall:.3f} ({rec}/{n_inf}), elapsed={time.perf_counter()-t0:.2f}s", flush=True
        )
    stage_means: dict[str, float] = {s: (sum(v) / len(v)) if v else 0.0 for s, v in per_stage.items()}
    return {
        "width": n_features,
        "n_seeds": n_seeds,
        "wall": _agg(walls),
        "recall": _agg(recalls),
        "stage_means": stage_means,
        "per_seed": per_seed,
    }


def print_multi_seed_table(results: list[dict]) -> None:
    """Print the cross-seed mean +- std table for recall and wall, plus per-seed detail."""
    print("\n=== ShapProxiedFS multi-seed recall / wall (mean +- std across seeds) ===")
    header = f"{'width':>8}{'n_seeds':>10}{'recall mean':>14}{'recall std':>12}{'recall min':>12}{'recall max':>12}{'wall mean(s)':>14}{'wall std(s)':>13}"
    print(header)
    print("-" * len(header))
    for r in results:
        rec = r["recall"]
        wall = r["wall"]
        print(f"{r['width']:>8}{r['n_seeds']:>10}"
              f"{rec['mean']:>14.3f}{rec['std']:>12.3f}{rec['min']:>12.3f}{rec['max']:>12.3f}"
              f"{wall['mean']:>14.2f}{wall['std']:>13.2f}", flush=True)
    print("-" * len(header))
    # Per-seed detail (recall + wall) so the variance source is visible.
    print("\n=== per-seed detail ===")
    for r in results:
        print(f"  width={r['width']}:")
        for ps in r["per_seed"]:
            print(f"    seed={ps['seed']}: recall={ps['recall']:.3f} "
                  f"({ps['recovered']}/{ps['n_informative']}), wall={ps['wall']:.2f}s, "
                  f"selected={ps['n_selected']}")


def print_stage_breakdown(results: list[dict]) -> None:
    """Cross-seed-mean stage breakdown table (one column per width)."""
    widths = [r["width"] for r in results]
    print("\n=== ShapProxiedFS stage-breakdown (cross-seed mean seconds) ===")
    header = f"{'stage':<22}" + "".join(f"{w:>12}" for w in widths)
    print(header)
    print("-" * len(header))
    for stage in _STAGE_ORDER:
        row = f"{stage:<22}"
        for r in results:
            t = r["stage_means"].get(stage, 0.0)
            row += f"{t:>12.3f}" if t else f"{'-':>12}"
        print(row)
    print("-" * len(header))
    row_sum = f"{'measured sum':<22}"
    row_tot = f"{'TOTAL fit (mean)':<22}"
    for r in results:
        row_sum += f"{sum(r['stage_means'].values()):>12.3f}"
        row_tot += f"{r['wall']['mean']:>12.3f}"
    print(row_sum)
    print(row_tot)


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
    # Default widths kept SHORT (3k + 5k) so the default multi-seed run completes <10min on the
    # watchdog; full 1k/5k/10k sweep is opt-in via --widths.
    ap.add_argument("--widths", default="3000,5000")
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--n_seeds", type=int, default=3, help="number of seeds (0..n_seeds-1) to aggregate at each width")
    ap.add_argument("--snr", type=float, default=5.0)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    widths = [int(w) for w in args.widths.split(",") if w.strip()]

    print(f"=== ShapProxiedFS width-scaling bench " f"(n_rows={args.rows}, n_seeds={args.n_seeds}, snr={args.snr}) ===", flush=True)
    results: list[dict] = []
    for w in widths:
        print(f"\n--- width={w}: running {args.n_seeds} seeds ---", flush=True)
        r = bench_width_multi_seed(w, n_rows=args.rows, n_seeds=args.n_seeds, snr=args.snr)
        results.append(r)
        # Print per-cell summary AS SOON AS the cell is done (watchdog visibility).
        print(f"--- width={w}: recall mean={r['recall']['mean']:.3f} std={r['recall']['std']:.3f}, "
              f"wall mean={r['wall']['mean']:.2f}s std={r['wall']['std']:.2f}s ---", flush=True)

    print_multi_seed_table(results)
    print_stage_breakdown(results)

    if args.profile:
        profile_widest(max(widths), n_rows=args.rows)


if __name__ == "__main__":
    main()
