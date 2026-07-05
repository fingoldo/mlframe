"""Multi-scenario bench: is BaselineDiagnostics ablation n_estimators over-provisioned?

The ablation diagnostic fits 1 baseline + N drop LightGBM models per target and ranks
features by ablation delta. At 200k it is ~74% of the full-config suite wall, and the
per-fit ``quick_model_n_estimators`` default (200) looks over-provisioned (a single-fixture
probe showed ~1.25x with the verdict unchanged). Per CLAUDE.md "Variant defaults: most
ACCURATE first" + "Don't ship validated-only-on-this-fixture defaults", a single-fixture win
does NOT justify flipping the default -- the verdict must hold across a WIDER test bed.

This bench runs the ablation diagnostic on >=6 diverse synthetic scenarios x >=3 seeds, at the
CURRENT default n_estimators (200) plus a reduced grid (0.25/0.5/0.75x) and a couple of sample_n
reductions, and records per cell:
  (a) the VERDICT -- the dominant feature and the top-k feature ranking by ablation delta;
  (b) wall time.

KEY question: at what reduced n_estimators (or sample_n) does the verdict stay IDENTICAL
(same dominant feature, same top-k ranking, no rank inversions) across ALL scenarios+seeds vs
the default? Verdict-equal -> speed breaks the tie (accuracy-then-speed rule). If lowering
flips the verdict on any scenario, the default is at its floor and must NOT be reduced.

Run CPU-only (< 4 min):
    cd "<mlframe>" && CUDA_VISIBLE_DEVICES="" python src/mlframe/training/baselines/_benchmarks/bench_ablation_n_estimators_provisioning.py

Output: _results/ablation_n_estimators_provisioning.json
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Each scenario builds a frame with a KNOWN dominant signal feature ("x0" by
# convention) so the ground-truth verdict is unambiguous. We then ask: does the
# ablation diagnostic recover x0 (and a stable ranking of the strong features)
# at reduced provisioning, on every scenario+seed?
N_ROWS = 4000  # per scenario; large enough for a stable LightGBM signal, small enough for budget
N_FEATURES = 8
SEEDS = (0, 1, 2)
DEFAULT_N_ESTIMATORS = 200  # current BaselineDiagnosticsConfig.quick_model_n_estimators
DEFAULT_SAMPLE_N = N_ROWS  # use full frame; sample_n reductions are tested as separate cells
ABLATION_TOP_K = 5
TOPK_VERDICT = 3  # how many leading ablation entries define the "ranking verdict"


def _scenario_linear(rng):
    """Linear-dominant: y is a clean linear combo, x0 dominates."""
    X = {f"x{i}": rng.normal(size=N_ROWS) for i in range(N_FEATURES)}
    df = pd.DataFrame(X)
    df["y"] = 6.0 * df["x0"] + 1.5 * df["x1"] + 0.6 * df["x2"] + 0.3 * rng.normal(size=N_ROWS)
    return df, [], "regression"


def _scenario_interaction(rng):
    """Interaction-dominant: x0*x1 drives y; x0 still the single strongest mover."""
    X = {f"x{i}": rng.normal(size=N_ROWS) for i in range(N_FEATURES)}
    df = pd.DataFrame(X)
    df["y"] = 4.0 * df["x0"] * df["x1"] + 2.5 * df["x0"] + 0.8 * df["x2"] + 0.3 * rng.normal(size=N_ROWS)
    return df, [], "regression"


def _scenario_redundant(rng):
    """Redundant-correlated: x0 and its near-copies x1,x2 share signal."""
    base = rng.normal(size=N_ROWS)
    X = {}
    X["x0"] = base + 0.05 * rng.normal(size=N_ROWS)
    X["x1"] = base + 0.20 * rng.normal(size=N_ROWS)
    X["x2"] = base + 0.40 * rng.normal(size=N_ROWS)
    for i in range(3, N_FEATURES):
        X[f"x{i}"] = rng.normal(size=N_ROWS)
    df = pd.DataFrame(X)
    df["y"] = 5.0 * base + 1.0 * df["x3"] + 0.3 * rng.normal(size=N_ROWS)
    return df, [], "regression"


def _scenario_noisy_weak(rng):
    """Noisy weak-signal: small SNR, x0 just barely the strongest."""
    X = {f"x{i}": rng.normal(size=N_ROWS) for i in range(N_FEATURES)}
    df = pd.DataFrame(X)
    df["y"] = 1.2 * df["x0"] + 0.6 * df["x1"] + 3.0 * rng.normal(size=N_ROWS)
    return df, [], "regression"


def _scenario_highcard_cat(rng):
    """High-card categorical: x0 is a categorical with a per-level mean signal."""
    n_levels = 60
    cat = rng.integers(0, n_levels, size=N_ROWS)
    level_means = rng.normal(size=n_levels) * 3.0
    X = {"x0": cat.astype(np.int32)}
    for i in range(1, N_FEATURES):
        X[f"x{i}"] = rng.normal(size=N_ROWS)
    df = pd.DataFrame(X)
    df["x0"] = df["x0"].astype("category")
    df["y"] = level_means[cat] + 1.0 * df["x1"] + 0.4 * rng.normal(size=N_ROWS)
    return df, ["x0"], "regression"


def _scenario_mixed_binary(rng):
    """Mixed binary-classification: x0 drives a logit, mixed informative features."""
    X = {f"x{i}": rng.normal(size=N_ROWS) for i in range(N_FEATURES)}
    df = pd.DataFrame(X)
    logit = 3.0 * df["x0"] + 1.2 * df["x1"] - 0.8 * df["x2"] + 0.3 * rng.normal(size=N_ROWS)
    p = 1.0 / (1.0 + np.exp(-logit))
    df["y"] = (rng.uniform(size=N_ROWS) < p).astype(np.int32)
    return df, [], "binary_classification"


SCENARIOS = {
    "linear_dominant": _scenario_linear,
    "interaction_dominant": _scenario_interaction,
    "redundant_correlated": _scenario_redundant,
    "noisy_weak_signal": _scenario_noisy_weak,
    "highcard_categorical": _scenario_highcard_cat,
    "mixed_binary": _scenario_mixed_binary,
}

# Grid of provisioning cells. "default" is the reference; the rest are reductions.
# Each cell varies either n_estimators (0.25/0.5/0.75x of 200) or sample_n.
CELLS = [
    {"name": "default", "n_estimators": DEFAULT_N_ESTIMATORS, "sample_n": DEFAULT_SAMPLE_N},
    {"name": "ne_150_0.75x", "n_estimators": 150, "sample_n": DEFAULT_SAMPLE_N},
    {"name": "ne_100_0.50x", "n_estimators": 100, "sample_n": DEFAULT_SAMPLE_N},
    {"name": "ne_50_0.25x", "n_estimators": 50, "sample_n": DEFAULT_SAMPLE_N},
    {"name": "sample_n_2000", "n_estimators": DEFAULT_N_ESTIMATORS, "sample_n": 2000},
    {"name": "sample_n_1000", "n_estimators": DEFAULT_N_ESTIMATORS, "sample_n": 1000},
]


def _run_cell(df, cat_features, target_type, n_estimators, sample_n, seed):
    """Run the ablation diagnostic once; return (dominant_feature, topk_ranking, wall)."""
    from mlframe.training.baselines.diagnostics import BaselineDiagnostics
    from mlframe.training.configs import BaselineDiagnosticsConfig

    feature_cols = [c for c in df.columns if c != "y"]
    cfg = BaselineDiagnosticsConfig(
        enabled=True,
        ablation_top_k=ABLATION_TOP_K,
        quick_model_family="lightgbm",
        quick_model_n_estimators=n_estimators,
        sample_n=sample_n,
        random_state=seed,
    )
    diag = BaselineDiagnostics(cfg)
    t0 = time.perf_counter()
    report = diag.fit_and_report(
        train_df=df, train_target=df["y"],
        feature_cols=feature_cols, cat_features=cat_features,
        target_type=target_type, target_name="y",
    )
    wall = time.perf_counter() - t0
    dom = report.dominant_features  # list of {"feature","score","rank"} by ablation delta
    ranking = [d["feature"] for d in dom]
    dominant = ranking[0] if ranking else None
    return dominant, ranking, wall, bool(report.skipped)


def main():
    results = []
    grand_t0 = time.perf_counter()

    for scen_name, builder in SCENARIOS.items():
        for seed in SEEDS:
            rng = np.random.default_rng(1000 + seed)
            df, cat_features, target_type = builder(rng)

            ref = None
            for cell in CELLS:
                dom, ranking, wall, skipped = _run_cell(
                    df, cat_features, target_type,
                    cell["n_estimators"], cell["sample_n"], seed,
                )
                row = {
                    "scenario": scen_name,
                    "seed": seed,
                    "target_type": target_type,
                    "cell": cell["name"],
                    "n_estimators": cell["n_estimators"],
                    "sample_n": cell["sample_n"],
                    "dominant_feature": dom,
                    "topk_ranking": ranking[:TOPK_VERDICT],
                    "full_ranking": ranking,
                    "wall_s": round(wall, 4),
                    "skipped": skipped,
                    "ground_truth_dominant": "x0",
                    "dominant_correct": (dom == "x0"),
                }
                if cell["name"] == "default":
                    ref = row
                    row["dominant_same_as_default"] = True
                    row["topk_same_as_default"] = True
                else:
                    row["dominant_same_as_default"] = dom == ref["dominant_feature"]
                    row["topk_same_as_default"] = ranking[:TOPK_VERDICT] == ref["topk_ranking"]
                results.append(row)
                print(
                    f"{scen_name:22s} seed={seed} {cell['name']:14s} "
                    f"dom={str(dom):8s} top{TOPK_VERDICT}={ranking[:TOPK_VERDICT]} "
                    f"wall={wall:.3f}s "
                    f"dom_same={row['dominant_same_as_default']} "
                    f"topk_same={row['topk_same_as_default']}"
                )

    total_wall = time.perf_counter() - grand_t0

    # --- aggregate verdict-stability per reduced cell ---
    reduced = [c["name"] for c in CELLS if c["name"] != "default"]
    summary = {}
    for cell_name in reduced:
        rows = [r for r in results if r["cell"] == cell_name]
        n = len(rows)
        dom_same = sum(r["dominant_same_as_default"] for r in rows)
        topk_same = sum(r["topk_same_as_default"] for r in rows)
        dom_correct = sum(r["dominant_correct"] for r in rows)
        # wall win vs default, averaged over scenario+seed pairs
        speedups = []
        for r in rows:
            ref = next(d for d in results if d["scenario"] == r["scenario"] and d["seed"] == r["seed"] and d["cell"] == "default")
            if r["wall_s"] > 0:
                speedups.append(ref["wall_s"] / r["wall_s"])
        summary[cell_name] = {
            "cells": n,
            "dominant_same_as_default": f"{dom_same}/{n}",
            "topk_same_as_default": f"{topk_same}/{n}",
            "dominant_correct_vs_ground_truth": f"{dom_correct}/{n}",
            "mean_speedup_vs_default": round(float(np.mean(speedups)), 3) if speedups else None,
            "verdict_fully_stable": (dom_same == n and topk_same == n),
            "dominant_fully_stable": (dom_same == n),
        }

    # default cell: dominant-correct rate (ground-truth recovery baseline)
    default_rows = [r for r in results if r["cell"] == "default"]
    default_dom_correct = sum(r["dominant_correct"] for r in default_rows)

    out = {
        "config": {
            "n_rows": N_ROWS,
            "n_features": N_FEATURES,
            "seeds": list(SEEDS),
            "default_n_estimators": DEFAULT_N_ESTIMATORS,
            "ablation_top_k": ABLATION_TOP_K,
            "topk_verdict": TOPK_VERDICT,
            "scenarios": list(SCENARIOS.keys()),
            "cells": CELLS,
        },
        "default_dominant_correct": f"{default_dom_correct}/{len(default_rows)}",
        "summary_per_reduced_cell": summary,
        "total_wall_s": round(total_wall, 2),
        "rows": results,
    }

    out_dir = Path(__file__).parent / "_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "ablation_n_estimators_provisioning.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== verdict-stability summary (reduced cell vs default) ===")
    for cell_name, s in summary.items():
        print(
            f"{cell_name:14s} dom_same={s['dominant_same_as_default']:6s} "
            f"topk_same={s['topk_same_as_default']:6s} "
            f"dom_correct={s['dominant_correct_vs_ground_truth']:6s} "
            f"speedup={s['mean_speedup_vs_default']}x "
            f"verdict_stable={s['verdict_fully_stable']}"
        )
    print(f"\ndefault dominant-correct vs ground-truth: {default_dom_correct}/{len(default_rows)}")
    print(f"total wall: {total_wall:.1f}s")
    print(f"results -> {out_path}")
    return out


if __name__ == "__main__":
    main()
