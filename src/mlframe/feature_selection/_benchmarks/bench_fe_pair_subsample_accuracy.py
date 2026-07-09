"""Benchmark: does subsampling rows in check_prospective_fe_pairs lose accuracy?

CRITICAL #2 follow-up. After the memory dispatcher (which avoids the OOM crash by
falling back to recompute-from-metadata at full n), the user asked whether
subsampling rows for the MI sweep is also safe -- subsampling would drop the
buffer memory linearly with sample size, which on n=4M means 200k subsample is
20x smaller buffer than full n.

The risk is statistical: MI estimates on smaller n are noisier, so the survivor
ranking under subsample may disagree with the full-n ranking. This bench drives
the function at several sample sizes against a synthetic dataset with KNOWN
engineered features (so we can grade against ground-truth) and reports per
sample size:

- ``jaccard_top_k``: Jaccard of (transformations_pair, bin_func_name) survivor
  sets between the subsampled run and the full-n reference; 1.0 = identical
  survivor choice.
- ``best_winner_match``: 1 if the single best survivor at subsample matches
  the full-n best; 0 otherwise. The single most-important survivor is what
  MRMR ends up appending to ``data`` -- if this disagrees, downstream feature
  ordering shifts.
- ``ground_truth_hit``: 1 if the synthetic ground-truth pair is in the
  recommended set; 0 otherwise. Sanity check against drift to spurious pairs.
- ``wall_time_s``: walltime per run.

Output: CSV in ``_results/`` next to the other FE benchmark logs.

Run:
    python -m mlframe.feature_selection._benchmarks.bench_fe_pair_subsample_accuracy
"""
from __future__ import annotations

import csv
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from mlframe.feature_selection.filters.feature_engineering import (
    check_prospective_fe_pairs,
    create_binary_transformations,
    create_unary_transformations,
)
from mlframe.feature_selection.filters.info_theory import merge_vars
from mlframe.feature_selection.filters.discretization import discretize_array


def _build_synthetic(n: int, seed: int = 0, noise_scale: float = 1.0) -> tuple[pd.DataFrame, np.ndarray]:
    """Build n-row data with several competing engineered-pair signals + noise.

    Three pair signals of comparable but different strength so the survivor
    ranking is INFORMATIVE (not just "the one good signal beats noise"):

    - (a, b) via multiply  -- strongest
    - (c, d) via add  -- medium
    - (a, c) via divide  -- weak

    The target combines all three with Gaussian noise of ``noise_scale``;
    higher noise = noisier per-sample MI estimates = harder for small
    subsamples to recover the full-n ranking.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "a": rng.uniform(-3.0, 3.0, n).astype(np.float32),
            "b": rng.uniform(-3.0, 3.0, n).astype(np.float32),
            "c": rng.uniform(0.5, 5.0, n).astype(np.float32),  # positive -> divide is well defined
            "d": rng.uniform(-2.0, 2.0, n).astype(np.float32),
            "n0": rng.standard_normal(n).astype(np.float32),
            "n1": rng.standard_normal(n).astype(np.float32),
        }
    )
    a = df["a"].to_numpy()
    b = df["b"].to_numpy()
    c = df["c"].to_numpy()
    d = df["d"].to_numpy()
    signal = 1.2 * (a * b) + 0.7 * (c + d) + 0.3 * (a / (c + 1e-3))  # strongest  # medium  # weak (a/c)
    noise = noise_scale * rng.standard_normal(n)
    target = (signal + noise > np.median(signal + noise)).astype(np.int32)
    return df, target


def _prepare_inputs(df: pd.DataFrame, target: np.ndarray, nbins: int = 4):
    """Build classes_y / freqs_y / classes_y_safe + original_cols for check_prospective_fe_pairs."""
    data_disc = np.column_stack([discretize_array(df[c].to_numpy(), n_bins=nbins, method="quantile", dtype=np.int32) for c in df.columns])
    data_disc = np.column_stack([data_disc, target.astype(np.int32)])
    nb = np.array([nbins] * df.shape[1] + [2], dtype=np.int64)
    target_indices = np.array([df.shape[1]], dtype=np.int64)
    classes_y, freqs_y, _ = merge_vars(
        factors_data=data_disc, vars_indices=target_indices,
        var_is_nominal=None, factors_nbins=nb, dtype=np.int32,
    )
    return {
        "classes_y": classes_y,
        "classes_y_safe": classes_y.copy(),
        "freqs_y": freqs_y,
        "original_cols": {i: i for i in range(df.shape[1])},
        "cols_names": list(df.columns),
        "nbins": nb,
    }


def _extract_survivor_configs(res: dict) -> list[tuple]:
    """Flatten res into a sorted, comparable list of survivor configs.

    Strips the ephemeral buffer index ``i`` from each (transformations_pair,
    bin_func_name, i) tuple so the same pair / binary combo is identifiable
    across runs that hit different buffer positions.
    """
    out = []
    for raw_pair, (this_pair_features, _vals, _cols, _nbins, _msgs) in res.items():
        for config, _j in this_pair_features:
            transformations_pair, bin_func_name, _i = config
            # Sort the unary pair to canonicalise (a-op, b-op) vs (b-op, a-op) for symmetric ops.
            key = (tuple(sorted(transformations_pair)), bin_func_name)
            out.append((raw_pair, key))
    return sorted(out)


def _run_one(n_eff: int, full_df: pd.DataFrame, full_target: np.ndarray, *, unary_preset: str, binary_preset: str, prospective_pairs: dict, seed: int = 42):
    """Run check_prospective_fe_pairs on a sampled subset of size n_eff and return
    (survivor_set, wall_time_s)."""
    if n_eff < len(full_df):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(full_df), size=n_eff, replace=False))
        df = full_df.iloc[idx].reset_index(drop=True)
        tgt = full_target[idx]
    else:
        df, tgt = full_df, full_target
    inputs = _prepare_inputs(df, tgt)
    unary = create_unary_transformations(preset=unary_preset)
    binary = create_binary_transformations(preset=binary_preset)
    times_spent: dict = defaultdict(float)
    t0 = time.perf_counter()
    res = check_prospective_fe_pairs(
        prospective_pairs=prospective_pairs,
        X=df,
        unary_transformations=unary,
        binary_transformations=binary,
        classes_y=inputs["classes_y"],
        classes_y_safe=inputs["classes_y_safe"],
        freqs_y=inputs["freqs_y"],
        num_fs_steps=0,
        cols=inputs["cols_names"],
        original_cols=inputs["original_cols"],
        fe_max_steps=1,
        fe_npermutations=1,
        fe_max_pair_features=4,
        fe_print_best_mis_only=True,
        fe_min_nonzero_confidence=0.0,
        fe_min_engineered_mi_prevalence=0.0,
        fe_good_to_best_feature_mi_threshold=0.5,
        fe_max_external_validation_factors=0,
        numeric_vars_to_consider=list(range(df.shape[1])),
        quantization_nbins=4,
        quantization_method="quantile",
        quantization_dtype=np.int32,
        times_spent=times_spent,
        verbose=0,
    )
    elapsed = time.perf_counter() - t0
    return _extract_survivor_configs(res), elapsed


def _jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def main():
    UNARY = "minimal"
    BINARY = "default"  # default preset has multiply, divide, add, subtract -> realistic competition
    FULL_N = 200_000  # full reference; large enough to be the "truth" but tractable
    SAMPLES = [2_000, 5_000, 10_000, 25_000, 50_000, 100_000, FULL_N]
    SEEDS = [0, 1, 2, 3, 4]  # 5 seeds for stable variance estimates
    NOISE_SCALE = 1.5  # noisier signal so subsample ranking can disagree with full
    OUT_DIR = os.path.join(os.path.dirname(__file__), "_results")
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(OUT_DIR, f"bench_fe_pair_subsample_accuracy_{stamp}.csv")

    print(f"# bench_fe_pair_subsample_accuracy  ({datetime.now().isoformat()})")
    print(f"# full_n={FULL_N}, samples={SAMPLES}, seeds={SEEDS}, noise_scale={NOISE_SCALE}")
    print(f"# preset: unary={UNARY}, binary={BINARY}")
    print()

    rows: list[dict] = []
    for trial_seed in SEEDS:
        df, tgt = _build_synthetic(FULL_N, seed=trial_seed, noise_scale=NOISE_SCALE)
        # Three competing pairs so the discovery's choice MATTERS:
        # (a, b) is strongest, (c, d) medium, (a, c) weak. Indices: a=0, b=1, c=2, d=3.
        prospective_pairs = {
            ((0, 1), 1.0): 1.0,
            ((2, 3), 1.0): 1.0,
            ((0, 2), 1.0): 1.0,
        }
        # Reference: full n run.
        full_survivors, full_time = _run_one(
            FULL_N, df, tgt,
            unary_preset=UNARY, binary_preset=BINARY,
            prospective_pairs=prospective_pairs, seed=trial_seed,
        )
        full_pairs_with_bins = {(s[0], s[1][1]) for s in full_survivors}
        for n_eff in SAMPLES:
            sub_survivors, sub_time = _run_one(
                n_eff, df, tgt,
                unary_preset=UNARY, binary_preset=BINARY,
                prospective_pairs=prospective_pairs, seed=trial_seed,
            )
            sub_pairs_with_bins = {(s[0], s[1][1]) for s in sub_survivors}
            jacc = _jaccard(full_survivors, sub_survivors)
            jacc_loose = _jaccard(list(full_pairs_with_bins), list(sub_pairs_with_bins))
            best_match = 1 if (full_survivors and sub_survivors and full_survivors[0] == sub_survivors[0]) else 0
            # Ground-truth: the (a, b) pair via multiply is the strongest synthetic signal;
            # for the (a, b)-keyed entry, the binary func should be "multiply".
            gt_hit = 1 if any(s[0] == (0, 1) and s[1][1] == "multiply" for s in sub_survivors) else 0
            row = {
                "trial_seed": trial_seed,
                "n_eff": n_eff,
                "ratio_of_full": f"{n_eff / FULL_N:.3f}",
                "n_survivors": len(sub_survivors),
                "jaccard_strict": f"{jacc:.4f}",
                "jaccard_pair_bin": f"{jacc_loose:.4f}",
                "best_winner_match": best_match,
                "ground_truth_hit": gt_hit,
                "wall_time_s": f"{sub_time:.3f}",
                "speedup_vs_full": f"{full_time / sub_time:.2f}x" if sub_time > 0 else "inf",
            }
            rows.append(row)
            print(
                f"  seed={trial_seed} n_eff={n_eff:>7d} "
                f"jacc={row['jaccard_strict']} jacc_pb={row['jaccard_pair_bin']} "
                f"best_match={best_match} gt_hit={gt_hit} "
                f"t={row['wall_time_s']}s ({row['speedup_vs_full']})"
            )
        print()

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"# wrote {out_csv}")

    # Summary: average jaccard across seeds per sample size
    print()
    print("# SUMMARY (averaged across seeds)")
    for n_eff in SAMPLES:
        slc = [r for r in rows if r["n_eff"] == n_eff]
        avg_strict = np.mean([float(r["jaccard_strict"]) for r in slc])
        avg_loose = np.mean([float(r["jaccard_pair_bin"]) for r in slc])
        avg_best = np.mean([r["best_winner_match"] for r in slc])
        avg_gt = np.mean([r["ground_truth_hit"] for r in slc])
        avg_t = np.mean([float(r["wall_time_s"]) for r in slc])
        print(
            f"  n_eff={n_eff:>7d}  jacc_strict={avg_strict:.3f}  "
            f"jacc_pair_bin={avg_loose:.3f}  "
            f"best={avg_best:.2f}  gt={avg_gt:.2f}  "
            f"t={avg_t:.3f}s"
        )


if __name__ == "__main__":
    main()
