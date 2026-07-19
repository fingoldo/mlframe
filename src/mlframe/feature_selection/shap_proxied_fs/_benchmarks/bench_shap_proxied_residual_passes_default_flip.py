"""gt_09 default-flip bench: does ``residual_passes=1`` deserve to become the default?

Run: python -m mlframe.feature_selection.shap_proxied_fs._benchmarks.bench_shap_proxied_residual_passes_default_flip

Fixtures: mixed-strength bed (6 strong + 6 weak), pure-strong bed (6 strong + noise, no weak signal),
pure-noise bed (no signal at all), and 2 ``make_regime_dataset`` regimes (additive high-SNR,
redundancy) x 3 seeds. Arms: {0 passes, 1 pass rescue, 1 pass blend lambda=1, 1 pass hard-residual}.

Flip rule (gt_09 sec 5): wins weak-recall beds, ties pure beds (no noise inflation), wall-clock
overhead <=~1.8x acceptable. This script prints the full table; the verdict is recorded in the
module docstring of the CALLING session's summary, not computed here (REJECTED != DELETED: a losing
arm stays runnable, only the shipped default is decided elsewhere).
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time

import numpy as np
import pandas as pd


def _mixed_strength(seed=0, n=1500, p=1500, n_strong=6, n_weak=6, strong_weight=1.0, weak_weight=0.25):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    strong = list(range(n_strong))
    weak = list(range(50, 50 + n_weak))
    logit = strong_weight * X[:, strong].sum(axis=1) + weak_weight * X[:, weak].sum(axis=1)
    logit = logit / logit.std() * 2.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    cols = [f"f{i}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), pd.Series(y), set(strong), set(weak)


def _pure_strong(seed=0, n=1500, p=1500, n_strong=6, strong_weight=1.0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    strong = list(range(n_strong))
    logit = strong_weight * X[:, strong].sum(axis=1)
    logit = logit / logit.std() * 2.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    cols = [f"f{i}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), pd.Series(y), set(strong), set()


def _pure_noise(seed=0, n=1500, p=1500):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = (rng.random(n) < 0.5).astype(int)
    cols = [f"f{i}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), pd.Series(y), set(), set()


def _regime(seed, redundancy_rho=0.0, snr=8.0):
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    n_redundant = 5 if redundancy_rho > 0 else 0
    Xdf, y, roles = make_regime_dataset(
        n_samples=1500, n_informative=8, n_redundant=n_redundant, redundancy_rho=redundancy_rho,
        n_noise=200, snr=snr, task="binary", seed=seed,
    )
    informative = {c for c, r in roles.items() if r == "informative"}
    return Xdf, pd.Series(y), informative, set()


ARMS = dict(
    off=dict(residual_passes=0),
    rescue=dict(residual_passes=1, residual_merge="rescue"),
    blend=dict(residual_passes=1, residual_merge="blend", residual_lambda=1.0),
    hard=dict(residual_passes=1, residual_merge="rescue", residual_exclude_top=6),
)


def run_one(name, Xdf, y, informative, weak, seed, arm_kwargs):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sel = ShapProxiedFS(classification=True, random_state=seed, verbose=False, n_jobs=1, prescreen_ladder_mode="off", **arm_kwargs)
    t0 = time.perf_counter()
    sel.fit(Xdf, y)
    wall = time.perf_counter() - t0
    selected = set(sel.selected_features_)
    weak_recall = len(weak & selected) / max(1, len(weak)) if weak else float("nan")
    noise_selected = len(selected - informative - weak)
    return dict(bed=name, seed=seed, wall=wall, n_selected=len(selected), weak_recall=weak_recall, noise_selected=noise_selected)


def main():
    beds = []
    for seed in (0, 1, 2):
        beds.append(("mixed_strength", *_mixed_strength(seed=seed), seed))
        beds.append(("pure_strong", *_pure_strong(seed=seed), seed))
        beds.append(("pure_noise", *_pure_noise(seed=seed), seed))
        beds.append(("regime_additive_hisnr", *_regime(seed, redundancy_rho=0.0, snr=8.0), seed))
        beds.append(("regime_redundancy", *_regime(seed, redundancy_rho=0.8, snr=8.0), seed))

    rows = []
    for bed_name, Xdf, y, informative, weak, seed in beds:
        for arm_name, arm_kwargs in ARMS.items():
            row = run_one(bed_name, Xdf, y, informative, weak, seed, arm_kwargs)
            row["arm"] = arm_name
            rows.append(row)
            print(f"{bed_name:24s} seed={seed} arm={arm_name:8s} wall={row['wall']:.2f}s "
                  f"n_selected={row['n_selected']:3d} weak_recall={row['weak_recall']:.2f} "
                  f"noise_selected={row['noise_selected']}")

    df = pd.DataFrame(rows)
    print("\n=== summary (mean over seeds) ===")
    print(df.groupby(["bed", "arm"])[["wall", "n_selected", "weak_recall", "noise_selected"]].mean().to_string())


if __name__ == "__main__":
    main()
