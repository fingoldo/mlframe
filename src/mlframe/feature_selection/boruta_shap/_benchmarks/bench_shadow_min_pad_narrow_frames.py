"""Bench: does the canonical >=5 shadow pad help / hurt / stay neutral on NARROW frames? (B7, audit 2026-06-22)

The Boruta paper extends the system by at least 5 shadow attributes even when the real frame is narrower, so the
per-trial shadow-importance MAX (the gate threshold) is not estimated from 1-2 noisy draws. The shipped code made
exactly one shadow per real column (no pad) -- a thin null on 1-2 feature inputs. Changing the shadow count is
SELECTION-ALTERING, so per mlframe/CLAUDE.md it must be bench-gated before any default flip.

This multi-seed bench compares selection WITH the pad (``shadow_min_pad=5``, the new default) vs WITHOUT
(``shadow_min_pad=0``, legacy) on narrow synthetic frames where a known subset of columns is informative and the
rest are pure noise. Metric: per-config mean Jaccard of the accepted set vs the GROUND-TRUTH informative set
(higher = better), plus the false-accept rate on the noise columns (lower = better). A config "wins" if it recovers
more truth and/or admits fewer noise columns, averaged over seeds.

Run:  CUDA_VISIBLE_DEVICES="" python bench_shadow_min_pad_narrow_frames.py
Writes JSON to _results/shadow_min_pad_narrow_frames.json next to this file.

VERDICT (2026-06-22, 12 seeds, n=600, n_trials=40, gini): pad5 is neutral-or-better -- wins=1 / losses=0 /
neutral=3 on truth-recovery Jaccard. On the (n_cols=2, n_informative=1) narrow frame pad5 raised mean truth
Jaccard 0.9167 -> 0.9583 (+0.0417) AND halved the noise false-accept rate 0.1667 -> 0.0833 (-0.0833); the (1,1),
(3,2), (4,2) shapes were bit-for-bit neutral. Per mlframe/CLAUDE.md "enable corrective by default", the canonical
>=5 pad is the new DEFAULT (shadow_min_pad=5); legacy one-shadow-per-column stays available via shadow_min_pad=0.
Re-run this file to refresh; results in _results/shadow_min_pad_narrow_frames.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from mlframe.feature_selection.boruta_shap import BorutaShap


def _make_narrow_frame(n_cols: int, n_informative: int, n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_cols))
    cols = [f"f{i}" for i in range(n_cols)]
    informative = cols[:n_informative]
    # signal = sum of the informative columns + small noise -> a clean separable classification target
    signal = X[:, :n_informative].sum(axis=1)
    y = (signal + 0.25 * rng.standard_normal(n) > 0).astype(int)
    return pd.DataFrame(X, columns=cols), pd.Series(y), set(informative), set(cols) - set(informative)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 1.0


def run(seeds=range(12), configs=((1, 1), (2, 1), (3, 2), (4, 2)), n=600, n_trials=40):
    """configs: list of (n_cols, n_informative) narrow-frame shapes."""
    results = []
    for n_cols, n_inf in configs:
        for pad, label in ((0, "no_pad"), (5, "pad5")):
            recalls, false_accepts = [], []
            for seed in seeds:
                X, y, truth, noise = _make_narrow_frame(n_cols, n_inf, n, seed)
                bs = BorutaShap(
                    importance_measure="gini", classification=True, n_trials=n_trials,
                    random_state=seed, verbose=False, shadow_min_pad=pad,
                )
                bs.fit(X, y)
                accepted = set(bs.accepted)
                recalls.append(_jaccard(accepted, truth))
                false_accepts.append(len(accepted & noise) / max(1, len(noise)))
            results.append({
                "n_cols": n_cols, "n_informative": n_inf, "config": label, "shadow_min_pad": pad,
                "mean_truth_jaccard": float(np.mean(recalls)), "mean_noise_false_accept": float(np.mean(false_accepts)),
                "n_seeds": len(list(seeds)),
            })
    return results


def main():
    res = run()
    out_dir = Path(__file__).parent / "_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "shadow_min_pad_narrow_frames.json"
    out_path.write_text(json.dumps(res, indent=2, sort_keys=True))

    print("shadow_min_pad bench (narrow frames; higher truth_jaccard better, lower noise_false_accept better)")
    print(f"{'config':>10} {'pad':>4} {'truth_jacc':>11} {'noise_FA':>9}")
    for r in res:
        print(f"{r['config']:>10} {r['shadow_min_pad']:>4} {r['mean_truth_jaccard']:>11.4f} {r['mean_noise_false_accept']:>9.4f}")
    # per-shape pad-vs-nopad delta
    by_shape = {}
    for r in res:
        by_shape.setdefault((r["n_cols"], r["n_informative"]), {})[r["shadow_min_pad"]] = r
    print("\nper-shape delta (pad5 - no_pad):")
    n_truth_wins = n_truth_losses = 0
    for shape, d in by_shape.items():
        dt = d[5]["mean_truth_jaccard"] - d[0]["mean_truth_jaccard"]
        df = d[5]["mean_noise_false_accept"] - d[0]["mean_noise_false_accept"]
        if dt > 1e-9:
            n_truth_wins += 1
        elif dt < -1e-9:
            n_truth_losses += 1
        print(f"  shape {shape}: d_truth_jaccard={dt:+.4f}  d_noise_FA={df:+.4f}")
    print(f"\npad5 truth-recovery: wins={n_truth_wins} losses={n_truth_losses} neutral={len(by_shape) - n_truth_wins - n_truth_losses}")
    print(f"results -> {out_path}")


if __name__ == "__main__":
    main()
