"""Bench: cheap-first / escalate detection of PAIRWISE & n-way MODULAR relationships.

Measures whether ``_pairwise_modular_fe.detect_pairwise_modular`` (cheap modular scan + permutation
gate + fine-grid escalate) recovers a large MI lift on modular targets ``(a+b) mod m`` / ``(a*b) mod m``
/ n-way parity / single hidden integer period, while staying cheap and LOW false-positive on controls
(smooth, monotone, noise, ordinary multiplicative interaction).

Reports per scenario: detection (did a responded hit fire), the escalated modulus vs the true one, MI
lift of the residue over the raw-combiner baseline (the best a smooth basis could recover), and wall
time. Aggregates TP detection rate, FP rate on controls, and cost. Writes JSON to ``_results/``.

Run: ``CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 PYTHONPATH=src python -m
mlframe.feature_selection._benchmarks.bench_modular_period_detection``
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._pairwise_modular_fe import (
    cheap_modular_scan,
    detect_pairwise_modular,
)

SEEDS = (1, 7, 13, 42, 101)
RESULTS_DIR = Path(__file__).parent / "_results"


def _noise_cols(rng, n, k, lo=0, hi=50):
    return {f"n{i}": rng.integers(lo, hi, n) for i in range(k)}


def make_pair_add_mod(seed, n=2000, m=7):
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 100, n), rng.integers(0, 100, n)
    y = ((a + b) % m >= (m // 2)).astype(int)
    X = pd.DataFrame({"a": a, "b": b, **_noise_cols(rng, n, 2)})
    return X, y, {"op": "sum", "modulus": m, "cols": ("a", "b")}


def make_pair_mul_mod(seed, n=2000, m=5):
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 40, n), rng.integers(0, 40, n)
    y = ((a * b) % m == 0).astype(int)
    X = pd.DataFrame({"a": a, "b": b, **_noise_cols(rng, n, 2)})
    return X, y, {"op": "prod", "modulus": m, "cols": ("a", "b")}


def make_nway_parity(seed, n=2000, k=3):
    rng = np.random.default_rng(seed)
    cols = {f"a{i}": rng.integers(0, 1000, n) for i in range(k)}
    s = np.zeros(n, dtype=np.int64)
    for v in cols.values():
        s += v
    y = (s % 2).astype(int)  # parity of the sum -> (a0+a1+...) mod 2
    X = pd.DataFrame({**cols, **_noise_cols(rng, n, 1)})
    return X, y, {"op": "sum", "modulus": 2, "cols": tuple(sorted(cols)[:2])}


def make_single_hidden_period(seed, n=2000, m=11):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 5000, n)  # large magnitude; signal is in a mod 11 (not a calendar period)
    y = (a % m >= (m // 2)).astype(int)
    X = pd.DataFrame({"a": a, **_noise_cols(rng, n, 2)})
    return X, y, {"op": "self", "modulus": m, "cols": ("a",)}


# ---- controls (must NOT fire) ----
def make_smooth(seed, n=2000):
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 100, n), rng.integers(0, 100, n)
    y = ((a + 0.7 * b) > 85).astype(int)  # smooth monotone threshold
    return pd.DataFrame({"a": a, "b": b, **_noise_cols(rng, n, 2)}), y, None


def make_monotone(seed, n=2000):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 1000, n)
    y = (a > 500).astype(int)
    return pd.DataFrame({"a": a, **_noise_cols(rng, n, 3)}), y, None


def make_noise(seed, n=2000):
    rng = np.random.default_rng(seed)
    cols = _noise_cols(rng, n, 4)
    y = rng.integers(0, 2, n)
    return pd.DataFrame(cols), y, None


def make_ordinary_interaction(seed, n=2000):
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 50, n), rng.integers(0, 50, n)
    y = ((a * b) > 600).astype(int)  # multiplicative threshold, NOT modular
    return pd.DataFrame({"a": a, "b": b, **_noise_cols(rng, n, 2)}), y, None


TP_SCENARIOS = {
    "pair_add_mod7": make_pair_add_mod,
    "pair_mul_mod5": make_pair_mul_mod,
    "nway_parity3": make_nway_parity,
    "single_hidden_period11": make_single_hidden_period,
}
CONTROL_SCENARIOS = {
    "smooth_threshold": make_smooth,
    "monotone_step": make_monotone,
    "pure_noise": make_noise,
    "ordinary_mul_interaction": make_ordinary_interaction,
}


def _run_scenario(name, gen, is_tp):
    per_seed = []
    for s in SEEDS:
        X, y, truth = gen(s)
        t0 = time.perf_counter()
        hits = detect_pairwise_modular(X, y, top_k=4, seed=s)
        dt = time.perf_counter() - t0

        # cost reference: a plain cheap scan with no escalate (the "plain FE" overhead).
        t1 = time.perf_counter()
        _ = cheap_modular_scan(X, y, seed=s)
        scan_dt = time.perf_counter() - t1

        fired = len(hits) > 0
        rec = {"seed": s, "fired": fired, "wall_s": round(dt, 4),
               "cheap_scan_s": round(scan_dt, 4), "n_hits": len(hits)}
        if hits:
            top = hits[0]
            rec.update({"detected_modulus": top["modulus"], "detected_op": top["op"],
                        "residue_mi": round(top["residue_mi"], 4),
                        "baseline_mi": round(top["baseline_mi"], 4),
                        "mi_lift": round(top["margin"], 4)})
        if is_tp and truth is not None:
            rec["true_modulus"] = truth["modulus"]
            rec["modulus_correct"] = bool(
                hits and (hits[0]["modulus"] == truth["modulus"]
                          or hits[0]["modulus"] % truth["modulus"] == 0))
        per_seed.append(rec)

    fire_rate = float(np.mean([r["fired"] for r in per_seed]))
    lifts = [r["mi_lift"] for r in per_seed if "mi_lift" in r]
    out = {"scenario": name, "is_tp": is_tp, "fire_rate": fire_rate,
           "mean_mi_lift": round(float(np.mean(lifts)), 4) if lifts else None,
           "mean_wall_s": round(float(np.mean([r["wall_s"] for r in per_seed])), 4),
           "mean_cheap_scan_s": round(float(np.mean([r["cheap_scan_s"] for r in per_seed])), 4),
           "per_seed": per_seed}
    if is_tp:
        corr = [r.get("modulus_correct", False) for r in per_seed]
        out["modulus_accuracy"] = float(np.mean(corr))
    return out


def main():
    results = {"tp": [], "control": []}
    for name, gen in TP_SCENARIOS.items():
        results["tp"].append(_run_scenario(name, gen, is_tp=True))
    for name, gen in CONTROL_SCENARIOS.items():
        results["control"].append(_run_scenario(name, gen, is_tp=False))

    tp_detect = float(np.mean([r["fire_rate"] for r in results["tp"]]))
    fp_rate = float(np.mean([r["fire_rate"] for r in results["control"]]))
    tp_lifts = [r["mean_mi_lift"] for r in results["tp"] if r["mean_mi_lift"] is not None]
    mod_acc = float(np.mean([r["modulus_accuracy"] for r in results["tp"]]))
    summary = {
        "tp_detection_rate": round(tp_detect, 4),
        "control_false_positive_rate": round(fp_rate, 4),
        "mean_tp_mi_lift": round(float(np.mean(tp_lifts)), 4) if tp_lifts else None,
        "modulus_accuracy": round(mod_acc, 4),
        "mean_detect_wall_s": round(float(np.mean(
            [r["mean_wall_s"] for r in results["tp"] + results["control"]])), 4),
    }
    results["summary"] = summary

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"modular_period_detection_{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(json.dumps(summary, indent=2))
    print("\nTP scenarios:")
    for r in results["tp"]:
        print(f"  {r['scenario']:28s} fire={r['fire_rate']:.2f} lift={r['mean_mi_lift']} "
              f"mod_acc={r['modulus_accuracy']:.2f} wall={r['mean_wall_s']}s")
    print("Control scenarios:")
    for r in results["control"]:
        print(f"  {r['scenario']:28s} fire={r['fire_rate']:.2f} wall={r['mean_wall_s']}s")
    print(f"\nwrote {out_path}")
    return results


if __name__ == "__main__":
    main()
