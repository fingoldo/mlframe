"""Bench: cheap-first detection of CONDITIONAL-GATE (regime-switch / masked-interaction) and ROW-ARGMAX structure.

Measures whether ``_conditional_gate_fe_proto.scan_conditional_gate`` / ``scan_row_argmax`` recover the engineered column on
their natural targets (gate-select / gate-mask / row-argmax) with the right tau / triple, while staying LOW false-positive on
controls (smooth threshold, noise, ordinary multiplicative interaction). Reports per scenario: fire rate, MI of the recovered
column vs the operand floor (the best a raw / pairwise op could give), and wall time. Writes JSON to ``_results/``.

Run: ``CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 MLFRAME_DISABLE_GPU=1 PYTHONPATH=src python -m
mlframe.feature_selection._benchmarks.bench_conditional_gate_detection``
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._conditional_gate_fe_proto import (
    scan_conditional_gate,
    scan_row_argmax,
)

SEEDS = (1, 7, 13, 42, 101)
RESULTS_DIR = Path(__file__).parent / "_results"


def make_gate_select(seed, n=2000):
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
    sel = np.where(c > 0.0, a, b)
    y = (sel > np.median(sel)).astype(int)
    return pd.DataFrame({"a": a, "b": b, "c": c}), y


def make_gate_mask(seed, n=2000):
    rng = np.random.default_rng(seed)
    a, c = rng.normal(0, 1, n), rng.normal(0, 1, n)
    masked = (c > 0.0).astype(float) * a
    y = (masked > np.median(masked)).astype(int)
    return pd.DataFrame({"a": a, "b": rng.normal(0, 1, n), "c": c}), y


def make_argmax(seed, n=2000):
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
    y = np.argmax(np.stack([a, b, c], axis=1), axis=1)
    return pd.DataFrame({"a": a, "b": b, "c": c}), y


def make_smooth(seed, n=2000):
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
    return pd.DataFrame({"a": a, "b": b, "c": c}), ((a + 0.5 * b) > 0).astype(int)


def make_noise(seed, n=2000):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n), "c": rng.normal(0, 1, n)}), rng.integers(0, 2, n)


def make_ordinary(seed, n=2000):
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
    return pd.DataFrame({"a": a, "b": b, "c": c}), ((a * b) > 0).astype(int)


def _run(name, gen, scanner, is_tp):
    rows = []
    for s in SEEDS:
        X, y = gen(s)
        cols = list(X.columns)
        t0 = time.perf_counter()
        hits = scanner(X, y, cols, rng_seed=s)
        dt = time.perf_counter() - t0
        rec = {"seed": s, "fired": len(hits) > 0, "n_hits": len(hits), "wall_s": round(dt, 4)}
        if hits:
            rec["top_mi"] = round(hits[0]["mi"], 4)
            rec["operand_floor"] = round(hits[0]["operand_floor"], 4)
            rec["lift"] = round(hits[0]["mi"] - hits[0]["operand_floor"], 4)
        rows.append(rec)
    fire = float(np.mean([r["fired"] for r in rows]))
    lifts = [r["lift"] for r in rows if "lift" in r]
    return {"scenario": name, "is_tp": is_tp, "fire_rate": fire,
            "mean_lift": round(float(np.mean(lifts)), 4) if lifts else None,
            "mean_wall_s": round(float(np.mean([r["wall_s"] for r in rows])), 4), "per_seed": rows}


def main():
    results = {"gate": [], "argmax": []}
    results["gate"].append(_run("gate_select", make_gate_select, scan_conditional_gate, True))
    results["gate"].append(_run("gate_mask", make_gate_mask, scan_conditional_gate, True))
    for nm, gen in (("smooth", make_smooth), ("noise", make_noise), ("ordinary_mul", make_ordinary)):
        results["gate"].append(_run(f"gate_ctrl_{nm}", gen, scan_conditional_gate, False))

    results["argmax"].append(_run("argmax_row", make_argmax, scan_row_argmax, True))
    for nm, gen in (("smooth", make_smooth), ("noise", make_noise), ("ordinary_mul", make_ordinary)):
        results["argmax"].append(_run(f"argmax_ctrl_{nm}", gen, scan_row_argmax, False))

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"conditional_gate_detection_{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2))

    for grp in ("gate", "argmax"):
        print(f"=== {grp} ===")
        for r in results[grp]:
            tag = "TP " if r["is_tp"] else "CTL"
            print(f"  {tag} {r['scenario']:18s} fire={r['fire_rate']:.2f} lift={r['mean_lift']} wall={r['mean_wall_s']}s")
    print(f"\nwrote {out_path}")
    return results


if __name__ == "__main__":
    main()
