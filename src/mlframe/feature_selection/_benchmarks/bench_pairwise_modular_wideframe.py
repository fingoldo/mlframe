"""WIDE-FRAME validation bench that GATES the OFF->ON default flip of pairwise/n-way modular FE.

Three things this proves so the default can go ON safely:

1. COST -- the ADDED wall-time of the enabled modular scan as a fraction of a FULL MRMR fit, swept over
   integer-eligible column count p in {5, 15, 30, 31, 50, 100} and n in {2000, 20000}. p>=31 must hit the
   max_int_cols=30 budget skip -> ~0 added cost. Reports added-seconds AND added-% of the full fit per shape.

2. FALSE-POSITIVE at SCALE -- on a WIDE frame of pure-noise + ordinary-smooth integer columns (NO modular
   structure) at p=30 (the budget edge), the enabled scan must inject ZERO ``pmod_`` features. If any spurious
   modular feature appears, the default cannot go ON; this is the hard gate.

3. SIGNAL PRESERVED under realistic noise -- a frame with a real (a+b) mod 7 signal PLUS 25 noise int columns
   still detects+selects the modular residue with the default-ON settings, noise rejected.

Run: ``CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 PYTHONPATH=src python -m
mlframe.feature_selection._benchmarks.bench_pairwise_modular_wideframe``
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "_results"
P_GRID = (5, 15, 30, 31, 50, 100)
N_GRID = (2000, 20000)


def _noise_frame(p: int, n: int, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Pure-noise + ordinary-smooth integer columns, NO modular structure. y is a smooth linear
    threshold of two columns (the kind a poly/Fourier leg handles) so the modular scan must stay silent."""
    rng = np.random.default_rng(seed)
    cols = {f"c{i}": rng.integers(0, 100, n) for i in range(p)}
    X = pd.DataFrame(cols)
    y = ((X["c0"] + 0.7 * X["c1"]) > 85).astype(int).to_numpy()
    return X, y


def _full_fit_seconds(X, y, *, enable: bool) -> float:
    from mlframe.feature_selection.filters.mrmr import MRMR

    t0 = time.perf_counter()
    m = MRMR(fe_pairwise_modular_enable=enable, max_runtime_mins=1)
    m.fit(X, pd.Series(y, name="y"))
    return time.perf_counter() - t0, m


def _cost_table() -> list[dict]:
    rows = []
    for n in N_GRID:
        for p in P_GRID:
            X, y = _noise_frame(p, n, seed=0)
            off_s, _ = _full_fit_seconds(X, y, enable=False)
            on_s, m_on = _full_fit_seconds(X, y, enable=True)
            added_s = on_s - off_s
            added_pct = 100.0 * added_s / off_s if off_s > 0 else float("nan")
            pmod = list(getattr(m_on, "pairwise_modular_features_", []) or [])
            rows.append({
                "p": p, "n": n,
                "off_fit_s": round(off_s, 3), "on_fit_s": round(on_s, 3),
                "added_s": round(added_s, 3), "added_pct_of_fit": round(added_pct, 2),
                "n_pmod_injected": len(pmod),
                "budget_skip_expected": p > 30,
            })
            print(f"  n={n:6d} p={p:4d}  off={off_s:7.3f}s  on={on_s:7.3f}s  added={added_s:+7.3f}s "
                  f"({added_pct:+6.2f}%)  pmod_injected={len(pmod)}  skip_expected={p > 30}")
    return rows


def _fp_at_scale() -> dict:
    """HARD GATE: pure-noise wide frame at p=30 (budget edge) must inject ZERO pmod features."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    results = []
    ok = True
    for seed in (0, 1, 2):
        X, y = _noise_frame(30, 4000, seed=seed)
        m = MRMR(fe_pairwise_modular_enable=True, max_runtime_mins=1)
        m.fit(X, pd.Series(y, name="y"))
        pmod = list(getattr(m, "pairwise_modular_features_", []) or [])
        out = m.transform(X.iloc[:300])
        pmod_out = [c for c in out.columns if str(c).startswith("pmod_")]
        n_fp = max(len(pmod), len(pmod_out))
        ok = ok and (n_fp == 0)
        results.append({"seed": seed, "p": 30, "n": 4000, "n_pmod": n_fp, "pmod_cols": pmod_out})
        print(f"  FP@scale seed={seed} p=30 n=4000 -> pmod injected={n_fp}  {'OK' if n_fp == 0 else 'FAIL'}")
    return {"zero_fp_at_p30": ok, "runs": results}


def _mixed_realism() -> dict:
    """Real (a+b) mod 7 signal + 25 noise int columns -> modular feature still detected+selected with default-ON."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    runs = []
    all_ok = True
    for seed in (1, 7, 42):
        rng = np.random.default_rng(seed)
        n = 4000
        a = rng.integers(0, 100, n)
        b = rng.integers(0, 100, n)
        y = ((a + b) % 7 >= 3).astype(int)
        cols = {"a": a, "b": b}
        for i in range(25):
            cols[f"noise{i}"] = rng.integers(0, 100, n)
        X = pd.DataFrame(cols)
        m = MRMR(fe_pairwise_modular_enable=True, max_runtime_mins=1)
        m.fit(X, pd.Series(y, name="y"))
        out = m.transform(X.iloc[:400])
        pmod_cols = [c for c in out.columns if str(c).startswith("pmod_")]
        caught = len(pmod_cols) >= 1
        all_ok = all_ok and caught
        runs.append({"seed": seed, "n_pmod_selected": len(pmod_cols), "pmod_cols": pmod_cols, "caught": caught})
        print(f"  mixed-realism seed={seed} (a+b)mod7 + 25 noise -> pmod selected={len(pmod_cols)}  "
              f"{'CAUGHT' if caught else 'MISSED'}")
    return {"real_signal_caught_with_25_noise": all_ok, "runs": runs}


def main():
    print("== COST table (added-s + added-% of full MRMR fit) ==")
    cost = _cost_table()
    print("\n== FALSE-POSITIVE at scale (p=30, must be 0) ==")
    fp = _fp_at_scale()
    print("\n== MIXED-REALISM (real signal + 25 noise) ==")
    mixed = _mixed_realism()

    verdict_safe = bool(fp["zero_fp_at_p30"]) and bool(mixed["real_signal_caught_with_25_noise"])
    results = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "cost_table": cost,
        "fp_at_scale": fp,
        "mixed_realism": mixed,
        "gate_verdict_measured_safe": verdict_safe,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"pairwise_modular_wideframe_{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\ngate_verdict_measured_safe={verdict_safe}\nwrote {out_path}")
    return results


if __name__ == "__main__":
    main()
