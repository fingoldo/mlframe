"""WIDE-FRAME validation bench that GATES the OFF->ON default flip of pairwise integer-lattice FE (gcd / lcm / and).

Mirrors ``bench_pairwise_modular_wideframe`` for the integer-lattice sibling. Three things this proves so the default
can go ON safely:

1. COST -- the ADDED wall-time of the enabled lattice scan as a fraction of a FULL MRMR fit, swept over integer-eligible
   column count p in {15, 30, 31, 50} and n in {2000, 20000}. p>=31 must hit the max_int_cols=30 budget skip -> ~0 added
   cost. The sweep is pairs-only (cheaper than the modular pairs+triples sweep). Reports added-seconds AND added-% per shape.

2. FALSE-POSITIVE at SCALE -- on a WIDE frame of pure-noise + ordinary-smooth integer columns (NO lattice structure) at
   p=30 (the budget edge), the enabled scan must inject ZERO ``il_`` features over 3 seeds. Hard gate.

3. SIGNAL PRESERVED under realistic noise -- a frame with a real gcd-shared-factor signal PLUS 25 noise int columns still
   detects+selects the gcd column with the default-ON settings, noise rejected.

Run: ``CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 PYTHONPATH=src python -m
mlframe.feature_selection._benchmarks.bench_integer_lattice_wideframe``
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "_results"
P_GRID = (15, 30, 31, 50)
N_GRID = (2000, 20000)


def _noise_frame(p: int, n: int, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Pure-noise + ordinary-smooth integer columns, NO lattice structure. y is a smooth linear threshold so the
    lattice scan must stay silent (no shared-factor / flag co-occurrence structure)."""
    rng = np.random.default_rng(seed)
    cols = {f"c{i}": rng.integers(0, 100, n) for i in range(p)}
    X = pd.DataFrame(cols)
    y = ((X["c0"] + 0.7 * X["c1"]) > 85).astype(int).to_numpy()
    return X, y


def _full_fit_seconds(X, y, *, enable: bool):
    from mlframe.feature_selection.filters.mrmr import MRMR

    t0 = time.perf_counter()
    m = MRMR(fe_integer_lattice_enable=enable, fe_pairwise_modular_enable=False, max_runtime_mins=1)
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
            il = list(getattr(m_on, "integer_lattice_features_", []) or [])
            rows.append({
                "p": p, "n": n,
                "off_fit_s": round(off_s, 3), "on_fit_s": round(on_s, 3),
                "added_s": round(added_s, 3), "added_pct_of_fit": round(added_pct, 2),
                "n_il_injected": len(il),
                "budget_skip_expected": p > 30,
            })
            print(f"  n={n:6d} p={p:4d}  off={off_s:7.3f}s  on={on_s:7.3f}s  added={added_s:+7.3f}s "
                  f"({added_pct:+6.2f}%)  il_injected={len(il)}  skip_expected={p > 30}")
    return rows


def _fp_at_scale() -> dict:
    """HARD GATE: pure-noise wide frame at p=30 (budget edge) must inject ZERO il features."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    results = []
    ok = True
    for seed in (0, 1, 2):
        X, y = _noise_frame(30, 4000, seed=seed)
        m = MRMR(fe_integer_lattice_enable=True, fe_pairwise_modular_enable=False, max_runtime_mins=1)
        m.fit(X, pd.Series(y, name="y"))
        il = list(getattr(m, "integer_lattice_features_", []) or [])
        out = m.transform(X.iloc[:300])
        il_out = [c for c in out.columns if str(c).startswith("il_")]
        n_fp = max(len(il), len(il_out))
        ok = ok and (n_fp == 0)
        results.append({"seed": seed, "p": 30, "n": 4000, "n_il": n_fp, "il_cols": il_out})
        print(f"  FP@scale seed={seed} p=30 n=4000 -> il injected={n_fp}  {'OK' if n_fp == 0 else 'FAIL'}")
    return {"zero_fp_at_p30": ok, "runs": results}


def _mixed_realism() -> dict:
    """Real gcd-shared-factor signal + 25 noise int columns -> gcd column still detected+selected with default-ON."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    runs = []
    all_ok = True
    for seed in (1, 7, 42):
        rng = np.random.default_rng(seed)
        n = 4000
        a = rng.integers(1, 60, n)
        b = rng.integers(1, 60, n)
        y = (np.gcd(a, b) >= 3).astype(int)
        cols = {"a": a, "b": b}
        for i in range(25):
            cols[f"noise{i}"] = rng.integers(0, 100, n)
        X = pd.DataFrame(cols)
        m = MRMR(fe_integer_lattice_enable=True, fe_pairwise_modular_enable=False, max_runtime_mins=1)
        m.fit(X, pd.Series(y, name="y"))
        out = m.transform(X.iloc[:400])
        il_cols = [c for c in out.columns if str(c).startswith("il_")]
        caught = len(il_cols) >= 1
        all_ok = all_ok and caught
        runs.append({"seed": seed, "n_il_selected": len(il_cols), "il_cols": il_cols, "caught": caught})
        print(f"  mixed-realism seed={seed} gcd(a,b)>=3 + 25 noise -> il selected={len(il_cols)}  " f"{'CAUGHT' if caught else 'MISSED'}")
    return {"real_signal_caught_with_25_noise": all_ok, "runs": runs}


def main():
    print("== COST table (added-s + added-% of full MRMR fit) ==")
    cost = _cost_table()
    print("\n== FALSE-POSITIVE at scale (p=30, must be 0) ==")
    fp = _fp_at_scale()
    print("\n== MIXED-REALISM (real gcd signal + 25 noise) ==")
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
    out_path = RESULTS_DIR / f"integer_lattice_wideframe_{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\ngate_verdict_measured_safe={verdict_safe}\nwrote {out_path}")
    return results


if __name__ == "__main__":
    main()
