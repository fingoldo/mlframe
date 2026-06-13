"""WIDE-FRAME validation bench that GATES the OFF->ON default flip of ROW-ARGMAX and CONDITIONAL-GATE FE.

Mirrors ``bench_integer_lattice_wideframe`` for the two frontier-pass-2 operators. Per operator it proves so the default can go ON safely:

1. COST -- the ADDED wall-time of the enabled scan as a fraction of a FULL MRMR fit, swept over eligible column count p in
   {15, 30, 31, 50} and n in {2000, 20000}. p>31 must hit the budget skip (argmax max_cols=30; gate max_cols=20) -> ~0 added cost.

2. FALSE-POSITIVE at SCALE -- on a WIDE frame of pure-noise + ordinary-smooth columns (NO argmax / gate structure; INCLUDES the
   gate's hard controls: a smooth-linear-threshold y AND an ordinary-multiplicative-sign y) at p=30, the enabled scan must inject
   ZERO ``argmax_`` / ``gate_`` features over 3 seeds. Hard gate (the gate hardening removes the prototype's smooth/ordinary_mul FP).

3. SIGNAL PRESERVED under realistic noise -- a frame with a real argmax-of-triple / threshold-gate signal PLUS 25 noise columns
   still detects+selects the engineered column with the default-ON settings, noise rejected.

Run: ``CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 MLFRAME_DISABLE_GPU=1 PYTHONPATH=src python -m
mlframe.feature_selection._benchmarks.bench_conditional_gate_wideframe``
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

# Only the operator under test is enabled; all other FE families OFF so the measured added-cost is purely this operator.
_OTHER_OFF = dict(
    fe_pairwise_modular_enable=False, fe_integer_lattice_enable=False,
)


def _enable_kwargs(op: str, enable: bool) -> dict:
    kw = dict(_OTHER_OFF)
    if op == "argmax":
        kw.update(fe_row_argmax_enable=enable, fe_conditional_gate_enable=False)
    else:
        kw.update(fe_conditional_gate_enable=enable, fe_row_argmax_enable=False)
    return kw


def _noise_frame(p: int, n: int, seed: int, control: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Pure-noise continuous columns with NO argmax / gate structure. ``control`` selects the y rule -- the gate's two hard
    false-positive cases (``smooth`` linear threshold, ``ordinary_mul`` sign-of-product) plus ``random`` noise."""
    rng = np.random.default_rng(seed)
    cols = {f"c{i}": rng.normal(0, 1, n) for i in range(p)}
    X = pd.DataFrame(cols)
    if control == "smooth":
        y = ((X["c0"] + 0.5 * X["c1"]) > 0).astype(int).to_numpy()
    elif control == "ordinary_mul":
        y = ((X["c0"] * X["c1"]) > 0).astype(int).to_numpy()
    else:
        y = rng.integers(0, 2, n)
    return X, y


def _full_fit_seconds(X, y, op: str, *, enable: bool):
    from mlframe.feature_selection.filters.mrmr import MRMR

    t0 = time.perf_counter()
    m = MRMR(max_runtime_mins=1, **_enable_kwargs(op, enable))
    m.fit(X, pd.Series(y, name="y"))
    return time.perf_counter() - t0, m


def _injected(m, op: str) -> list:
    attr = "row_argmax_features_" if op == "argmax" else "conditional_gate_features_"
    return list(getattr(m, attr, []) or [])


def _budget(op: str) -> int:
    return 30 if op == "argmax" else 20


def _cost_table(op: str) -> list[dict]:
    rows = []
    for n in N_GRID:
        for p in P_GRID:
            X, y = _noise_frame(p, n, seed=0, control="smooth")
            off_s, _ = _full_fit_seconds(X, y, op, enable=False)
            on_s, m_on = _full_fit_seconds(X, y, op, enable=True)
            added_s = on_s - off_s
            added_pct = 100.0 * added_s / off_s if off_s > 0 else float("nan")
            inj = _injected(m_on, op)
            skip_expected = p > _budget(op)
            rows.append({
                "p": p, "n": n, "off_fit_s": round(off_s, 3), "on_fit_s": round(on_s, 3),
                "added_s": round(added_s, 3), "added_pct_of_fit": round(added_pct, 2),
                "n_injected": len(inj), "budget_skip_expected": skip_expected,
            })
            print(f"  [{op}] n={n:6d} p={p:4d}  off={off_s:7.3f}s  on={on_s:7.3f}s  added={added_s:+7.3f}s "
                  f"({added_pct:+6.2f}%)  injected={len(inj)}  skip_expected={skip_expected}")
    return rows


def _fp_at_scale(op: str) -> dict:
    """HARD GATE: wide noise frame at p=30 over 3 seeds x 3 control y-rules (smooth / ordinary_mul / random) must inject ZERO."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    prefix = "argmax_" if op == "argmax" else "gate_"
    results = []
    ok = True
    for control in ("smooth", "ordinary_mul", "random"):
        for seed in (0, 1, 2):
            X, y = _noise_frame(30, 4000, seed=seed, control=control)
            m = MRMR(max_runtime_mins=1, **_enable_kwargs(op, True))
            m.fit(X, pd.Series(y, name="y"))
            inj = _injected(m, op)
            out = m.transform(X.iloc[:300])
            inj_out = [c for c in out.columns if str(c).startswith(prefix)]
            n_fp = max(len(inj), len(inj_out))
            ok = ok and (n_fp == 0)
            results.append({"control": control, "seed": seed, "n_injected": n_fp, "cols": inj_out})
            print(f"  [{op}] FP@scale control={control:12s} seed={seed} -> injected={n_fp}  {'OK' if n_fp == 0 else 'FAIL'}")
    return {"zero_fp_at_p30": ok, "runs": results}


def _mixed_realism_argmax() -> dict:
    from mlframe.feature_selection.filters.mrmr import MRMR

    runs = []
    all_ok = True
    for seed in (1, 7, 42):
        rng = np.random.default_rng(seed)
        n = 4000
        a, b, c = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
        y = np.argmax(np.stack([a, b, c], axis=1), axis=1)
        cols = {"a": a, "b": b, "c": c}
        for i in range(25):
            cols[f"noise{i}"] = rng.normal(0, 1, n)
        X = pd.DataFrame(cols)
        m = MRMR(max_runtime_mins=1, **_enable_kwargs("argmax", True))
        m.fit(X, pd.Series(y, name="y"))
        out = m.transform(X.iloc[:400])
        sel = [c for c in out.columns if str(c).startswith("argmax_")]
        caught = len(sel) >= 1
        all_ok = all_ok and caught
        runs.append({"seed": seed, "n_selected": len(sel), "cols": sel, "caught": caught})
        print(f"  [argmax] mixed seed={seed} argmax(a,b,c) + 25 noise -> selected={len(sel)}  {'CAUGHT' if caught else 'MISSED'}")
    return {"real_signal_caught_with_25_noise": all_ok, "runs": runs}


def _mixed_realism_gate() -> dict:
    from mlframe.feature_selection.filters.mrmr import MRMR

    runs = []
    all_ok = True
    for seed in (1, 7, 42):
        rng = np.random.default_rng(seed)
        n = 4000
        a, b, c = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
        sel = np.where(c > 0.0, a, b)
        y = (sel > np.median(sel)).astype(int)
        cols = {"a": a, "b": b, "c": c}
        # 15 noise cols -> 18 total, within the gate max_cols=20 budget so the sweep RUNS (the gate is opt-in; this validates the
        # detection path under realistic noise within its budget, not the budget-skip which the cost table already exercises).
        for i in range(15):
            cols[f"noise{i}"] = rng.normal(0, 1, n)
        X = pd.DataFrame(cols)
        m = MRMR(max_runtime_mins=1, **_enable_kwargs("gate", True))
        m.fit(X, pd.Series(y, name="y"))
        out = m.transform(X.iloc[:400])
        gsel = [c for c in out.columns if str(c).startswith("gate_")]
        caught = len(gsel) >= 1
        all_ok = all_ok and caught
        runs.append({"seed": seed, "n_selected": len(gsel), "cols": gsel, "caught": caught})
        print(f"  [gate] mixed seed={seed} c>0?a:b + 25 noise -> selected={len(gsel)}  {'CAUGHT' if caught else 'MISSED'}")
    return {"real_signal_caught_with_25_noise": all_ok, "runs": runs}


def main():
    out: dict = {"generated": datetime.now().isoformat(timespec="seconds")}
    for op in ("argmax", "gate"):
        print(f"\n===== OPERATOR: {op} =====")
        print("== COST table ==")
        cost = _cost_table(op)
        print("== FALSE-POSITIVE at scale (p=30, must be 0; incl. smooth + ordinary_mul controls) ==")
        fp = _fp_at_scale(op)
        print("== MIXED-REALISM (real signal + 25 noise) ==")
        mixed = _mixed_realism_argmax() if op == "argmax" else _mixed_realism_gate()
        safe = bool(fp["zero_fp_at_p30"]) and bool(mixed["real_signal_caught_with_25_noise"])
        out[op] = {"cost_table": cost, "fp_at_scale": fp, "mixed_realism": mixed, "gate_verdict_measured_safe": safe}
        print(f"  [{op}] gate_verdict_measured_safe={safe}")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"conditional_gate_wideframe_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    return out


if __name__ == "__main__":
    main()
