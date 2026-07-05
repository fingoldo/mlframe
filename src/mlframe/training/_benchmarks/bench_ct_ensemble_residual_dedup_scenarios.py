"""Bench: residual-correlation dedup before cross-target ensembling -- multi-scenario sweep.

Extends ``bench_ct_ensemble_residual_dedup`` (which tests ONLY the NNLS-stack + near-duplicate-of-strongest case) with
two more scenarios so the default-flip decision rests on more than a single regime:

- ``nnls_dup``: NNLS stack, several near-duplicates of the strongest member (the original bench's case).
- ``mean_dup``: uniform-weight (``mean`` strategy) ensemble, same redundant pool. Uniform averaging is where redundant
  members SHOULD hurt most -- a redundant cluster of size d gets d/K of the average weight regardless of merit, so
  dedup has the clearest mechanical reason to win here.
- ``nnls_tight``: NNLS stack with MORE + TIGHTER duplicates (jitter 0.005, n_dup=8) -- stress the redundancy.

Decision rule (project policy): flip ``ct_ensemble_dedup_enabled`` ON only if dedup wins (lower test RMSE) on the
MAJORITY of seeds in the MAJORITY of scenarios. Otherwise keep default OFF; the knob stays in prod either way.

Usage::

    python -m mlframe.training._benchmarks.bench_ct_ensemble_residual_dedup_scenarios
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
from scipy.optimize import nnls

from mlframe.training.composite import residual_dedup_indices


def _nnls_stack_test_rmse(oof_preds, y_oof, test_preds, y_test):
    w, _ = nnls(oof_preds, y_oof)
    pred = test_preds @ w
    return float(np.sqrt(np.mean((pred - y_test) ** 2)))


def _mean_test_rmse(test_preds, y_test):
    pred = test_preds.mean(axis=1)
    return float(np.sqrt(np.mean((pred - y_test) ** 2)))


def _make_pool(seed: int, n: int, n_dup: int, jitter: float):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 2.0 * x + 1.5 * z + rng.normal(scale=1.0, size=n)
    m_x = 2.0 * x + rng.normal(scale=0.8, size=n)
    m_z = 1.5 * z + rng.normal(scale=0.8, size=n)
    m_both = 1.8 * x + 1.3 * z + rng.normal(scale=1.2, size=n)
    members = [m_x, m_z, m_both]
    for _ in range(n_dup):
        members.append(m_both + rng.normal(scale=jitter, size=n))
    return np.column_stack(members), y


def _split(M, y, frac=0.5):
    cut = int(M.shape[0] * frac)
    return M[:cut], y[:cut], M[cut:], y[cut:]


_SCENARIOS = {
    "nnls_dup": dict(strategy="nnls", n_dup=4, jitter=0.02),
    "mean_dup": dict(strategy="mean", n_dup=4, jitter=0.02),
    "nnls_tight": dict(strategy="nnls", n_dup=8, jitter=0.005),
}


def _bench_seed(seed: int, *, strategy: str, n_dup: int, jitter: float, corr_threshold: float = 0.95) -> dict:
    M, y = _make_pool(seed, n=4000, n_dup=n_dup, jitter=jitter)
    M_oof, y_oof, M_test, y_test = _split(M, y)
    oof_rmses = np.sqrt(np.mean((M_oof - y_oof[:, None]) ** 2, axis=0))
    resid = M_oof - y_oof[:, None]
    keep, drop = residual_dedup_indices(resid, oof_rmses, corr_threshold=corr_threshold)
    if strategy == "nnls":
        rmse_full = _nnls_stack_test_rmse(M_oof, y_oof, M_test, y_test)
        rmse_dedup = _nnls_stack_test_rmse(M_oof[:, keep], y_oof, M_test[:, keep], y_test)
    else:
        rmse_full = _mean_test_rmse(M_test, y_test)
        rmse_dedup = _mean_test_rmse(M_test[:, keep], y_test)
    return {"seed": seed, "rmse_full": rmse_full, "rmse_dedup": rmse_dedup, "n_dropped": len(drop), "dedup_wins": rmse_dedup < rmse_full}


def main() -> None:
    seeds = list(range(10))
    out_scenarios = {}
    scen_verdicts = []
    print("CT-ensemble residual-dedup MULTI-SCENARIO bench (test RMSE, full pool vs dedup)\n")
    for name, cfg in _SCENARIOS.items():
        rows = [_bench_seed(s, **cfg) for s in seeds]
        wins = sum(r["dedup_wins"] for r in rows)
        mean_full = float(np.mean([r["rmse_full"] for r in rows]))
        mean_dedup = float(np.mean([r["rmse_dedup"] for r in rows]))
        verdict = "ON" if wins > len(seeds) / 2 else "OFF"
        scen_verdicts.append(verdict)
        print(f"## scenario={name} ({cfg})")
        print("| seed | rmse_full | rmse_dedup | dropped | dedup_wins |")
        print("|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['seed']} | {r['rmse_full']:.5f} | {r['rmse_dedup']:.5f} | {r['n_dropped']} | {r['dedup_wins']} |")
        print(f"mean rmse_full={mean_full:.5f}  mean rmse_dedup={mean_dedup:.5f}  dedup wins {wins}/{len(seeds)} -> {verdict}\n")
        out_scenarios[name] = {"cfg": cfg, "rows": rows, "mean_rmse_full": mean_full, "mean_rmse_dedup": mean_dedup, "dedup_wins": wins, "verdict": verdict}
    n_on = sum(v == "ON" for v in scen_verdicts)
    overall = "ON" if n_on > len(scen_verdicts) / 2 else "OFF"
    print(f"OVERALL DECISION: default ct_ensemble_dedup_enabled = {overall} "
          f"(majority-of-scenarios rule; {n_on}/{len(scen_verdicts)} scenarios ON).")
    out = {"bench": "ct_ensemble_residual_dedup_scenarios",
           "timestamp": datetime.now().isoformat(timespec="seconds"),
           "seeds": seeds, "scenarios": out_scenarios, "decision_default": overall}
    _dir = os.path.join(os.path.dirname(__file__), "_results")
    os.makedirs(_dir, exist_ok=True)
    _path = os.path.join(_dir, "bench_ct_ensemble_residual_dedup_scenarios.json")
    with open(_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"wrote {_path}")


if __name__ == "__main__":
    main()
