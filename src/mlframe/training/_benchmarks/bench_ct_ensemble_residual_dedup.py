"""Bench: residual-correlation dedup before cross-target stacking (A3-07).

Question: does dropping near-duplicate ensemble members (|honest-OOF residual corr| > 0.95, keeping the lower-RMSE
one) before the NNLS stack improve test RMSE on a pool that contains redundant near-duplicate components?

Synthetic: y = signal + noise; a small set of genuinely diverse base predictors PLUS several near-duplicates of the
strongest predictor (its preds + tiny jitter). The duplicates split the NNLS weight among themselves and let a
redundant cluster dominate; dedup should remove them and let the diverse members carry the stack.

Decision rule (project policy): flip ``ct_ensemble_dedup_enabled`` ON only if dedup wins (lower test RMSE) on the
MAJORITY of seeds. Otherwise keep default OFF; the knob stays in prod either way.

Usage::

    python -m mlframe.training._benchmarks.bench_ct_ensemble_residual_dedup
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

import numpy as np
from scipy.optimize import nnls

from mlframe.training.composite import residual_dedup_indices


def _nnls_stack_test_rmse(oof_preds, y_oof, test_preds, y_test):
    """Fit raw NNLS on OOF preds, evaluate on test preds (no renorm -- matches deploy)."""
    w, _ = nnls(oof_preds, y_oof)
    pred = test_preds @ w
    return float(np.sqrt(np.mean((pred - y_test) ** 2)))


def _make_pool(seed: int, n: int = 4000, n_dup: int = 4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 2.0 * x + 1.5 * z + rng.normal(scale=1.0, size=n)
    # Diverse members: each captures part of the signal with its own error.
    m_x = 2.0 * x + rng.normal(scale=0.8, size=n)
    m_z = 1.5 * z + rng.normal(scale=0.8, size=n)
    m_both = 1.8 * x + 1.3 * z + rng.normal(scale=1.2, size=n)
    members = [m_x, m_z, m_both]
    # Near-duplicates of the strongest member (m_both): tiny jitter -> residuals ~ identical.
    for _ in range(n_dup):
        members.append(m_both + rng.normal(scale=0.02, size=n))
    M = np.column_stack(members)
    return M, y


def _split(M, y, frac=0.5):
    n = M.shape[0]
    cut = int(n * frac)
    return M[:cut], y[:cut], M[cut:], y[cut:]


def _bench_seed(seed: int, corr_threshold: float = 0.95) -> dict:
    M, y = _make_pool(seed)
    M_oof, y_oof, M_test, y_test = _split(M, y)
    # Per-member OOF RMSE (used as the dedup tiebreak).
    oof_rmses = np.sqrt(np.mean((M_oof - y_oof[:, None]) ** 2, axis=0))

    rmse_full = _nnls_stack_test_rmse(M_oof, y_oof, M_test, y_test)

    resid = M_oof - y_oof[:, None]
    keep, drop = residual_dedup_indices(resid, oof_rmses, corr_threshold=corr_threshold)
    rmse_dedup = _nnls_stack_test_rmse(M_oof[:, keep], y_oof, M_test[:, keep], y_test)
    return {
        "seed": seed,
        "rmse_full": rmse_full,
        "rmse_dedup": rmse_dedup,
        "n_dropped": len(drop),
        "dedup_wins": rmse_dedup < rmse_full,
    }


def main() -> None:
    """Benchmark the residual-correlation diversity-dedup gate against the full cross-target stack across seeds; writes the OFF/ON verdict JSON to _results/."""
    seeds = list(range(10))
    rows = [_bench_seed(s) for s in seeds]
    wins = sum(r["dedup_wins"] for r in rows)
    mean_full = float(np.mean([r["rmse_full"] for r in rows]))
    mean_dedup = float(np.mean([r["rmse_dedup"] for r in rows]))

    print("CT-ensemble residual-dedup bench (NNLS stack test RMSE, full pool vs dedup)\n")
    print("| seed | rmse_full | rmse_dedup | dropped | dedup_wins |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['seed']} | {r['rmse_full']:.4f} | {r['rmse_dedup']:.4f} | " f"{r['n_dropped']} | {r['dedup_wins']} |")
    print(f"\nmean rmse_full={mean_full:.4f}  mean rmse_dedup={mean_dedup:.4f}  " f"dedup wins {wins}/{len(seeds)} seeds")
    verdict = "ON" if wins > len(seeds) / 2 else "OFF"
    print(f"DECISION: default ct_ensemble_dedup_enabled = {verdict} " f"(majority-of-seeds rule).")

    out = {
        "bench": "ct_ensemble_residual_dedup",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seeds": seeds,
        "rows": rows,
        "mean_rmse_full": mean_full,
        "mean_rmse_dedup": mean_dedup,
        "dedup_wins": wins,
        "n_seeds": len(seeds),
        "decision_default": verdict,
    }
    _dir = os.path.join(os.path.dirname(__file__), "_results")
    os.makedirs(_dir, exist_ok=True)
    _path = os.path.join(_dir, "bench_ct_ensemble_residual_dedup.json")
    with open(_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"wrote {_path}")


if __name__ == "__main__":
    main()
