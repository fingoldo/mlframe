"""Wall time of the wave-2 operators at n = 100k and 300k (CPU, single process) versus the existing 1734-combo pair table; 1M is a linear extrapolation printed next to the measurement.

Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.wave2.cost``. Output: ``wave2/results/cost.json``.
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np

from ..brainstorm.h import UN, _best, clean, ybins
from ..brainstorm.ops_M import resid_oof
from ..common._paths import results_dir
from .rowstats import fit_rowstat, replay_rowstat
from ..brainstorm.ops_M import screen
from .run_M import _qcodes, all_pair_scores, cell_screen
from .warp_service import oof_fit, replay


def timed(f, *a, **k) -> float:
    """Wall time of one call."""
    t = time.perf_counter()
    f(*a, **k)
    return time.perf_counter() - t


def measure(n: int) -> dict:
    """Seconds per component at ``n`` rows (p = 8 uniform columns)."""
    rng = np.random.default_rng(0)
    X = rng.random((n, 8))
    y = X[:, 0] * X[:, 1] + np.sin(6 * X[:, 2]) + 0.1 * rng.standard_normal(n)
    yb = ybins(y, y)
    UA = np.array([clean(UN[k](X[:, 0])) for k in UN])
    UB = np.array([clean(UN[k](X[:, 1])) for k in UN])
    _best(UA[:, :100], UB[:, :100], yb[0][:100], yb[2])
    out = {"n": n, "pair_table_1_pair": timed(_best, UA, UB, yb[0], yb[2])}
    out["pair_table_28_pairs_extrap"] = 28 * out["pair_table_1_pair"]
    out["B_1_pair_K6_K10"] = timed(oof_fit, "cell2d", X, y, [0, 1], K=6) + timed(oof_fit, "cell2d", X, y, [0, 1], K=10)
    _, rec = oof_fit("cell2d", X, y, [0, 1], K=10)
    out["B_replay_full"] = timed(replay, rec, X)
    out["C_1_col"] = timed(oof_fit, "warp1d", X, y, [0], nb=20)
    out["C_8_cols"] = sum(timed(oof_fit, "warp1d", X, y, [j], nb=20) for j in range(8))
    fit_rowstat(X[:3000], y[:3000], list(range(8)))
    out["G_fit_p8_scan20k"] = timed(fit_rowstat, X, y, list(range(8)))
    rec_g = fit_rowstat(X, y, list(range(8)))
    out["G_replay"] = timed(replay_rowstat, rec_g, X)
    pairs = list(itertools.combinations(range(8), 2))
    r = np.empty(n)
    out["M_resid_oof_p8"] = timed(lambda: r.__setitem__(slice(None), resid_oof(X, y, rng)))
    out["M_screens_28_pairs_all4"] = timed(all_pair_scores, X, y, r, pairs)
    Q = np.column_stack([_qcodes(X[:, j], 6) for j in range(8)])
    out["M_cell_screen_28_pairs_one_target"] = timed(cell_screen, Q, r, pairs)
    out["M_qcodes_8_cols"] = timed(lambda: [_qcodes(X[:, j], 6) for j in range(8)])
    out["M_preset_screen_28_pairs_one_target"] = timed(screen, X, r, pairs)
    return out


def main() -> None:
    """Measure at 100k and 300k and write the json."""
    res = [measure(n) for n in (100_000, 300_000)]
    for r in res:
        print(json.dumps({k: round(v, 3) for k, v in r.items()}))
    (results_dir("wave2") / "cost.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
