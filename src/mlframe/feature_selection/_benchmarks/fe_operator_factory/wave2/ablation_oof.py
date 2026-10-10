"""Out-of-fold versus in-sample table values for the training rows of operator B (2-D cell table), at small n and with weak or no shrinkage: does cross-fitting matter downstream?

A model is trained on the fit half with the engineered column computed out of fold or in sample, and scored on the held-out half where the column comes from the full table
(``oof`` and ``insample``) or from the average of the k fold tables (``oof_foldavg``).
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.wave2.ablation_oof``. Output: ``wave2/results/ablation_oof.json``.
"""

from __future__ import annotations

import json

import numpy as np

from ..brainstorm.h import MI, ybins
from ..common._paths import results_dir
from .harness import leak_ablation, prep
from .targets import CASES
from .warp_service import oof_fit, replay


def run(case: str, n: int, m: float, K: int, seeds: int = 8) -> dict:
    """Mean relative MAE / RMSE improvement (ridge, HGB) of the out-of-fold and in-sample variants over ``seeds`` seeds."""
    gen, _ = CASES["B"][case]
    acc = []
    for s in range(seeds):
        X, y, _ = gen(np.random.default_rng(900 + s), n)
        h = n // 2
        XA, XB, ya, yB = X[:h], X[h:], y[:h], y[h:]
        fa, rec = oof_fit("cell2d", XA, ya, [0, 1], seed=s, K=K, m=m)
        fin = replay(rec, XA)
        fb = replay(rec, XB)
        fb_avg = replay(rec, XB, mode="foldavg")
        ab = {
            "raw": (XA, XB),
            "oof": (np.column_stack([XA, prep(fa, fa)]), np.column_stack([XB, prep(fa, fb)])),
            "oof_foldavg": (np.column_stack([XA, prep(fa, fa)]), np.column_stack([XB, prep(fa, fb_avg)])),
            "insample": (np.column_stack([XA, prep(fin, fin)]), np.column_stack([XB, prep(fa, fb)])),
        }
        yb = ybins(ya, yB)
        d = leak_ablation(ab, ya, yB)
        d["mi|train_insample"] = MI(fin, fb, yb)[0]
        d["mi|train_oof"] = MI(fa, fb, yb)[0]
        d["mi|heldout_full_table"] = MI(fa, fb, yb)[1]
        acc.append(d)
    keys = [k for k in acc[0] if not k.endswith("|raw")]
    return {k: [float(np.mean([a[k] for a in acc])), float(np.std([a[k] for a in acc]))] for k in keys}


def main() -> None:
    """Run the grid (case x n x shrinkage) and write the json."""
    out = {}
    for case in ("W", "H"):
        for n in (1500, 4000):
            for m in (0.0, 2.0, 5.0, 10.0, 20.0):
                out[f"B_{case}|n={n}|m={m}|K=10"] = r = run(case, n, m, 10)
                print(case, n, m, {k: f"{v[0]:+.3f}" for k, v in r.items() if k.startswith(("mae", "mi"))}, flush=True)
    (results_dir("wave2") / "ablation_oof.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
