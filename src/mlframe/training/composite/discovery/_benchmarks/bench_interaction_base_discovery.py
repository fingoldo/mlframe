"""Bench: interaction-aware base-pair discovery on a PURE-interaction DGP.

DGP (no main effects): ``y = a * b + noise`` with ``a, b ~ N(0,1)`` independent.
Here ``MI(y, a) ~ MI(y, b) ~ 0`` (each base alone is uninformative -- the product
is symmetric in sign), but ``MI(y, a*b)`` is large. A single/additive base spec
(``linear_residual`` on the best single base) cannot capture the structure; the
synthetic interaction base ``a__mul__b`` can.

What this measures
------------------
1. SCORER: ``score_interaction_pairs`` ranks ``a__mul__b`` top with a large
   positive synergy gain (mi_z >> max(mi_a, mi_b)) and flags it ``qualifies``.
2. BIZ_VALUE (OOS RMSE): a ridge fit on the interaction spec's residual feature
   (``a*b``) vs the best single base (``a`` or ``b``), scored on a held-out OOS
   split. The interaction base must measurably beat the single base on OOS RMSE.

VERDICT (this Windows host, py3.14, n=4000 train / 4000 OOS, seeds 0..4):
  filled in by ``main()`` -- see ``_results/interaction_base_discovery.json``.
  The pure-interaction synthetic is the textbook case where the interaction
  base wins; if it does NOT beat the single base by the gate margin, the step
  is rejected as default (kept as opt-in research code + this bench).

Usage::

    python -m mlframe.training.composite.discovery._benchmarks.bench_interaction_base_discovery
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np

from mlframe.training.composite.discovery._interaction_bases import (
    discover_interaction_bases,
    score_interaction_pairs,
)

_N = 4000
_NBINS = 12


def _make_pure_interaction(seed: int, n: int = _N):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    # Two decoy features with weak marginal signal so the single-base path has a
    # plausible (but losing) candidate to pick.
    c = rng.standard_normal(n)
    y = a * b + 0.1 * rng.standard_normal(n)
    return {"a": a, "b": b, "c": c}, y


def _ridge_oos_rmse(feat_tr, y_tr, feat_oos, y_oos, lam: float = 1e-3) -> float:
    """1-D ridge (intercept + slope) fit on train, RMSE on OOS."""
    X = np.column_stack([np.ones_like(feat_tr), feat_tr])
    A = X.T @ X + lam * np.eye(2)
    coef = np.linalg.solve(A, X.T @ y_tr)
    pred = coef[0] + coef[1] * feat_oos
    return float(np.sqrt(np.mean((y_oos - pred) ** 2)))


def main() -> None:
    seeds = list(range(5))
    scorer_ok = []
    single_rmses = []
    inter_rmses = []
    for seed in seeds:
        cand_tr, y_tr = _make_pure_interaction(seed)
        cand_oos, y_oos = _make_pure_interaction(seed + 100)
        scored = score_interaction_pairs(
            cand_tr, y_tr, ops=("mul",), top_k=3, nbins=_NBINS,
        )
        top = scored[0] if scored else None
        scorer_ok.append(top is not None and top["op"] == "mul" and set(top["parents"]) == {"a", "b"} and top["qualifies"])
        # Best single base by marginal MI proxy: pick the candidate whose ridge
        # OOS RMSE is lowest (the honest "single/additive spec" baseline).
        best_single = min(
            ("a", "b", "c"),
            key=lambda c: _ridge_oos_rmse(
                cand_tr[c], y_tr, cand_oos[c], y_oos,
            ),
        )
        single_rmse = _ridge_oos_rmse(
            cand_tr[best_single], y_tr, cand_oos[best_single], y_oos,
        )
        synth, _recs = discover_interaction_bases(
            cand_tr, y_tr, ops=("mul",), top_k=3, nbins=_NBINS, max_pairs=1,
        )
        if synth:
            sname = next(iter(synth))
            inter_tr = synth[sname]
            # Reconstruct the OOS synthetic by the same parents/op.
            pa, pb = sname.split("__mul__")
            inter_oos = cand_oos[pa] * cand_oos[pb]
            inter_rmse = _ridge_oos_rmse(inter_tr, y_tr, inter_oos, y_oos)
        else:
            inter_rmse = single_rmse  # no synthetic found -> no improvement
        single_rmses.append(single_rmse)
        inter_rmses.append(inter_rmse)

    single_mean = float(np.mean(single_rmses))
    inter_mean = float(np.mean(inter_rmses))
    rel_improve = (single_mean - inter_mean) / single_mean if single_mean else 0.0
    scorer_pass = sum(scorer_ok)
    win_seeds = sum(1 for s, i in zip(single_rmses, inter_rmses) if i < s - 1e-9)
    verdict = "SHIP-as-default" if win_seeds >= 3 and rel_improve >= 0.10 else "REJECT-keep-as-optin"

    print(f"interaction-base discovery bench  n={_N} nbins={_NBINS} " f"py={sys.version.split()[0]}")
    print(f"  scorer top-pair correct + qualifies: {scorer_pass}/{len(seeds)}")
    print(f"  single-base OOS RMSE (mean):   {single_mean:.4f}")
    print(f"  interaction OOS RMSE (mean):   {inter_mean:.4f}")
    print(f"  relative improvement:          {rel_improve * 100:+.1f}%")
    print(f"  seeds where interaction wins:  {win_seeds}/{len(seeds)}")
    print(f"  VERDICT: {verdict}")

    out_dir = os.path.join(os.path.dirname(__file__), "_results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "interaction_base_discovery.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "ts": datetime.now().isoformat(),
            "n": _N, "nbins": _NBINS, "seeds": seeds,
            "scorer_pass": scorer_pass,
            "single_rmse_mean": single_mean,
            "inter_rmse_mean": inter_mean,
            "rel_improve": rel_improve,
            "win_seeds": win_seeds,
            "single_rmses": single_rmses,
            "inter_rmses": inter_rmses,
            "verdict": verdict,
        }, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
