"""qual-24 confirmation bench for the large-n regression adaptive-quantization gate.

The 180-cell ``mrmr_largeN_campaign.jsonl`` already established the win (reg n=100k: fixed-20-bin quantile beats MDLP 15/15 seeds,
holdout R2 +0.116 / F1 +0.242; loses at reg n=20k and on classification). This small bench (<=8 MRMR fits) confirms the SHIPPED GATE
(``adaptive_nbins_large_n_reg=True``) actually FIRES on reg n>=50k and reproduces the campaign-winner config (nbins_strategy=None,
quantization_nbins=20), and does NOT fire on reg n<50k or on classification (where MDLP must stay). Run:

    PYTHONPATH=<repo>/src python qual24_adaptive_nbins_large_n_reg_confirm.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mrmr_largeN_campaign import _make_dgp  # noqa: E402


def _f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def _build(seed: int):
    from mlframe.feature_selection.filters import MRMR

    return MRMR(
        fe_max_steps=0, interactions_max_order=1, full_npermutations=3, baseline_npermutations=2,
        random_seed=seed, use_gpu=False, n_jobs=1, verbose=0, cv=2,
    )


def _run(scenario: str, n: int, seed: int) -> dict:
    X, y, rel = _make_dgp(n=n, p=300, scenario=scenario, seed=seed)
    m = _build(seed)
    m.fit(X, y)
    sup = list(m.get_support(indices=True))
    tp = len(set(sup) & set(rel))
    prec = tp / max(1, len(sup))
    rec = tp / len(rel)
    return {
        "scenario": scenario, "n": n, "seed": seed,
        "fired": bool(getattr(m, "_adaptive_nbins_large_n_reg_fired_", False)),
        "nbins_strategy": m.nbins_strategy, "quantization_nbins": m.quantization_nbins,
        "nsel": len(sup), "precision": prec, "recall": rec, "f1": _f1(prec, rec),
    }


def main() -> None:
    cells = [
        ("regression", 100_000, 0), ("regression", 100_000, 1),
        ("regression", 20_000, 0),
        ("classification", 100_000, 0), ("classification", 20_000, 0),
    ]
    print(f"{'scenario':<14}{'n':>8}{'seed':>5}{'fired':>7}{'strat':>7}{'nbins':>6}{'nsel':>5}{'prec':>6}{'rec':>6}{'F1':>6}")
    for scn, n, seed in cells:
        r = _run(scn, n, seed)
        print(f"{r['scenario']:<14}{r['n']:>8}{r['seed']:>5}{str(r['fired']):>7}{str(r['nbins_strategy']):>7}"
              f"{r['quantization_nbins']:>6}{r['nsel']:>5}{r['precision']:>6.3f}{r['recall']:>6.3f}{r['f1']:>6.3f}")


if __name__ == "__main__":
    main()
