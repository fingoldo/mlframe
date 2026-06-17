"""D-flip bench harness: measure FS levers across scenarios before flipping any default (Workstream D).

Per the accuracy-first variant-default rule, a surfaced FS lever (D-surface) is flipped default-ON ONLY
after a multi-scenario x multi-seed win on the honest holdout -- and additive stages must be benched in
COMBINATION, not by independent per-lever wins. This script is that measurement harness: it runs a fixed
scenario suite x seeds for {baseline, each variant-slot lever, greedy-combined config} and reports the
honest-holdout downstream metric per config, so the flip decision is made on numbers, not assumption.

Committed per REJECTED!=DELETED: re-running this is how a future agent re-validates a flip. It does NOT
change any default by itself -- it only prints/saves the deltas. Run:

    python -m mlframe.feature_selection._benchmarks.bench_fs_levers_dflip            # full
    python -m mlframe.feature_selection._benchmarks.bench_fs_levers_dflip --quick    # 2 scenarios x 2 seeds

Smoke run (--quick, 2 scenarios x 2 seeds, Ridge downstream) -- DIRECTIONAL ONLY, not a flip basis:
  high_cardinality : all levers ~neutral (+/-0.0002).
  synergistic_pair : jmim ALONE +0.436 R^2 (big); su ALONE +0.048; su+jmim COMBINED only +0.048
                     -- i.e. the two interact NEGATIVELY: stacking su onto jmim forfeits jmim's win.
This confirms the accuracy-first rule's premise: a per-lever win does NOT imply a combined-config win, so
defaults must be chosen by the COMBINED greedy bench, never by independent per-lever deltas. No default is
flipped on this smoke run; the full multi-seed campaign (and the combined-vs-leave-one-out comparison) is
required before any flip, with the numbers recorded in CHANGELOG.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np


def _scenarios(rng_seed: int):
    """Return {name: (X, y, task)} synthetics where specific levers should help."""
    rng = np.random.default_rng(rng_seed)
    out: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}

    # High-cardinality integer features (SU normalization should reduce cardinality bias).
    n = 1500
    hc = np.column_stack([rng.integers(0, c, n) for c in (50, 30, 3, 3, 2)]).astype(float)
    y_hc = (hc[:, 2] + hc[:, 3] + 0.3 * rng.standard_normal(n)).astype(float)
    out["high_cardinality"] = (hc, y_hc, "regression")

    # Synergistic pair (XOR-like): jmim / synergy-aware aggregation should keep both operands.
    xa = rng.integers(0, 2, n)
    xb = rng.integers(0, 2, n)
    noise = rng.standard_normal((n, 6))
    Xsyn = np.column_stack([xa, xb, noise]).astype(float)
    y_syn = (xa ^ xb).astype(float) + 0.2 * rng.standard_normal(n)
    out["synergistic_pair"] = (Xsyn, y_syn, "regression")

    # Many-noise-dims (any debias / prescreen should help downstream).
    p = 40
    Xn = rng.standard_normal((n, p))
    w = np.zeros(p)
    w[:4] = [2.0, -1.5, 1.0, -0.8]
    y_n = Xn @ w + 0.5 * rng.standard_normal(n)
    out["noisy_wide"] = (Xn, y_n, "regression")
    return out


_LEVER_CONFIGS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "mi_normalization=su": {"mi_normalization": "su"},
    "redundancy_aggregator=jmim": {"redundancy_aggregator": "jmim"},
    "combined": {"mi_normalization": "su", "redundancy_aggregator": "jmim"},
}


def _honest_metric(X, y, lever_kwargs, seed: int) -> float:
    """Fit MRMR with the lever kwargs on train, train Ridge on the selected features, return holdout R^2."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    from mlframe.feature_selection.filters import MRMR

    import pandas as pd

    cols = [f"f{i}" for i in range(X.shape[1])]
    Xdf = pd.DataFrame(X, columns=cols)
    Xtr, Xte, ytr, yte = train_test_split(Xdf, y, test_size=0.3, random_state=seed)
    try:
        sel = MRMR(n_workers=1, verbose=0, fe_max_steps=0, max_runtime_mins=2, **lever_kwargs)
        sel.fit(Xtr, ytr)
        Xtr_s = sel.transform(Xtr)
        Xte_s = sel.transform(Xte)
    except Exception:  # harness must not die on one config
        return float("nan")
    model = Ridge().fit(np.asarray(Xtr_s), ytr)
    return float(r2_score(yte, model.predict(np.asarray(Xte_s))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 scenarios x 2 seeds for a smoke run")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    seeds = list(range(2 if args.quick else args.seeds))
    results: dict[str, dict[str, list[float]]] = {}
    for s in seeds:
        scen = _scenarios(s)
        items = list(scen.items())[: 2 if args.quick else None]
        for name, (X, y, _task) in items:
            for cfg_name, kw in _LEVER_CONFIGS.items():
                r2 = _honest_metric(X, y, kw, s)
                results.setdefault(name, {}).setdefault(cfg_name, []).append(r2)

    print("\n=== D-flip FS-lever honest-holdout R^2 (mean over seeds) ===")
    for scen_name, by_cfg in results.items():
        base = float(np.nanmean(by_cfg.get("baseline", [float("nan")])))
        print(f"\n[{scen_name}] baseline={base:.4f}")
        for cfg_name, vals in by_cfg.items():
            if cfg_name == "baseline":
                continue
            m = float(np.nanmean(vals))
            print(f"  {cfg_name:32s} {m:.4f}  (delta {m - base:+.4f})")
    print(
        "\nNOTE: flip a default ON only if the COMBINED config wins the MAJORITY of scenarios x seeds; "
        "record the numbers in CHANGELOG. This harness changes no default."
    )

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({k: {c: list(map(float, v)) for c, v in d.items()} for k, d in results.items()}, fh, indent=2)


if __name__ == "__main__":
    main()
