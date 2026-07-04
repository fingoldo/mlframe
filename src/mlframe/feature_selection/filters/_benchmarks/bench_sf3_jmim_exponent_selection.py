"""Bench (critique S-F3): does the JMIM ``**(nexisting+1)`` discount exponent, which AMPLIFIES a joint MI > 1, change selection?

The Fleuret/JMIM confirmation path (evaluation.py evaluate_gain, use_jmim branch) applies the same ``nexisting`` exponent the
CMI branch uses. For a conditional MI in [0,1] that exponent is a discount; for a JMIM JOINT MI I({X,Z};Y) -- routinely > 1 nat --
``x**k`` AMPLIFIES, the opposite of a discount (S-F3). The exponent is round-constant (nexisting is the same for every candidate in a
round) and ``x**k`` is monotone, so it preserves the WITHIN-ROUND candidate ranking; the only way it can move selection is via the
ABSOLUTE ``min_relevance_gain`` floor (a value < 1 shrinks below the floor, a value > 1 inflates above it).

This bench measures selection quality with the exponent as-is vs the discount-only correction
(``MLFRAME_JMIM_EXPONENT_DISCOUNT_ONLY=1`` clamps the exponent so it can only shrink the joint MI, never amplify it), multi-seed, on a
synergy fixture where JMIM should help. Metric: recall of the planted informative features in the selected set (higher = better).

The env var is read at import as a numba compile-time constant, so the two arms MUST run in separate processes. This script fits ONE
arm (whichever the env var selects) and prints per-seed selected sets + recall. Drive both arms + the verdict via:
  python -m mlframe.feature_selection.filters._benchmarks.bench_sf3_jmim_exponent_selection --run-both
"""
import os
import sys
import json
import subprocess

import numpy as np


def _make_synergy_fixture(seed, n=1500):
    """Plant informative features among noise. z0..z2 are noisy reflections of a latent driver (JMIM's strong suit);
    a,b form a synergistic (XOR-like) pair; the rest are noise. y depends on the latent driver AND the synergy pair."""
    rng = np.random.default_rng(seed)
    latent = rng.integers(0, 3, n)
    z0 = np.where(rng.random(n) < 0.85, latent, rng.integers(0, 3, n))
    z1 = np.where(rng.random(n) < 0.80, latent, rng.integers(0, 3, n))
    z2 = np.where(rng.random(n) < 0.75, latent, rng.integers(0, 3, n))
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    synergy = a ^ b
    noise = [rng.integers(0, 4, n) for _ in range(6)]
    y = ((latent >= 1).astype(int) ^ synergy).astype(int)
    cols = {"z0": z0, "z1": z1, "z2": z2, "a": a, "b": b}
    for i, c in enumerate(noise):
        cols[f"n{i}"] = c
    import pandas as pd
    X = pd.DataFrame(cols)
    informative = {"z0", "z1", "z2", "a", "b"}
    return X, y.astype(np.int64), informative


def _run_arm(seeds):
    from mlframe.feature_selection.filters.mrmr._mrmr_class import MRMR
    from mlframe.feature_selection.filters.evaluation import _JMIM_EXPONENT_DISCOUNT_ONLY

    out = {"discount_only": bool(_JMIM_EXPONENT_DISCOUNT_ONLY), "per_seed": [], "recalls": []}
    for seed in seeds:
        X, y, informative = _make_synergy_fixture(seed)
        sel = MRMR(redundancy_aggregator="jmim", max_runtime_mins=0.5, verbose=0)
        sel.fit(X, y)
        picked = list(sel.get_feature_names_out())
        # FE is on by default, so an informative raw column may be recovered inside an engineered composite (as an operand token).
        # Credit an informative feature when it appears as a token (raw name or delimited operand) in any picked feature name.
        import re
        joined = " ".join(picked)
        hit = {c for c in informative if re.search(rf"(?<![A-Za-z0-9_]){re.escape(c)}(?![A-Za-z0-9_])", joined)}
        recall = len(hit) / len(informative)
        out["per_seed"].append({"seed": seed, "picked": list(picked), "recall": recall})
        out["recalls"].append(recall)
    out["mean_recall"] = float(np.mean(out["recalls"])) if out["recalls"] else 0.0
    return out


def _drive_both():
    seeds = list(range(8))
    env_base = dict(os.environ)
    env_base["CUDA_VISIBLE_DEVICES"] = ""
    results = {}
    for arm, val in (("exponent", "0"), ("discount_only", "1")):
        env = dict(env_base)
        env["MLFRAME_JMIM_EXPONENT_DISCOUNT_ONLY"] = val
        env["_SF3_ARM_SEEDS"] = json.dumps(seeds)
        proc = subprocess.run([sys.executable, "-m",
                               "mlframe.feature_selection.filters._benchmarks.bench_sf3_jmim_exponent_selection",
                               "--arm"], env=env, capture_output=True, text=True)
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("ARM_RESULT ")]
        if not line:
            print(proc.stdout[-2000:]); print(proc.stderr[-2000:]); raise SystemExit(f"arm {arm} produced no result")
        results[arm] = json.loads(line[-1][len("ARM_RESULT "):])

    exp, disc = results["exponent"], results["discount_only"]
    n = len(seeds)
    disc_better = sum(1 for i in range(n) if disc["recalls"][i] > exp["recalls"][i])
    exp_better = sum(1 for i in range(n) if exp["recalls"][i] > disc["recalls"][i])
    same = n - disc_better - exp_better
    print("=== S-F3 JMIM exponent vs discount-only correction (recall of planted informative features) ===")
    for i, s in enumerate(seeds):
        print(f"seed={s}: exponent recall={exp['recalls'][i]:.3f} picked={exp['per_seed'][i]['picked']}")
        print(f"        discount recall={disc['recalls'][i]:.3f} picked={disc['per_seed'][i]['picked']}")
    print(f"\nmean recall  exponent={exp['mean_recall']:.4f}  discount_only={disc['mean_recall']:.4f}")
    print(f"per-seed wins  discount_only_better={disc_better}  exponent_better={exp_better}  identical_selection={same}/{n}")
    verdict = ("SELECTION-NEUTRAL -> DOC (keep exponent)" if disc_better == 0 and exp_better == 0
               else ("SHIP discount-only (equal-or-better majority)" if disc["mean_recall"] >= exp["mean_recall"] and disc_better >= exp_better
                     else "KEEP exponent (correction regresses)"))
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    if "--arm" in sys.argv:
        seeds = json.loads(os.environ.get("_SF3_ARM_SEEDS", "[0,1,2,3]"))
        print("ARM_RESULT " + json.dumps(_run_arm(seeds)))
    else:
        _drive_both()
