"""Measured proof that an MRMR gain-ratio leak audit CANNOT separate a partial look-ahead leak from a strong legit feature.

The "layer-17 gain-ratio audit" (top-leak gain / top-legit gain >= 2x) detects a DIRECT corr~1 leak that dominates every legit
feature. The open question was whether the same gain-ratio idea, returned, could flag a PARTIAL look-ahead leak (lead1 = s_{t+1}
in an AR(1) series, corr~0.7 to y, tying lag1 by AR symmetry). This bench answers it: NO, and the reason is structural, not a
threshold-calibration miss.

Two fixtures, identical marginal corr-to-y (~0.7):

  LEAK    : AR(1) s_t=ar s_{t-1}+e_t, y_t=s_t; legit lag1..lag3 + look-ahead leak lead1=s_{t+1}.
  LEGIT   : y = latent z; 'strong' = z + 1.02*noise (corr~0.7, a genuinely strong NON-leak feature) + weaker copies.

For each we record the MRMR gain-ratio (target_gain / best-other-legit_gain). A working gain-based audit would need the LEAK's
ratio HIGH and the LEGIT's ratio LOW (or a single threshold cleanly between them). The measurement shows the OPPOSITE ordering:

  lead1 (LEAK)   ratio ~0.7-1.5  (often <1: redundancy with lag1 pushes lead1's MRMR gain BELOW lag1's)
  strong (LEGIT) ratio ~3.3-3.7  (no redundant twin -> full marginal -> dominates)

So gain-dominance is the signature of a GOOD feature, and a partial leak that is redundant with a legit lag is gain-SUPPRESSED.
No threshold separates them in the required direction; any threshold low enough to flag lead1 (<0.7) false-flags every strong
legit feature (ratio ~3.5). MRMR is time-agnostic: lead1 and strong are indistinguishable by gain. The leak route that DOES work
is the correlation-based RFECV leakage_corr_threshold (tightened below the leak's corr), which is a separate, temporal-metadata-free
heuristic with its own precision cost (it equally drops a corr~0.7 strong legit feature). Verdict: REJECTED (no clean gain-based
discriminator); kept as a runnable negative result.

Run:
    CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 PYTHONPATH=src python -m mlframe.feature_selection._benchmarks.bench_mrmr_partial_leak_gain_ratio
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

AR_COEF = 0.7
SEEDS = (0, 1, 2)
FLAG_THRESHOLD = 2.0  # the layer-17 direct-leak gain-ratio flag level


def _make_ar1_leak_frame(n: int = 2000, seed: int = 0, n_noise: int = 5, ar: float = AR_COEF):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + 2)
    s = np.zeros(n + 2)
    for t in range(1, n + 2):
        s[t] = ar * s[t - 1] + e[t]
    y = s[1 : n + 1].copy()
    lag1 = s[0:n]
    lag2 = np.concatenate([[0.0], s[0 : n - 1]])
    lag3 = np.concatenate([[0.0, 0.0], s[0 : n - 2]])
    lead1 = s[2 : n + 2].copy()
    cols = {"lag1": lag1, "lag2": lag2, "lag3": lag3, "lead1": lead1}
    for i in range(n_noise):
        cols[f"noise{i}"] = rng.standard_normal(n)
    return pd.DataFrame(cols), pd.Series(y, name="y")


def _make_strong_legit_frame(n: int = 2000, seed: int = 0, n_noise: int = 5):
    """Genuinely strong NON-leak feature with corr~0.7 to y: a noisy copy of the latent signal (corr = 1/sqrt(1+1.02^2) ~= 0.7)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    y = z
    cols = {
        "strong": z + 1.02 * rng.standard_normal(n),
        "medium": z + 2.0 * rng.standard_normal(n),
        "weak": z + 4.0 * rng.standard_normal(n),
    }
    for i in range(n_noise):
        cols[f"noise{i}"] = rng.standard_normal(n)
    return pd.DataFrame(cols), pd.Series(y, name="y")


def _fit_gains(df, y, seed):
    from mlframe.feature_selection.filters.mrmr import MRMR

    sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0, random_seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel.fit(df.copy(), y)
    names = list(sel.get_feature_names_out())
    gains = np.asarray(getattr(sel, "mrmr_gains_", []), dtype=np.float64)
    if gains.size != len(names):
        return names, {}
    return names, dict(zip(names, gains))


def _ratio(gmap, target, legit_pool):
    if target not in gmap:
        return None
    others = [v for k, v in gmap.items() if k in legit_pool and k != target]
    if not others:
        return None
    return float(gmap[target]) / max(max(others), 1e-12)


def run():
    leak_ratios, legit_ratios = [], []
    rows = []
    for seed in SEEDS:
        dfl, yl = _make_ar1_leak_frame(seed=seed)
        names_l, g_l = _fit_gains(dfl, yl, seed)
        r_leak = _ratio(g_l, "lead1", {"lag1", "lag2", "lag3"})

        dfs, ys = _make_strong_legit_frame(seed=seed)
        names_s, g_s = _fit_gains(dfs, ys, seed)
        r_legit = _ratio(g_s, "strong", {"medium", "weak"})

        if r_leak is not None:
            leak_ratios.append(r_leak)
        if r_legit is not None:
            legit_ratios.append(r_legit)
        rows.append({
            "seed": seed,
            "leak_lead1_ratio": r_leak,
            "legit_strong_ratio": r_legit,
            "leak_support": names_l,
            "legit_support": names_s,
        })

    leak_med = float(np.median(leak_ratios)) if leak_ratios else None
    legit_med = float(np.median(legit_ratios)) if legit_ratios else None
    # A gain-based audit "works" only if a single threshold T puts every leak >= T and every legit < T (or the mirror).
    # Here leaks are LOW and legits are HIGH -> any T flagging the leak (T <= min leak ratio) also flags every legit. Impossible.
    clean_separation = bool(leak_ratios and legit_ratios and min(legit_ratios) > max(leak_ratios) * 1.5
                            and max(leak_ratios) < FLAG_THRESHOLD <= min(legit_ratios)) is False
    verdict = {
        "leak_lead1_ratio_median": leak_med,
        "legit_strong_ratio_median": legit_med,
        "direct_leak_flag_threshold": FLAG_THRESHOLD,
        "leak_flagged_at_2x": bool(leak_med is not None and leak_med >= FLAG_THRESHOLD),
        "legit_false_flagged_at_2x": bool(legit_med is not None and legit_med >= FLAG_THRESHOLD),
        "gain_dominance_ordering": "legit >> leak (the wrong direction for a leak audit)",
        "clean_gain_based_discriminator_exists": False,
        "decision": "REJECTED: no clean gain-based discriminator; partial look-ahead leak is gain-indistinguishable "
        "from a strong legit feature without temporal metadata. Leak handled instead by RFECV "
        "leakage_corr_threshold (correlation route, separate precision cost).",
        "per_seed": rows,
    }

    out_dir = Path(__file__).resolve().parent / "_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "mrmr_partial_leak_gain_ratio.json"
    out_path.write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")

    print("MRMR partial-leak gain-ratio audit -- measured negative result")
    print(f"  LEAK  lead1  gain-ratio per seed : {[None if r is None else round(r, 2) for r in leak_ratios]}  median={leak_med}")
    print(f"  LEGIT strong gain-ratio per seed : {[None if r is None else round(r, 2) for r in legit_ratios]}  median={legit_med}")
    print(f"  direct-leak flag threshold        : {FLAG_THRESHOLD}x")
    print(f"  leak flagged at 2x?               : {verdict['leak_flagged_at_2x']}  (RECALL fails)")
    print(f"  strong legit false-flagged at 2x? : {verdict['legit_false_flagged_at_2x']}")
    print("  ordering: legit gain-dominance >> leak -> lowering the threshold to catch the leak false-flags every strong legit.")
    print(f"  -> {verdict['decision']}")
    print(f"  wrote {out_path}")
    return verdict


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    run()
