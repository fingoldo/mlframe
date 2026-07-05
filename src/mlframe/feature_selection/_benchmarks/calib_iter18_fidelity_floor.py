"""iter18 calibration study: pick the default ``fidelity_floor`` for ``proxy_trust_guard``.

The composite ``proxy_fidelity_score = 0.6*spearman + 0.4*recall@k`` (iter17 weights) gates the
``trustworthy`` boolean. The legacy ``spearman_floor=0.6`` default was set against the raw-Spearman
scale (pre-iter16); on the composite scale it is too conservative and trips on the partial-recovery
``interaction_heavy`` regime (recovery_rate 0.75, a real partial success worth surfacing as caution
rather than a hard fail).

Principled choice (the study):
  - Reuse iter17's 5-regime set (additive_highSNR / redundancy_heavy / interaction_heavy /
    xor_interaction / noise_heavy).
  - Per regime: fit ShapProxiedFS, record ``proxy_fidelity_score`` and ``recovery_rate =
    recovered / planted_informative``.
  - Define "acceptable" recovery as ``recovery_rate >= 0.7`` (70% of informatives kept).
  - Floor = the LOWEST ``proxy_fidelity_score`` of any PASS regime, rounded down to a clean value.
  - Sanity check: highest composite of any regime with ``recovery_rate < 0.5`` must be BELOW the
    floor; else the floor doesn't separate good from bad.

Watchdog-safe: 5 regimes, width<=1200, prints per regime, ~70s total wall on dev HW.

Run with the worktree on PYTHONPATH:
  $env:PYTHONPATH='<worktree>\\src'
  D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.calib_iter18_fidelity_floor
"""
from __future__ import annotations

import time
import numpy as np

from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
from mlframe.feature_selection._benchmarks.calib_iter17_fidelity_weights import REGIMES
from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS


def _hb(msg: str) -> None:
    print(f"[calib-iter18 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Threshold for "acceptable" recovery: 70% of planted informatives kept. Tunable; documented in the
# docstring. Below this the regime is a "failure" we want the gate to flag; at-or-above it the gate
# must NOT trip (a partial-recovery regime like interaction_heavy at 0.75 is still a real win).
RECOVERY_PASS_THRESHOLD = 0.70
# Sanity threshold: regimes with ``recovery_rate < 0.5`` are considered "broken" and the floor MUST
# sit above their highest composite for a clean separation. If not we report inconclusive.
RECOVERY_FAIL_THRESHOLD = 0.50


def run_one(regime: dict, time_budget_s: float = 90.0) -> dict:
    name = regime["name"]
    kwargs = regime["kwargs"]
    rs = int(regime.get("random_state", 0))
    _hb(f"regime {name}: building dataset ({kwargs.get('n_samples')}s x "
        f"{kwargs.get('n_informative') + kwargs.get('n_redundant', 0) + kwargs.get('n_noise')} cols)")
    X, y, roles = make_regime_dataset(**kwargs)
    informatives = {c for c, r in roles.items() if r == "informative"}

    fs = ShapProxiedFS(
        classification=False, metric="rmse",
        min_features=2, max_features=8, top_n=5,
        n_models=1, n_splits=3, holdout_size=0.3,
        # Set a permissive floor so we observe the raw composite without the gate filtering anchors;
        # the calibration cares about the score VALUE not the boolean. (The boolean is downstream
        # advisory; the selector still proceeds and reports trust=False -- recovery is unaffected.)
        trust_guard=True, n_anchors=30, fidelity_floor=0.0,
        revalidate=True, n_revalidation_models=1,
        run_importance_ablation=False,
        use_bias_corrector=True,
        prefilter_top=200, prefilter_method="auto",
        cluster_features="auto",
        random_state=rs, verbose=False, tqdm=False, n_jobs=-1,
    )

    t0 = time.time()
    fs.fit(X, y)
    elapsed = time.time() - t0

    trust = fs.shap_proxy_report_.get("trust", {})
    sp = float(trust.get("spearman", float("nan")))
    rc = float(trust.get("recall_at_k", float("nan")))
    comp = float(trust.get("proxy_fidelity_score", float("nan")))
    selected = set(fs.selected_features_)
    recovery = len(informatives & selected)
    n_inf = len(informatives)
    recovery_rate = recovery / max(1, n_inf)
    _hb(f"regime {name}: spearman={sp:.4f} recall@k={rc:.4f} composite={comp:.4f} "
        f"recovery={recovery}/{n_inf} rate={recovery_rate:.3f} elapsed={elapsed:.1f}s")
    if elapsed > time_budget_s:
        _hb(f"  WARNING: regime {name} exceeded soft budget {time_budget_s:.0f}s " f"(actual {elapsed:.1f}s); consider narrower kwargs")
    return dict(name=name, spearman=sp, recall_at_k=rc, composite=comp, recovery=recovery, n_informative=n_inf, recovery_rate=recovery_rate, elapsed=elapsed)


def main():
    rows = []
    overall_t0 = time.time()
    for reg in REGIMES:
        rows.append(run_one(reg))
        _hb(f"  cumulative wall: {time.time() - overall_t0:.1f}s")

    print("\nper-regime table (iter18 composite + recovery_rate):", flush=True)
    print(f"{'regime':<20} {'spearman':>10} {'recall@k':>10} {'composite':>10} " f"{'recovery':>10} {'rate':>7} {'sec':>7}", flush=True)
    for r in rows:
        print(f"{r['name']:<20} {r['spearman']:>10.4f} {r['recall_at_k']:>10.4f} "
              f"{r['composite']:>10.4f} {r['recovery']:>5}/{r['n_informative']:<4} "
              f"{r['recovery_rate']:>7.3f} {r['elapsed']:>7.1f}", flush=True)

    # Partition regimes by recovery_rate.
    pass_rows = [r for r in rows if r["recovery_rate"] >= RECOVERY_PASS_THRESHOLD]
    fail_rows = [r for r in rows if r["recovery_rate"] < RECOVERY_FAIL_THRESHOLD]
    ambig_rows = [r for r in rows if RECOVERY_FAIL_THRESHOLD <= r["recovery_rate"] < RECOVERY_PASS_THRESHOLD]

    print(f"\nrecovery_rate >= {RECOVERY_PASS_THRESHOLD} (PASS group): " f"{[r['name'] for r in pass_rows]}", flush=True)
    print(f"recovery_rate <  {RECOVERY_FAIL_THRESHOLD} (FAIL group): " f"{[r['name'] for r in fail_rows]}", flush=True)
    if ambig_rows:
        print(f"ambiguous (between thresholds, NOT used for floor): " f"{[r['name'] for r in ambig_rows]}", flush=True)

    if not pass_rows or not fail_rows:
        print("\nINCONCLUSIVE: need at least one PASS and one FAIL regime to bracket the floor; " "keeping legacy 0.5 as a fallback default.", flush=True)
        return dict(rows=rows, proposed=0.5, inconclusive=True)

    pass_floor = min(r["composite"] for r in pass_rows)
    fail_ceiling = max(r["composite"] for r in fail_rows)
    print(f"\nPASS group min composite (the FLOOR candidate)        = {pass_floor:.4f} "
          f"({min(pass_rows, key=lambda r: r['composite'])['name']})", flush=True)
    print(f"FAIL group max composite (sanity-check upper bound)   = {fail_ceiling:.4f} "
          f"({max(fail_rows, key=lambda r: r['composite'])['name']})", flush=True)

    if fail_ceiling >= pass_floor:
        print("\nINCONCLUSIVE: FAIL ceiling >= PASS floor; the composite does not cleanly separate "
              "PASS/FAIL on this regime set. Recommend keeping legacy 0.5 and revisiting with more "
              "regimes / a different composite weighting.", flush=True)
        return dict(rows=rows, proposed=0.5, inconclusive=True,
                    pass_floor=pass_floor, fail_ceiling=fail_ceiling)

    # Pick a clean published floor: round DOWN the PASS floor to one decimal so we leave a safety
    # margin. If that rounding crosses the FAIL ceiling we'd lose separation, so back off in
    # 0.05 increments until we sit above the ceiling.
    candidate = int(pass_floor * 10) / 10  # round-down to 1 decimal
    while candidate <= fail_ceiling + 1e-9 and candidate < pass_floor:
        candidate += 0.05
    proposed = round(candidate, 2)
    print(f"\nPROPOSED default fidelity_floor = {proposed:.2f} "
          f"(PASS margin {pass_floor - proposed:+.4f}, FAIL margin {proposed - fail_ceiling:+.4f})",
          flush=True)

    overall = time.time() - overall_t0
    print(f"\ntotal wall: {overall:.1f}s", flush=True)
    return dict(rows=rows, proposed=proposed, inconclusive=False, pass_floor=pass_floor, fail_ceiling=fail_ceiling)


if __name__ == "__main__":
    main()
