"""Large-n MRMR CAMPAIGN #2 -- can a stricter stopping floor fix jmim's over-selection? (checkpointed / resumable)

MOTIVATION (from campaign #1, mrmr_largeN_campaign.py)
------------------------------------------------------
Campaign #1 found ``redundancy_aggregator='jmim'`` (Bennasar 2015) WINS downstream holdout (33/1/26 paired) yet LOSES F1
(9/51) at large n: JMIM preserves synergy CMIM discards, so it keeps recalling genuine drivers -- but it ALSO keeps adding
low-marginal-gain features (the correlated decoys), tanking precision. jmim was shipped as an opt-in, not a default, precisely
because of that over-selection. This campaign tests the obvious remedy: pair jmim with a STRICTER SELECTION STOPPING FLOOR so
the decoy tail is cut while the synergy-preserving recall is kept -- i.e. recover jmim's holdout win AT a precision that finally
clears the default on F1 too. If a (jmim + floor) pair wins the MAJORITY of paired cells on BOTH holdout AND F1 across the
majority of (scenario, n) groups, it is a genuine default-flip candidate; otherwise jmim stays an opt-in and we say so with data.

THE KNOB (chosen after a calibration probe -- NOT the relevance floor)
----------------------------------------------------------------------
The intuitive lever -- raising the marginal-gain stopping floor ``min_relevance_gain_frac`` -- was PROBED and REJECTED: jmim's
over-selected decoys are driver-COPIES (driver + small noise), so they carry HIGH marginal relevance and the floor (which cuts
LOW-gain noise) leaves them untouched (jmim_floor2x/5x selected an identical set to jmim across seeds). The decoy problem is a
REDUNDANCY/uniqueness problem, not a relevance-magnitude one, so we use the levers that actually bit in the probe:

  * ``bur_lambda`` (MRwMR-BUR unique-relevance bonus, Gao 2022): additive bonus for relevance that NO already-selected feature can
    explain. A decoy's relevance is fully explained by the driver it copies -> ~zero unique relevance -> demoted below the genuine
    drivers. Probe (n=4000, seeds 0-2): ``bur_lambda=0.5`` dropped every decoy -> precision 1.0 (vs jmim 0.67), recall kept.
  * ``cmi_perm_stop`` (CMI-permutation stopping, Yu-Principe 2019): replaces the floor with a permutation null test on each
    candidate's conditional MI. Probe: bit jmim's tail variably, occasionally ideal (full driver recovery, prec=rec=1.0).
  * ``relaxmrmr_alpha`` (RelaxMRMR 3-D redundancy) was probed at 0.5/1.0 and was INERT (identical to jmim) -> excluded.

VARIANTS SWEPT (bounded, single-knob deltas on jmim; all verified to change selection AND time-bounded at n=100k: ~140-266s)
---------------------------------------------------------------------------------------------------------------------------
  * ``default``       -- production defaults. Baseline every variant is scored against (reused from campaign #1 semantics).
  * ``jmim``          -- ``redundancy_aggregator='jmim'``. Reproduces campaign #1's holdout-winning-but-over-selecting jmim.
  * ``jmim_bur05``  -- jmim + ``bur_lambda=0.5`` (unique-relevance bonus; the leading precision-fix hypothesis).
  * ``jmim_cmiperm``  -- jmim + ``cmi_perm_stop=True`` (permutation-null stopping; the alternative tail-cutter).

DGP / METRICS / SPLIT / CHECKPOINT are REUSED VERBATIM from mrmr_largeN_campaign.py (same known-ground-truth drivers + decoys +
mixed-cardinality noise; exact precision/recall vs the driver set; disjoint-holdout GBM R2/AUC; per-cell JSONL append + fsync +
resume-skip). Only the variant set and the result file differ, so campaign #1 and #2 results never collide and stay comparable.

BOUNDED COMPUTE
---------------
fits = variants(4) x scenarios(2) x n(2: 20k,100k) x p(1: 300) x seeds(15) = 240 fits (~5 h wall, inside the authorized budget).
Smoke: --smoke (n=2000, p=40, 2 seeds, regression only). Resume: re-run the same command. Summarize: --summarize.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Reuse campaign #1's DGP, downstream scorer, checkpoint I/O and F1 helper verbatim -- identical evaluation, comparable results.
from mlframe.feature_selection._benchmarks.fs_quality.mrmr_largeN_campaign import (
    _append_row,
    _downstream_score,
    _f1,
    _load_done,
    _make_dgp,
)

RESULTS_DIR = Path(__file__).resolve().parent / "_results"
FULL_JSONL = RESULTS_DIR / "mrmr_largeN_campaign2_jmim_threshold.jsonl"
SMOKE_JSONL = RESULTS_DIR / "mrmr_largeN_campaign2_jmim_threshold_smoke.jsonl"

# Single-knob deltas layered on the shared base kwargs. ``default`` is the baseline.
VARIANTS: dict[str, dict] = {
    "default": {},
    "jmim": {"redundancy_aggregator": "jmim"},
    "jmim_bur05": {"redundancy_aggregator": "jmim", "bur_lambda": 0.5},
    "jmim_cmiperm": {"redundancy_aggregator": "jmim", "cmi_perm_stop": True},
}

SCENARIOS = ("regression", "classification")


def _build_mrmr(variant: str, seed: int):
    from mlframe.feature_selection.filters import MRMR

    base = dict(
        fe_max_steps=0,  # raw-index-only support_ so precision/recall vs the known driver set stays exact
        interactions_max_order=1,
        full_npermutations=3,
        baseline_npermutations=2,
        random_seed=seed,
        use_gpu=False,
        n_jobs=1,
        verbose=0,
        cv=2,
    )
    base.update(VARIANTS[variant])
    return MRMR(**base)


def _run_cell(variant: str, scenario: str, n: int, p: int, seed: int) -> dict:
    X, y, relevant = _make_dgp(n=n, p=p, scenario=scenario, seed=seed)

    rng = np.random.default_rng(seed + 7919)
    perm = rng.permutation(n)
    cut = int(0.7 * n)
    tr_idx, ho_idx = perm[:cut], perm[cut:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_ho, y_ho = X[ho_idx], y[ho_idx]

    sel = _build_mrmr(variant, seed)
    t0 = time.perf_counter()
    sel.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0

    sel_idx = sorted(int(i) for i in sel.get_support(indices=True).tolist())
    relevant_set, sel_set = set(relevant), set(sel_idx)
    tp = len(sel_set & relevant_set)
    precision = tp / len(sel_set) if sel_set else 0.0
    recall = tp / len(relevant_set) if relevant_set else 0.0
    holdout = _downstream_score(X_tr, y_tr, X_ho, y_ho, sel_idx, scenario)

    return {
        "variant": variant, "scenario": scenario, "n": n, "p": p, "seed": seed,
        "n_selected": len(sel_idx), "selected": sel_idx, "relevant": relevant,
        "tp": tp, "precision": precision, "recall": recall, "holdout_score": holdout, "fit_s": fit_s,
    }


def _cell_key(row: dict) -> tuple:
    return (row["variant"], row["scenario"], int(row["n"]), int(row["p"]), int(row["seed"]))


def run_campaign(smoke: bool) -> None:
    path = SMOKE_JSONL if smoke else FULL_JSONL
    if smoke:
        n_values, p_values, seeds, scenarios = (2000,), (40,), list(range(2)), ("regression",)
    else:
        n_values = (20000, 100000)
        p_values = (150, 300) if os.environ.get("FULL_P_SWEEP") else (300,)
        seeds, scenarios = list(range(15)), SCENARIOS

    cells = [(v, s, n, p, seed) for v in VARIANTS for s in scenarios for n in n_values for p in p_values for seed in seeds]
    done = _load_done(path)
    total = len(cells)
    print(f"[campaign2] mode={'smoke' if smoke else 'full'} total_cells={total} already_done={len(done)} -> {path.name}", flush=True)

    for i, (variant, scenario, n, p, seed) in enumerate(cells, 1):
        key = (variant, scenario, n, p, seed)
        if key in done:
            print(f"[{i}/{total}] SKIP (done) {key}", flush=True)
            continue
        t0 = time.perf_counter()
        row = _run_cell(variant=variant, scenario=scenario, n=n, p=p, seed=seed)
        _append_row(path, row)
        done.add(key)
        print(
            f"[{i}/{total}] {variant:>13} {scenario:<14} n={n:<7} p={p:<4} seed={seed:<2} "
            f"prec={row['precision']:.3f} rec={row['recall']:.3f} hold={row['holdout_score']} "
            f"nsel={row['n_selected']} fit={row['fit_s']:.1f}s wall={time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    print(f"[campaign2] complete: {total} cells in {path.name}", flush=True)


def summarize(smoke: bool) -> None:
    import json

    path = SMOKE_JSONL if smoke else FULL_JSONL
    if not path.exists():
        print(f"[summarize] no results file: {path}")
        return
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        print(f"[summarize] empty: {path}")
        return

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["variant"], r["scenario"], int(r["n"]), int(r["p"]))].append(r)

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n=== per-(variant, scenario, n, p) MEANS ===")
    header = f"{'variant':>13} {'scenario':<14} {'n':>7} {'p':>4} {'prec':>6} {'rec':>6} {'F1':>6} {'hold':>7} {'nsel':>5} {'cells':>5}"
    print(header)
    print("-" * len(header))
    for key in sorted(groups):
        g = groups[key]
        prec, rec = _mean([r["precision"] for r in g]), _mean([r["recall"] for r in g])
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        v, s, n, p = key
        print(f"{v:>13} {s:<14} {n:>7} {p:>4} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {_mean([r['holdout_score'] for r in g]):>7.4f} {_mean([r['n_selected'] for r in g]):>5.1f} {len(g):>5}")

    by_cellseed: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_cellseed[(r["scenario"], int(r["n"]), int(r["p"]), int(r["seed"]))][r["variant"]] = r

    print("\n=== WIN COUNTS vs 'default' (paired per scenario/n/p/seed) ===")
    win_hold: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    win_f1: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for _cs, variants in by_cellseed.items():
        if "default" not in variants:
            continue
        base = variants["default"]
        base_hold, base_f1 = base["holdout_score"], _f1(base)
        for v, r in variants.items():
            if v == "default":
                continue
            if r["holdout_score"] is not None and base_hold is not None:
                d = r["holdout_score"] - base_hold
                win_hold[v][0 if d > 1e-4 else (2 if d < -1e-4 else 1)] += 1
            df1 = _f1(r) - base_f1
            win_f1[v][0 if df1 > 1e-4 else (2 if df1 < -1e-4 else 1)] += 1

    for v in [x for x in VARIANTS if x != "default"]:
        w, t, lo = win_hold[v]
        w2, t2, lo2 = win_f1[v]
        print(f"  {v:>13}: holdout W/T/L = {w}/{t}/{lo}   |   F1 W/T/L = {w2}/{t2}/{lo2}")

    print("\n[verdict] A (jmim + floor) variant is a DEFAULT-FLIP candidate only if it wins the MAJORITY of paired cells on BOTH")
    print("          holdout AND F1 across the majority of (scenario, n) groups. The whole point: keep jmim's holdout win while")
    print("          the stricter floor lifts F1 over default. Single-seed / single-group wins do NOT count (selectors are high-variance).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Large-n MRMR campaign #2: jmim x stricter stopping floor (checkpointed / resumable).")
    ap.add_argument("--smoke", action="store_true", help="Tiny smoke run (n=2000, p=40, 2 seeds, regression only).")
    ap.add_argument("--summarize", action="store_true", help="Aggregate the JSONL into a verdict-ready table (no fitting).")
    args = ap.parse_args(argv)
    if args.summarize:
        summarize(smoke=args.smoke)
        return 0
    run_campaign(smoke=args.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
