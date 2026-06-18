"""Large-n MRMR selector-variant quality CAMPAIGN (checkpointed / resumable).

GOAL
----
Find a SELECTOR-VARIANT DEFAULT that measurably improves selection quality at LARGE n / MANY features, the regime where the
quick-loop's tractable-scale probes were inert. We compare two promising variant defaults against the current MRMR defaults on a
known-ground-truth DGP and score them on honest metrics (precision / recall vs the known relevant set AND downstream holdout R2 / AUC).

VARIANTS SWEPT (and why)
------------------------
We deliberately sweep a SMALL, bounded set (not a cartesian) of the knobs most plausibly mis-tuned at large n:

  * ``default``   -- current production defaults (nbins_strategy='mdlp', redundancy_aggregator=None, quantization_nbins=10).
                     This is the BASELINE every other variant is scored against.
  * ``jmim``      -- ``redundancy_aggregator='jmim'`` (Bennasar 2015). The Fleuret CMIM redundancy gate uses ``min_k I(X_k;Y|Z_j)``
                     which rejects features sharing signal with an already-selected one; JMIM uses ``min_j I(X_k,X_j;Y)`` which
                     PRESERVES synergy that CMIM discards. On many-feature data with correlated-decoy clusters + interaction drivers
                     the synergy-preserving aggregator is the leading hypothesis for better recall of the genuine drivers.
  * ``nbins20``   -- ``quantization_nbins=20`` (keeping mdlp off via ``nbins_strategy=None`` so the fixed bin count actually bites).
                     At large n the plug-in MI is far better sampled, so a finer quantization can resolve nonlinear drivers that a
                     coarse 10-bin grid blurs into noise. This is the leading hypothesis for better recall of the NONLINEAR drivers.

We chose redundancy_aggregator and quantization (nbins) because (a) they are single-knob flips with a clear large-n mechanism, (b) they
are orthogonal (one targets synergy/redundancy, the other targets MI resolution), and (c) the MI-estimator bin-vs-knn and the
knee/stability selection-rule knobs are NOT wired into MRMR.fit's hot path (per the _mrmr_class docstring, alternative MI families live
in sibling modules for ad-hoc use only), so sweeping them here would not measure a shippable DEFAULT.

DGP (known relevant set)
------------------------
Large n (sweep n in {20000, 100000}); p in {150, 300}. K genuine drivers = a mix of linear + a couple nonlinear/interaction terms.
Around the drivers we add correlated-decoy clusters (columns that SHARE signal with a driver but carry no independent information -- the
classic redundancy trap) plus pure-noise columns of mixed cardinality (low-card integer + high-card continuous). The relevant index set
(the genuine drivers ONLY -- decoys are NOT relevant) is KNOWN, so precision / recall are exact. Both regression and classification.

HONEST METRICS (per variant, scenario, n, seed)
------------------------------------------------
  * precision / recall of the selected raw-feature index set vs the known driver set (FE disabled via fe_max_steps=0 so support_ holds
    raw column indices only; decoys count as false positives, which is the whole point of the redundancy test).
  * downstream holdout R2 (regression) / AUC (classification): fit a GBM on the TRAIN split restricted to the selected features, score
    on a DISJOINT holdout split. This is the honest "did the selection help a real model" signal.

BOUNDED COMPUTE
---------------
total MRMR.fit calls = variants(3) x scenarios(2: reg+clf) x n-values(2: 20k,100k) x p-values(1) x seeds(15) = 180 fits.
At ~1-3 min per n=100k fit the full campaign is ~3-6 h wall -- inside the authorized <8 h budget. Each cell is checkpointed (below), so a
crash/kill loses at most ONE cell. p is fixed at 300 (the harder many-feature regime) rather than swept, to keep the fit count bounded;
re-run with FULL_P_SWEEP=1 to also cover p=150 (doubles to 360 fits / ~6-12 h).

CHECKPOINT / RESUME
-------------------
After EACH (variant, scenario, n, seed) cell, its result row is appended to ``_results/mrmr_largeN_campaign.jsonl`` and the file is
flushed + fsync'd. On restart the script reads the file, builds the set of already-completed cell keys, and SKIPS them -- so re-running
the exact same command resumes from where it died. A one-line progress marker (cell index / total + key + timing) prints per cell.

USAGE
-----
  Smoke   (n=2000, p=40, 2 seeds, 1 scenario, <2 min)   : python mrmr_largeN_campaign.py --smoke
  Full    (the 180-fit campaign, launched by the loop)  : python mrmr_largeN_campaign.py
  Resume  (same command; completed cells are skipped)   : python mrmr_largeN_campaign.py
  Summarize the JSONL into a verdict-ready table        : python mrmr_largeN_campaign.py --summarize

The full run writes to ``_results/mrmr_largeN_campaign.jsonl``; the smoke run writes to ``_results/mrmr_largeN_campaign_smoke.jsonl`` so
the two never collide. ``--summarize`` reads whichever file matches the (--smoke or full) mode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "_results"
FULL_JSONL = RESULTS_DIR / "mrmr_largeN_campaign.jsonl"
SMOKE_JSONL = RESULTS_DIR / "mrmr_largeN_campaign_smoke.jsonl"

# Variant defaults. Each maps to MRMR-constructor overrides layered on top of the shared base kwargs. ``default`` is the baseline.
VARIANTS: dict[str, dict] = {
    "default": {},
    "jmim": {"redundancy_aggregator": "jmim"},
    "nbins20": {"nbins_strategy": None, "quantization_nbins": 20},
}

SCENARIOS = ("regression", "classification")


def _make_dgp(n: int, p: int, scenario: str, seed: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build a known-ground-truth dataset.

    Returns (X, y, relevant_indices). ``relevant_indices`` are the GENUINE drivers only -- correlated decoys are deliberately excluded so
    that picking a decoy counts as a false positive (the redundancy test). Layout of the p columns:
      [0]            linear driver A
      [1]            linear driver B
      [2]            linear driver C
      [3]            nonlinear driver (squared)
      [4], [5]       interaction operands (driver only via their product x4*x5)
      [6..6+nd-1]    correlated decoys: each = a driver + small noise (share signal, no independent info)  -> NOT relevant
      rest           pure noise: half low-cardinality integer, half high-cardinality continuous           -> NOT relevant
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float64)

    # Drivers.
    a, b, c = X[:, 0], X[:, 1], X[:, 2]
    nl = X[:, 3] ** 2 - 1.0  # nonlinear (centered)
    inter = X[:, 4] * X[:, 5]  # interaction
    relevant = [0, 1, 2, 3, 4, 5]

    signal = 1.5 * a - 1.0 * b + 0.8 * c + 1.2 * nl + 1.0 * inter

    # Correlated decoys: copies of drivers with added noise -- they share signal but add nothing once the driver is selected.
    nd = min(6, max(2, p // 25))
    decoy_start = 6
    driver_cols = [0, 1, 2, 3]
    for i in range(nd):
        src = driver_cols[i % len(driver_cols)]
        X[:, decoy_start + i] = X[:, src] + 0.3 * rng.standard_normal(n)

    # Pure noise columns: mixed cardinality. Low-card integer block + high-card continuous block.
    noise_start = decoy_start + nd
    noise_cols = list(range(noise_start, p))
    half = len(noise_cols) // 2
    for j, col in enumerate(noise_cols):
        if j < half:
            X[:, col] = rng.integers(0, 5, size=n).astype(np.float64)  # low-card integer noise
        # else: leave as the high-card continuous standard-normal already there.

    if scenario == "regression":
        y = (signal + 0.5 * rng.standard_normal(n)).astype(np.float64)
    else:
        logits = signal + 0.5 * rng.standard_normal(n)
        prob = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.uniform(size=n) < prob).astype(np.int64)
    return X, y, relevant


def _build_mrmr(variant: str, scenario: str, seed: int):
    from mlframe.feature_selection.filters import MRMR

    base = dict(
        # FE disabled so support_ holds RAW column indices only -- precision/recall against the known driver index set stays exact.
        fe_max_steps=0,
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


def _downstream_score(X_tr, y_tr, X_ho, y_ho, sel_idx, scenario: str) -> float | None:
    """Fit a GBM on selected features (train split), score on the DISJOINT holdout. R2 for regression, AUC for classification."""
    if len(sel_idx) == 0:
        return None
    cols = list(sel_idx)
    try:
        if scenario == "regression":
            from sklearn.ensemble import HistGradientBoostingRegressor
            from sklearn.metrics import r2_score

            model = HistGradientBoostingRegressor(max_iter=120, random_state=0)
            model.fit(X_tr[:, cols], y_tr)
            return float(r2_score(y_ho, model.predict(X_ho[:, cols])))
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.metrics import roc_auc_score

            model = HistGradientBoostingClassifier(max_iter=120, random_state=0)
            model.fit(X_tr[:, cols], y_tr)
            # single-class holdout guard.
            if len(np.unique(y_ho)) < 2:
                return None
            return float(roc_auc_score(y_ho, model.predict_proba(X_ho[:, cols])[:, 1]))
    except Exception as exc:  # noqa: BLE001 -- record the failure rather than abort the whole campaign on one cell
        return None if not os.environ.get("MRMR_CAMPAIGN_RAISE") else (_ for _ in ()).throw(exc)


def _run_cell(variant: str, scenario: str, n: int, p: int, seed: int) -> dict:
    X, y, relevant = _make_dgp(n=n, p=p, scenario=scenario, seed=seed)

    # Disjoint train / holdout split. MRMR fits on TRAIN; downstream model trains on TRAIN-selected and scores on HOLDOUT.
    rng = np.random.default_rng(seed + 7919)
    perm = rng.permutation(n)
    cut = int(0.7 * n)
    tr_idx, ho_idx = perm[:cut], perm[cut:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_ho, y_ho = X[ho_idx], y[ho_idx]

    sel = _build_mrmr(variant, scenario, seed)
    t0 = time.perf_counter()
    sel.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0

    sel_idx = sorted(int(i) for i in sel.get_support(indices=True).tolist())
    relevant_set = set(relevant)
    sel_set = set(sel_idx)
    tp = len(sel_set & relevant_set)
    precision = tp / len(sel_set) if sel_set else 0.0
    recall = tp / len(relevant_set) if relevant_set else 0.0

    holdout = _downstream_score(X_tr, y_tr, X_ho, y_ho, sel_idx, scenario)

    return {
        "variant": variant,
        "scenario": scenario,
        "n": n,
        "p": p,
        "seed": seed,
        "n_selected": len(sel_idx),
        "selected": sel_idx,
        "relevant": relevant,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "holdout_score": holdout,
        "fit_s": fit_s,
    }


def _cell_key(row: dict) -> tuple:
    return (row["variant"], row["scenario"], int(row["n"]), int(row["p"]), int(row["seed"]))


def _load_done(path: Path) -> set[tuple]:
    done: set[tuple] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(_cell_key(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a partial last line from a kill mid-write
    return done


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _campaign_grid(smoke: bool):
    if smoke:
        n_values = (2000,)
        p = 40
        seeds = list(range(2))
        scenarios = ("regression",)
    else:
        n_values = (20000, 100000)
        p = None  # p sweeps per-n below; set in the cell loop
        seeds = list(range(15))
        scenarios = SCENARIOS
    return n_values, p, seeds, scenarios


def run_campaign(smoke: bool) -> None:
    path = SMOKE_JSONL if smoke else FULL_JSONL
    n_values, smoke_p, seeds, scenarios = _campaign_grid(smoke)

    # smoke pins one p; full fixes p=300 (bounded 180 fits). Opt into the p=150 sweep (360 fits) via FULL_P_SWEEP=1.
    if smoke:
        p_values = (smoke_p,)
    else:
        p_values = (150, 300) if os.environ.get("FULL_P_SWEEP") else (300,)

    cells: list[tuple] = []
    for variant in VARIANTS:
        for scenario in scenarios:
            for n in n_values:
                for p in p_values:
                    for seed in seeds:
                        cells.append((variant, scenario, n, p, seed))

    done = _load_done(path)
    total = len(cells)
    print(f"[campaign] mode={'smoke' if smoke else 'full'} total_cells={total} already_done={len(done)} -> {path.name}", flush=True)

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
            f"[{i}/{total}] {variant:>8} {scenario:<14} n={n:<7} p={p:<4} seed={seed:<2} "
            f"prec={row['precision']:.3f} rec={row['recall']:.3f} hold={row['holdout_score']} "
            f"nsel={row['n_selected']} fit={row['fit_s']:.1f}s wall={time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    print(f"[campaign] complete: {total} cells in {path.name}", flush=True)


def summarize(smoke: bool) -> None:
    path = SMOKE_JSONL if smoke else FULL_JSONL
    if not path.exists():
        print(f"[summarize] no results file: {path}")
        return
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        print(f"[summarize] empty results file: {path}")
        return

    # Per-(variant, scenario, n, p) means.
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["variant"], r["scenario"], int(r["n"]), int(r["p"]))].append(r)

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n=== per-(variant, scenario, n, p) MEANS ===")
    header = f"{'variant':>8} {'scenario':<14} {'n':>7} {'p':>4} {'prec':>6} {'rec':>6} {'F1':>6} {'hold':>7} {'nsel':>5} {'cells':>5}"
    print(header)
    print("-" * len(header))
    for key in sorted(groups):
        g = groups[key]
        prec, rec = _mean([r["precision"] for r in g]), _mean([r["recall"] for r in g])
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        hold = _mean([r["holdout_score"] for r in g])
        nsel = _mean([r["n_selected"] for r in g])
        v, s, n, p = key
        print(f"{v:>8} {s:<14} {n:>7} {p:>4} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {hold:>7.4f} {nsel:>5.1f} {len(g):>5}")

    # Per-cell win counts vs the default baseline (paired by scenario/n/p/seed).
    by_cellseed: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_cellseed[(r["scenario"], int(r["n"]), int(r["p"]), int(r["seed"]))][r["variant"]] = r

    print("\n=== WIN COUNTS vs 'default' (paired per scenario/n/p/seed) ===")
    win_hold: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # [wins, ties, losses]
    win_f1: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for cellseed, variants in by_cellseed.items():
        if "default" not in variants:
            continue
        base = variants["default"]
        base_hold = base["holdout_score"]
        base_f1 = _f1(base)
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
        print(f"  {v:>8}: holdout W/T/L = {w}/{t}/{lo}   |   F1 W/T/L = {w2}/{t2}/{lo2}")

    print("\n[verdict] A variant is a DEFAULT-FLIP candidate only if it wins the MAJORITY of paired cells on BOTH holdout AND F1,")
    print("          across the majority of (scenario, n, p) groups -- single-seed/single-group wins do NOT count (selectors are high-variance).")


def _f1(r: dict) -> float:
    prec, rec = r["precision"], r["recall"]
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Large-n MRMR selector-variant quality campaign (checkpointed / resumable).")
    ap.add_argument("--smoke", action="store_true", help="Run the tiny smoke version (n=2000, p=40, 2 seeds, regression only, <2 min).")
    ap.add_argument("--summarize", action="store_true", help="Aggregate the JSONL into a verdict-ready table (no fitting).")
    args = ap.parse_args(argv)

    if args.summarize:
        summarize(smoke=args.smoke)
        return 0
    run_campaign(smoke=args.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
