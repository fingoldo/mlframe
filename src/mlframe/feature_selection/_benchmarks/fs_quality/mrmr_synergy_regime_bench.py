"""SYNERGY-vs-ADDITIVE regime bench: should a synergy-aware redundancy aggregator AUTO-engage? (multi-seed, honest holdout)

WHY THIS BENCH EXISTS
---------------------
``fleuret.py`` documents a known weakness: the default Fleuret/CMIM redundancy gate ``gain = I(X;Y) - max_k I(X;Y|S_k)`` REJECTS
synergistic features (an operand useless alone but informative jointly with an already-selected partner scores ~0/negative gain).
MRMR ships a synergy-aware aggregator (``redundancy_aggregator='jmim'``, Bennasar 2015) but only as OPT-IN.

Prior campaigns (``mrmr_largeN_campaign.py`` / ``mrmr_largeN_campaign2_jmim_threshold.py``) measured JMIM on an ADDITIVE /
main-effect-plus-decoy DGP and found JMIM WINS downstream holdout (33/1/26 paired) but LOSES F1 decisively (9/0/51): it
over-selects correlated decoys (precision 1.0 -> 0.44, nsel 3-5 -> 8-10). Hence JMIM stayed opt-in. What those campaigns did NOT
contain is a SYNERGISTIC regime where the planted ground truth is XOR / sign-product PAIRS -- exactly the case the docstring says
Fleuret mishandles. This bench fills that gap and asks the precise question for an AUTO gate:

  * SYNERGISTIC regime: does default Fleuret actually MISS the XOR operands, and does JMIM RECOVER them (recovery + holdout)?
  * ADDITIVE regime: does JMIM regress (over-select, lower parsimony, lower/equal holdout)? -- reproduces the prior finding cheaply.

If JMIM clearly wins synergistic recovery+holdout AND the additive regression is real, an "auto" gate (route to JMIM only when the
data is synergistic) is justified. If not, we DO NOT auto-switch and record the numbers.

DGP
---
  SYNERGISTIC (classification): K planted XOR/sign-product PAIRS. Each operand is ~marginally-useless (balanced bits / symmetric
    sign) but the pair drives y. Plus correlated decoys of ONE operand + mixed-cardinality pure noise. Ground-truth relevant set =
    all 2K operands.
  ADDITIVE (classification): K independent main-effect features (linear logit) + correlated decoys + noise. Ground truth = the K
    main effects. This is the regime where plain Fleuret is correct and JMIM may over-select the collinear decoy group.

METRICS (per variant/regime/seed, fixed 60/40 split)
  * operand/main-effect recovery: recall of the planted relevant set, precision, n_selected.
  * downstream holdout AUC: LightGBM AND logistic regression on TRAIN-selected features, scored on the disjoint 40% holdout.

USAGE
  Smoke   : python -m mlframe.feature_selection._benchmarks.fs_quality.mrmr_synergy_regime_bench --smoke
  Full    : python -m mlframe.feature_selection._benchmarks.fs_quality.mrmr_synergy_regime_bench
  Summary : python -m mlframe.feature_selection._benchmarks.fs_quality.mrmr_synergy_regime_bench --summarize
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
FULL_JSONL = RESULTS_DIR / "mrmr_synergy_regime.jsonl"
SMOKE_JSONL = RESULTS_DIR / "mrmr_synergy_regime_smoke.jsonl"

# Single-knob deltas. ``auto`` exercises the data-dependent gate shipped in this change.
VARIANTS: dict[str, dict] = {
    "default": {},
    "jmim": {"redundancy_aggregator": "jmim"},
    "auto": {"redundancy_aggregator": "auto"},
}
REGIMES = ("synergistic", "additive")


def _make_synergistic(n: int, n_pairs: int, n_decoy: int, n_noise: int, seed: int):
    """K XOR/sign-product pairs; each operand ~useless alone, pair drives y. Returns X, y, relevant_idx."""
    rng = np.random.default_rng(seed)
    cols = []
    relevant = []
    logit = np.zeros(n)
    for k in range(n_pairs):
        if k % 2 == 0:
            a = rng.integers(0, 2, n).astype(np.float64)
            b = rng.integers(0, 2, n).astype(np.float64)
            contrib = (a.astype(int) ^ b.astype(int)).astype(np.float64) * 2.0 - 1.0
        else:
            a = rng.standard_normal(n)
            b = rng.standard_normal(n)
            contrib = np.sign(a) * np.sign(b)
        ia, ib = len(cols), len(cols) + 1
        cols.append(a + 0.05 * rng.standard_normal(n))
        cols.append(b + 0.05 * rng.standard_normal(n))
        relevant.extend([ia, ib])
        logit += 2.5 * contrib
    # correlated decoys of the first operand (redundancy trap, but NOT relevant)
    op0 = cols[0]
    for j in range(n_decoy):
        cols.append(op0 + (0.3 + 0.05 * j) * rng.standard_normal(n))
    for _ in range(n_noise):
        cols.append(rng.standard_normal(n))
    X = np.column_stack(cols)
    p = 1.0 / (1.0 + np.exp(-logit - 0.1 * rng.standard_normal(n)))
    y = (rng.random(n) < p).astype(np.int64)
    return X, y, sorted(relevant)


def _make_additive(n: int, n_main: int, n_decoy: int, n_noise: int, seed: int):
    """K independent linear main effects + correlated decoys + noise. Returns X, y, relevant_idx."""
    rng = np.random.default_rng(seed)
    cols = []
    relevant = []
    logit = np.zeros(n)
    for k in range(n_main):
        f = rng.standard_normal(n)
        cols.append(f)
        relevant.append(len(cols) - 1)
        logit += 1.5 * f
    main0 = cols[0]
    for j in range(n_decoy):
        cols.append(main0 + (0.3 + 0.05 * j) * rng.standard_normal(n))
    for _ in range(n_noise):
        cols.append(rng.standard_normal(n))
    X = np.column_stack(cols)
    p = 1.0 / (1.0 + np.exp(-logit - 0.1 * rng.standard_normal(n)))
    y = (rng.random(n) < p).astype(np.int64)
    return X, y, sorted(relevant)


def _holdout_auc(X_tr, y_tr, X_ho, y_ho, sel_idx):
    from sklearn.metrics import roc_auc_score
    if not sel_idx or len(np.unique(y_ho)) < 2 or len(np.unique(y_tr)) < 2:
        return {"lgbm": None, "logit": None}
    Xtr, Xho = X_tr[:, sel_idx], X_ho[:, sel_idx]
    out = {}
    try:
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(n_estimators=80, num_leaves=15, verbose=-1, random_state=0)
        m.fit(Xtr, y_tr)
        out["lgbm"] = float(roc_auc_score(y_ho, m.predict_proba(Xho)[:, 1]))
    except Exception:
        out["lgbm"] = None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=500)
        m.fit(Xs.transform(Xtr), y_tr)
        out["logit"] = float(roc_auc_score(y_ho, m.predict_proba(Xs.transform(Xho))[:, 1]))
    except Exception:
        out["logit"] = None
    return out


def _build_mrmr(variant: str, seed: int):
    from mlframe.feature_selection.filters import MRMR
    base = dict(
        fe_max_steps=0, interactions_max_order=1,
        full_npermutations=3, baseline_npermutations=2,
        random_seed=seed, use_gpu=False, n_jobs=1, verbose=0, cv=2,
    )
    base.update(VARIANTS[variant])
    return MRMR(**base)


def _run_cell(variant, regime, n, seed, dims):
    n_k, n_decoy, n_noise = dims
    if regime == "synergistic":
        X, y, relevant = _make_synergistic(n, n_k, n_decoy, n_noise, seed)
    else:
        X, y, relevant = _make_additive(n, n_k, n_decoy, n_noise, seed)
    rng = np.random.default_rng(seed + 7919)
    perm = rng.permutation(n)
    cut = int(0.6 * n)
    tr, ho = perm[:cut], perm[cut:]
    sel = _build_mrmr(variant, seed)
    t0 = time.perf_counter()
    sel.fit(X[tr], y[tr])
    fit_s = time.perf_counter() - t0
    sel_idx = sorted(int(i) for i in sel.get_support(indices=True).tolist())
    rel_set, sel_set = set(relevant), set(sel_idx)
    tp = len(sel_set & rel_set)
    auc = _holdout_auc(X[tr], y[tr], X[ho], y[ho], sel_idx)
    return {
        "variant": variant, "regime": regime, "n": n, "seed": seed,
        "n_selected": len(sel_idx), "selected": sel_idx, "relevant": relevant,
        "tp": tp, "precision": tp / len(sel_set) if sel_set else 0.0,
        "recall": tp / len(rel_set) if rel_set else 0.0,
        "auc_lgbm": auc["lgbm"], "auc_logit": auc["logit"], "fit_s": fit_s,
    }


def _load_done(path):
    done = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        done.add((r["variant"], r["regime"], int(r["n"]), int(r["seed"])))
                    except json.JSONDecodeError:
                        pass
    return done


def _append(path, row):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def run(smoke):
    path = SMOKE_JSONL if smoke else FULL_JSONL
    if smoke:
        n_values, seeds = (1500,), list(range(2))
        dims = (2, 3, 6)
    else:
        n_values, seeds = (4000, 8000), list(range(10))
        dims = (3, 6, 12)
    cells = [(v, r, n, s) for v in VARIANTS for r in REGIMES for n in n_values for s in seeds]
    done = _load_done(path)
    total = len(cells)
    print(f"[synergy-bench] mode={'smoke' if smoke else 'full'} total={total} done={len(done)} -> {path.name}", flush=True)
    for i, (v, r, n, s) in enumerate(cells, 1):
        if (v, r, n, s) in done:
            print(f"[{i}/{total}] SKIP {(v, r, n, s)}", flush=True)
            continue
        t0 = time.perf_counter()
        row = _run_cell(v, r, n, s, dims)
        _append(path, row)
        done.add((v, r, n, s))
        print(f"[{i}/{total}] {v:>8} {r:<12} n={n:<6} seed={s:<2} prec={row['precision']:.3f} "
              f"rec={row['recall']:.3f} auc_lgbm={row['auc_lgbm']} auc_logit={row['auc_logit']} "
              f"nsel={row['n_selected']} fit={row['fit_s']:.1f}s wall={time.perf_counter()-t0:.1f}s", flush=True)
    print("[synergy-bench] complete", flush=True)


def summarize(smoke):
    path = SMOKE_JSONL if smoke else FULL_JSONL
    if not path.exists():
        print(f"[summarize] no file {path}")
        return
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        print("[summarize] empty")
        return

    def _mean(vs):
        vs = [v for v in vs if v is not None]
        return float(np.mean(vs)) if vs else float("nan")

    groups = defaultdict(list)
    for r in rows:
        groups[(r["regime"], r["variant"], int(r["n"]))].append(r)
    print("\n=== per (regime, variant, n) MEANS ===")
    h = f"{'regime':<12} {'variant':>8} {'n':>6} {'prec':>6} {'rec':>6} {'F1':>6} {'auc_lgbm':>9} {'auc_logit':>10} {'nsel':>5} {'cells':>5}"
    print(h); print("-" * len(h))
    for key in sorted(groups):
        g = groups[key]
        prec, rec = _mean([r["precision"] for r in g]), _mean([r["recall"] for r in g])
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        reg, v, n = key
        print(f"{reg:<12} {v:>8} {n:>6} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} "
              f"{_mean([r['auc_lgbm'] for r in g]):>9.4f} {_mean([r['auc_logit'] for r in g]):>10.4f} "
              f"{_mean([r['n_selected'] for r in g]):>5.1f} {len(g):>5}")

    # paired wins vs default, per regime
    by = defaultdict(dict)
    for r in rows:
        by[(r["regime"], int(r["n"]), int(r["seed"]))][r["variant"]] = r
    print("\n=== PAIRED WINS vs default (per regime) ===")
    for target in ("jmim", "auto"):
        agg = defaultdict(lambda: {"rec": [0, 0, 0], "auc": [0, 0, 0]})
        for (reg, n, s), vs in by.items():
            if "default" not in vs or target not in vs:
                continue
            b, t = vs["default"], vs[target]
            dr = t["recall"] - b["recall"]
            agg[reg]["rec"][0 if dr > 1e-9 else (2 if dr < -1e-9 else 1)] += 1
            if t["auc_lgbm"] is not None and b["auc_lgbm"] is not None:
                da = t["auc_lgbm"] - b["auc_lgbm"]
                agg[reg]["auc"][0 if da > 1e-4 else (2 if da < -1e-4 else 1)] += 1
        for reg in REGIMES:
            rc, ac = agg[reg]["rec"], agg[reg]["auc"]
            print(f"  {target:>5} vs default [{reg:<12}] recall W/T/L={rc[0]}/{rc[1]}/{rc[2]}  auc_lgbm W/T/L={ac[0]}/{ac[1]}/{ac[2]}")

    print("\n[decision rule] AUTO ships as default ONLY if: (synergistic) auto matches jmim recovery+auc gains AND")
    print("                (additive) auto matches default (no over-selection regression). Else auto ships opt-in.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(argv)
    if a.summarize:
        summarize(a.smoke)
    else:
        run(a.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
