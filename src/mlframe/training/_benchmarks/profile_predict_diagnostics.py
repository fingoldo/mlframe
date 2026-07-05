"""Discovery cProfile of the PREDICT + honest_diagnostics + REPORT path.

The train-core harness (``profile_training_core_hotpath.py``) isolates the
FTE -> split -> preprocess -> fit -> metrics path. This sibling isolates the
ALWAYS-ON finalize path that runs AFTER training on a fitted suite:

  predict (prod-shape ~200k rows) -> run_honest_diagnostics(ctx, models, meta)
  -> report/provenance rendering.

``run_honest_diagnostics`` (training/honest_diagnostics.py) is the aggregator
that the suite invokes at finalize. It has four blocks:
  1. bootstrap_ci  -- 1000-resample bootstrap of roc_auc/brier/log_loss/ece per
                      model entry (numba metric kernels).
  2. drift_psi     -- categorical PSI across train/val/test
                      (compute_categorical_drift_psi). The drift-PSI value_counts
                      blowup on ndarray-cell object columns was fixed; this
                      harness re-profiles what is the top mlframe-side cost NOW.
  3. calibration   -- pick_best_calibrator on oof (500-resample bootstrap + plot).
  4. provenance    -- format_provenance_table.

We profile against a realistically fitted single HGB classifier whose
test_probs / oof_probs are PROD-shape so the bootstrap loops dominate the way
they do on a real 100k-500k holdout. CPU-only.

cProfile attribution caveat: cProfile inflates deep pandas/sklearn/numba-
dispatch stacks ~10-13x vs a wall microbench. Every hotspot flagged here is
re-checked with an isolated wall microbench in ``--microbench`` mode before
being declared a real lead.

Run (CPU-only):
    CUDA_VISIBLE_DEVICES="" python -m mlframe.training._benchmarks.profile_predict_diagnostics

Output: JSON summary -> sibling ``_results/predict_diagnostics.json``.
"""
from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Native DLL-load-order guard (see profile_training_core_hotpath docstring).
import mlframe.training.pipeline as _pipeline_preimport  # noqa: E402,F401

_RESULTS_DIR = Path(__file__).resolve().parent / "_results"


class _FakeSplitCfg:
    random_seed = 42


class _FakeEntry:
    """Minimal model-entry shim exposing what run_honest_diagnostics reads."""

    def __init__(self, model_name, test_target, test_probs, oof_probs, oof_target):
        self.model_name = model_name
        self.model = None
        self.test_target = test_target
        self.test_probs = test_probs
        self.test_preds = (test_probs[:, 1] >= 0.5).astype(int)
        self.oof_probs = oof_probs
        self.oof_target = oof_target


class _FakeCtx:
    def __init__(self, train_df, val_df, test_df):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.split_config = _FakeSplitCfg()
        self.data_dir = ""
        self.models_dir = ""
        self.metadata = {}


def _build_state(n_rows: int, n_cat_cols: int, seed: int):
    """Build a prod-shape (ctx, models, metadata) finalize state.

    test_probs / oof_probs are prod-length so the 1000-resample bootstrap loops
    dominate as on a real holdout. train/val/test frames carry mid-card string
    categorical columns so the PSI drift block has real work.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    # Binary classification holdout: realistic separable-ish probs.
    y_test = (rng.random(n_rows) < 0.35).astype(int)
    logits = 1.2 * y_test + rng.normal(0, 1.0, n_rows)
    p1 = 1.0 / (1.0 + np.exp(-logits))
    test_probs = np.column_stack([1.0 - p1, p1])

    n_oof = n_rows  # oof typically train-aligned; same order of magnitude
    y_oof = (rng.random(n_oof) < 0.35).astype(int)
    logits_o = 1.1 * y_oof + rng.normal(0, 1.05, n_oof)
    po = 1.0 / (1.0 + np.exp(-logits_o))
    oof_probs = np.column_stack([1.0 - po, po])

    entry = _FakeEntry("hgb", y_test, test_probs, oof_probs, y_oof)
    models = {"binary_classification": {"y": [entry]}}

    # Frames for PSI drift: mid-card string categoricals with a slight
    # train->test level shift so PSI has signal.
    def _mk_df(n, shift):
        cats = {}
        for c in range(n_cat_cols):
            card = 8 + c * 4
            levels = np.array([f"L{c}_{k}" for k in range(card)])
            probs = rng.random(card) + shift * (np.arange(card) / card)
            probs = probs / probs.sum()
            cats[f"cat_{c}"] = rng.choice(levels, size=n, p=probs)
        # a couple numeric cols too (PSI block skips them but they're realistic)
        cats["num_0"] = rng.normal(0, 1, n)
        cats["num_1"] = rng.normal(5, 2, n)
        return pd.DataFrame(cats)

    train_df = _mk_df(n_rows, 0.0)
    val_df = _mk_df(max(n_rows // 4, 1000), 0.1)
    test_df = _mk_df(max(n_rows // 4, 1000), 0.2)

    ctx = _FakeCtx(train_df, val_df, test_df)
    return ctx, models, ctx.metadata


def _profile_one(n_rows: int, n_cat_cols: int, seed: int, top_n: int) -> dict[str, Any]:
    from mlframe.training.honest_diagnostics import run_honest_diagnostics

    ctx, models, meta = _build_state(n_rows, n_cat_cols, seed)

    # Warm numba metric kernels once (JIT compile is not the steady-state cost).
    run_honest_diagnostics(ctx, models, dict(meta))

    profiler = cProfile.Profile()
    status = "OK"
    t0 = time.perf_counter()
    profiler.enable()
    try:
        run_honest_diagnostics(ctx, models, meta)
    except Exception as e:  # noqa: BLE001
        status = f"{type(e).__name__}: {e}"[:200]
    finally:
        profiler.disable()
    wall = time.perf_counter() - t0

    def _table(sort_key: str) -> str:
        s = io.StringIO()
        pstats.Stats(profiler, stream=s).sort_stats(sort_key).print_stats(top_n)
        return s.getvalue()

    return {
        "n_rows": n_rows,
        "n_cat_cols": n_cat_cols,
        "seed": seed,
        "wall_s": round(wall, 3),
        "status": status,
        "cumulative_top": _table("cumulative"),
        "tottime_top": _table("tottime"),
    }


def _microbench() -> dict[str, Any]:
    """Isolated wall-time microbenches for the leads cProfile surfaces.

    Each returns per-call / total wall so we can separate a REAL hotspot from
    cProfile deep-stack attribution noise.
    """
    import pandas as pd

    out: dict[str, Any] = {}
    rng = np.random.default_rng(0)
    n = 200_000

    # 1) The bootstrap loop on numba metric kernels (the dominant block).
    from mlframe.evaluation.bootstrap import bootstrap_metrics
    from mlframe.metrics.core import (
        fast_roc_auc_unstable as _fast_auc,
        fast_brier_score_loss as _fast_brier,
        fast_log_loss as _fast_ll,
    )
    from mlframe.calibration.policy import _ece_score

    y = (rng.random(n) < 0.35).astype(int)
    p = 1.0 / (1.0 + np.exp(-(1.2 * y + rng.normal(0, 1, n))))

    def _auc(yy, pp):
        return float(_fast_auc(yy, pp))

    def _brier(yy, pp):
        return float(_fast_brier(yy.astype(np.float64, copy=False), pp.astype(np.float64, copy=False)))

    def _ll(yy, pp):
        return float(_fast_ll(yy.astype(np.float64, copy=False), pp.astype(np.float64, copy=False)))

    mfns = {"roc_auc": _auc, "brier": _brier, "log_loss": _ll, "ece": lambda yy, pp: _ece_score(yy, pp)}
    bootstrap_metrics(y, p, mfns, n_bootstrap=50, alpha=0.05, stratify=y, random_state=0)  # warm
    t0 = time.perf_counter()
    bootstrap_metrics(y, p, mfns, n_bootstrap=1000, alpha=0.05, stratify=y, random_state=0)
    out["bootstrap_metrics_1000_n200k_wall_s"] = round(time.perf_counter() - t0, 4)

    # 1b) stratified resample index generation alone (the per-resample slice).
    from numpy.random import default_rng as _drng
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    g = _drng(0)
    t0 = time.perf_counter()
    for _ in range(1000):
        i = np.concatenate([g.choice(pos, pos.size, replace=True), g.choice(neg, neg.size, replace=True)])
        _ = y[i]
        _ = p[i]
    out["stratified_resample_1000_n200k_wall_s"] = round(time.perf_counter() - t0, 4)

    # 2) ece in the bootstrap loop (binning) -- 1000 calls.
    _ece_score(y, p)  # warm
    t0 = time.perf_counter()
    for _ in range(1000):
        _ece_score(y, p)
    out["ece_score_x1000_n200k_wall_s"] = round(time.perf_counter() - t0, 4)

    # 3) calibration block: pick_best_calibrator (500-resample + ece).
    from mlframe.calibration.policy import pick_best_calibrator
    t0 = time.perf_counter()
    pick_best_calibrator(probs=None, y=None, oof_probs=np.column_stack([1 - p, p]), oof_y=y, n_bootstrap=500, random_state=0, emit_plot=False)
    out["pick_best_calibrator_500_n200k_wall_s"] = round(time.perf_counter() - t0, 4)

    # 4) categorical PSI drift on mid-card string cols.
    from mlframe.training.feature_drift_report import compute_categorical_drift_psi
    cats = {}
    for c in range(6):
        card = 8 + c * 4
        lv = np.array([f"L{c}_{k}" for k in range(card)])
        cats[f"cat_{c}"] = rng.choice(lv, size=n)
    df = pd.DataFrame(cats)
    compute_categorical_drift_psi(df, df, df)  # warm
    t0 = time.perf_counter()
    compute_categorical_drift_psi(df, df, df)
    out["psi_drift_6cols_n200k_wall_s"] = round(time.perf_counter() - t0, 4)

    return out


def main(top_n: int = 25) -> dict[str, Any]:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    shapes = [
        (100_000, 6, 7),
        (300_000, 6, 7),
    ]
    results = []
    for n, ncat, seed in shapes:
        print(f"=== profiling predict+diagnostics n_rows={n:_} cat_cols={ncat} (CPU-only) ===", flush=True)
        r = _profile_one(n, ncat, seed, top_n)
        print(f"  wall={r['wall_s']}s status={r['status']}", flush=True)
        print(r["cumulative_top"], flush=True)
        print("--- tottime ---", flush=True)
        print(r["tottime_top"], flush=True)
        results.append(r)

    print("=== microbench (wall-time, real-vs-attribution) ===", flush=True)
    mb = _microbench()
    print(json.dumps(mb, indent=2), flush=True)

    out = {"label": "profile-predict-diagnostics", "results": results, "microbench": mb}
    out_path = _RESULTS_DIR / "predict_diagnostics.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}", flush=True)
    return out


if __name__ == "__main__":
    main()
