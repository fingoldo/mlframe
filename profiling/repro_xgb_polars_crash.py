"""Standalone reproducer for the XGBoost "Python has stopped working"
crash observed 2026-04-20 on the 9M-row prod_jobsdetails training.

Hypotheses tested, in order:

  (A) Polars Categorical via XGB fastpath (enable_categorical=True) is
      the trigger on Windows. Casting to pandas Categorical first should
      work around it.

  (B) High-cardinality categorical feature (_raw_tags / occupation /
      _raw_languages ~ 1000-2000 unique) combined with
      max_cat_threshold=100 trips native code.

  (C) Scale (7M rows × 19 cat features) alone causes the crash —
      reproduces only at full size, not at sample.

Mirrors prod params from mlframe.training.helpers:
  XGB_GENERAL_PARAMS: enable_categorical=True, max_cat_threshold=100,
  tree_method="hist", n_jobs=physical_cores, n_estimators=500, es_rounds=100.

Data shape matches the 2026-04-20 prod log:
  19 categoricals after text-feature drop, cardinalities {_raw_languages:1769,
  occupation:~500, ...}, 95 numeric (Float32/Int16/Bool), binary target.

Row counts: 500_000 (smoke) → 2_000_000 → 7_000_000. Stops at the first
size that crashes. ``__MISSING__`` fill is applied to match prod's
upstream pre-fit step for Polars Categorical cat_features.

Usage:
    python -m mlframe.profiling.repro_xgb_polars_crash
    # or a specific phase:
    XGB_REPRO_MODE=polars python -m mlframe.profiling.repro_xgb_polars_crash
    XGB_REPRO_MODE=pandas python -m mlframe.profiling.repro_xgb_polars_crash
    XGB_REPRO_ROWS=2000000 python -m mlframe.profiling.repro_xgb_polars_crash
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# Prod cardinalities from 2026-04-20 log, in descending order.
# _raw_languages is the real big one (1769). The 4 text features
# (skills_text etc.) are dropped before XGB sees the frame so they're
# not here.
PROD_CAT_CARDINALITIES = [
    # Bumped to stress-test: prod might have _raw_tags / occupation /
    # _raw_segmentation-leftovers in 5k-20k range. If real cardinality is
    # lower, no harm — XGB handles whatever's there.
    ("_raw_tags", 20_000),       # stress: guild/tag-like column
    ("occupation", 10_000),      # stress
    ("_raw_languages", 1_769),   # exact from prod log
    ("category", 500),
    ("category_group", 50),
    ("workload", 15),
    ("hourly_budget_type", 10),
    ("contractor_tier", 10),
    ("job_post_type", 10),
    ("job_post_device", 8),
    ("job_post_browser", 20),
    ("job_post_source", 12),
    ("job_post_flow_type", 8),
    ("job_type", 5),
    ("job_urgency", 6),
    ("desc_ai_opted_in", 3),
    ("job_local_flexibility", 4),
    ("qual_type", 8),
    ("job_req_english", 5),
]
# 19 cat features — matches prod "Tier (False, False)" XGB branch exactly


def build_synthetic_frame(n_rows: int, seed: int = 42) -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    cols = {}

    # 19 Polars Categoricals. 15 of them are filled-null with "__MISSING__"
    # to match prod's upstream fill step.
    for i, (name, k) in enumerate(PROD_CAT_CARDINALITIES):
        pool = [f"{name}_v{j}" for j in range(k)]
        vals = rng.choice(pool, size=n_rows).tolist()
        # Match prod: 15 cats are "nullable" and got filled with "__MISSING__"
        # upstream. First 4 (highest cardinality) have no fill — they're
        # not in the fill list in the log. Emulate by sprinkling a few
        # "__MISSING__" sentinel values into the rest.
        if i >= 4:
            null_frac = 0.05 + 0.1 * rng.random()
            null_mask = rng.random(n_rows) < null_frac
            for idx in np.where(null_mask)[0]:
                vals[idx] = "__MISSING__"
        cols[name] = pl.Series(name, vals, dtype=pl.String).cast(pl.Categorical)

    # ~40 Float32 (prod had 38 Float32)
    for i in range(40):
        cols[f"num_f{i}"] = rng.standard_normal(n_rows).astype(np.float32)

    # ~35 Int16
    for i in range(35):
        cols[f"num_i{i}"] = rng.integers(-1000, 1000, size=n_rows).astype(np.int16)

    # 15 Boolean
    for i in range(15):
        cols[f"bool_{i}"] = rng.random(n_rows) > 0.5

    # 2 Int32, 2 Int64, 2 UInt32 (prod mix)
    for i in range(2):
        cols[f"num_i32_{i}"] = rng.integers(0, 1_000_000, size=n_rows).astype(np.int32)
        cols[f"num_i64_{i}"] = rng.integers(0, 10_000_000, size=n_rows).astype(np.int64)
        cols[f"num_u32_{i}"] = rng.integers(0, 1_000_000, size=n_rows).astype(np.uint32)

    # Binary target with mild signal from num_f0
    logits = cols["num_f0"].astype(np.float64) + 0.2 * cols["num_f1"].astype(np.float64)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n_rows) < probs).astype(np.int8)

    df = pl.DataFrame(cols)
    return df, y


def _log_cat_cardinalities(X, label: str) -> None:
    """Print cardinality of every categorical column so we can see what
    XGB is about to consume. Polars and pandas frames handled uniformly."""
    print(f"\n[{label}] categorical column cardinalities:")
    if isinstance(X, pl.DataFrame):
        for name, dtype in X.schema.items():
            if dtype == pl.Categorical:
                print(f"  {name}: n_unique={X[name].n_unique():_}  dtype=pl.Categorical")
    else:  # pandas
        for name in X.columns:
            dt = X[name].dtype
            if str(dt).startswith("category"):
                print(f"  {name}: n_unique={X[name].cat.categories.size:_}  dtype=pd.Categorical")


def fit_xgb(X_train, y_train, X_val, y_val, label: str, n_estimators: int = 50) -> None:
    """Fit XGB with the exact production params. Smaller n_estimators
    (50 not 500) so we crash early rather than waste time training — the
    crash (if any) reproduces during DMatrix build or the first few
    iterations, not at the end.

    Verbosity=3 (debug) prints XGB C++ internals to stderr: DMatrix
    build, categorical handling, split finding. Plus verbose=True on
    fit() shows per-iter eval metrics so we see exactly which iteration
    triggers the crash (if any)."""
    import psutil
    from xgboost import XGBClassifier

    n_jobs = psutil.cpu_count(logical=False) or 4

    _log_cat_cardinalities(X_train, label)

    print(f"\n[{label}] fitting XGBClassifier on shape={X_train.shape}, n_jobs={n_jobs}, n_est={n_estimators}")
    print(f"[{label}] dtype counts: {dict(zip(*np.unique([str(t) for t in (X_train.dtypes if hasattr(X_train, 'dtypes') else X_train.schema.values())], return_counts=True)))}")

    # XGBOOST_VERBOSITY=3 is redundant with verbosity=3 below but belts
    # and braces in case the classifier init-path misses it.
    os.environ.setdefault("XGBOOST_VERBOSITY", "3")

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=0.1,
        enable_categorical=True,
        max_cat_to_onehot=1,
        max_cat_threshold=100,
        tree_method="hist",
        device="cpu",
        n_jobs=n_jobs,
        early_stopping_rounds=20,
        random_state=42,
        objective="binary:logistic",
        eval_metric="logloss",
        verbosity=3,  # 0=silent, 1=warning, 2=info, 3=debug
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    t = time.perf_counter() - t0
    print(f"[{label}] ✓ fit OK in {t:.1f}s — best_iter={model.best_iteration}")


def run_for_rows(n_rows: int, mode: str) -> bool:
    """Returns True if this configuration completed without crashing."""
    print(f"\n{'=' * 80}")
    print(f"Phase: n_rows={n_rows:_}, mode={mode}")
    print(f"{'=' * 80}")

    t0 = time.perf_counter()
    df_pl, y = build_synthetic_frame(n_rows)
    print(f"Built synthetic frame {df_pl.shape} in {time.perf_counter() - t0:.1f}s, "
          f"est. size={df_pl.estimated_size() / 1e9:.2f}GB")

    # 90/10 split to match prod-ish val fraction
    n_val = max(1000, n_rows // 10)
    n_train = n_rows - n_val
    y_train = y[:n_train]
    y_val = y[n_train:]
    df_train_pl = df_pl.slice(0, n_train)
    df_val_pl = df_pl.slice(n_train, n_val)

    if mode == "polars":
        # Polars fastpath — exactly what XGBoostStrategy does in prod
        try:
            fit_xgb(df_train_pl, y_train, df_val_pl, y_val, label="POLARS")
            return True
        except BaseException as e:
            print(f"[POLARS] ✗ raised: {type(e).__name__}: {e}")
            traceback.print_exc()
            return False

    elif mode == "pandas":
        # pandas conversion to test hypothesis (A): is Polars specifically the trigger?
        t0 = time.perf_counter()
        pdf_train = df_train_pl.to_pandas()
        pdf_val = df_val_pl.to_pandas()
        # Ensure cat dtypes survived
        for c in [n for n, _ in PROD_CAT_CARDINALITIES]:
            if c in pdf_train.columns and not str(pdf_train[c].dtype).startswith("category"):
                pdf_train[c] = pdf_train[c].astype("category")
                pdf_val[c] = pdf_val[c].astype("category")
        print(f"to_pandas() took {time.perf_counter() - t0:.1f}s")
        try:
            fit_xgb(pdf_train, y_train, pdf_val, y_val, label="PANDAS")
            return True
        except BaseException as e:
            print(f"[PANDAS] ✗ raised: {type(e).__name__}: {e}")
            traceback.print_exc()
            return False

    else:
        raise ValueError(f"unknown mode: {mode}")


def main():
    mode_env = os.environ.get("XGB_REPRO_MODE", "both").lower()
    rows_env = os.environ.get("XGB_REPRO_ROWS")

    if rows_env:
        row_schedule = [int(rows_env)]
    else:
        # Smoke first, then scale. Stops at first row-count where Polars
        # crashes — that's the row-count we use for the pandas comparison.
        row_schedule = [500_000, 2_000_000, 5_000_000]

    if mode_env == "polars":
        modes = ["polars"]
    elif mode_env == "pandas":
        modes = ["pandas"]
    else:
        modes = ["polars", "pandas"]

    import xgboost
    import polars
    print(f"xgboost {xgboost.__version__}, polars {polars.__version__}, "
          f"python {sys.version.split()[0]}")

    for n in row_schedule:
        polars_ok = True
        for mode in modes:
            ok = run_for_rows(n, mode)
            if mode == "polars":
                polars_ok = ok
                if not ok:
                    print(f"\n*** POLARS CRASHED at n_rows={n:_}. Testing pandas at same size... ***")
            if not ok and mode == "pandas":
                print(f"\n*** PANDAS ALSO CRASHED at n_rows={n:_} — hypothesis (A) disproved ***")
                return
        if not polars_ok:
            # Polars crashed. If pandas succeeded in the inner loop we'd
            # have printed the good news above; stop scaling — we have
            # enough data.
            print("\n=== Stopping scale-up: crash already reproduced. ===")
            return

    print("\n=== All sizes completed without crash. Hypothesis (C) disproved at the tested scales. ===")


if __name__ == "__main__":
    main()
