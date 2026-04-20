"""Check whether the pure pipeline crashes without ANY value replacement."""
import sys, time, numpy as np, polars as pl
from xgboost import XGBClassifier
if sys.platform == "win32":
    import ctypes; ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
try:
    import faulthandler; faulthandler.enable(all_threads=True)
except Exception: pass
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--parquet", required=True)
args = ap.parse_args()

print(f"polars {pl.__version__}, using_string_cache={pl.using_string_cache()}",
      flush=True)

t0 = time.perf_counter()
df = pl.read_parquet(args.parquet, columns=["skills_text", "category", "job_posted_at"])
print(f"loaded {df.shape} in {time.perf_counter()-t0:.1f}s", flush=True)

df = df.sort("job_posted_at").drop("job_posted_at")
df = df.with_columns([
    pl.col("skills_text").cast(pl.String).cast(pl.Categorical),
    pl.col("category").cast(pl.String).cast(pl.Categorical),
])
print(f"after cast: skills n_unique={df['skills_text'].n_unique():_}, "
      f"category n_unique={df['category'].n_unique()}", flush=True)
print(f"  category codes before fill_null: "
      f"max={df['category'].to_physical().drop_nulls().max()}, "
      f"min={df['category'].to_physical().drop_nulls().min()}", flush=True)

df = df.drop("skills_text")
df = df.with_columns(pl.col("category").fill_null("__MISSING__"))
print(f"  category codes after fill_null: "
      f"max={df['category'].to_physical().max()}, "
      f"distinct={df['category'].to_physical().n_unique()}", flush=True)

train = df[:211_168]
val   = df[211_168:211_168+100_000]
rng = np.random.default_rng(42)
y_tr = rng.integers(0, 2, train.height).astype(np.int8)
y_v  = rng.integers(0, 2, val.height).astype(np.int8)
print(f"train={train.shape}, val={val.shape}", flush=True)

print("fitting XGB...", flush=True)
m = XGBClassifier(n_estimators=5, enable_categorical=True, tree_method="hist",
                  device="cpu", n_jobs=-1, verbosity=1,
                  max_cat_to_onehot=1, max_cat_threshold=100,
                  early_stopping_rounds=3,
                  objective="binary:logistic", eval_metric="logloss")
t0 = time.perf_counter()
try:
    m.fit(train, y_tr, eval_set=[(val, y_v)], verbose=False)
    print(f"FIT_OK in {time.perf_counter()-t0:.1f}s", flush=True)
except BaseException as e:
    print(f"RAISED {type(e).__name__}: {e}", flush=True)
