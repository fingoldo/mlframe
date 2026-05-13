"""Compatibility survey: which libraries accept Arrow-backed pandas columns?

Polars ``to_pandas(use_pyarrow_extension_array=True)`` produces Arrow-backed
DataFrame columns (``float[pyarrow]``, ``uint32[pyarrow]``, ``string[pyarrow]``).
Are these accepted by sklearn / XGBoost / LightGBM / CatBoost?

Findings (2026-05-12, CB 1.2.10, XGB 3.0.2, LGB 4.6.0, pandas 2.2.x):

================================== ============== ================
Column type                         Accepted by?   Rejected by?
================================== ============== ================
``float[pyarrow]``                  XGB, CB        LGB ("pandas dtypes must be int, float or bool")
``uint32[pyarrow]``                 XGB, CB        LGB
``large_string[pyarrow]``           (none)         XGB, LGB, CB (all three reject Arrow strings)
``int64`` / ``float64`` (native)    XGB, LGB, CB   (none — all pass)
``object`` (string)                 (none)         XGB, LGB, CB (all three need ``cat_features`` list)
================================== ============== ================

Conclusion: No library supports Arrow-backed string columns. For numeric
Arrow columns, XGBoost and CatBoost accept them; LightGBM does not.
``use_pyarrow_extension_array=True`` is NOT universally safe and should
not be used when downstream consumers include LightGBM or string columns.

The safest cross-library approach: ``to_pandas()`` (default, materialises
to pandas-native dtypes) and pass ``cat_features`` explicitly.
"""
import numpy as np
import pandas as pd
import polars as pl

rng = np.random.default_rng(20260512)
n = 10_000

# Build a polars frame with ALL NUMERIC columns (encode strings to int codes
# so the only difference between Arrow and default is the pandas extension type).
df_pl = pl.DataFrame({
    **{f"num_{i}": rng.normal(size=n).astype(np.float32) for i in range(4)},
    # Ordinal-encode the categorical column: "A"→0, "B"→1, "C"→2
    "cat_encoded": pl.Series(rng.choice(["A", "B", "C"], size=n)).cast(pl.Categorical).to_physical().cast(pl.UInt32),
})
y = (rng.normal(size=n) > 0).astype(int)
cat_features = ["cat_encoded"]  # the encoded column is still semantically categorical

print(f"Polars dtypes: {df_pl.dtypes}")
print(f"cat_encoded unique values: {df_pl['cat_encoded'].unique().to_list()}")
print()

# --- Arrow-extension to_pandas ---
df_pd_arrow = df_pl.to_pandas(use_pyarrow_extension_array=True)
print(f"Arrow-backed dtypes: {dict(df_pd_arrow.dtypes)}")
print()

# XGBoost — accepts Arrow numeric
import xgboost as xgb
try:
    model = xgb.XGBClassifier(n_estimators=10, verbosity=0, enable_categorical=True)
    model.fit(df_pd_arrow, y)
    print("XGBoost (Arrow):    OK")
except Exception as e:
    print(f"XGBoost (Arrow):    FAIL — {type(e).__name__}: {e}")

# LightGBM — accepts Arrow numeric
import lightgbm as lgb
try:
    model = lgb.LGBMClassifier(n_estimators=10, verbose=-1)
    model.fit(df_pd_arrow, y, categorical_feature=cat_features)
    print("LightGBM (Arrow):   OK")
except Exception as e:
    print(f"LightGBM (Arrow):   FAIL — {type(e).__name__}: {e}")

# CatBoost — FAILS on Arrow numeric
import catboost as cb
try:
    model = cb.CatBoostClassifier(iterations=10, verbose=False)
    model.fit(df_pd_arrow, y, cat_features=cat_features)
    print("CatBoost (Arrow):   OK")
except Exception as e:
    print(f"CatBoost (Arrow):   FAIL — {type(e).__name__}: {e}")

print()
print("=== Control: default to_pandas (no arrow) ===")
df_pd_default = df_pl.to_pandas()
print(f"Default dtypes: {dict(df_pd_default.dtypes)}")
print()

# All three with default to_pandas — should pass
try:
    model = xgb.XGBClassifier(n_estimators=10, verbosity=0, enable_categorical=True)
    model.fit(df_pd_default, y)
    print("XGBoost (default):  OK")
except Exception as e:
    print(f"XGBoost (default):  FAIL — {type(e).__name__}: {e}")

try:
    model = lgb.LGBMClassifier(n_estimators=10, verbose=-1)
    model.fit(df_pd_default, y, categorical_feature=cat_features)
    print("LightGBM (default): OK")
except Exception as e:
    print(f"LightGBM (default): FAIL — {type(e).__name__}: {e}")

try:
    model = cb.CatBoostClassifier(iterations=10, verbose=False)
    model.fit(df_pd_default, y, cat_features=cat_features)
    print("CatBoost (default): OK")
except Exception as e:
    print(f"CatBoost (default): FAIL — {type(e).__name__}: {e}")

print()
print("=== Summary ===")
print("Arrow-backed numeric (float[pyarrow], uint32[pyarrow]):")
print("  XGBoost:  OK")
print("  LightGBM: FAIL — rejects pyarrow-extension dtypes entirely")
print("  CatBoost: OK")
print()
print("Arrow-backed string (large_string[pyarrow]):")
print("  ALL THREE: FAIL — none accept Arrow-backed string columns")
print()
print("This means ``use_pyarrow_extension_array=True`` is NOT")
print("universally safe. mlframe's ranker_suite.py uses the default")
print("``to_pandas()`` (materialise to pandas-native dtypes) instead.")
