"""Fix 12 — c0098 root-cause dig: trace value lifecycle through the suite."""
import sys, functools
sys.path.insert(0, r"D:/Upd/Programming/PythonCodeRepository/mlframe")
sys.path.insert(0, r"D:/Upd/Programming/PythonCodeRepository/mlframe/tests/training")

# Monkeypatch MRMR.fit to print X state on entry.
from mlframe.feature_selection.filters import MRMR
_orig_fit = MRMR.fit
def _spy_fit(self, X, y=None, *args, **kwargs):
    try:
        import polars as pl
        import pandas as pd
        print(f"[SPY MRMR.fit] module={type(X).__module__} qualname={type(X).__qualname__} isinstance pl={isinstance(X, pl.DataFrame)} isinstance pd={isinstance(X, pd.DataFrame)}", file=sys.stderr, flush=True)
        if hasattr(X, "columns"):
            cols = list(X.columns) if not hasattr(X, "schema") else list(X.schema.keys())
            for c in cols:
                if c.startswith("cat_"):
                    if isinstance(X, pl.DataFrame):
                        sample = X[c].head(5).to_list()
                        dt = X.schema[c]
                    elif isinstance(X, pd.DataFrame):
                        sample = X[c].iloc[:5].tolist()
                        dt = X[c].dtype
                    else:
                        sample = "?"
                        dt = "?"
                    has_nan = False
                    try:
                        import numpy as np
                        if isinstance(X, pd.DataFrame):
                            has_nan = bool(X[c].isna().any())
                        else:
                            has_nan = bool(X[c].is_null().any())
                    except Exception:
                        pass
                    print(f"[SPY MRMR.fit] type(X)={type(X).__name__} col={c} dtype={dt} first5={sample} has_nan={has_nan}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[SPY failed: {e}]", file=sys.stderr, flush=True)
    return _orig_fit(self, X, y, *args, **kwargs)
MRMR.fit = _spy_fit

from _fuzz_combo import FuzzCombo, build_frame_for_combo
from shared import SimpleFeaturesAndTargetsExtractor

combo = FuzzCombo(
    models=("lgb", "xgb"), input_type="polars_utf8", n_rows=300,
    cat_feature_count=1, null_fraction_cats=0.0, use_mrmr_fs=True,
    weight_schemas=("uniform",), target_type="binary_classification",
    auto_detect_cats=True, align_polars_categorical_dicts=False, seed=98,
)
df, target_col, _ = build_frame_for_combo(combo)
fte = SimpleFeaturesAndTargetsExtractor(target_column=target_col, regression=False)

from mlframe.training.core import train_mlframe_models_suite
try:
    trained, _ = train_mlframe_models_suite(
        df=df, target_name="repro", model_name="repro",
        features_and_targets_extractor=fte,
        mlframe_models=["lgb", "xgb"],
        hyperparams_config={"iterations": 3,
            "xgb_kwargs": {"device": "cpu", "verbosity": 0},
            "lgb_kwargs": {"device_type": "cpu", "verbose": -1}},
        init_common_params={"drop_columns": [], "verbose": 1},
        use_ordinary_models=True, use_mlframe_ensembles=False,
        data_dir=r"D:/Temp/repro_c0098", models_dir="models", verbose=0,
        use_mrmr_fs=True, mrmr_kwargs={"verbose": 0, "max_runtime_mins": 1, "n_workers": 1,
            "quantization_nbins": 5, "use_simple_mode": True,
            "min_nonzero_confidence": 0.9, "max_consec_unconfirmed": 3, "full_npermutations": 3},
    )
    print("PASS")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
