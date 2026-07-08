"""cProfile harness for the feature-selector wiring in train_mlframe_models_suite.

Profiles the FULL pipeline (train + predict + save + load) of an MRMR-driven
suite so the FS integration hotspots (selector instantiate / fit_transform /
report build / pickle-of-fitted-selector / predict-time pre_pipeline reuse)
are attributed. Run:

    CUDA_VISIBLE_DEVICES="" D:/ProgramData/anaconda3/python.exe -m mlframe.training._profile_fs_suite_integration

Not a test; lives in the package so any maintainer can rerun the same shape.
"""
from __future__ import annotations

import cProfile
import os
import pstats
import tempfile
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd


def _make_frame(n=1500, p=20, seed=0):
    """Build a synthetic binary-classification frame (``p`` gaussian features, the first three informative via a linear-logit rule) sized to keep MRMR fit time low while still exercising the FS-integration code paths."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float64)
    logits = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.8 * X[:, 2] + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = (logits > 0).astype(np.int8)
    return df


def _run_once():
    """Run one full train+predict cycle through ``train_mlframe_models_suite`` with MRMR feature selection enabled, so the profiler attributes time to selector fit_transform / report build / pickle / predict-time reuse rather than to the harness itself. Predict failures are swallowed and printed (not fatal to the profile)."""
    from mlframe.training.core import train_mlframe_models_suite, predict_from_models
    from mlframe.training import FeatureSelectionConfig, OutputConfig
    from tests.training.shared import SimpleFeaturesAndTargetsExtractor

    df = _make_frame()
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    fs_cfg = FeatureSelectionConfig(
        use_mrmr_fs=True,
        mrmr_kwargs={"verbose": 0, "use_simple_mode": True, "max_runtime_mins": 0.3},
    )
    with tempfile.TemporaryDirectory() as d:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = train_mlframe_models_suite(
                df=df,
                target_name="target",
                model_name="prof_fs",
                features_and_targets_extractor=fte,  # type: ignore[arg-type]  # tests.training.shared.SimpleFeaturesAndTargetsExtractor is a duck-typed test double, not a real FeaturesAndTargetsExtractor subclass
                mlframe_models=["lgb"],
                feature_selection_config=fs_cfg,
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                output_config=OutputConfig(data_dir=d, models_dir="models"),
                verbose=0,
            )
            models = res[0] if isinstance(res, tuple) else getattr(res, "models", None)
            metadata = res[1] if isinstance(res, tuple) else getattr(res, "metadata", None)
            try:
                predict_from_models(df=df, models=models, metadata=metadata, features_and_targets_extractor=fte, verbose=0)  # type: ignore[arg-type]  # duck-typed test double, see above
            except Exception as e:
                print("predict skipped:", type(e).__name__, str(e)[:120])
    return res


def main():
    """Profile ``_run_once`` under cProfile and print the top-30 cumulative-time and top-20 self-time (tottime) frames -- the standard entry point for rerunning this harness from the command line."""
    pr = cProfile.Profile()
    pr.enable()
    _run_once()
    pr.disable()
    st = pstats.Stats(pr)
    st.sort_stats("cumulative")
    print("=== top 30 cumulative ===")
    st.print_stats(30)
    st.sort_stats("tottime")
    print("=== top 20 tottime ===")
    st.print_stats(20)


if __name__ == "__main__":
    main()
