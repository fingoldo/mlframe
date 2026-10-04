"""cProfile (cumtime top 25, stats saved under _results/) of small default fits: MRMR with FE, the default training suite, RFECV, ShapProxiedFS.

Run one mode per process: ``OMP_NUM_THREADS=2 python -m mlframe.feature_selection._benchmarks.profile_default_fits mrmr_pandas``
Modes: mrmr_pandas, mrmr_polars, suite_pandas, suite_polars, rfecv, shapproxied.
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

_OUT = os.path.join(os.path.dirname(__file__), "_results")


def _frame(n: int, p: int, seed: int = 0):
    """Synthetic regression/classification frame with a nonlinear pair interaction and noise columns."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (X[:, 0] * X[:, 1] + np.sin(X[:, 2]) + 0.5 * X[:, 3] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(p)]), y


def _run(mode: str) -> None:
    """Execute one fit for ``mode``."""
    if mode.startswith("mrmr"):
        from mlframe.feature_selection.filters import MRMR

        X, y = _frame(20_000, 40)
        if mode.endswith("polars"):
            import polars as pl

            X = pl.from_pandas(X)
        MRMR(verbose=0, n_jobs=1).fit(X, y)
    elif mode.startswith("suite"):
        from mlframe.training.configs import OutputConfig, ReportingConfig, TrainingSplitConfig
        from mlframe.training.core import train_mlframe_models_suite
        from mlframe.training._benchmarks._earlystop_bench_shared import make_fte

        X, y = _frame(1500, 12)
        df = X.assign(target=y)
        if mode.endswith("polars"):
            import polars as pl

            df = pl.from_pandas(df)
        with tempfile.TemporaryDirectory() as d:
            train_mlframe_models_suite(
                df=df, target_name="t", model_name="prof", features_and_targets_extractor=make_fte(False),
                mlframe_models=["lgb"], reporting_config=ReportingConfig(show_perf_chart=False, show_fi=False),
                use_ordinary_models=True, use_mlframe_ensembles=False, output_config=OutputConfig(data_dir=d, models_dir="models"),
                verbose=0, split_config=TrainingSplitConfig(test_size=0.1, val_size=0.1),
                hyperparams_config={"iterations": 30, "learning_rate": 0.1},
            )
    elif mode == "rfecv":
        from sklearn.linear_model import LogisticRegression
        from mlframe.feature_selection.wrappers import RFECV

        X, y = _frame(2000, 40)
        RFECV(estimator=LogisticRegression(max_iter=200), cv=3, verbose=0).fit(X, y)
    elif mode == "shapproxied":
        from sklearn.ensemble import HistGradientBoostingClassifier
        from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

        X, y = _frame(800, 25)
        ShapProxiedFS(model=HistGradientBoostingClassifier(max_iter=30), random_state=0, verbose=False).fit(X, y)
    else:
        raise SystemExit(f"unknown mode {mode}")


def main(mode: str) -> None:
    """Profile ``mode`` and save the .prof file plus the cumtime top-25 text."""
    warnings.filterwarnings("ignore")
    os.makedirs(_OUT, exist_ok=True)
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    _run(mode)
    pr.disable()
    wall = time.perf_counter() - t0
    pr.dump_stats(os.path.join(_OUT, f"profile_default_{mode}.prof"))
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
    txt = f"mode={mode} wall={wall:.1f}s\n" + s.getvalue()
    with open(os.path.join(_OUT, f"profile_default_{mode}.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1])
