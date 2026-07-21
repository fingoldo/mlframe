"""Bench: does lgb_shim's module-level Dataset cache actually save wall-clock time in the
composite-ensemble OOF refit pattern (repeated ``sklearn.clone()`` on the SAME X, once per
component/weight-schema)?

xgb_shim's own module docstring cites a measured "20+s wasted per ensemble round" for the
equivalent XGB DMatrix rebuild before its module-level cache landed. This bench reproduces the
identical clone-per-component loop for BOTH shims side by side and reports the real number for
LGB, with the cache enabled vs a forced-disabled control (``MLFRAME_LGB_CACHE_DISABLE=1``).

Usage::

    python -m mlframe.training._benchmarks.bench_lgb_shim_clone_cache_reuse
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from sklearn.base import clone

from mlframe.training.lgb_shim import LGBMRegressorWithDatasetReuse, _lgb_cache_clear
from mlframe.training.xgb_shim import XGBRegressorWithDMatrixReuse, _xgb_cache_clear

N_ROWS = 200_000
N_COLS = 30
N_COMPONENTS = 4  # mirrors a typical composite-ensemble (raw + a few composite targets)


def _make_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(N_ROWS, N_COLS)), columns=[f"f{i}" for i in range(N_COLS)])
    y = rng.normal(size=N_ROWS)
    return X, y


def _time_clone_per_component_loop(template) -> float:
    """Fit ``template`` once (warms its instance + module cache), then time ``N_COMPONENTS``
    ``clone()`` + ``.fit()`` rounds on the SAME X/y -- the composite-ensemble OOF refit shape."""
    X, y = _make_data()
    template.fit(X, y)
    t0 = time.perf_counter()
    for _ in range(N_COMPONENTS):
        m = clone(template)
        m.fit(X, y)
    return time.perf_counter() - t0


def run() -> None:
    print(f"n_rows={N_ROWS} n_cols={N_COLS} n_components={N_COMPONENTS}\n")

    for label, shim_cls, cache_clear, disable_env in (
        ("LGB", LGBMRegressorWithDatasetReuse, _lgb_cache_clear, "MLFRAME_LGB_CACHE_DISABLE"),
        ("XGB", XGBRegressorWithDMatrixReuse, _xgb_cache_clear, "MLFRAME_XGB_CACHE_DISABLE"),
    ):
        cache_clear()
        os.environ.pop(disable_env, None)
        t_cached = _time_clone_per_component_loop(shim_cls(n_estimators=50))
        cache_clear()

        os.environ[disable_env] = "1"
        t_uncached = _time_clone_per_component_loop(shim_cls(n_estimators=50))
        os.environ.pop(disable_env, None)
        cache_clear()

        saved = t_uncached - t_cached
        print(
            f"{label}: cache ON {t_cached:.2f}s, cache OFF {t_uncached:.2f}s, "
            f"saved {saved:.2f}s over {N_COMPONENTS} clone+fit rounds "
            f"({saved / max(N_COMPONENTS, 1):.2f}s/round)."
        )


if __name__ == "__main__":
    run()
