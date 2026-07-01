"""iter135 SUITE-ORCHESTRATION profiling driver.

Profiles the FULL ``train_mlframe_models_suite`` orchestration (train + predict + save + load)
at a RAM-fittable scale, isolating mlframe-OWN orchestration glue from leaf numeric kernels and
external model .fit. Uses the ridge model so external (CatBoost/LightGBM) fit time stays out of
the profile and the suite glue dominates. The first call is a warm-up (numba JIT etc.); the second
is the one cProfile measures.

Run::

    python -m tests.perf._iter135_suite_orchestration --n-rows 5000 --top 40
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import shutil
import sys
import tempfile
from pathlib import Path

# conftest-style import order so the cold ``mlframe.training.core`` import does not segfault on py3.14.
import scipy.stats  # noqa: F401
import numba  # noqa: F401
sys.modules.setdefault("cupy", None)

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlframe.training import OutputConfig  # noqa: E402
from mlframe.training.core import train_mlframe_models_suite  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "tests"))
from training.shared import SimpleFeaturesAndTargetsExtractor  # noqa: E402


def _make_regression_df(n_rows: int, n_features: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features))
    coefs = rng.normal(size=n_features)
    y = X @ coefs + 0.1 * rng.normal(size=n_rows)
    cols = {f"f{i}": X[:, i] for i in range(n_features)}
    cols["target"] = y
    return pd.DataFrame(cols)


def run_once(n_rows: int) -> None:
    df = _make_regression_df(n_rows=n_rows)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)
    tmp_dir = tempfile.mkdtemp(prefix="mlframe_iter135_")
    try:
        train_mlframe_models_suite(
            df=df,
            target_name="perf_target",
            model_name="perf_model",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=tmp_dir, models_dir="models"),
            verbose=0,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=5000)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "iter135_suite_orch.prof")
    args = parser.parse_args()

    for _ in range(args.warmups):
        run_once(n_rows=args.n_rows)
        print("warmup done", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pr = cProfile.Profile()
    pr.enable()
    run_once(n_rows=args.n_rows)
    pr.disable()
    pr.dump_stats(str(args.output))

    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf).strip_dirs()
    st.sort_stats("tottime").print_stats(args.top)
    print("===== TOTTIME TOP =====", flush=True)
    print(buf.getvalue(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
