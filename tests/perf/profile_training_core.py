"""cProfile harness for ``mlframe.training.core.train_mlframe_models_suite``.

Runs a tiny single-target regression suite end-to-end and writes a ``.prof``
artefact under ``tests/perf/results/`` so the training-core hotspots become
visible to ``snakeviz`` / ``pyprof2calltree`` / ``gprof2dot``.

Usage::

    python -m tests.perf.profile_training_core
    python -m tests.perf.profile_training_core --n-rows 5000
    python -m tests.perf.profile_training_core --output custom.prof

The .prof artefact is regenerated on every run; consumers should commit only
the harness, not its outputs.

cProfile attribution inflates pandas-internal call timings; treat the dump as
a relative-ranking tool, not an absolute cost report (see the matching note
in ``_profile_dummy_baselines`` for the same caveat).
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlframe.training import OutputConfig  # noqa: E402
from mlframe.training.core import train_mlframe_models_suite  # noqa: E402

# Pull the test-side mock FTE so the harness mirrors the smoke-test path
# rather than constructing a separate ad-hoc extractor that could drift.
sys.path.insert(0, str(REPO_ROOT / "tests"))
from training.shared import SimpleFeaturesAndTargetsExtractor  # noqa: E402


def _make_regression_df(n_rows: int, n_features: int = 6, seed: int = 0) -> pd.DataFrame:
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
    tmp_dir = tempfile.mkdtemp(prefix="mlframe_perf_")
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


def profile(n_rows: int, output_path: Path, top: int = 30) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pr = cProfile.Profile()
    pr.enable()
    try:
        run_once(n_rows=n_rows)
    finally:
        pr.disable()
    pr.dump_stats(str(output_path))
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).strip_dirs().sort_stats("cumulative").print_stats(top)
    print(buf.getvalue())
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "training_core.prof",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    out = profile(n_rows=args.n_rows, output_path=args.output, top=args.top)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
