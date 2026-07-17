"""iter145 fuzz-profile harness: an MRMR-FS-on regression suite (richer than the ridge-only
baseline harness) so the profile surfaces the feature-selection + FE leg, not just linear fit.

Run: CUDA_VISIBLE_DEVICES="" python -m tests.perf.bench_mrmr_fs_regression_suite --n-rows 4000
Prints top mlframe-own cumulative frames (src/mlframe paths) so the loop can pick a hotspot.
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

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from mlframe.training import OutputConfig
from mlframe.training.core import train_mlframe_models_suite
from mlframe.training._feature_selection_config import FeatureSelectionConfig
from training.shared import SimpleFeaturesAndTargetsExtractor


def _make_df(n_rows: int, n_features: int = 14, seed: int = 145, classification: bool = False) -> pd.DataFrame:
    """Helper that make df."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features))
    # a few informative + interaction structure so MRMR/FE has real work
    score = X[:, 0] * 1.5 + X[:, 1] * X[:, 2] + np.sin(X[:, 3]) + 0.3 * rng.normal(size=n_rows)
    cols = {f"f{i}": X[:, i] for i in range(n_features)}
    # add a couple of redundant + noisy columns
    cols["f_dup"] = X[:, 0] + 1e-6 * rng.normal(size=n_rows)
    cols["target"] = (score > np.median(score)).astype(np.int64) if classification else score
    return pd.DataFrame(cols)


def run_once(
    n_rows: int, classification: bool = False, n_features: int = 14, seed: int = 145, no_mrmr: bool = False, ensembles: bool = False, rfecv: bool = False
) -> None:
    """Helper that run once."""
    df = _make_df(n_rows=n_rows, n_features=n_features, seed=seed, classification=classification)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=not classification)
    tmp_dir = tempfile.mkdtemp(prefix="mlframe_iter145_")
    try:
        if rfecv:
            fs_cfg = FeatureSelectionConfig(use_mrmr_fs=False, rfecv_models=["lgb_rfecv"])
        else:
            fs_cfg = FeatureSelectionConfig(use_mrmr_fs=not no_mrmr)
        train_mlframe_models_suite(
            df=df,
            target_name="iter145",
            model_name="iter145",
            features_and_targets_extractor=fte,
            mlframe_models=(["ridge", "lightgbm"] if (no_mrmr or rfecv) else ["ridge"]),
            use_ordinary_models=True,
            use_mlframe_ensembles=ensembles,
            feature_selection_config=fs_cfg,
            output_config=OutputConfig(data_dir=tmp_dir, models_dir="models"),
            verbose=0,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def profile(
    n_rows: int,
    top: int,
    classification: bool = False,
    n_features: int = 14,
    seed: int = 145,
    no_mrmr: bool = False,
    ensembles: bool = False,
    rfecv: bool = False,
) -> None:
    """Helper that profile."""
    out = Path(__file__).parent / "results" / ("iter146.prof" if classification else "iter145.prof")
    out.parent.mkdir(parents=True, exist_ok=True)
    pr = cProfile.Profile()
    pr.enable()
    try:
        run_once(n_rows=n_rows, classification=classification, n_features=n_features, seed=seed, no_mrmr=no_mrmr, ensembles=ensembles, rfecv=rfecv)
    finally:
        pr.disable()
    pr.dump_stats(str(out))
    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf).sort_stats("tottime")
    st.print_stats(r"mlframe", top)
    print(buf.getvalue())
    print(f"wrote {out}")


def main() -> int:
    """Helper that main."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows", type=int, default=4000)
    ap.add_argument("--top", type=int, default=35)
    ap.add_argument("--classification", action="store_true")
    ap.add_argument("--n-features", type=int, default=14)
    ap.add_argument("--seed", type=int, default=145)
    ap.add_argument("--no-mrmr", action="store_true")
    ap.add_argument("--ensembles", action="store_true")
    ap.add_argument("--rfecv", action="store_true")
    args = ap.parse_args()
    profile(
        n_rows=args.n_rows,
        top=args.top,
        classification=args.classification,
        n_features=args.n_features,
        seed=args.seed,
        no_mrmr=args.no_mrmr,
        ensembles=args.ensembles,
        rfecv=args.rfecv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
