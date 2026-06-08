"""Baseline + cProfile hotspot harness for ``mlframe.feature_selection.boruta_shap.BorutaShap``.

Run:
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.profile_boruta_shap

Representative SHAP-driven config (the path that does real work -- the default
``importance_measure='gini'`` reads ``model.feature_importances_`` and never calls
SHAP, so it would not exercise the dominant TreeExplainer cost):

    n=2407, p=120 (20 informative), LightGBM 50-tree binary classifier,
    importance_measure='shap', n_trials=30, deterministic random_state=0.

Outputs:
  - wall time (plain, un-profiled) of a full ``fit``,
  - the accept / reject / tentative feature SETS and per-feature accumulated
    HIT COUNTS (the bit-identity golden for every optimisation gate),
  - a cProfile top-N hotspot table.

The golden (sets + hit vector) is written to ``_results/boruta_shap_golden.json``
so optimisation candidates can be diffed against it for bit-identity.
"""
from __future__ import annotations

import cProfile
import io
import json
import pstats
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

warnings.filterwarnings("ignore")


def build_scene(n: int = 2407, p: int = 120, informative: int = 20, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the representative classification scene (deterministic make_classification) for the BorutaShap profile."""
    X, y = make_classification(
        n_samples=n, n_features=p, n_informative=informative,
        n_redundant=0, n_classes=2, n_clusters_per_class=1,
        random_state=seed, shuffle=False,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(p)]), y


def make_selector() -> "object":
    """Construct the representative SHAP-driven BorutaShap (the path that actually calls TreeExplainer)."""
    import lightgbm as lgb
    from mlframe.feature_selection.boruta_shap import BorutaShap

    model = lgb.LGBMClassifier(n_estimators=50, num_leaves=31, random_state=0, verbose=-1, n_jobs=1)
    return BorutaShap(
        model=model, importance_measure="shap", classification=True,
        n_trials=30, random_state=0, verbose=False, normalize=True,
    )


def extract_golden(bs: "object") -> dict:
    """Extract the bit-identity golden (accept/reject/tentative sets + per-feature hit vector) from a fitted BorutaShap."""
    return {
        "accepted": sorted(bs.accepted),
        "rejected": sorted(bs.rejected),
        "tentative": sorted([str(t) for t in bs.tentative]),
        "hits": [float(h) for h in bs.hits],
        "n_trials_run": int(bs.n_trials_run_),
        "selected_features": sorted(bs.selected_features_),
    }


def run_once_walltime() -> tuple[float, dict]:
    """Run one un-profiled fit and return (wall seconds, golden) -- the honest baseline number."""
    X, y = build_scene()
    bs = make_selector()
    t0 = time.perf_counter()
    bs.fit(X, y)
    dt = time.perf_counter() - t0
    return dt, extract_golden(bs)


def run_once_profiled(out_dir: Path) -> tuple[str, dict]:
    """Run one cProfile'd fit, dump the .prof + top-tottime table to out_dir, and return (table text, golden)."""
    X, y = build_scene()
    bs = make_selector()
    profiler = cProfile.Profile()
    profiler.enable()
    bs.fit(X, y)
    profiler.disable()

    out_dir.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(out_dir / "boruta_shap.prof"))

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("tottime")
    stats.print_stats(25)
    (out_dir / "boruta_shap_profile.txt").write_text(stream.getvalue(), encoding="utf-8")
    return stream.getvalue(), extract_golden(bs)


def main() -> None:
    """Print the BorutaShap baseline wall + accept/reject summary, write the golden, then the cProfile hotspot table."""
    out_dir = Path(__file__).parent / "_results"
    print("# BorutaShap baseline + hotspot scan")

    # 1) plain wall (no profiler) -> the honest baseline number.
    wall, golden = run_once_walltime()
    print(f"\n[plain wall] full fit = {wall:.2f}s")
    print(f"  accepted={len(golden['accepted'])} rejected={len(golden['rejected'])} "
          f"tentative={len(golden['tentative'])} trials_run={golden['n_trials_run']} "
          f"hits_sum={sum(golden['hits']):.0f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "boruta_shap_golden.json").write_text(
        json.dumps(golden, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"  golden -> {out_dir / 'boruta_shap_golden.json'}")

    # 2) profiled wall -> measure cProfile overhead + hotspots.
    prof_text, golden_prof = run_once_profiled(out_dir)
    # cProfile run does not print its own wall; recompute via a quick second plain pass not needed --
    # overhead is shown by comparing the profiled cumtime header in the table to the plain wall above.
    assert golden_prof["accepted"] == golden["accepted"], "profiled run diverged from plain run (non-determinism!)"
    assert golden_prof["hits"] == golden["hits"], "profiled run hit vector diverged (non-determinism!)"
    print("\n[cProfile top hotspots, sorted tottime]")
    print(prof_text)


if __name__ == "__main__":
    main()
