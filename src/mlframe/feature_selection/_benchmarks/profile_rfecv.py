"""cProfile-based hotspot identification for mlframe.feature_selection.wrappers.RFECV.

Run:
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.profile_rfecv

Outputs a stats dump and a top-N table to stdout.

Profiles a single deterministic RFECV.fit() on a moderate problem (n=600, p=80,
n_informative=10) using LogisticRegression - chosen for stable timings without
the noise of CB/RF inner training. The function-level cost shows where any
future optimisation should target.
"""
from __future__ import annotations

import cProfile
import io
import pstats
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.wrappers import RFECV


def _build_problem(n: int = 600, p: int = 80, n_informative: int = 10, seed: int = 0):
    X, y = make_classification(
        n_samples=n, n_features=p, n_informative=n_informative,
        n_redundant=0, n_classes=2, n_clusters_per_class=1,
        random_state=seed, shuffle=False,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(p)]), y


def _profile_once(out_dir: Path, n: int, p: int, label: str) -> Path:
    X, y = _build_problem(n=n, p=p)
    rfecv = RFECV(
        estimator=LogisticRegression(max_iter=400, random_state=0),
        cv=3,
        max_refits=15,
        verbose=0,
        random_state=0,
    )
    profiler = cProfile.Profile()
    profiler.enable()
    rfecv.fit(X, y)
    profiler.disable()

    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / f"profile_{label}.prof"
    profiler.dump_stats(str(stats_path))

    # Render text top-N table - cumulative time, restricted to wrappers.py and
    # close-by callsites. Filter out sklearn / numpy / pandas internals which
    # are out of our control.
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative")
    stats.print_stats(40)

    text_path = out_dir / f"profile_{label}.txt"
    text_path.write_text(stream.getvalue(), encoding="utf-8")

    print(f"\n=== {label}  (n={n}, p={p})  n_features_selected={rfecv.n_features_} ===")
    print(stream.getvalue())
    return stats_path


def main():
    out_dir = Path(__file__).parent / "_results"
    print(f"# cProfile RFECV hotspot scan -> {out_dir}")

    # Two scales: small (CI-fast) and medium (real-world-ish)
    _profile_once(out_dir, n=400, p=40, label="small")
    _profile_once(out_dir, n=600, p=80, label="medium")
    _profile_once(out_dir, n=1000, p=200, label="large")


if __name__ == "__main__":
    main()
