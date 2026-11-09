"""Baseline for the FE column-pool plan (``audits/2026-11-09/fe_candidate_pool_plan.md``): per planted-signal generator, does an all-families fit select a column built from every planted source?

Usage: ``python bench_pool_planted_recall.py [ROWS] [COLS] [SEEDS]`` (defaults 20000 100 1), run from a repository checkout (the generators are loaded from ``tests/feature_selection/_synth/pool_planted.py``). For each generator it prints the wall time, the planted
columns, the selected columns that mention a planted source, and ``recall``: 1 when the selected set holds a column whose name mentions every planted source (a raw column counts only when one source is planted),
else 0. Run it once per pool arm and compare the lines; one process, so the timings share the same warm kernels.
"""

import sys
import time
import warnings

warnings.simplefilter("ignore")

from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("pool_planted", Path(__file__).resolve().parents[5] / "tests" / "feature_selection" / "_synth" / "pool_planted.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
GENERATORS = _module.GENERATORS


def run(rows: int, cols: int, seeds: int) -> None:
    """Fit every generator at ``seeds`` seeds and print one line per fit."""
    kw = F.all_families_kwargs()
    kw["nbins_strategy"] = "mdlp"
    hits: dict = {name: 0 for name in GENERATORS}
    for name, make in GENERATORS.items():
        for seed in range(seeds):
            X, y, planted = make(n=rows, p=cols, seed=seed)
            MRMR._FIT_CACHE.clear()
            t = time.perf_counter()
            m = MRMR(verbose=0, random_seed=seed, n_jobs=1, **kw).fit(X, y)
            wall = time.perf_counter() - t
            selected = list(m.get_feature_names_out())
            mention = [c for c in selected if any(s in c for s in planted)]
            recall = int(any(all(s in c for s in planted) for c in selected))
            hits[name] += recall
            print(f"{name:30s} seed {seed} wall {wall:6.1f}s recall {recall} planted {planted} selected {len(selected)} mention {mention[:6]}", flush=True)
    print("recall per generator:", {k: f"{v}/{seeds}" for k, v in hits.items()})


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 20000, int(sys.argv[2]) if len(sys.argv) > 2 else 100, int(sys.argv[3]) if len(sys.argv) > 3 else 1)
