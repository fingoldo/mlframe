"""Paired A/B of a whole MRMR fit (the F2 example, GPU strict-resident mode) with the new FE families (``FAMILY_FLAGS``: offset product, out-of-fold warp, row statistics, 2-D cell table) all off and all on.

Usage: ``python -m mlframe.feature_selection._benchmarks.offset_product.ab_fit <n_rows> [pairs]``. Every pair gets novel data (MRMR memoises by content hash and parameters, so a repeated frame with the
same flag would time the cache replay); both arms of a pair fit the SAME frame back to back, the order alternating between pairs (the flag is part of the memo key, so the second arm is a real fit).
Prints each wall time, the medians, and how many pairs selected the same features in both arms. Run it on a quiet box: a concurrent GPU job moves every figure by more than the effect measured here.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

for _k, _v in {
    "MLFRAME_FE_GPU_STRICT": "1",
    "MLFRAME_CMI_GPU": "1",
    "MLFRAME_FE_VRAM_F32": "1",
    "MLFRAME_FE_GPU_DISCRETIZE": "1",
    "MLFRAME_FE_GPU_BINNING": "1",
}.items():
    os.environ.setdefault(_k, _v)
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR

DEFAULT_PAIRS = 3
FAMILY_FLAGS = ("fe_offset_product_enable", "fe_oof_warp_enable", "fe_row_stat_enable", "fe_oof_cell2d_enable")  # switched together by the arms


def _make(n: int, seed: int):
    """The F2 example on fresh random numbers."""
    rng = np.random.default_rng(seed)
    a, b, c, d, e, f = (rng.uniform(0.1, 1.1, n) for _ in range(6))
    df = pd.DataFrame(dict(zip("abcde", (a, b, c, d, e))))
    return df, a**2 / b + f / 5.0 + np.log(np.abs(c) + 1e-9) * np.sin(d)


def _fit(n: int, seed: int, flag: bool):
    """One fit; returns ``(wall seconds, selected feature names)``."""
    import cupy as cp

    df, y = _make(n, seed)
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    fs = MRMR(
        full_npermutations=10, baseline_npermutations=20, fe_max_steps=2, fe_min_pair_mi_prevalence=1.05, verbose=0, n_jobs=1, random_seed=seed, **dict.fromkeys(FAMILY_FLAGS, flag)
    ).fit(df, y)
    cp.cuda.Device().synchronize()
    return time.perf_counter() - t0, [str(s) for s in fs.get_feature_names_out()]


def main() -> None:
    """Warm up, then run the alternating pairs and print the medians."""
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    pairs = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PAIRS
    _fit(n, 1, True)  # warm: jit, tuning cache, cupy kernels
    walls = {False: [], True: []}
    same = 0
    for k in range(pairs):
        names = {}
        for flag in (False, True) if k % 2 == 0 else (True, False):
            wall, names[flag] = _fit(n, 10 + k, flag)
            walls[flag].append(wall)
            print(f"n={n} offset_product={flag!s:5} fit {wall:.2f}s", flush=True)
        same += int(names[False] == names[True])
        if names[False] != names[True]:
            print("  selection differs: off", names[False], "on", names[True])
    off, on = float(np.median(walls[False])), float(np.median(walls[True]))
    print(f"n={n} median off {off:.2f}s  on {on:.2f}s  delta {on - off:+.2f}s ({(on - off) / off:+.1%});  same selection in {same} of {pairs} pairs")


if __name__ == "__main__":
    main()
