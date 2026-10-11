"""Paired A/B of a whole MRMR fit (the F2 example, GPU strict-resident mode) with three arms: ``off`` (the FE families of ``FAMILY_FLAGS`` and the residual rescue all off), ``families`` (the four
FE families on, the rescue off) and ``all`` (everything on, the constructor default).

Usage: ``python -m mlframe.feature_selection._benchmarks.offset_product.ab_fit <n_rows> [pairs]``. Every round gets novel data (MRMR memoises by content hash and parameters, so a repeated frame with the
same flags would time the cache replay); the three arms of a round fit the SAME frame back to back, the order rotating between rounds (the flags are part of the memo key, so every arm is a real fit).
Prints each wall time, the medians per arm, the paired deltas against ``off`` and how many rounds selected the same features as ``off``. Run it on a quiet box: a concurrent GPU job moves every figure by
more than the effect measured here.
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
RESCUE_FLAG = "fe_residual_rescue_enable"
ARMS = {"off": (False, False), "families": (True, False), "all": (True, True)}  # (families, rescue)


def _make(n: int, seed: int):
    """The F2 example on fresh random numbers."""
    rng = np.random.default_rng(seed)
    a, b, c, d, e, f = (rng.uniform(0.1, 1.1, n) for _ in range(6))
    df = pd.DataFrame(dict(zip("abcde", (a, b, c, d, e))))
    return df, a**2 / b + f / 5.0 + np.log(np.abs(c) + 1e-9) * np.sin(d)


def _fit(n: int, seed: int, arm: str):
    """One fit; returns ``(wall seconds, selected feature names)``."""
    import cupy as cp

    families, rescue = ARMS[arm]
    df, y = _make(n, seed)
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    fs = MRMR(
        full_npermutations=10,
        baseline_npermutations=20,
        fe_max_steps=2,
        fe_min_pair_mi_prevalence=1.05,
        verbose=0,
        n_jobs=1,
        random_seed=seed,
        **{**dict.fromkeys(FAMILY_FLAGS, families), RESCUE_FLAG: rescue},
    ).fit(df, y)
    cp.cuda.Device().synchronize()
    return time.perf_counter() - t0, [str(s) for s in fs.get_feature_names_out()]


def main() -> None:
    """Warm up, then run the rotating rounds and print the medians."""
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    pairs = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PAIRS
    for arm in ARMS:
        _fit(n, 1, arm)  # warm: jit, tuning cache, cupy kernels, per arm
    walls = {arm: [] for arm in ARMS}
    same = dict.fromkeys(ARMS, 0)
    arm_list = list(ARMS)
    for k in range(pairs):
        names = {}
        for arm in arm_list[k % 3 :] + arm_list[: k % 3]:
            wall, names[arm] = _fit(n, 10 + k, arm)
            walls[arm].append(wall)
            print(f"n={n} arm={arm:8} fit {wall:.2f}s", flush=True)
        for arm in ARMS:
            same[arm] += int(names[arm] == names["off"])
        for arm in ("families", "all"):
            if names[arm] != names["off"]:
                print(f"  selection differs ({arm}):", names["off"], "->", names[arm])
    med = {arm: float(np.median(w)) for arm, w in walls.items()}
    for arm in ("families", "all"):
        deltas = np.array(walls[arm]) - np.array(walls["off"])
        print(f"n={n} median off {med['off']:.2f}s  {arm} {med[arm]:.2f}s  paired delta median {np.median(deltas):+.2f}s ({np.median(deltas) / med['off']:+.1%});  same selection as off in {same[arm]} of {pairs} rounds")


if __name__ == "__main__":
    main()
