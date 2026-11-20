"""Times batch_mi_with_noise_gate v1/v2/v3 with npermutations=0 (observed-MI pass only, the worst case for a per-row range check); run once per checkout (cwd = its root) and compare the printed best-of-N process times."""

import sys
import time
from typing import Any

import numpy as np

from mlframe.feature_selection.filters.info_theory import _batch_kernels as bk

v3: Any
try:
    from mlframe.feature_selection.filters.info_theory import _batch_kernels_v3 as v3
except ImportError:
    v3 = None


def _best(fn, reps: int) -> float:
    """Best-of-``reps`` process time of ``fn`` in milliseconds."""
    best = float("inf")
    for _ in range(reps):
        t = time.process_time()
        fn()
        best = min(best, time.process_time() - t)
    return best * 1e3


def main() -> None:
    """Print per-size best process time of each kernel variant."""
    rng = np.random.default_rng(0)
    nb = 12
    for n, k in ((200_000, 64), (200_000, 512), (1_000_000, 64), (1_000_000, 256), (2_000_000, 64)):
        d = rng.integers(0, nb, (n, k)).astype(np.int8)
        y = rng.integers(0, 2, n).astype(np.int8)
        fy = np.bincount(y, minlength=2) / n
        nbins = np.full(k, nb, dtype=np.int64)
        args = (d, nbins, y, y, fy, 0, np.uint64(1), 0.99, False)
        fns = {"v1": lambda: bk.batch_mi_with_noise_gate(*args), "v2": lambda: bk.batch_mi_with_noise_gate_v2(*args)}
        if v3 is not None:
            fns["v3"] = lambda: v3.batch_mi_with_noise_gate_v3(*args)
        res: dict[str, list[float]] = {}
        for name, fn in fns.items():
            fn()
            res[name] = []
        for _ in range(3):
            for name, fn in fns.items():
                res[name].append(_best(fn, 2))
        print(sys.argv[1] if len(sys.argv) > 1 else "run", n, k, {a: round(min(b), 1) for a, b in res.items()}, flush=True)


if __name__ == "__main__":
    main()
