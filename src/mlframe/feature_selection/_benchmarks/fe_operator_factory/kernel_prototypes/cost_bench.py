"""Cost of scoring 10 forms x 8 offsets (80 columns) on the CPU against the full 1734-combo pair table at n = 100k and 1M, plus the 20k-subsample variant.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.kernel_prototypes.cost_bench``."""

import time
import warnings

warnings.simplefilter("ignore")
import numpy as np

from mlframe.feature_selection.filters._fe_cpu_batch import cpu_fe_batch_mi
from mlframe.feature_selection.filters._usability_njit_pool import _NJIT_UNARY_OP_CODES, njit_binary_codes_or_none, njit_unary_codes_or_none, score_pair_combos


def y_terms_of(yc):
    """Target-entropy tuple the pair-table scorer expects: (None, H(y), classes)."""
    k = int(yc.max()) + 1
    p = np.bincount(yc, minlength=k) / yc.size
    p = p[p > 0]
    return (None, float(-(p * np.log(p)).sum()), k)


def best(f, reps=3):
    """Best-of-``reps`` wall time of ``f()`` and its result."""
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        r = f()
        ts.append(time.perf_counter() - t)
    return min(ts), r


def main(sizes=(100_000, 1_000_000)) -> None:
    """Print the cost of scoring 10 forms x 8 offsets against the full 1734-combo pair table at each size in ``sizes``."""
    for n in sizes:
        rng = np.random.default_rng(0)
        c, d = rng.random(n), rng.random(n)
        y = np.log(2 * c) * np.sin(d / 3) + rng.random(n) / 5
        yc = np.searchsorted(np.quantile(y, np.linspace(0, 1, 11)[1:-1]), y).astype(np.int64)
        # K=10 forms x G=8 offsets = 80 columns; mul(u+t, v) with u=log(c), v=sin(d)
        ts = np.linspace(-0.5, 1.5, 8)
        u, v = np.log(c), np.sin(d)
        cols = np.empty((n, 80))
        for k in range(10):
            for g in range(8):
                cols[:, k * 8 + g] = (u * (1 + 0.01 * k) + ts[g]) * v
        cpu_fe_batch_mi(cols[:2000, :4].copy(), yc[:2000], 10)  # warm
        t80, _ = best(lambda: cpu_fe_batch_mi(cols, yc, 10))
        t1, _ = best(lambda: cpu_fe_batch_mi(cols[:, :8].copy(), yc, 10))

        # build cost of 80 columns (numpy)
        def build():
            """Materialise the 80 candidate columns (numpy)."""
            o = np.empty((n, 80))
            for k in range(10):
                for g in range(8):
                    o[:, k * 8 + g] = (u * (1 + 0.01 * k) + ts[g]) * v
            return o

        tb, _ = best(build)
        # full existing pair-combo table (medium unary x minimal binary = 1734 combos)
        ua = njit_unary_codes_or_none(list(_NJIT_UNARY_OP_CODES))
        bn = njit_binary_codes_or_none(["mul", "add", "sub", "div", "max", "min"])
        yt = y_terms_of(yc)
        score_pair_combos(c[:3000], d[:3000], yc[:3000], yt, 10, ua, ua, bn)
        tt, _ = best(lambda: score_pair_combos(c, d, yc, yt, 10, ua, ua, bn), reps=2)
        # subsample design: 80 cols on 20k subsample
        sub = np.arange(0, n, max(1, n // 20000))[:20000]
        ts20, _ = best(lambda: cpu_fe_batch_mi(np.ascontiguousarray(cols[sub]), yc[sub], 10))
        print(
            f"n={n}: 80-col MI batch {t80:.3f}s (8 cols {t1:.3f}s), build80 {tb:.3f}s, full 1734-combo pair table {tt:.3f}s, 80-col MI on 20k subsample {ts20:.3f}s; ratio 80col/(table)={(t80 + tb) / tt:.3f}"
        )


if __name__ == "__main__":
    main()
