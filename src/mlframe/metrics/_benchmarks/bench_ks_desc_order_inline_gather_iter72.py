"""iter72 @200k: KS large-n desc_order path -- pre-gathered scan vs inline-indexed scan.

In ``fast_calibration_report`` KS is called with ``desc_order`` (the AUC score-desc argsort) already in hand. The
current large-n (n >= _KS_FUSED_MAX_N) path reverses it to ascending and feeds ``_ks_statistic_kernel(yt[order],
ys[order])`` -- two N-length fancy-index gather temporaries (int64 + float64). ``_ks_statistic_kernel_ordered`` indexes
through ``order`` inline, allocating neither gather (only a single contiguous copy of the order array). The prior
``bench_ks_statistic_njit`` rejected the ordered kernel for the STANDALONE-replace case (it also argsorts there);
this bench isolates the desc_order case where ``order`` already exists, so the only delta is gather-vs-inline.

Run: python -m mlframe.metrics._benchmarks.bench_ks_desc_order_inline_gather_iter72
"""
import numpy as np
from time import perf_counter as timer

from mlframe.metrics.classification._classification_extras import (
    _ks_statistic_kernel, _ks_statistic_kernel_ordered,
)


def old_path(yt, ys, desc_order):
    order = desc_order[::-1]
    return float(_ks_statistic_kernel(yt[order], ys[order]))


def new_path(yt, ys, desc_order):
    order = np.ascontiguousarray(desc_order[::-1])
    return float(_ks_statistic_kernel_ordered(order, yt, ys))


def bench(n, reps=200, trials=30):
    rng = np.random.default_rng(0)
    logit = rng.normal(0, 1.3, n)
    p = 1.0 / (1.0 + np.exp(-logit))
    yt = (rng.random(n) < p).astype(np.int64)
    ys = np.clip(p + rng.normal(0, 0.05, n), 1e-6, 1 - 1e-6)
    desc_order = np.argsort(ys)[::-1].copy()  # descending, contiguous (as AUC returns)

    v_old = old_path(yt, ys, desc_order)
    v_new = new_path(yt, ys, desc_order)
    assert v_old == v_new, f"NOT bit-identical: old={v_old!r} new={v_new!r}"

    def best(fn):
        out = []
        for _ in range(trials):
            t0 = timer()
            for _ in range(reps):
                fn(yt, ys, desc_order)
            out.append((timer() - t0) / reps)
        return min(out), float(np.median(out))

    o_min, o_med = best(old_path)
    n_min, n_med = best(new_path)
    print(f"n={n:>7}: OLD min {o_min*1e6:8.1f}us med {o_med*1e6:8.1f}us | "
          f"NEW min {n_min*1e6:8.1f}us med {n_med*1e6:8.1f}us | "
          f"min-speedup {o_min/n_min:.2f}x  med {o_med/n_med:.2f}x | identical={v_old==v_new}")


if __name__ == "__main__":
    for n in (2048, 10000, 50000, 200000, 500000):
        bench(n)


def bench_standalone(n, reps=100, trials=25):
    """The _reporting_probabilistic per-class path: ks_statistic(yt, ys) with NO desc_order -> own argsort + (gather|inline)."""
    from mlframe.metrics.classification import _classification_extras as ex
    rng = np.random.default_rng(1)
    logit = rng.normal(0, 1.3, n); p = 1.0 / (1.0 + np.exp(-logit))
    yt = (rng.random(n) < p).astype(np.int64)
    ys = np.clip(p + rng.normal(0, 0.05, n), 1e-6, 1 - 1e-6)
    orig = ex._KS_INLINE_ORDERED_MIN_N

    def run():
        return ex.ks_statistic(yt, ys)

    ex._KS_INLINE_ORDERED_MIN_N = 10**18; v_old = run()
    ex._KS_INLINE_ORDERED_MIN_N = 150_000; v_new = run()
    assert v_old == v_new, (v_old, v_new)

    def best():
        out = []
        for _ in range(trials):
            t0 = timer()
            for _ in range(reps):
                run()
            out.append((timer() - t0) / reps)
        return min(out), float(np.median(out))

    ex._KS_INLINE_ORDERED_MIN_N = 10**18; o_min, o_med = best()
    ex._KS_INLINE_ORDERED_MIN_N = 150_000; n_min, n_med = best()
    ex._KS_INLINE_ORDERED_MIN_N = orig
    print(f"standalone n={n:>7}: OLD min {o_min*1e6:8.1f}us | NEW min {n_min*1e6:8.1f}us | "
          f"min {o_min/n_min:.3f}x med {o_med/n_med:.3f}x | identical={v_old==v_new}")
