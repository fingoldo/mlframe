"""CPX30 bench: hoist alpha-invariant APS test-side argsort+cumsum out of the alpha loop
in conformal_classification_report.

Per-alpha the OLD code recomputed np.argsort(-tp) + take_along_axis + cumsum(tp) -- all
invariant across alpha (depend only on test probs tp). NEW hoists them once above the loop.

Run: CUDA_VISIBLE_DEVICES="" python src/mlframe/training/_benchmarks/bench_cpx30_conformal_finalize.py
Warm, best-of-N. `python` on PATH (3.14.3).
"""

import time
import numpy as np

from mlframe.training._conformal_finalize import conformal_classification_report


def make_data(n_test=120_000, n_cal=20_000, k=12, seed=0):
    rng = np.random.default_rng(seed)
    test_logits = rng.standard_normal((n_test, k))
    calib_logits = rng.standard_normal((n_cal, k))
    test_probs = np.exp(test_logits); test_probs /= test_probs.sum(1, keepdims=True)
    calib_probs = np.exp(calib_logits); calib_probs /= calib_probs.sum(1, keepdims=True)
    classes = np.arange(k)
    test_target = rng.integers(0, k, n_test)
    calib_target = rng.integers(0, k, n_cal)
    return test_probs, test_target, calib_probs, calib_target, classes


def bench(reps=7):
    tp, tt, cp, ct, cls = make_data()
    alphas = [round(0.02 * i, 4) for i in range(1, 16)]  # 15 alpha levels

    def run():
        return conformal_classification_report(
            test_probs=tp, test_target=tt, calib_probs=cp, calib_target=ct,
            classes=cls, alphas=alphas, score="aps",
        )

    run()  # warm
    times = []
    for _ in range(reps):
        t0 = time.perf_counter(); res = run(); times.append(time.perf_counter() - t0)
    best = min(times)
    print(f"score=aps  test={tp.shape}  cal={cp.shape}  k={cls.size}  n_alphas={len(alphas)}")
    print(f"best-of-{reps}: {best*1000:.2f} ms  (median {sorted(times)[reps//2]*1000:.2f} ms)")
    return best


if __name__ == "__main__":
    bench()
