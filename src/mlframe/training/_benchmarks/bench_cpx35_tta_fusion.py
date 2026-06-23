"""CPX35: bench the fused TTA point/mean/spread (one streaming pass, n model calls) vs the legacy three-sweep path (2n+1 calls).

Legacy: evaluate_tta_quality called predict_fn(X) for `point`, tta_predict(...) (which recomputes the clean pass + n-1 jittered),
and tta_predict_spread(...) (another clean pass + n-1 jittered) -> 3 clean + 2*(n-1) jittered = 2n+1 model calls, and materialized two
full (n, rows) stacks for np.mean / np.std. The fused tta_point_mean_spread does 1 clean + (n-1) jittered = n calls via Welford.

Warm + best-of-N wall A/B, plus identity check (mean/spread within ~1e-9 of the two-pass reference) and a model-call counter to prove
the call-count reduction. CUDA_VISIBLE_DEVICES="" set by harness; stub model is a deterministic numpy fn.
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.training._tta import tta_predict, tta_predict_spread, tta_point_mean_spread


def make_model():
    w = np.array([1.3, -0.7, 0.5, 2.0, -1.1])
    calls = {"n": 0}

    def predict(Z):
        calls["n"] += 1
        return Z[:, :5] @ w + 0.4 * np.sin(7.0 * Z[:, 5 % Z.shape[1]])

    return predict, calls


def legacy(predict, X, n, sigma):
    point = np.asarray(predict(X), dtype=np.float64)
    mean = np.asarray(tta_predict(predict, X, n=n, sigma_scale=sigma, seed=0), dtype=np.float64)
    spread = np.asarray(tta_predict_spread(predict, X, n=n, sigma_scale=sigma, seed=0), dtype=np.float64)
    return point, mean, spread


def best_of(fn, reps=5):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def main():
    n_rows, n_aug, sigma = 50_000, 20, 0.02
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_rows, 8))

    p_old, c_old = make_model()
    p_new, c_new = make_model()

    # warm
    legacy(p_old, X, n_aug, sigma)
    tta_point_mean_spread(p_new, X, n=n_aug, sigma_scale=sigma, seed=0)
    calls_old, calls_new = c_old["n"], c_new["n"]

    t_old = best_of(lambda: legacy(p_old, X, n_aug, sigma))
    t_new = best_of(lambda: tta_point_mean_spread(p_new, X, n=n_aug, sigma_scale=sigma, seed=0))

    # identity vs two-pass reference (legacy outputs)
    po, mo, so = legacy(p_old, X, n_aug, sigma)
    pn, mn, sn = tta_point_mean_spread(p_new, X, n=n_aug, sigma_scale=sigma, seed=0)
    d_point = float(np.max(np.abs(po - pn)))
    d_mean = float(np.max(np.abs(mo - mn)))
    d_spread = float(np.max(np.abs(so - sn)))

    print(f"shape rows={n_rows} n_aug={n_aug} sigma={sigma}")
    print(f"model calls per run: legacy={calls_old} (expect 2n+1={2*n_aug+1})  fused={calls_new} (expect n={n_aug})")
    print(f"wall best-of-5: legacy={t_old*1e3:.2f} ms  fused={t_new*1e3:.2f} ms  speedup={t_old/t_new:.3f}x")
    print(f"identity max abs diff: point={d_point:.2e}  mean={d_mean:.2e}  spread={d_spread:.2e}")


if __name__ == "__main__":
    main()
