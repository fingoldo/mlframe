"""Bench cb_logits_to_probs_multiclass softmax with (K,N) vs transposed (N,K) input.

bench-attempt-rejected (2026-05-22, c0108 / iter165): transpose-and-
contiguous-row approach is 11-21% SLOWER at every benchmark size.
Modern CPU prefetchers handle stride-N reads well for small K=3-8;
the 24-100 MB full-transpose memcpy upfront never earns itself back.

    N=  50000 K=3: strided=  2.03ms  transposed=  2.34ms  (0.87x)
    N= 200000 K=3: strided=  7.74ms  transposed=  9.80ms  (0.79x)
    N=1000000 K=3: strided= 40.45ms  transposed= 47.47ms  (0.85x)
    N= 200000 K=8: strided= 18.09ms  transposed= 21.22ms  (0.85x)
    N=1000000 K=8: strided= 94.44ms  transposed=105.88ms  (0.89x)

Bit-equivalent output (max_diff = 0.0 at all sizes).

iter140 originally flagged the strided layout as suspicious; this
bench confirms the strided path IS the right choice. Numpy/numba
prefetcher behaviour at K=3-8 makes the cache-miss concern theoretical.

Documented per ``feedback_document_failed_optimization_attempts`` so
the next agent doesn't re-test this hypothesis.

Run: ``python profiling/bench_cb_logits_softmax_layout.py``
"""
import time
import numpy as np
import numba


NUMBA_NJIT_PARAMS = dict(fastmath=False, cache=True, nogil=True)


@numba.njit(**NUMBA_NJIT_PARAMS)
def softmax_strided(logits_kn):
    """Current production form: (K, N) input, fixed-i strided-c reads."""
    n_classes, n_samples = logits_kn.shape
    probs = np.empty((n_samples, n_classes), dtype=np.float64)
    for i in range(n_samples):
        max_logit = logits_kn[0, i]
        for c in range(1, n_classes):
            if logits_kn[c, i] > max_logit:
                max_logit = logits_kn[c, i]
        exp_sum = 0.0
        for c in range(n_classes):
            probs[i, c] = np.exp(logits_kn[c, i] - max_logit)
            exp_sum += probs[i, c]
        for c in range(n_classes):
            probs[i, c] /= exp_sum
    return probs


@numba.njit(**NUMBA_NJIT_PARAMS)
def softmax_transposed(logits_kn):
    """Transpose (K, N) -> (N, K) once, then access contiguous per-sample."""
    n_classes, n_samples = logits_kn.shape
    # Transpose: K*N float64 copy. For K=3, N=1M -> 24 MB.
    logits_nk = np.ascontiguousarray(logits_kn.T)
    probs = np.empty((n_samples, n_classes), dtype=np.float64)
    for i in range(n_samples):
        row = logits_nk[i]
        max_logit = row[0]
        for c in range(1, n_classes):
            if row[c] > max_logit:
                max_logit = row[c]
        exp_sum = 0.0
        for c in range(n_classes):
            probs[i, c] = np.exp(row[c] - max_logit)
            exp_sum += probs[i, c]
        for c in range(n_classes):
            probs[i, c] /= exp_sum
    return probs


def bench(label, fn, arg, n_iter=20):
    fn(arg); fn(arg)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(n_iter):
            fn(arg)
        times.append((time.perf_counter() - t) / n_iter)
    return min(times) * 1e3, label


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for N, K in [(50_000, 3), (200_000, 3), (1_000_000, 3), (200_000, 8), (1_000_000, 8)]:
        logits = rng.standard_normal((K, N)).astype(np.float64)
        t_str, _ = bench("strided", softmax_strided, logits)
        t_tr, _ = bench("transposed", softmax_transposed, logits)
        p_str = softmax_strided(logits)
        p_tr = softmax_transposed(logits)
        diff = np.max(np.abs(p_str - p_tr))
        speedup = t_str / t_tr
        print(f"N={N:>7} K={K}: strided={t_str:6.2f}ms  transposed={t_tr:6.2f}ms  ({speedup:.2f}x)  max_diff={diff:.1e}")
