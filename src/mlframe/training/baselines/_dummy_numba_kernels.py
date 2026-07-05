"""Numba @njit kernels for ``mlframe.training.dummy_baselines``.

Split out of ``dummy_baselines.py`` so the bootstrap-CI sibling
(``_dummy_bootstrap.py``) can depend on these kernels without re-entering
the parent. The parent file re-exports each kernel under its historical
name so the in-module call sites continue to resolve.

Numba is an optional dep here: import failure flips the
``_NUMBA_AVAILABLE`` flag to ``False`` and the kernels stay unbound, which
matches the parent's prior behaviour. Callers must check the flag before
dispatching to a kernel.

What lives here:
  - Macro / micro multilabel log-loss kernels.
  - Within-group descending rank for LTR.
  - Bootstrap-RMSE / -MAE / -log-loss kernels (paired + single-sample
    variants).
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange  # noqa: F401
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


if _NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_macro_log_loss(y_int, p, n, K):
        """Per-label-averaged binary log-loss; skips all-constant labels.

        ``y_int`` is (N, K) int64 with values in {0, 1}; ``p`` is (N, K)
        float64 probability of class 1. Returns NaN when no label has
        both classes present in ``y_int``.
        """
        eps = 1e-15
        label_lls = np.empty(K, dtype=np.float64)
        for k in prange(K):
            s = 0.0
            n_pos = 0
            n_neg = 0
            for i in range(n):
                yi = y_int[i, k]
                pi = p[i, k]
                if pi < eps:
                    pi = eps
                elif pi > 1.0 - eps:
                    pi = 1.0 - eps
                if yi == 1:
                    s -= np.log(pi)
                    n_pos += 1
                else:
                    s -= np.log(1.0 - pi)
                    n_neg += 1
            if n_pos > 0 and n_neg > 0:
                label_lls[k] = s / n
            else:
                label_lls[k] = -1.0  # sentinel: skip (single-class)
        total = 0.0
        valid_count = 0
        for k in range(K):
            if label_lls[k] >= 0:
                total += label_lls[k]
                valid_count += 1
        if valid_count == 0:
            return np.nan
        return total / valid_count

    @njit(fastmath=True, cache=True)
    def _numba_micro_log_loss(y_int, p, n, K):
        """Pooled (micro) binary log-loss across all (N, K) cells."""
        eps = 1e-15
        s = 0.0
        for k in range(K):
            for i in range(n):
                yi = y_int[i, k]
                pi = p[i, k]
                if pi < eps:
                    pi = eps
                elif pi > 1.0 - eps:
                    pi = 1.0 - eps
                if yi == 1:
                    s -= np.log(pi)
                else:
                    s -= np.log(1.0 - pi)
        return s / (n * K)

    @njit(cache=True)
    def _numba_within_group_descending_rank(group_ids: np.ndarray) -> np.ndarray:
        """Descending within-group rank: row 0 of each group -> highest score.

        Single-pass over a stable-sorted index. Output[i] = -within_group_idx
        so the first row of each group has the highest score. Robust
        against non-contiguous group_ids; works on any integer dtype.
        Replaces the prior dict-based 2-pass (which produced a numba
        ``unsafe cast from int64 to undefined`` warning at module load
        from the typed-dict default-value type inference).
        """
        n = len(group_ids)
        out = np.empty(n, dtype=np.float64)
        if n == 0:
            return out
        # argsort is stable; sequential scan over sorted indices counts
        # within-group position via prev-group equality check.
        order = np.argsort(group_ids, kind="mergesort")
        prev_g = group_ids[order[0]]
        c = 0
        for k in range(n):
            i = order[k]
            g = group_ids[i]
            if g != prev_g:
                c = 0
                prev_g = g
            out[i] = -float(c)
            c += 1
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_paired_bootstrap_rmse(y, p1, p2, n_resamples, seed):
        """Paired bootstrap on RMSE between two predictors.

        Returns ndarray of length ``n_resamples`` with
        ``RMSE(p1) - RMSE(p2)`` per resample (negative when p1 wins under
        minimize-metric convention). ~20-50x faster than sklearn's
        ``mean_squared_error`` per-call loop on n=1500 x 1000 resamples
        (current Python loop ~1100ms -> numba ~30ms measured).
        """
        n = len(y)
        out = np.empty(n_resamples, dtype=np.float64)
        # Per-resample independent -- prange parallel.
        for i in prange(n_resamples):
            # Per-iteration LCG for index draws (avoids np.random global
            # state under prange; reproducible from (seed, i) pair).
            state = np.uint64(seed) ^ np.uint64(i) * np.uint64(2862933555777941757) + np.uint64(3037000493)
            sse1 = 0.0
            sse2 = 0.0
            for _k in range(n):
                # LCG step + mod n for index in [0, n)
                state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                idx = int(state >> np.uint64(33)) % n
                d1 = y[idx] - p1[idx]
                d2 = y[idx] - p2[idx]
                sse1 += d1 * d1
                sse2 += d2 * d2
            out[i] = np.sqrt(sse1 / n) - np.sqrt(sse2 / n)
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_paired_bootstrap_mae(y, p1, p2, n_resamples, seed):
        """MAE-paired-bootstrap counterpart of _numba_paired_bootstrap_rmse."""
        n = len(y)
        out = np.empty(n_resamples, dtype=np.float64)
        for i in prange(n_resamples):
            state = np.uint64(seed) ^ np.uint64(i) * np.uint64(2862933555777941757) + np.uint64(3037000493)
            sae1 = 0.0
            sae2 = 0.0
            for _k in range(n):
                state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                idx = int(state >> np.uint64(33)) % n
                sae1 += abs(y[idx] - p1[idx])
                sae2 += abs(y[idx] - p2[idx])
            out[i] = sae1 / n - sae2 / n
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_bootstrap_rmse_samples(y, p, n_resamples, seed):
        """Bootstrap CI on a single predictor's RMSE.

        Returns ndarray of length ``n_resamples`` with bootstrap samples
        of RMSE. Caller computes 2.5/97.5 percentiles for the CI.
        """
        n = len(y)
        out = np.empty(n_resamples, dtype=np.float64)
        for i in prange(n_resamples):
            state = np.uint64(seed) ^ np.uint64(i) * np.uint64(2862933555777941757) + np.uint64(3037000493)
            sse = 0.0
            for _k in range(n):
                state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                idx = int(state >> np.uint64(33)) % n
                d = y[idx] - p[idx]
                sse += d * d
            out[i] = np.sqrt(sse / n)
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_bootstrap_mae_samples(y, p, n_resamples, seed):
        n = len(y)
        out = np.empty(n_resamples, dtype=np.float64)
        for i in prange(n_resamples):
            state = np.uint64(seed) ^ np.uint64(i) * np.uint64(2862933555777941757) + np.uint64(3037000493)
            sae = 0.0
            for _k in range(n):
                state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                idx = int(state >> np.uint64(33)) % n
                sae += abs(y[idx] - p[idx])
            out[i] = sae / n
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_paired_bootstrap_logloss_binary(y_int, p1, p2, n_resamples, seed):
        """Binary cross-entropy paired bootstrap.

        ``y_int`` is (N,) int64 in {0, 1}; ``p1`` / ``p2`` are (N,)
        probability of class 1 (float64). Eps-clipping mirrors
        sklearn's ``log_loss`` (eps=1e-15). Returns ``log_loss(p1) -
        log_loss(p2)`` per resample. Numba kernel is ~30x faster than
        the sklearn loop (sklearn's log_loss does input validation +
        label_binarize per call; the inner-loop is the same arithmetic).
        """
        n = len(y_int)
        eps = 1e-15
        out = np.empty(n_resamples, dtype=np.float64)
        for i in prange(n_resamples):
            state = np.uint64(seed) ^ np.uint64(i) * np.uint64(2862933555777941757) + np.uint64(3037000493)
            ll1 = 0.0
            ll2 = 0.0
            for _k in range(n):
                state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                idx = int(state >> np.uint64(33)) % n
                yi = y_int[idx]
                # Predictor 1
                pi1 = p1[idx]
                if pi1 < eps:
                    pi1 = eps
                elif pi1 > 1.0 - eps:
                    pi1 = 1.0 - eps
                if yi == 1:
                    ll1 -= np.log(pi1)
                else:
                    ll1 -= np.log(1.0 - pi1)
                # Predictor 2
                pi2 = p2[idx]
                if pi2 < eps:
                    pi2 = eps
                elif pi2 > 1.0 - eps:
                    pi2 = 1.0 - eps
                if yi == 1:
                    ll2 -= np.log(pi2)
                else:
                    ll2 -= np.log(1.0 - pi2)
            out[i] = ll1 / n - ll2 / n
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_bootstrap_logloss_binary_samples(y_int, p, n_resamples, seed):
        """Bootstrap CI samples for binary log-loss on a single predictor."""
        n = len(y_int)
        eps = 1e-15
        out = np.empty(n_resamples, dtype=np.float64)
        for i in prange(n_resamples):
            state = np.uint64(seed) ^ np.uint64(i) * np.uint64(2862933555777941757) + np.uint64(3037000493)
            ll = 0.0
            for _k in range(n):
                state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                idx = int(state >> np.uint64(33)) % n
                yi = y_int[idx]
                pi = p[idx]
                if pi < eps:
                    pi = eps
                elif pi > 1.0 - eps:
                    pi = 1.0 - eps
                if yi == 1:
                    ll -= np.log(pi)
                else:
                    ll -= np.log(1.0 - pi)
            out[i] = ll / n
        return out
