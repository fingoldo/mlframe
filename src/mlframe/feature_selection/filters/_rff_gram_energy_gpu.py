"""Device twin of ``_gradient_interaction_seeder._rff_analytic_mixed_partial_energy_gram``.

The mixed-partial energy of an RFF + ridge surrogate for a pair ``(a, b)`` is ``coef' G coef`` with ``G = cos(Xs W + b)' cos(Xs W + b) / n`` (``D x D``) and
``coef[r] = c_r * sqrt(2 / D) * W[a, r] * W[b, r]``. On the host it is BLAS over a ``(D, D) x (D, 1024)`` block per chunk of pairs, about 46 ms per call at ``n = 2000``,
``D = 400`` and a 100-column pool (4 950 pairs); here the same products run in float32 on the card in one launch sequence and the energies come back in one copy.

Float32 is enough for this use: the energies of the pairs are ranked and compared with a permutation-null floor of the same statistic (computed by the same code), the product
sums are over ``D = 400`` terms, and the relative error of the float32 sums is about 1e-6 (measured in ``tests/feature_selection/gpu/test_rff_gram_energy_device.py``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

#: Number of pairs times features below which the host BLAS is as fast as the upload and the launches (pre-sweep floor; the device twin wins 4x already at 18 000 (45 pairs): `D:/Temp/exp/rff_gram_cross.py`, see the tracker).
DEVICE_FALLBACK_WORK = 10_000


def _device_wanted(n_pairs: int, n_features: int) -> bool:
    """Whether the device twin runs: a GPU is usable, ``MLFRAME_RFF_GRAM_DEVICE`` (``0`` / ``1``) does not force a side, and the work reaches the floor."""
    from ._gpu_policy import cuda_available_for_run, gpu_globally_disabled

    if gpu_globally_disabled() or not cuda_available_for_run():
        return False
    forced = os.environ.get("MLFRAME_RFF_GRAM_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    return int(n_pairs) * int(n_features) >= DEVICE_FALLBACK_WORK


def rff_gram_energies_device(rbf: Any, ridge: Any, Xs: np.ndarray, pairs: Any, chunk: int = 1024) -> Optional[dict]:
    """``{pair: energy}`` as the host Gram kernel returns it, computed on the device; ``None`` when the device route does not apply or fails (the caller runs the host kernel)."""
    pairs = list(pairs)
    W_h = np.asarray(rbf.random_weights_)
    if not _device_wanted(len(pairs), int(W_h.shape[1])) or not pairs:
        return None
    try:
        import cupy as cp

        n_features = int(W_h.shape[1])
        scale = np.sqrt(2.0 / n_features)
        cc = cp.asarray(np.asarray(ridge.coef_, dtype=np.float64).ravel() * scale, dtype=cp.float32)
        W = cp.asarray(W_h, dtype=cp.float32)  # (p, D)
        b = cp.asarray(np.asarray(rbf.random_offset_), dtype=cp.float32)
        X = cp.asarray(np.ascontiguousarray(Xs), dtype=cp.float32)
        mcos = cp.cos(X @ W + b)  # (n, D)
        gram = (mcos.T @ mcos) / np.float32(mcos.shape[0])  # (D, D)
        del mcos
        a_idx = cp.asarray(np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs)))
        b_idx = cp.asarray(np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs)))
        energies = cp.empty(len(pairs), dtype=cp.float32)
        wt = W.T  # (D, p)
        for start in range(0, len(pairs), chunk):
            stop = min(len(pairs), start + chunk)
            coef = cc[:, None] * wt[:, a_idx[start:stop]] * wt[:, b_idx[start:stop]]  # (D, k)
            energies[start:stop] = cp.einsum("dk,dk->k", coef, gram @ coef)
        host = cp.asnumpy(energies)
        return {p: float(host[i]) for i, p in enumerate(pairs)}
    except Exception as exc:
        log_throttle(logger, "rff_gram_energy_device.failed", logging.WARNING, "device RFF Gram energies failed (%s: %s); the host kernel runs", type(exc).__name__, exc)
        return None


def rff_gram_null_maxima_device(rbfs: Any, ridges: Any, Xs: np.ndarray, pairs: Any, chunk: int = 1024) -> Optional[np.ndarray]:
    """The largest pair energy of every ``(rbf, ridge)`` surrogate of ``rbfs`` / ``ridges`` (``(k,)`` host array), each as :func:`rff_gram_energies_device` gives it for that surrogate alone, with the ``k`` surrogates
    of a permutation null run together: the cosine features, the ``D x D`` Grams and the pair products are batched matmuls over a stack of surrogates (a few dozen launches for the whole null instead of about
    forty per surrogate), only the maxima come back. ``None`` when the device route does not apply or fails (the caller scores the surrogates one by one)."""
    pairs = list(pairs)
    rbfs = list(rbfs)
    if not pairs or not rbfs:
        return None
    W0 = np.asarray(rbfs[0].random_weights_)
    if not _device_wanted(len(pairs), int(W0.shape[1])):
        return None
    try:
        import cupy as cp

        n_features = int(W0.shape[1])
        n = int(Xs.shape[0])
        scale = np.sqrt(2.0 / n_features)
        X = cp.asarray(np.ascontiguousarray(Xs), dtype=cp.float32)
        a_idx = cp.asarray(np.fromiter((q[0] for q in pairs), dtype=np.int64, count=len(pairs)))
        b_idx = cp.asarray(np.fromiter((q[1] for q in pairs), dtype=np.int64, count=len(pairs)))
        group = max(1, int((256 << 20) // max(1, n * n_features * 4)))  # surrogates per batch: the (g, n, D) cosine block stays near 256 MB
        out = np.empty(len(rbfs), dtype=np.float64)  # Intentional float64: the null maxima are compared with a quantile on the host
        for g0 in range(0, len(rbfs), group):
            rb = rbfs[g0 : g0 + group]
            rd = ridges[g0 : g0 + group]
            W = cp.asarray(np.stack([np.asarray(r.random_weights_) for r in rb]), dtype=cp.float32)  # (g, p, D)
            off = cp.asarray(np.stack([np.asarray(r.random_offset_) for r in rb]), dtype=cp.float32)  # (g, D)
            cc = cp.asarray(np.stack([np.asarray(r.coef_, dtype=np.float64).ravel() * scale for r in rd]), dtype=cp.float32)  # (g, D)  # Intentional float64: scaled in double, stored float32 as the single route does
            mcos = cp.cos(cp.matmul(X, W) + off[:, None, :])  # (g, n, D)
            gram = cp.matmul(mcos.transpose(0, 2, 1), mcos) / np.float32(n)  # (g, D, D)
            del mcos
            wt = W.transpose(0, 2, 1)  # (g, D, p)
            best = None
            for start in range(0, len(pairs), chunk):
                stop = min(len(pairs), start + chunk)
                coef = cc[:, :, None] * wt[:, :, a_idx[start:stop]] * wt[:, :, b_idx[start:stop]]  # (g, D, k)
                energy = (coef * cp.matmul(gram, coef)).sum(axis=1)  # (g, k)
                top = energy.max(axis=1)
                best = top if best is None else cp.maximum(best, top)
            out[g0 : g0 + len(rb)] = cp.asnumpy(best)
        return out
    except Exception as exc:
        log_throttle(logger, "rff_gram_null_maxima_device.failed", logging.WARNING, "device RFF Gram null maxima failed (%s: %s); the surrogates run one by one", type(exc).__name__, exc)
        return None
