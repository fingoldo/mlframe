"""Optional CUDA (cupy) backend for the exact subset-sum proxy scan.

The subset-scan is the *cheapest* stage of the pipeline (the research clocked ~0.005 s for the
proven case; SHAP compute + honest re-validation dominate). So this backend earns its keep only for
very large combo counts (>~1e7) on a GPU box - it is opt-in via ``optimizer="bruteforce_gpu"`` and
the facade falls back to the numba path when cupy / a device is unavailable.

One CUDA thread per combination computes its proxy loss by accumulating ``base[i] + sum phi[i, comb]``
per sample and reducing the chosen metric (same integer codes as the numba kernel). Top-N is taken
on-device via argpartition. Lazy kernel compile is guarded by ``multiprocessing.Lock`` + double-
checked locking and cached as a module global, mirroring ``filters/gpu.py`` (Windows-spawn safe).
"""

from __future__ import annotations

import logging
import math
import multiprocessing

import numpy as np

from mlframe.feature_selection.filters._fe_gpu_batch._devices import assert_kernel_buffers, crit_float_dtype
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import METRIC_CODES, resolve_metric
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import generate_combinations, _merge_topn
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_gpu import _cuda_demote_errors

logger = logging.getLogger(__name__)

_KERNEL_INIT_LOCK = multiprocessing.Lock()

# Hand-tuned default; overridden by kernel_tuning_cache when available (guarded for None).
_DEFAULT_BLOCK_SIZE = 256

_KERNEL_SRC = r"""
extern "C" __global__
void subset_loss(const {REAL}* phi, const {REAL}* base, const {REAL}* y,
                 const int* combos, const int n, const int f, const int r,
                 const long long C, const int metric_code, {REAL}* out){
    long long c = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int* comb = combos + c * r;
    double s = 0.0;
    for (int i = 0; i < n; i++){
        double m = base[i];
        for (int jj = 0; jj < r; jj++){
            m += phi[(long long)i * f + comb[jj]];
        }
        double v = y[i];
        if (metric_code == 0){            // MAE
            double d = v - m; s += d < 0 ? -d : d;
        } else if (metric_code == 1){     // MSE
            double d = v - m; s += d * d;
        } else {                          // sigmoid-based
            double p;
            if (m >= 0){ double z = exp(-m); p = 1.0 / (1.0 + z); }
            else       { double z = exp(m);  p = z / (1.0 + z); }
            if (metric_code == 2){        // Brier
                double d = p - v; s += d * d;
            } else {                      // log-loss (clipped)
                double eps = 1e-7;
                if (p < eps) p = eps; else if (p > 1.0 - eps) p = 1.0 - eps;
                s += -(v * log(p) + (1.0 - v) * log(1.0 - p));
            }
        }
    }
    out[c] = ({REAL})(s / (double)n);
}
"""


def gpu_available() -> bool:
    """Report whether a CUDA device is usable for the subset-loss kernel, demoting to False (with a debug log, not a raised error) on any cupy/CUDA failure."""
    try:
        import cupy as cp

        return bool(cp.cuda.runtime.getDeviceCount() > 0)
    except _cuda_demote_errors() as exc:
        logger.debug("GPU subset-loss kernel unavailable, demoting to CPU: %s", exc)
        return False


_SUBSET_LOSS_KERNELS: dict = {}


def _ensure_kernel(real_t: str = "double"):
    """Lazily compile and cache the ``subset_loss`` RawKernel for the buffer element type ``real_t`` (``"float"`` or ``"double"``).

    The loss accumulates in double either way; only the buffers are templated on ``{REAL}``.
    """
    with _KERNEL_INIT_LOCK:
        kern = _SUBSET_LOSS_KERNELS.get(real_t)
        if kern is None:
            import cupy as cp

            kern = cp.RawKernel(_KERNEL_SRC.replace("{REAL}", real_t), "subset_loss")
            _SUBSET_LOSS_KERNELS[real_t] = kern
        return kern


def _block_size() -> int:
    """Hardware-tuned CUDA block size for the subset-loss kernel, or ``_DEFAULT_BLOCK_SIZE`` on a tuning-cache miss."""
    from ._gpu_block_size import block_size_from_tuning_cache

    return block_size_from_tuning_cache("shap_proxy_subset_loss", "block_size", _DEFAULT_BLOCK_SIZE)


def brute_force_top_n_gpu(
    phi: np.ndarray,
    base: np.ndarray,
    y: np.ndarray,
    *,
    classification: bool,
    metric: str | None = None,
    min_card: int = 1,
    max_card: int | None = None,
    top_n: int = 30,
) -> list[tuple[float, tuple[int, ...]]]:
    """GPU exact subset scan; identical contract to ``brute_force_top_n`` (numba)."""
    import cupy as cp

    metric_name = resolve_metric(classification, metric)
    if metric_name == "auc":
        raise ValueError("GPU backend does not support metric='auc'; use brier/logloss.")
    metric_code = METRIC_CODES[metric_name]

    buf_t = crit_float_dtype()
    _real = "float" if buf_t == cp.float32 else "double"
    kernel = _ensure_kernel(_real)
    block = _block_size()

    phi_d = cp.asarray(np.ascontiguousarray(phi, dtype=buf_t))
    base_d = cp.asarray(np.ascontiguousarray(base, dtype=buf_t))
    y_d = cp.asarray(np.ascontiguousarray(y, dtype=buf_t))
    n, f = phi.shape
    max_card = f if max_card is None else min(max_card, f)
    seq = np.arange(f, dtype=np.int32)

    candidates: list[tuple[float, tuple[int, ...]]] = []
    for r in range(min_card, max_card + 1):
        if math.comb(f, r) == 0:
            continue
        combos = generate_combinations(seq, r)
        C = combos.shape[0]
        if C == 0:
            continue
        combos_d = cp.asarray(np.ascontiguousarray(combos, dtype=np.int32).ravel())
        out_d = cp.empty(C, dtype=buf_t)
        grid = ((C + block - 1) // block,)
        assert_kernel_buffers("subset_loss", _real, phi=phi_d, base=base_d, y=y_d, out=out_d)
        kernel(grid, (block,), (phi_d, base_d, y_d, combos_d, np.int32(n), np.int32(f), np.int32(r), np.int64(C), np.int32(metric_code), out_d))
        # SR1 (mirror of the CPU path): map non-finite losses to +inf so a NaN cannot be argpartition-
        # selected as "top"; lower=better, so +inf sinks. Cannot be exercised on this CPU-only box.
        out_d[~cp.isfinite(out_d)] = cp.inf
        k = min(top_n, C)
        sel = cp.argpartition(out_d, k - 1)[:k]
        sel = sel[cp.argsort(out_d[sel])]
        sel_host = cp.asnumpy(sel)
        loss_host = cp.asnumpy(out_d[sel])
        for pos, ci in enumerate(sel_host):
            candidates.append((float(loss_host[pos]), tuple(int(x) for x in combos[ci])))
    merged = _merge_topn(candidates, top_n)
    if metric_name == "rmse":
        merged = [(float(np.sqrt(loss)), comb) for loss, comb in merged]
    return merged
