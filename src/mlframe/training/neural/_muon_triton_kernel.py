"""F-43 (2026-05-31): Optional Triton-accelerated Newton-Schulz for the
Muon optimizer.

Per the 2026-05-31 PyTorch optimization audit (Agent C top-1
recommendation), vendoring flash-muon's Triton kernel for Newton-Schulz
yields 1.5-2x on the NS path by exploiting:

  1. ``X @ X.T`` symmetry (SYRK pattern) -- compute only the upper
     triangle of A, mirror to the lower; PyTorch's matmul doesn't
     have a Python-exposed SYRK so this is the main Triton-only win.
  2. Larger BLOCK sizes than cuBLAS's default heuristic chooses for
     the small-K matmuls Newton-Schulz hits (typical Muon parameter
     shape: 256 x 256 to 2048 x 2048).

This module is GATED:
  1. ``mlframe.training.neural._triton_bootstrap.ensure_triton_loaded()``
     must succeed (Triton importable on this host)
  2. CUDA device, 2D, and matrix size large enough that Triton's launch
     overhead amortises (small matrices stay on torch.matmul)
  3. A one-shot per-device calibration must show Triton actually beating
     eager cuBLAS on THIS GPU. Compute capability alone is not trusted:
     low-end Ampere+/Ada laptop parts have TensorCores but still lose to
     cuBLAS, so the decision is measured per device and cached, not
     hardcoded from the compute capability.

Fallback: when either gate fails, returns the original PyTorch impl
from ``_muon_optimizer._zeropower_via_newtonschulz5``. Muon callers
don't need to know which backend ran.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional

import torch

from ._triton_bootstrap import ensure_triton_loaded

logger = logging.getLogger(__name__)

# Compiled-once kernel handle cache. Set on first successful
# ``_get_triton_ns_fn`` call; None means "couldn't compile, use eager".
_TRITON_NS_FN: Optional[Callable] = None
_TRITON_LOAD_ATTEMPTED: bool = False

# Below this matrix dim, eager torch.matmul wins on every GPU: the Triton
# kernel-launch overhead exceeds the SYRK gain on small matrices.
_MIN_DIM_FOR_TRITON_NS: int = 256

# Cheap pre-filter before the (expensive) Triton compile + calibration.
# Pre-Ampere cards have no TensorCores, so cuBLAS's BF16 GEMM always beats
# the Triton SYRK kernel there; skip them without paying the compile cost.
# Ampere+ (cc >= 8.0) is NOT assumed to win: low-end Ada/Ampere laptop parts
# (few SMs, narrow memory bus) still lose. That call is made empirically per
# device by the calibration below, never from the compute capability alone.
_MIN_COMPUTE_CAPABILITY: tuple = (8, 0)

# Triton must beat eager by at least this factor on the one-shot calibration
# before we route the Muon NS path through it. A bare > 1.0 would flip on
# near-ties and take on the SYRK NaN-init / launch risk for no real gain.
_TRITON_WIN_MARGIN: float = 1.10

# Per-(device_index, size_bucket) empirical verdict, measured once per
# process: True -> Triton was faster on this GPU+shape, use it; False -> eager
# won, stay on torch.matmul. Cleared between processes; cheap to repopulate.
_TRITON_VERDICT: dict = {}

# Override the empirical gate: "0"/"off"/"false" force eager, "1"/"on"/"true"
# force Triton (skips calibration), anything else / unset -> auto-calibrate.
_TRITON_ENV_VAR: str = "MLFRAME_MUON_TRITON"


def _build_triton_ns_fn() -> Optional[Callable]:
    """Compile the Triton Newton-Schulz quintic on first call.

    Returns a callable ``(G, steps) -> X`` that runs entirely on GPU,
    OR returns None if compilation fails."""
    if not ensure_triton_loaded():
        return None

    try:
        import triton
        import triton.language as tl
    except Exception as _imp_err:
        logger.debug("F-43: Triton import failed (%s)", _imp_err)
        return None

    # SYRK kernel: A = X @ X.T (X is M x N, A is M x M, symmetric).
    # We compute only the upper triangle then mirror.
    @triton.jit
    def _syrk_upper_kernel(
        x_ptr, a_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_am, stride_ak,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        # 2-D grid over (M, N) tiles of the output A.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        # Upper-triangle skip: if the tile is strictly below the diagonal,
        # don't compute (we'll mirror from the upper tile later).
        # Tile-level skip uses the tile's UPPER-LEFT corner; can be
        # imprecise on the diagonal tile but correctness is preserved
        # because we mirror after.
        if pid_n * BLOCK_N + BLOCK_N <= pid_m * BLOCK_M:
            return
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # K-loop: accumulate X[:, k] @ X[:, k].T as a tile.
        for k in range(0, K, BLOCK_K):
            x_block_a = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + (k + offs_k)[None, :] * stride_xk,
                mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K),
                other=0.0,
            )
            x_block_b = tl.load(
                x_ptr + offs_n[:, None] * stride_xm + (k + offs_k)[None, :] * stride_xk,
                mask=(offs_n[:, None] < M) & ((k + offs_k)[None, :] < K),
                other=0.0,
            )
            acc += tl.dot(x_block_a, tl.trans(x_block_b))
        tl.store(
            a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_ak,
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < M),
        )

    def _syrk_via_triton(X: torch.Tensor) -> torch.Tensor:
        """Compute X @ X.T using the upper-triangle Triton kernel + mirror.

        X: (M, N) float / bf16 contiguous on CUDA.
        Returns A: (M, M) same dtype, symmetric.
        """
        M, N = X.shape
        # F-60 (2026-05-31): zero-init A, NOT torch.empty. The upper-tri
        # kernel SKIPS strict-lower-triangle tiles (line 96-97 early return)
        # AND masks out off-tile positions inside the kernel, so the LOWER
        # half + masked positions of A are NEVER written. ``torch.triu(A) +
        # torch.triu(A, diagonal=1).T`` then reads those uninitialised
        # positions: usually finite garbage that gets zeroed by triu, but
        # CUDA's caching allocator can reuse a freed buffer whose previous
        # owner left NaN/Inf bit patterns. ``0 * NaN = NaN`` propagates
        # through triu's multiply -> the SYRK output gets NaN -> Newton-
        # Schulz orthogonalisation gets NaN -> Muon's update tensor gets
        # NaN -> silent training corruption with the Muon optimiser.
        # Same root cause as F-58 (kernel writes subset, caller reads whole).
        # Currently dormant: the path is hard-gated to Ampere+ (cc >= 8.0)
        # so Pascal dev hosts never hit it; will bite the next Ampere+
        # rebench. One cudaMemsetAsync per NS step is negligible vs the
        # matmul, so the zero-init is the cheap robust default.
        A = torch.zeros((M, M), dtype=X.dtype, device=X.device)
        # Tile sizes -- tuned for Ampere+. Pre-Ampere uses smaller blocks.
        BLOCK_M = 128 if M >= 256 else 64
        BLOCK_N = BLOCK_M
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(M, BLOCK_N))
        _syrk_upper_kernel[grid](
            X, A,
            M, M, N,
            X.stride(0), X.stride(1),
            A.stride(0), A.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        # Mirror the upper triangle to fill the lower (so callers see
        # the full symmetric A).
        A = torch.triu(A) + torch.triu(A, diagonal=1).T
        return A

    def _newton_schulz_triton(G: torch.Tensor, steps: int = 4) -> torch.Tensor:
        """Triton-backed Newton-Schulz quintic.

        Same numerical contract as ``_muon_optimizer._zeropower_via_newtonschulz5``
        but the X @ X.T step uses the Triton SYRK kernel above.
        """
        assert G.ndim == 2 and G.is_cuda, "Triton NS path requires 2D CUDA tensor"  # nosec B101 - internal invariant check in src/mlframe/training/neural, not reachable with untrusted input
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.to(torch.bfloat16) if G.is_cuda else G.to(torch.float32)
        transposed = X.size(0) > X.size(1)
        if transposed:
            X = X.transpose(0, 1).contiguous()
        X = X / (X.norm() + 1e-7)
        for _ in range(steps):
            # SYRK: A = X @ X.T (symmetric)
            A = _syrk_via_triton(X.contiguous())
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        if transposed:
            X = X.transpose(0, 1)
        return X.to(dtype=G.dtype)

    return _newton_schulz_triton


def get_triton_ns_fn() -> Optional[Callable]:
    """Lazy-load and cache the Triton Newton-Schulz function."""
    global _TRITON_NS_FN, _TRITON_LOAD_ATTEMPTED
    if _TRITON_LOAD_ATTEMPTED:
        return _TRITON_NS_FN
    _TRITON_LOAD_ATTEMPTED = True
    _TRITON_NS_FN = _build_triton_ns_fn()
    if _TRITON_NS_FN is not None:
        logger.info(
            "Triton Newton-Schulz kernel compiled; eligible for Muon matrices " ">= %d on a side, subject to the per-device calibration.",
            _MIN_DIM_FOR_TRITON_NS,
        )
    return _TRITON_NS_FN


def _size_bucket(min_dim: int) -> int:
    """Round up to a power-of-two bucket so one calibration is reused across
    nearby Muon parameter shapes instead of re-measured for every distinct
    matrix dim."""
    b = 1
    while b < min_dim:
        b <<= 1
    return b


def _env_force() -> Optional[bool]:
    """Read the ``MLFRAME_MUON_TRITON`` override. True/False force a backend;
    None means auto-calibrate."""
    val = os.environ.get(_TRITON_ENV_VAR, "").strip().lower()
    if val in ("0", "off", "false", "no"):
        return False
    if val in ("1", "on", "true", "yes"):
        return True
    return None


def _calibrate_triton_vs_eager(fn, shape, dtype, device, steps) -> float:
    """Measure eager vs Triton Newton-Schulz once on a synthetic tensor of the
    target shape. Returns eager_time / triton_time (> 1 means Triton is faster);
    0.0 if anything fails, which the caller reads as "use eager"."""
    import time

    from ._muon_optimizer import _zeropower_via_newtonschulz5

    warmup, iters = 3, 20  # enough to settle clocks; this runs once per device+bucket
    try:
        probe = torch.randn(*shape, device=device, dtype=dtype)
        for _ in range(warmup):  # warm up compile + caches on both paths
            _zeropower_via_newtonschulz5(probe, steps=steps)
            fn(probe, steps)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            _zeropower_via_newtonschulz5(probe, steps=steps)
        torch.cuda.synchronize(device)
        eager_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(probe, steps)
        torch.cuda.synchronize(device)
        triton_t = time.perf_counter() - t0
        return (eager_t / triton_t) if triton_t > 0 else 0.0
    except Exception as _cal_err:
        logger.debug("Muon Triton calibration failed (%s); using eager.", _cal_err)
        return 0.0


def maybe_newton_schulz_triton(
    G: torch.Tensor, steps: int = 4,
) -> Optional[torch.Tensor]:
    """Try the Triton Newton-Schulz; return None to tell the caller to use the
    eager torch.matmul path. Gates, in order:

      * G must be a 2D CUDA tensor with min(shape) >= _MIN_DIM_FOR_TRITON_NS
      * ``MLFRAME_MUON_TRITON`` override (force eager / force Triton)
      * compute capability >= 8.0 (cheap pre-filter; pre-Ampere always loses)
      * a one-shot per-device calibration: Triton runs only if it actually
        beats eager by _TRITON_WIN_MARGIN on this GPU. Low-end Ampere+/Ada
        laptop cards measure a loss here and stay on eager.
    """
    if not (G.is_cuda and G.ndim == 2):
        return None
    if min(G.shape) < _MIN_DIM_FOR_TRITON_NS:
        return None

    forced = _env_force()
    if forced is False:
        return None

    try:
        dev_index = G.device.index if G.device.index is not None else torch.cuda.current_device()
        cc = torch.cuda.get_device_capability(dev_index)
    except Exception:
        return None
    if forced is not True and cc < _MIN_COMPUTE_CAPABILITY:
        return None

    fn = get_triton_ns_fn()
    if fn is None:
        return None

    if forced is not True:
        key = (dev_index, _size_bucket(min(G.shape)))
        verdict = _TRITON_VERDICT.get(key)
        if verdict is None:
            speedup = _calibrate_triton_vs_eager(fn, tuple(G.shape), G.dtype, G.device, steps)
            verdict = speedup >= _TRITON_WIN_MARGIN
            _TRITON_VERDICT[key] = verdict
            logger.info(
                "Muon Triton calibration on %s (bucket %d): %.2fx vs eager -> %s",
                torch.cuda.get_device_name(dev_index), key[1], speedup,
                "Triton" if verdict else "eager",
            )
        if not verdict:
            return None

    try:
        return fn(G, steps)
    except Exception as _run_err:
        logger.warning(
            "Muon Triton Newton-Schulz failed at runtime (%s); falling back " "to eager torch.matmul.",
            _run_err,
        )
        return None
