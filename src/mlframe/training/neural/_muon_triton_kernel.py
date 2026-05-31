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

This module is GATED two ways:
  1. ``mlframe.training.neural._triton_bootstrap.ensure_triton_loaded()``
     must succeed (Triton importable on this host)
  2. CUDA device + matrix size large enough that Triton's overhead
     amortises (small matrices stay on torch.matmul)

Fallback: when either gate fails, returns the original PyTorch impl
from ``_muon_optimizer._zeropower_via_newtonschulz5``. Muon callers
don't need to know which backend ran.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

from ._triton_bootstrap import ensure_triton_loaded

logger = logging.getLogger(__name__)

# Compiled-once kernel handle cache. Set on first successful
# ``_get_triton_ns_fn`` call; None means "couldn't compile, use eager".
_TRITON_NS_FN: Optional[Callable] = None
_TRITON_LOAD_ATTEMPTED: bool = False

# Below this matrix dim, eager torch.matmul wins (cuBLAS heuristic +
# Triton compile overhead exceeds the SYRK gain).
_MIN_DIM_FOR_TRITON_NS: int = 256

# F-43 honest measurement (2026-05-31, GTX 1050 Ti / Pascal / cc 6.1):
#   [256x256]   eager 2.67ms vs triton 12.97ms  (0.21x — 5x SLOWER)
#   [512x512]   eager 3.35ms vs triton 46.98ms  (0.07x — 14x SLOWER)
#   [1024x1024] eager 22.66ms vs triton 307.78ms (0.07x — 14x SLOWER)
# Correctness: max_diff 0.002-0.008 (well within BF16 numerical noise).
#
# Root cause: Pascal has no TensorCores; cuBLAS's hand-tuned FP16/BF16
# GEMM kernels are far ahead of a naive Triton kernel. flash-muon's
# reported 1.5-2x wins are all on Ampere+ (compute capability 8.0+)
# where TensorCores accelerate the BF16 matmul and the SYRK upper-
# triangle skip starts paying off.
#
# Hard-gate the Triton path to Ampere+ — on Pascal/Turing/Volta-non-
# tensorcore the eager cuBLAS path wins. Once we have an Ampere+ host
# for re-bench, lift the gate (or auto-detect via cc>=8).
_MIN_COMPUTE_CAPABILITY: tuple = (8, 0)


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
        assert G.ndim == 2 and G.is_cuda, "Triton NS path requires 2D CUDA tensor"
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
            "F-43: Triton Newton-Schulz kernel compiled and ready (Muon "
            "optimizer fast path enabled for matrices >= %d on a side).",
            _MIN_DIM_FOR_TRITON_NS,
        )
    return _TRITON_NS_FN


def maybe_newton_schulz_triton(
    G: torch.Tensor, steps: int = 4,
) -> Optional[torch.Tensor]:
    """Try the Triton Newton-Schulz; return None if not applicable
    (eager fallback at the caller). Gates:
      * G must be a 2D CUDA tensor
      * min(G.shape) >= _MIN_DIM_FOR_TRITON_NS
      * Triton kernel compiled successfully
    """
    if not (G.is_cuda and G.ndim == 2):
        return None
    if min(G.shape) < _MIN_DIM_FOR_TRITON_NS:
        return None
    # F-43 hardware gate: only fire on Ampere+ where TensorCores
    # make the Triton path actually faster than cuBLAS (see bench notes
    # at top of file). Pre-Ampere falls back to torch.matmul.
    try:
        _cc = torch.cuda.get_device_capability(G.device.index)
        if _cc < _MIN_COMPUTE_CAPABILITY:
            return None
    except Exception:
        return None
    fn = get_triton_ns_fn()
    if fn is None:
        return None
    try:
        return fn(G, steps)
    except Exception as _run_err:
        logger.warning(
            "F-43: Triton Newton-Schulz failed at runtime (%s); falling "
            "back to eager torch.matmul.",
            _run_err,
        )
        return None
