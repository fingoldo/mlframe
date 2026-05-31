"""Portable bench: Triton-backed Newton-Schulz vs eager torch.matmul
for the Muon optimizer hot path (F-43 audit).

Run on any CUDA host to find the speedup ratio on your GPU. On Ampere+
(compute capability >= 8.0) the Triton SYRK kernel should beat cuBLAS
by 1.5-2x per flash-muon's reported numbers. On Pascal / Turing / Volta
without TensorCores the eager path wins by 5-14x (measured on a
GTX 1050 Ti 2026-05-31).

═══════════════════════════════════════════════════════════════════════
INSTALLATION (Windows + Linux)
═══════════════════════════════════════════════════════════════════════

LINUX (Ubuntu/Debian):
    pip install triton
    # That's it — Triton's Linux wheels ship with everything.

WINDOWS — IMPORTANT, NON-OBVIOUS STEPS:

    1. Install triton-windows:
           pip install triton-windows

    2. Test if it imports out of the box:
           python -c "import triton; print(triton.__version__)"

    3. If you get `WinError 1114: A dynamic link library (DLL)
       initialization routine failed` — this is a known Windows
       DLL search path issue. The fix:

       a. Add this preload to the TOP of your script (BEFORE any
          other imports that touch CUDA / Triton):

              import ctypes
              import site, os
              for sp in site.getsitepackages():
                  pyd = os.path.join(sp, "triton", "_C", "libtriton.pyd")
                  if os.path.exists(pyd):
                      ctypes.WinDLL(pyd, winmode=0x8)  # LOAD_WITH_ALTERED_SEARCH_PATH
                      break

       b. OR use mlframe's built-in bootstrap which does the above
          for you:

              from mlframe.training.neural._triton_bootstrap import ensure_triton_loaded
              ensure_triton_loaded()

    4. Verify with a smoke test:
           import triton, triton.language as tl
           # If no ImportError, you're set.

WINDOWS PREREQUISITES (often missed):
    * CUDA Toolkit 12.x installed (check: nvcc --version)
    * Visual C++ Redistributable 2017+ (provides VCRUNTIME140_1.dll)
    * Matching PyTorch CUDA version (triton-windows 3.7 expects PT 2.7+)

═══════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════

    python bench_muon_triton_vs_eager.py

    Optional flags:
        --sizes 256,512,1024,2048   matrix dims to test (square)
        --steps 4                    Newton-Schulz iterations
        --warmup 5                   per-config warmup count
        --iters 50                   bench iteration count
"""
from __future__ import annotations

import argparse
import os
import sys
import time


# === Windows-specific Triton preload (no-op on Linux/Mac) ====================

def _bootstrap_triton_windows() -> bool:
    """Preload libtriton.pyd via WinDLL(winmode=0x8) so Python's import
    can find its delay-loaded deps. Idempotent + safe on non-Windows."""
    if sys.platform != "win32":
        return True
    try:
        import triton  # noqa: F401
        return True  # already works, no bootstrap needed
    except ImportError:
        pass
    try:
        import ctypes
        import site
        candidates = []
        for sp in site.getsitepackages():
            candidates.append(os.path.join(sp, "triton", "_C", "libtriton.pyd"))
        try:
            candidates.append(os.path.join(site.getusersitepackages(), "triton", "_C", "libtriton.pyd"))
        except Exception:
            pass
        for pyd in candidates:
            if not os.path.exists(pyd):
                continue
            ctypes.WinDLL(pyd, winmode=0x8)
            import triton  # noqa: F401
            return True
        return False
    except Exception as e:
        print(f"[bootstrap] Triton preload failed: {e}")
        return False


# === Core kernel + eager reference =========================================

_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def _eager_newton_schulz(G, steps: int = 4):
    """Reference impl identical to mlframe._zeropower_via_newtonschulz5
    minus the Triton dispatch. Pure torch.matmul."""
    import torch
    a, b, c = _NS_COEFFS
    X = G.to(torch.bfloat16) if G.is_cuda else G.to(torch.float32)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.transpose(0, 1).contiguous()
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(0, 1)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.transpose(0, 1)
    return X.to(dtype=G.dtype)


def _build_triton_ns():
    """Compile + return the Triton Newton-Schulz function. None on failure."""
    try:
        import triton
        import triton.language as tl
        import torch
    except Exception:
        return None

    @triton.jit
    def _syrk_upper_kernel(
        x_ptr, a_ptr, M, N, K,
        stride_xm, stride_xk, stride_am, stride_ak,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        if pid_n * BLOCK_N + BLOCK_N <= pid_m * BLOCK_M:
            return
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            xa = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + (k + offs_k)[None, :] * stride_xk,
                mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K),
                other=0.0,
            )
            xb = tl.load(
                x_ptr + offs_n[:, None] * stride_xm + (k + offs_k)[None, :] * stride_xk,
                mask=(offs_n[:, None] < M) & ((k + offs_k)[None, :] < K),
                other=0.0,
            )
            acc += tl.dot(xa, tl.trans(xb))
        tl.store(
            a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_ak,
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < M),
        )

    def syrk(X):
        M, N = X.shape
        A = torch.empty((M, M), dtype=X.dtype, device=X.device)
        BLOCK_M = 128 if M >= 256 else 64
        BLOCK_N = BLOCK_M
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(M, BLOCK_N))
        _syrk_upper_kernel[grid](
            X, A, M, M, N,
            X.stride(0), X.stride(1), A.stride(0), A.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return torch.triu(A) + torch.triu(A, diagonal=1).T

    def triton_ns(G, steps=4):
        a, b, c = _NS_COEFFS
        X = G.to(torch.bfloat16)
        transposed = X.size(0) > X.size(1)
        if transposed:
            X = X.transpose(0, 1).contiguous()
        X = X / (X.norm() + 1e-7)
        for _ in range(steps):
            A = syrk(X.contiguous())
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        if transposed:
            X = X.transpose(0, 1)
        return X.to(dtype=G.dtype)

    return triton_ns


# === Bench loop ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="256,512,1024,2048",
                        help="comma-separated square matrix dims")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    print("=" * 70)
    print("Triton Newton-Schulz bench")
    print("=" * 70)

    print(f"Platform: {sys.platform}")
    print(f"Python:   {sys.version.split()[0]}")
    if not _bootstrap_triton_windows():
        print("WARNING: Triton not importable. Bench will only run eager path.")

    import torch
    print(f"PyTorch:  {torch.__version__}")
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        name = torch.cuda.get_device_name(dev)
        cc = torch.cuda.get_device_capability(dev)
        print(f"GPU:      {name}  (compute capability {cc[0]}.{cc[1]})")
        ampere_plus = cc >= (8, 0)
        print(f"Ampere+:  {'YES (TensorCores available)' if ampere_plus else 'NO (no TensorCores; Triton SYRK likely loses)'}")
    else:
        print("No CUDA device — abort.")
        return

    triton_ns = _build_triton_ns()
    if triton_ns is None:
        print("Triton compile failed — bench will skip Triton column.")

    sizes = [int(s) for s in args.sizes.split(",")]
    print()
    print(f"{'Shape':>12} | {'Eager (ms)':>10} | {'Triton (ms)':>11} | {'Speedup':>8} | {'Diff':>8}")
    print("-" * 70)
    for n in sizes:
        G = torch.randn(n, n, device="cuda", dtype=torch.float32)

        # Correctness
        out_e = _eager_newton_schulz(G, steps=args.steps)
        if triton_ns is not None:
            try:
                out_t = triton_ns(G, steps=args.steps)
                diff = (out_e - out_t).abs().max().item()
            except Exception as e:
                print(f"{f'{n}x{n}':>12} | Triton failed: {e}")
                continue
        else:
            diff = float("nan")

        # Bench eager
        for _ in range(args.warmup):
            _ = _eager_newton_schulz(G, steps=args.steps)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            _ = _eager_newton_schulz(G, steps=args.steps)
        torch.cuda.synchronize()
        eager_t = (time.perf_counter() - t0) / args.iters * 1000

        # Bench triton
        if triton_ns is not None:
            for _ in range(args.warmup):
                _ = triton_ns(G, steps=args.steps)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                _ = triton_ns(G, steps=args.steps)
            torch.cuda.synchronize()
            triton_t = (time.perf_counter() - t0) / args.iters * 1000
            speedup = eager_t / triton_t if triton_t > 0 else 0
            print(f"{f'{n}x{n}':>12} | {eager_t:>10.2f} | {triton_t:>11.2f} | {speedup:>7.2f}x | {diff:>8.4f}")
        else:
            print(f"{f'{n}x{n}':>12} | {eager_t:>10.2f} | {'N/A':>11} | {'N/A':>8} | {'N/A':>8}")

    print()
    print("Notes:")
    print("  * 'Diff' = max abs difference between eager and Triton outputs.")
    print("    BF16 noise level is ~0.01; values < 0.1 mean numerically equivalent.")
    print("  * On Ampere+ Triton should win 1.5-2x (per flash-muon github).")
    print("  * On pre-Ampere (no TensorCores) eager cuBLAS dominates.")


if __name__ == "__main__":
    main()
