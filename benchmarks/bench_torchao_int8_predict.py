"""F-57 (2026-05-31): portable bench for torchao Int8WeightOnly on a
representative mlframe tabular MLP predict path.

Per Agent C (2026-05-31 audit), torchao's ``Int8WeightOnlyConfig`` was the
top-1 quantization technique worth prototyping:

  * Memory-BW win composes cleanly with F-40 CUDA-graph predict cache
  * No calibration data required (post-hoc weight quantization)
  * Pascal-compatible per docs (CC >= 6.1)

Honest measurement on Pascal (GTX 1050 Ti, CC 6.1, 2026-05-31):

    bs=  64 in= 32: eager=  0.737ms  int8=  1.950ms  0.38x slower
    bs= 256 in= 64: eager=  0.764ms  int8=  1.551ms  0.49x slower
    bs=1024 in=128: eager=  0.603ms  int8=  1.638ms  0.37x slower
    bs=4096 in=256: eager=  1.864ms  int8=  2.446ms  0.76x slower

Why Pascal loses: torchao's CUDA cpp extensions require torch >= 2.11
(we have 2.7.1). Without cpp extensions torchao falls back to a pure-
Python int8 path; the integer conversion overhead per-call exceeds the
memory-BW savings the actual int8 mm would provide. The TensorCores +
int8 GEMM kernels that win on Ampere+ don't exist on Pascal.

Correctness was fine throughout: max_diff vs fp32 eager < 5e-4 (BF16-
equivalent quantization noise; under MSE-loss training noise).

Recommendation:
  * Pascal hosts: stay on eager fp32 (+ F-40 CUDA-graph) for predict
  * Ampere+ hosts: re-run this bench, expect 1.1-1.5x lift per Agent C
  * Re-evaluate after PyTorch >= 2.11 + torchao cpp extensions land

═══════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════

    pip install torchao
    python bench_torchao_int8_predict.py

For Ampere+ with PyTorch 2.11+ you should see speedups in the 1.1-1.5x
range on small batches (memory-bound regime). bs=4096 may not lift if
the GEMM becomes compute-bound -- Int8WeightOnly only quantizes weights,
activations stay bf16/fp32, so compute path stays at the activation
dtype.
"""
from __future__ import annotations

import copy
import time

import torch
import torch.nn as nn


def build_tabular_mlp(in_dim: int, hidden: int = 256, layers: int = 4, out: int = 1) -> nn.Module:
    """Approximate mlframe.generate_mlp output for a regression head."""
    mods: list[nn.Module] = []
    cur = in_dim
    for _ in range(layers):
        mods += [nn.Linear(cur, hidden), nn.ReLU()]
        cur = hidden
    mods.append(nn.Linear(cur, out))
    return nn.Sequential(*mods)


def bench_predict(model: nn.Module, X: torch.Tensor,
                  n_warmup: int = 5, n_iter: int = 50) -> float:
    """Return per-call wall time in milliseconds."""
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(X)
        if X.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = model(X)
        if X.is_cuda:
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iter * 1000


def main() -> None:
    print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        cc = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name()
        print(f"GPU: {name} (compute capability {cc[0]}.{cc[1]})")
        ampere_plus = cc >= (8, 0)
        print(f"Ampere+: {'YES' if ampere_plus else 'NO (expect Pascal-style regression per file header)'}")

    torch.manual_seed(0)

    try:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
    except ImportError as e:
        print(f"ERROR: torchao import failed ({e}); install with `pip install torchao`.")
        return

    print()
    print(f"{'shape':>14} | {'eager (ms)':>10} | {'int8 (ms)':>10} | {'speedup':>8} | {'max_diff':>10}")
    print("-" * 70)
    for batch_size, in_dim in [(64, 32), (256, 64), (1024, 128), (4096, 256)]:
        model = build_tabular_mlp(in_dim=in_dim).to(device).eval()
        X = torch.randn(batch_size, in_dim, device=device)

        eager_ms = bench_predict(model, X)

        model_q = copy.deepcopy(model)
        quantize_(model_q, Int8WeightOnlyConfig())

        with torch.no_grad():
            out_eager = model(X)
            out_q = model_q(X)
            max_diff = (out_eager - out_q).abs().max().item()

        q_ms = bench_predict(model_q, X)
        speedup = eager_ms / q_ms if q_ms > 0 else float("nan")
        shape = f"{batch_size}x{in_dim}"
        print(f"{shape:>14} | {eager_ms:>10.3f} | {q_ms:>10.3f} | {speedup:>7.2f}x | {max_diff:>10.5f}")

    print()
    print("Notes:")
    print("  * max_diff up to ~5e-4 is BF16-equivalent noise; well under MSE training noise.")
    print("  * Speedup < 1.0 means int8 is SLOWER -- the regime Pascal currently hits.")
    print("  * Ampere+ with PyTorch 2.11+ should show 1.1-1.5x lift on small batches.")


if __name__ == "__main__":
    main()
