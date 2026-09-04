"""Why MLPRanker's FIRST fit in a process differs from every later one.

Background in ``audits/full_audit_2026-09-01/known_complications.md``: fitting the same data with the same
seed four times in one process gives bit-identical INITIAL WEIGHTS every time, and predictions from fits 2-4
that are bit-identical to each other, but fit 1 differs from them by exactly ``0.00631118`` on every run.

This file runs the arms that narrow it down. Each is a separate process by design: the divergence is a
cold-process effect, so anything that warms the process first (importing torch under pytest, a previous fit)
destroys the very thing being measured. That is what invalidated an earlier thread-count probe.

Arms::

    python benchmarks/mlp_ranker_cold_fit_probe.py                 # control
    python benchmarks/mlp_ranker_cold_fit_probe.py --strict        # deterministic algs + cuBLAS workspace
    python benchmarks/mlp_ranker_cold_fit_probe.py --warm          # device generator seeded and drawn first
    python benchmarks/mlp_ranker_cold_fit_probe.py --empty-cache   # allocator emptied before every fit
    CUDA_VISIBLE_DEVICES="" python benchmarks/mlp_ranker_cold_fit_probe.py   # CPU only

Results as of 2026-09-04 (16-thread host, CUDA present): CPU-only is bit-identical across all four fits, and
every other arm reproduces ``0.00631118`` unchanged. The pre-fit CUDA RNG state is byte-identical before all
four fits, and no global precision flag is mutated by the first fit. See the log for what that rules out.
"""

from __future__ import annotations

import hashlib
import os
import sys

STRICT = "--strict" in sys.argv
if STRICT:
    # Must precede the torch import: cuBLAS reads it when its handle is created.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

if STRICT:
    torch.use_deterministic_algorithms(True)

WARM = "--warm" in sys.argv
EMPTY = "--empty-cache" in sys.argv


def rng_fingerprint() -> str:
    """Short hash of the CUDA generator state, or a marker when CUDA is absent."""
    if not torch.cuda.is_available():
        return "no-cuda"
    return hashlib.blake2b(torch.cuda.get_rng_state().cpu().numpy().tobytes(), digest_size=8).hexdigest()


def precision_flags() -> dict:
    """The global precision/determinism switches a first fit could plausibly mutate."""
    return {
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "fp32_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def main() -> None:
    """Fit the same data with the same seed four times and report what differs."""
    from mlframe.training.neural.ranker import MLPRanker

    if WARM and torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.manual_seed_all(0)
        _ = torch.randn(8, device="cuda")
        torch.cuda.synchronize()

    rng = np.random.default_rng(0)
    n = 120
    X = rng.normal(size=(n, 5)).astype(np.float32)
    group_ids = np.repeat(np.arange(n // 6), 6)
    y = rng.integers(0, 4, size=n).astype(np.float32)

    before_flags = precision_flags()
    preds, fingerprints = [], []
    for _ in range(4):
        if EMPTY and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        torch.manual_seed(999)
        fingerprints.append(rng_fingerprint())
        model = MLPRanker(seed=7, n_estimators=3, hidden_layers=(8,), early_stopping_patience=None)
        model.fit(X, y, group_ids)
        preds.append(np.asarray(model.predict(X), dtype=np.float64))
    after_flags = precision_flags()

    arms = [name for flag, name in ((STRICT, "strict"), (WARM, "warm"), (EMPTY, "empty-cache")) if flag]
    print(f"mode: {'+'.join(arms) if arms else 'control'}   cuda={torch.cuda.is_available()}")
    print(f"  pre-fit CUDA rng state identical across all four fits: {len(set(fingerprints)) == 1}")
    changed = {k: (before_flags[k], after_flags[k]) for k in before_flags if before_flags[k] != after_flags[k]}
    print(f"  global precision flags mutated by the fits: {changed or 'none'}")
    for i in range(3):
        d = float(np.max(np.abs(preds[i] - preds[i + 1])))
        print(f"  fit {i + 1} vs fit {i + 2}: {'bit-identical' if d == 0.0 else f'maxdiff {d:.8f}'}")


if __name__ == "__main__":
    main()
