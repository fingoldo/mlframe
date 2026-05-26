"""Micro-bench: CUDA permutation FI optimization variants.

Compares the current shipped CUDA-batched kernel (F-current) against
three optimization candidates and their combination:

  F-current: rng.permutation(n) on CPU + H2D per (feature, repeat);
             chunk_size = 16 fixed.
  V2:        torch.randperm(n, device=device) -- no H2D for indices.
  V3:        Adaptive chunk_size from torch.cuda.mem_get_info().
  V4:        Gather-based batched permutation -- all permutations in
             a chunk built as one (k * n_repeats, n) index matrix,
             one gather per chunk instead of k * n_repeats writes.
  V5:        V2 + V3 + V4 combined.

Each scenario is run hot (one warmup pass first) so the CUDA DLL
JIT + cache state is comparable. Reports wall time + speedup vs
F-current + Spearman agreement (must stay > 0.95 vs F-current to
prove the optimization didn't break correctness).

Run:
    D:/ProgramData/anaconda3/python.exe profiling/bench_cuda_perm_variants.py
"""
from __future__ import annotations

import gc
import os
import time

# Reduce fragmentation pressure on a 4 GB consumer GPU. Must be set
# BEFORE torch is imported anywhere in the process.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np


_RNG = np.random.default_rng(20260526)


def _gpu_reset():
    """Aggressively free GPU memory between scenarios."""
    try:
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


class _TorchMLPRegressor:
    def __init__(self, n_features: int, hidden: int = 128):
        import torch
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.net.eval()

    def fit(self, X, y, epochs: int = 20):
        import torch
        import torch.nn as nn
        opt = torch.optim.Adam(self.net.parameters(), lr=3e-3)
        loss_fn = nn.MSELoss()
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
        self.net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(self.net(X_t), y_t)
            loss.backward()
            opt.step()
        self.net.eval()
        return self


def _make_task(n_rows: int, n_features: int):
    X = _RNG.standard_normal((n_rows, n_features)).astype(np.float32)
    w = np.zeros(n_features, dtype=np.float32)
    w[:10] = _RNG.uniform(0.5, 2.0, size=10)
    y = X @ w + 0.1 * _RNG.standard_normal(n_rows).astype(np.float32)
    return X, y


# --------------------------------------------------------------------------
# Common helpers
# --------------------------------------------------------------------------

def _baseline_score(net, X_t, y_arr):
    import torch
    from sklearn.metrics import r2_score
    with torch.no_grad():
        pred = net(X_t).reshape(-1).cpu().numpy().astype(np.float64)
    return float(r2_score(y_arr, pred))


def _adaptive_chunk_size(n: int, n_features: int, n_repeats: int,
                         *, safety_fraction: float = 0.2,
                         dtype_bytes: int = 4) -> int:
    """Pick chunk_size that fits ``safety_fraction`` of free GPU memory.

    Memory for one chunk's batched tensor: ``k * n_repeats * n * n_features * dtype_bytes``.
    The 0.2 safety fraction accounts for activations (typically 2-3x the input
    batch size for a small MLP, more for deeper nets), workspace, and the
    PyTorch caching allocator overhead. Cap chunk_size at 64 so very large
    free-memory pools don't push the batched tensor into a regime where
    a single forward pass exceeds the kernel-launch sweet spot.
    """
    import torch
    try:
        free, _ = torch.cuda.mem_get_info()
    except Exception:
        return 16
    per_slot = n_repeats * n * n_features * dtype_bytes
    if per_slot == 0:
        return 16
    k_max = max(1, int((free * safety_fraction) // per_slot))
    return min(k_max, n_features, 64)


# --------------------------------------------------------------------------
# F-current: shipped baseline
# --------------------------------------------------------------------------

def variant_current(net, X, y, *, n_repeats: int, chunk_size: int = 16, seed: int = 0):
    import torch
    from sklearn.metrics import r2_score
    n, n_features = X.shape
    device = torch.device("cuda")
    rng = np.random.default_rng(seed)

    net = net.to(device)
    net.eval()
    X_t = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
    y_arr = np.asarray(y, dtype=np.float64)
    baseline_score = _baseline_score(net, X_t, y_arr)
    importances = np.zeros(n_features, dtype=np.float64)

    t0 = time.perf_counter()
    with torch.no_grad():
        for chunk_start in range(0, n_features, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_features)
            k = chunk_end - chunk_start
            batched = X_t.repeat(k * n_repeats, 1)
            for slot, j in enumerate(range(chunk_start, chunk_end)):
                for r in range(n_repeats):
                    offset = (slot * n_repeats + r) * n
                    perm = torch.as_tensor(rng.permutation(n), dtype=torch.long, device=device)
                    batched[offset:offset + n, j] = X_t[perm, j]
            preds = net(batched).reshape(-1).cpu().numpy().astype(np.float64)
            for slot, j in enumerate(range(chunk_start, chunk_end)):
                scores = np.empty(n_repeats, dtype=np.float64)
                for r in range(n_repeats):
                    offset = (slot * n_repeats + r) * n
                    scores[r] = r2_score(y_arr, preds[offset:offset + n])
                importances[j] = baseline_score - scores.mean()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    try:
        net.to("cpu")
        del batched, X_t
        torch.cuda.empty_cache()
    except Exception:
        pass
    return importances, elapsed


# --------------------------------------------------------------------------
# V2: GPU-side permutation generation (torch.randperm on device)
# --------------------------------------------------------------------------

def variant_v2_gpu_perm(net, X, y, *, n_repeats: int, chunk_size: int = 16, seed: int = 0):
    import torch
    from sklearn.metrics import r2_score
    n, n_features = X.shape
    device = torch.device("cuda")
    torch.manual_seed(seed)

    net = net.to(device)
    net.eval()
    X_t = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
    y_arr = np.asarray(y, dtype=np.float64)
    baseline_score = _baseline_score(net, X_t, y_arr)
    importances = np.zeros(n_features, dtype=np.float64)

    t0 = time.perf_counter()
    with torch.no_grad():
        for chunk_start in range(0, n_features, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_features)
            k = chunk_end - chunk_start
            batched = X_t.repeat(k * n_repeats, 1)
            for slot, j in enumerate(range(chunk_start, chunk_end)):
                for r in range(n_repeats):
                    offset = (slot * n_repeats + r) * n
                    perm = torch.randperm(n, device=device)  # GPU-side
                    batched[offset:offset + n, j] = X_t[perm, j]
            preds = net(batched).reshape(-1).cpu().numpy().astype(np.float64)
            for slot, j in enumerate(range(chunk_start, chunk_end)):
                scores = np.empty(n_repeats, dtype=np.float64)
                for r in range(n_repeats):
                    offset = (slot * n_repeats + r) * n
                    scores[r] = r2_score(y_arr, preds[offset:offset + n])
                importances[j] = baseline_score - scores.mean()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    try:
        net.to("cpu")
        del batched, X_t
        torch.cuda.empty_cache()
    except Exception:
        pass
    return importances, elapsed


# --------------------------------------------------------------------------
# V3: Adaptive chunk_size
# --------------------------------------------------------------------------

def variant_v3_adaptive_chunk(net, X, y, *, n_repeats: int, seed: int = 0):
    import torch
    n, n_features = X.shape
    chunk_size = _adaptive_chunk_size(n, n_features, n_repeats)
    return variant_current(net, X, y, n_repeats=n_repeats, chunk_size=chunk_size, seed=seed) + (chunk_size,)


# --------------------------------------------------------------------------
# V4: Gather-based batched permutation (all permutations as a matrix)
# --------------------------------------------------------------------------

def variant_v4_gather(net, X, y, *, n_repeats: int, chunk_size: int = 16, seed: int = 0):
    import torch
    from sklearn.metrics import r2_score
    n, n_features = X.shape
    device = torch.device("cuda")
    rng = np.random.default_rng(seed)

    net = net.to(device)
    net.eval()
    X_t = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
    y_arr = np.asarray(y, dtype=np.float64)
    baseline_score = _baseline_score(net, X_t, y_arr)
    importances = np.zeros(n_features, dtype=np.float64)

    t0 = time.perf_counter()
    with torch.no_grad():
        for chunk_start in range(0, n_features, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_features)
            k = chunk_end - chunk_start
            # Build all k * n_repeats permutations as one (k*n_rep, n) index matrix on CPU,
            # ship once. (Doing torch.randperm in a vectorised fashion isn't directly
            # supported; we still call it n_perm times -- but the per-call kernel work is
            # small and the H2D is amortised across the whole chunk via stack.)
            perms_np = np.stack([rng.permutation(n) for _ in range(k * n_repeats)])
            perms = torch.as_tensor(perms_np, dtype=torch.long, device=device)
            batched = X_t.repeat(k * n_repeats, 1)
            # For each slot j, replace column j of the corresponding n-row slice with
            # the gathered values. Implemented as a single fancy-indexed write across
            # all slots: build the row index (broadcasted) and apply per-feature.
            for slot, j in enumerate(range(chunk_start, chunk_end)):
                # Slots for THIS feature j: slot * n_repeats ... slot * n_repeats + n_repeats - 1
                rep_start = slot * n_repeats
                # Build the (n_repeats, n) row indices into batched for these slots.
                rows = (torch.arange(rep_start, rep_start + n_repeats, device=device).unsqueeze(1) * n
                        + torch.arange(n, device=device).unsqueeze(0))
                # The values: X_t[perms[rep_start:rep_start+n_repeats], j] -> shape (n_repeats, n)
                src = X_t[perms[rep_start:rep_start + n_repeats], j]
                batched.view(-1, n_features)[rows.flatten(), j] = src.flatten()
            preds = net(batched).reshape(-1).cpu().numpy().astype(np.float64)
            for slot, j in enumerate(range(chunk_start, chunk_end)):
                scores = np.empty(n_repeats, dtype=np.float64)
                for r in range(n_repeats):
                    offset = (slot * n_repeats + r) * n
                    scores[r] = r2_score(y_arr, preds[offset:offset + n])
                importances[j] = baseline_score - scores.mean()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    try:
        net.to("cpu")
        del batched, X_t
        torch.cuda.empty_cache()
    except Exception:
        pass
    return importances, elapsed


# --------------------------------------------------------------------------
# V5: V2 + V3 + V4 combined
# --------------------------------------------------------------------------

def variant_v5_all(net, X, y, *, n_repeats: int, seed: int = 0):
    import torch
    from sklearn.metrics import r2_score
    n, n_features = X.shape
    device = torch.device("cuda")
    torch.manual_seed(seed)

    net = net.to(device)
    net.eval()
    X_t = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
    y_arr = np.asarray(y, dtype=np.float64)
    baseline_score = _baseline_score(net, X_t, y_arr)
    importances = np.zeros(n_features, dtype=np.float64)
    chunk_size = _adaptive_chunk_size(n, n_features, n_repeats)

    t0 = time.perf_counter()
    with torch.no_grad():
        for chunk_start in range(0, n_features, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_features)
            k = chunk_end - chunk_start
            # GPU-side permutations stacked into one (k*n_rep, n) matrix.
            # torch.randperm is sequential per-call, but the call sequence
            # stays on-device (no H2D), and small enough to be cheap.
            perm_list = [torch.randperm(n, device=device) for _ in range(k * n_repeats)]
            perms = torch.stack(perm_list)
            batched = X_t.repeat(k * n_repeats, 1)
            for slot, j in enumerate(range(chunk_start, chunk_end)):
                rep_start = slot * n_repeats
                rows = (torch.arange(rep_start, rep_start + n_repeats, device=device).unsqueeze(1) * n
                        + torch.arange(n, device=device).unsqueeze(0))
                src = X_t[perms[rep_start:rep_start + n_repeats], j]
                batched.view(-1, n_features)[rows.flatten(), j] = src.flatten()
            preds = net(batched).reshape(-1).cpu().numpy().astype(np.float64)
            for slot, j in enumerate(range(chunk_start, chunk_end)):
                scores = np.empty(n_repeats, dtype=np.float64)
                for r in range(n_repeats):
                    offset = (slot * n_repeats + r) * n
                    scores[r] = r2_score(y_arr, preds[offset:offset + n])
                importances[j] = baseline_score - scores.mean()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    try:
        net.to("cpu")
        del batched, X_t
        torch.cuda.empty_cache()
    except Exception:
        pass
    return importances, elapsed, chunk_size


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _spearman(a, b):
    from scipy.stats import spearmanr
    r, _ = spearmanr(a, b)
    return float(r)


def _run(n_rows: int, n_features: int, n_repeats: int):
    print(f"\nTask: n_rows={n_rows}, n_features={n_features}, n_repeats={n_repeats}")
    _gpu_reset()
    X, y = _make_task(n_rows, n_features)
    model = _TorchMLPRegressor(n_features=n_features).fit(X, y)
    # CUDA warmup -- run a throwaway pass through the current variant
    # so all subsequent timings are hot.
    _ = variant_current(model.net, X[:128], y[:128], n_repeats=2, chunk_size=4)
    _gpu_reset()

    imp_cur, t_cur = variant_current(model.net, X, y, n_repeats=n_repeats)
    _gpu_reset()
    print(f"  F-current (chunk=16, CPU rng): {t_cur:6.2f}s  baseline")

    imp_v2, t_v2 = variant_v2_gpu_perm(model.net, X, y, n_repeats=n_repeats)
    _gpu_reset()
    print(f"  V2 GPU perm (chunk=16):        {t_v2:6.2f}s  speed={t_cur/t_v2:.2f}x  "
          f"agree={_spearman(imp_cur, imp_v2):.4f}")

    imp_v3, t_v3, chunk_v3 = variant_v3_adaptive_chunk(model.net, X, y, n_repeats=n_repeats)
    _gpu_reset()
    print(f"  V3 adaptive chunk={chunk_v3:3d}:        {t_v3:6.2f}s  speed={t_cur/t_v3:.2f}x  "
          f"agree={_spearman(imp_cur, imp_v3):.4f}")

    imp_v4, t_v4 = variant_v4_gather(model.net, X, y, n_repeats=n_repeats)
    _gpu_reset()
    print(f"  V4 gather (chunk=16):          {t_v4:6.2f}s  speed={t_cur/t_v4:.2f}x  "
          f"agree={_spearman(imp_cur, imp_v4):.4f}")

    imp_v5, t_v5, chunk_v5 = variant_v5_all(model.net, X, y, n_repeats=n_repeats)
    _gpu_reset()
    print(f"  V5 all (chunk={chunk_v5:3d}, GPU+gather): {t_v5:6.2f}s  speed={t_cur/t_v5:.2f}x  "
          f"agree={_spearman(imp_cur, imp_v5):.4f}")


def main():
    _run(n_rows=5000, n_features=200, n_repeats=5)
    _run(n_rows=5000, n_features=200, n_repeats=10)
    _run(n_rows=5000, n_features=200, n_repeats=20)
    _run(n_rows=2000, n_features=500, n_repeats=5)
    _run(n_rows=1000, n_features=200, n_repeats=10)
    _run(n_rows=10000, n_features=100, n_repeats=10)


if __name__ == "__main__":
    main()
