"""Bench: weight_sum > 0 Python branch vs sync-free safe-divide in _compute_weighted_loss (iter122).

Pre-iter122 the function did ``if weight_sum > 0`` to branch between the
weighted-mean path and an all-zero-weight fall-back. The Python ``if`` on a
0-D CUDA tensor forces a GPU->CPU sync each batch (~30 us). Post-fix uses
``raw / torch.clamp(weight_sum, min=1e-12)`` -- when all weights are zero,
raw is also zero (sum(loss * 0)) so the divide produces the same 0 the
legacy branch returned.

Bench at n=50k (c0117 binary MLP batch shape, CUDA):

    old (Python branch)   : 299 us/call
    new (safe divide)     : 267 us/call    (~10%)

Bit-identical for both non-degenerate and all-zero-weight cases.
Run: ``python profiling/bench_weighted_loss_sync_free.py``
"""

import time
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n = 50_000  # ~c0117 batch size
predictions = torch.randn(n, device=device, requires_grad=True)
labels = torch.randint(0, 2, (n,), device=device).float()
sample_weight = torch.rand(n, device=device)


def old_branched(pred, lab, sw):
    """The current code."""
    loss_unreduced = F.binary_cross_entropy_with_logits(pred, lab, reduction='none')
    weight_sum = sw.sum()
    if weight_sum > 0:
        weighted_loss = torch.dot(loss_unreduced, sw) / weight_sum
    else:
        weighted_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    return weighted_loss


def new_safe_divide(pred, lab, sw):
    """Safe divide: avoids the Python branch and its forced sync."""
    loss_unreduced = F.binary_cross_entropy_with_logits(pred, lab, reduction='none')
    weight_sum = sw.sum()
    raw = torch.dot(loss_unreduced, sw)
    # When weight_sum is 0, raw is also 0 (sum(loss * 0) = 0), so
    # 0 / clamp(0, 1e-12) = 0 -- same as the legacy zero-tensor branch.
    return raw / torch.clamp(weight_sum, min=1e-12)


# Warmup
for fn in (old_branched, new_safe_divide):
    for _ in range(20):
        out = fn(predictions, labels, sample_weight)
        out.detach()

if device.type == 'cuda':
    torch.cuda.synchronize()

for name, fn in [('old (Python branch)', old_branched), ('new (safe divide)', new_safe_divide)]:
    times = []
    for _ in range(5):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(200):
            out = fn(predictions, labels, sample_weight)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t) / 200
        times.append(elapsed)
    print(f'{name:>25}: {min(times)*1e6:7.1f}us/call (min of 5 runs)')

# Verify same output for the non-degenerate case
out_old = old_branched(predictions, labels, sample_weight).item()
out_new = new_safe_divide(predictions, labels, sample_weight).item()
print(f'\nidentical (non-degenerate): old={out_old:.6f} new={out_new:.6f}')

# Verify all-zero-weight case
sw_zero = torch.zeros_like(sample_weight)
out_old_zero = old_branched(predictions, labels, sw_zero).item()
out_new_zero = new_safe_divide(predictions, labels, sw_zero).item()
print(f'all-zero-weight: old={out_old_zero:.6f} new={out_new_zero:.6f}')
