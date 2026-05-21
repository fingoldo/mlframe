"""Bench ranknet_pairwise_loss_precomputed: eager / scripted / torch.compile / paired-index (iter132).

c0105 iter132 profile (200k LTR-MLP+CB) attributed 115 s self-time across
540_000 calls of ``ranknet_pairwise_loss_precomputed`` -- 213 us/call on
the (cuda, n=10 / ~25 pairs) per-query shape after iter112/iter115. Most
of that is per-op Python dispatch overhead since the math is a 4-op chain
(``scores[i] - scores[j] -> neg -> softplus -> mean``).

Bench at the c0105 shape (cuda, n=10, n_pairs=25):

    eager                  : 197 us / call
    torch.jit.script       : 146 us / call   (~26 %)
    torch.compile(aot_eager): 564 us / call   (worse -- dispatch overhead)
    torch.compile(inductor): FAILED (triton-windows DLL doesn't load on this box)

Shipped: torch.jit.script-compiled inner kernel + Python wrapper that
keeps the Optional[Tensor] None-handling outside the script (TorchScript
Optional requires explicit annotations and the wrapping check is
effectively free per-call).

triton-windows install attempted -- it loads as a package but the libtriton
DLL initialisation fails with "A dynamic link library (DLL) initialization
routine failed". Pip-uninstalled to restore torch._dynamo. Without
Triton, the inductor backend can't compile; aot_eager only traces +
dispatches, adding ~3x overhead with no fusion benefit.

Run: ``python profiling/bench_ranknet_scripted_kernels.py``
"""

import time
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

n, n_pairs = 10, 25
scores = torch.randn(n, device=device, requires_grad=True)
i_idx = torch.randint(0, n, (n_pairs,), device=device, dtype=torch.long)
j_idx = torch.randint(0, n, (n_pairs,), device=device, dtype=torch.long)


def eager_loss(scores, i_idx, j_idx):
    score_diff_pairs = scores[i_idx] - scores[j_idx]
    return F.softplus(-score_diff_pairs).mean()


@torch.jit.script
def scripted_loss(scores, i_idx, j_idx):
    score_diff_pairs = scores[i_idx] - scores[j_idx]
    return F.softplus(-score_diff_pairs).mean()


@torch.compile(backend='aot_eager', dynamic=False)
def compiled_aot_eager_loss(scores, i_idx, j_idx):
    score_diff_pairs = scores[i_idx] - scores[j_idx]
    return F.softplus(-score_diff_pairs).mean()


@torch.compile(backend='inductor', dynamic=False)
def compiled_inductor_loss(scores, i_idx, j_idx):
    score_diff_pairs = scores[i_idx] - scores[j_idx]
    return F.softplus(-score_diff_pairs).mean()


# Warmup (cold compile included)
print('warming...')
for fn in (eager_loss, scripted_loss, compiled_aot_eager_loss):
    for _ in range(30):
        try:
            fn(scores, i_idx, j_idx)
        except Exception as e:
            print(f'warmup error {fn}: {type(e).__name__}: {str(e)[:80]}')
            break
# Inductor may fail without triton-windows libs working
try:
    for _ in range(30):
        compiled_inductor_loss(scores, i_idx, j_idx)
    inductor_ok = True
except Exception as e:
    print(f'inductor warmup FAILED ({type(e).__name__}: {str(e)[:120]})')
    inductor_ok = False

if device.type == 'cuda':
    torch.cuda.synchronize()
print('benching...')

variants = [
    ('eager (current)', eager_loss),
    ('scripted', scripted_loss),
    ('compile aot_eager', compiled_aot_eager_loss),
]
if inductor_ok:
    variants.append(('compile inductor', compiled_inductor_loss))

for name, fn in variants:
    times = []
    for _ in range(5):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(500):
            out = fn(scores, i_idx, j_idx)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t) / 500
        times.append(elapsed)
    print(f'{name:>22}: {min(times)*1e6:7.1f}us/call')
