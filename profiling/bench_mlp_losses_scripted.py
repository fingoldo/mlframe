"""Bench torch.jit.script vs eager for MLP loss kernels (iter132 follow-up).

Per user "может, то же самое можно применить к mlp/recurrent для всех видов
таргета (включая регрессию/классификацию?) побенчи на разных размерах":
extended the iter132 ranknet scripted experiment to the general MLP loss
paths (CE / BCE / MSE) used by classification + regression.

Bench results (cuda, all sizes 1k / 10k / 50k / 200k):

    Multiclass CE weighted (K=3) : 215 -> 186 us/call  (1.11-1.17x)
    Binary BCE weighted          : 235 -> 210 us/call  (1.08-1.13x)
    Regression MSE weighted      : 180 -> 145 us/call  (1.22-1.26x)
    ListNet (n=10-200)           : 150 -> 145 us/call  (1.02-1.04x, marginal)
    ListNet (n=1000)             : 153 -> 166 us/call  (0.92x, REGRESSION)

Decision: NOT shipped for general MLP losses. The per-call savings (20-35 us)
multiply out to ~3 ms / fit on typical batch counts (~100 calls), which
is below the noise floor of the surrounding GPU/CPU dispatch. The
ranknet path ships scripted because it pays ~50 us / call x 540k calls =
27 s on a single c0105 fit -- 4 orders of magnitude more amortisation.

ListNet specifically REGRESSES at n=1000 (TorchScript's broadcast-handling
on the (N,) ranks softmax adds overhead vs eager); kept on eager.

Run: ``python profiling/bench_mlp_losses_scripted.py``
"""

import time
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')


# === Per-sample weighted loss (the _compute_weighted_loss inner) ===
def eager_weighted_ce(predictions, labels, sample_weight):
    loss_unreduced = F.cross_entropy(predictions, labels, reduction='none')
    weight_sum = sample_weight.sum()
    raw = torch.dot(loss_unreduced, sample_weight)
    return raw / torch.clamp(weight_sum, min=1e-12)


@torch.jit.script
def scripted_weighted_ce(predictions: torch.Tensor, labels: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    loss_unreduced = F.cross_entropy(predictions, labels, reduction='none')
    weight_sum = sample_weight.sum()
    raw = torch.dot(loss_unreduced, sample_weight)
    return raw / torch.clamp(weight_sum, min=1e-12)


def eager_weighted_mse(predictions, labels, sample_weight):
    loss_unreduced = F.mse_loss(predictions, labels, reduction='none')
    weight_sum = sample_weight.sum()
    raw = torch.dot(loss_unreduced, sample_weight)
    return raw / torch.clamp(weight_sum, min=1e-12)


@torch.jit.script
def scripted_weighted_mse(predictions: torch.Tensor, labels: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    loss_unreduced = F.mse_loss(predictions, labels, reduction='none')
    weight_sum = sample_weight.sum()
    raw = torch.dot(loss_unreduced, sample_weight)
    return raw / torch.clamp(weight_sum, min=1e-12)


def eager_weighted_bce(predictions, labels, sample_weight):
    loss_unreduced = F.binary_cross_entropy_with_logits(predictions, labels, reduction='none')
    weight_sum = sample_weight.sum()
    raw = torch.dot(loss_unreduced, sample_weight)
    return raw / torch.clamp(weight_sum, min=1e-12)


@torch.jit.script
def scripted_weighted_bce(predictions: torch.Tensor, labels: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    loss_unreduced = F.binary_cross_entropy_with_logits(predictions, labels, reduction='none')
    weight_sum = sample_weight.sum()
    raw = torch.dot(loss_unreduced, sample_weight)
    return raw / torch.clamp(weight_sum, min=1e-12)


# === ListNet (1 query per batch, like ranknet) ===
def eager_listnet(scores, rel):
    if scores.dim() != 1:
        scores = scores.view(-1)
    rel = rel.view(-1).to(scores.dtype)
    true_p = F.softmax(rel, dim=0)
    pred_log_p = F.log_softmax(scores, dim=0)
    return -(true_p * pred_log_p).sum()


@torch.jit.script
def scripted_listnet(scores: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
    if scores.dim() != 1:
        scores = scores.view(-1)
    rel = rel.view(-1).to(scores.dtype)
    true_p = F.softmax(rel, dim=0)
    pred_log_p = F.log_softmax(scores, dim=0)
    return -(true_p * pred_log_p).sum()


sizes = [1_000, 10_000, 50_000, 200_000]
print()
print("== Multiclass CE (K=3) weighted ==")
for n in sizes:
    pred = torch.randn(n, 3, device=device, requires_grad=True)
    lab = torch.randint(0, 3, (n,), device=device)
    sw = torch.rand(n, device=device)
    # Warmup
    for _ in range(30):
        eager_weighted_ce(pred, lab, sw)
        scripted_weighted_ce(pred, lab, sw)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    et, st = [], []
    for _ in range(5):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t = time.perf_counter()
        for _ in range(200):
            eager_weighted_ce(pred, lab, sw)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        et.append((time.perf_counter()-t)/200)
    for _ in range(5):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t = time.perf_counter()
        for _ in range(200):
            scripted_weighted_ce(pred, lab, sw)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        st.append((time.perf_counter()-t)/200)
    e, s = min(et)*1e6, min(st)*1e6
    print(f'  n={n:>7}: eager {e:7.1f}us, scripted {s:7.1f}us  ({e/s:.2f}x)')

print()
print("== Binary BCE weighted ==")
for n in sizes:
    pred = torch.randn(n, device=device, requires_grad=True)
    lab = torch.randint(0, 2, (n,), device=device, dtype=torch.float32)
    sw = torch.rand(n, device=device)
    for _ in range(30):
        eager_weighted_bce(pred, lab, sw)
        scripted_weighted_bce(pred, lab, sw)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    et, st = [], []
    for _ in range(5):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t = time.perf_counter()
        for _ in range(200):
            eager_weighted_bce(pred, lab, sw)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        et.append((time.perf_counter()-t)/200)
    for _ in range(5):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t = time.perf_counter()
        for _ in range(200):
            scripted_weighted_bce(pred, lab, sw)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        st.append((time.perf_counter()-t)/200)
    e, s = min(et)*1e6, min(st)*1e6
    print(f'  n={n:>7}: eager {e:7.1f}us, scripted {s:7.1f}us  ({e/s:.2f}x)')

print()
print("== Regression MSE weighted ==")
for n in sizes:
    pred = torch.randn(n, device=device, requires_grad=True)
    lab = torch.randn(n, device=device)
    sw = torch.rand(n, device=device)
    for _ in range(30):
        eager_weighted_mse(pred, lab, sw)
        scripted_weighted_mse(pred, lab, sw)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    et, st = [], []
    for _ in range(5):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t = time.perf_counter()
        for _ in range(200):
            eager_weighted_mse(pred, lab, sw)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        et.append((time.perf_counter()-t)/200)
    for _ in range(5):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t = time.perf_counter()
        for _ in range(200):
            scripted_weighted_mse(pred, lab, sw)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        st.append((time.perf_counter()-t)/200)
    e, s = min(et)*1e6, min(st)*1e6
    print(f'  n={n:>7}: eager {e:7.1f}us, scripted {s:7.1f}us  ({e/s:.2f}x)')

print()
print("== ListNet (per-query, n=10-200 typical) ==")
for n in (10, 50, 200, 1000):
    scores = torch.randn(n, device=device, requires_grad=True)
    rel = torch.randint(0, 4, (n,), device=device, dtype=torch.float32)
    for _ in range(30):
        eager_listnet(scores, rel)
        scripted_listnet(scores, rel)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    et, st = [], []
    for _ in range(5):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t = time.perf_counter()
        for _ in range(500):
            eager_listnet(scores, rel)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        et.append((time.perf_counter()-t)/500)
    for _ in range(5):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t = time.perf_counter()
        for _ in range(500):
            scripted_listnet(scores, rel)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        st.append((time.perf_counter()-t)/500)
    e, s = min(et)*1e6, min(st)*1e6
    print(f'  n={n:>5}: eager {e:7.1f}us, scripted {s:7.1f}us  ({e/s:.2f}x)')
