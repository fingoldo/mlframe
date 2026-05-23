"""Bench wrapper-vs-direct-kernel for ranknet_pairwise_loss_precomputed.

c0027 iter183 profile: 540000 calls in 117.187s tottime ~= 217us / call.
The torchscript core kernel is ~146us / call (see ranker.py:155 bench note).
That leaves ~71us / call of pure-Python wrapper overhead from
``if scores.dim() != 1`` + ``if i_idx is None or j_idx is None or numel == 0``
+ function-call frame.

``GroupBatchSampler.__init__`` (ranker.py:230) already filters queries to
``len(indices) >= 2`` AND ``len(unique(rel_slice)) >= 2``. With >=2 distinct
relevance values, at least one (i,j) pair has rel_i > rel_j, so
i_idx.numel() > 0 for every query reaching the precomputed path.
``forward(x)`` (ranker.py:430) already does ``.view(-1)`` so scores is 1-D.

Both wrapper checks are therefore dead code on the hot path. We bench
direct-kernel-call against the safe wrapper.

Run: ``python profiling/bench_ranker_loss_wrapper_overhead.py``
"""
import time
import torch
import torch.nn.functional as F


@torch.jit.script
def _core(scores: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor) -> torch.Tensor:
    score_diff_pairs = scores[i_idx] - scores[j_idx]
    return F.softplus(-score_diff_pairs).mean()


def wrapper(scores, i_idx, j_idx):
    if scores.dim() != 1:
        scores = scores.view(-1)
    if i_idx is None or j_idx is None or i_idx.numel() == 0:
        return scores.new_zeros(())
    return _core(scores, i_idx, j_idx)


def bench(fn, *args, n_iter=20_000):
    for _ in range(200):
        fn(*args)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(n_iter):
            fn(*args)
        times.append((time.perf_counter() - t) / n_iter)
    return min(times) * 1e6


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    for n_docs in (10, 25, 100):
        scores = torch.randn(n_docs, device=device, requires_grad=True)
        rel = torch.randint(0, 4, (n_docs,), device=device)
        i_idx, j_idx = torch.where(rel.unsqueeze(1) > rel.unsqueeze(0))
        i_idx = i_idx.to(torch.long)
        j_idx = j_idx.to(torch.long)
        t_wrap = bench(wrapper, scores, i_idx, j_idx)
        t_core = bench(_core, scores, i_idx, j_idx)
        delta = t_wrap - t_core
        n_calls_540k = 540_000
        savings_s = delta * n_calls_540k / 1e6
        print(f"n_docs={n_docs:3d} npairs={i_idx.numel():4d}: wrapper={t_wrap:6.1f}us core={t_core:6.1f}us "
              f"delta={delta:5.1f}us  -> {savings_s:6.2f}s saved on 540k calls")
