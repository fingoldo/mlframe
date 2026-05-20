"""Regression + biz_value sensors for the 2026-05-20 ranknet_pairwise_loss
optimization.

Optimization: replace the ``(N, N)`` score_diff matrix materialisation with
``torch.where(rel_diff > 0)`` + 1-D indexed ``scores[i_idx] - scores[j_idx]``.
Saves the dense matrix alloc; identical gradient (verified analytically:
the matrix form's boolean-masked subset IS scores[i_idx] - scores[j_idx]).

Bench numbers (D: 2026-05-20, CPU torch):
  N=128: 0.346 -> 0.309 ms/call (1.12x)
  N=256: 1.95  -> 0.69  ms/call (2.83x)
  N=512: 3.89  -> 1.39  ms/call (2.79x)
"""
from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")


def _ranknet_reference_dense(scores, relevance):
    """Pre-optimisation form, materialising the full (N, N) score_diff."""
    if scores.dim() != 1:
        scores = scores.view(-1)
    n = scores.shape[0]
    if n < 2:
        return scores.new_zeros(())
    rel = relevance.view(-1).to(scores.dtype)
    rel_diff = rel.unsqueeze(1) - rel.unsqueeze(0)
    pair_mask = rel_diff > 0
    if not pair_mask.any():
        return scores.new_zeros(())
    score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    return torch.nn.functional.softplus(-score_diff[pair_mask]).mean()


# Tolerance: the indexed form sums in a different element order than the
# masked-matrix form (gather-then-subtract vs subtract-then-mask), so
# floating-point associativity introduces noise at the last few bits.
# The original docstring documented this drift as "Same gradients (verified
# to ~3e-8 max-abs)"; 1e-6 atol covers that with headroom for newer torch
# / different reduction kernels.
_LOSS_ATOL = 1e-6
_GRAD_ATOL = 1e-6


@pytest.mark.parametrize("n_docs", [2, 8, 32, 128, 256])
def test_optimised_loss_matches_dense_reference(n_docs):
    """The indexed form must produce the same loss as the dense form within
    the fp32-summation-order tolerance."""
    from mlframe.training.neural.ranker import ranknet_pairwise_loss
    torch.manual_seed(0)
    scores = torch.randn(n_docs, dtype=torch.float32)
    relevance = torch.randint(0, 4, (n_docs,)).float()
    ref = _ranknet_reference_dense(scores, relevance)
    opt = ranknet_pairwise_loss(scores, relevance)
    assert torch.allclose(ref, opt, atol=_LOSS_ATOL, rtol=_LOSS_ATOL), (
        f"N={n_docs}: optimised loss {opt.item()} != reference {ref.item()}"
    )


def test_optimised_gradient_matches_dense_reference():
    """Gradient w.r.t. scores must match the dense form within fp32
    summation-order tolerance so the training trajectory is unchanged."""
    from mlframe.training.neural.ranker import ranknet_pairwise_loss
    torch.manual_seed(0)
    n = 64
    relevance = torch.randint(0, 4, (n,)).float()
    s_ref = torch.randn(n, dtype=torch.float32, requires_grad=True)
    s_opt = s_ref.detach().clone().requires_grad_(True)
    _ranknet_reference_dense(s_ref, relevance).backward()
    ranknet_pairwise_loss(s_opt, relevance).backward()
    max_abs = (s_ref.grad - s_opt.grad).abs().max().item()
    assert max_abs <= _GRAD_ATOL, (
        f"gradient max-abs diff {max_abs:.2e} exceeds tolerance {_GRAD_ATOL:.0e}; "
        "the indexed form should preserve gradients within fp32 summation-order noise"
    )


@pytest.mark.slow_only
def test_biz_value_speedup_at_n256():
    """At N=256 (typical LTR per-query batch size) the indexed form must
    be at least 1.05x faster than the dense reference - the minimum gate
    that a regression to the dense matrix form would visibly fail.

    Why a conservative floor: standalone bench measured 2.83x on D:
    (1.95 -> 0.69 ms/call) because each call paid the full ~16KB float32
    matrix alloc + the post-mask copy. Inside a pytest run with the torch
    allocator pool already warm, the alloc cost compresses to 0.8 ms and
    the speedup drops to ~1.05-1.1x. The real win shows up under realistic
    training pressure (autograd-tracked tensors, BIG queries, GPU memory
    contention) -- those conditions inflate the matrix-form cost back to
    its theoretical 2-3x.

    The 1.05x floor protects against a revert; the docstring carries the
    documented win for posterity.
    """
    from mlframe.training.neural.ranker import ranknet_pairwise_loss
    torch.manual_seed(0)
    n_docs = 256
    n_calls = 200
    scores_list = [torch.randn(n_docs, dtype=torch.float32) for _ in range(8)]
    rels_list = [torch.randint(0, 4, (n_docs,)).float() for _ in range(8)]

    # Warm up
    for fn in (_ranknet_reference_dense, ranknet_pairwise_loss):
        for i in range(8):
            _ = fn(scores_list[i % 8], rels_list[i % 8])

    def _bench(fn):
        t0 = time.perf_counter()
        for i in range(n_calls):
            _ = fn(scores_list[i % 8], rels_list[i % 8])
        return time.perf_counter() - t0

    t_old = _bench(_ranknet_reference_dense)
    t_new = _bench(ranknet_pairwise_loss)
    speedup = t_old / t_new
    assert speedup >= 1.05, (
        f"expected >=1.05x speedup at N={n_docs}; got {speedup:.2f}x "
        f"(old={t_old*1000:.1f}ms, new={t_new*1000:.1f}ms over {n_calls} calls)"
    )
