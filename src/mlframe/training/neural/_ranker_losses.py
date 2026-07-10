"""Ranker loss functions + pair-index cache carved out of ranker.py.

RankNet (Burges 2005) pairwise BCE-with-logits via softplus, ListNet (Cao 2007)
listwise softmax cross-entropy, and the per-query pair-index cache that gives
~20x speedup on repeating relevance patterns across epochs.

Moved here so the parent module stays below the 1k-LOC monolith threshold;
ranker.py re-exports the public symbols.
"""
from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F

_RANKNET_MAX_PAIRS_PER_QUERY: int = 2_000_000  # ~16MB float32 per (i,j) tensor


# Per-query pair-index cache. The (i_idx, j_idx) tensors returned by
# torch.where(rel_i > rel_j) depend ONLY on the relevance pattern; the same
# query repeats every epoch, so caching keyed on the relevance tuple gives
# ~20x speedup on cache hits. Disabled when N > MAX_N (tuple hashing gets
# expensive) and when the subsampling path runs (torch.randperm produces a
# fresh pair distribution per call). MAX_N=256 covers small/medium queries;
# SIZE=65536 fits a 200k-row LTR fit (~58 MB).
_RANKNET_PAIR_CACHE_MAX_N: int = 256
_RANKNET_PAIR_CACHE_SIZE: int = 65536
_ranknet_pair_cache: dict = {}

# Binary-relevance short-circuit threshold. For rel in {0, 1}, the pair set
# {(i, j) : rel[i] > rel[j]} is exactly Cartesian product (positives x
# negatives), built in O(N + n_pairs) vs torch.where's O(N^2) (N, N) bool
# mask materialisation. Below ~N=200 the torch.where path is faster on CPU
# (single fused kernel beats nonzero+repeat_interleave+repeat); above it the
# binary path scales much better. Bench 2026-05-28 (modern torch, CPU):
#   N=200 ~1.4x, N=500 4.5x, N=1000 9.8x, N=2000 14.2x at 10%-positive rel.
# The MIN_N gate also keeps the install_pair_index_cache fast path
# (typical fuzz N=10-15) unaffected.
_BINARY_PAIR_SHORTCUT_MIN_N: int = 200


def _binary_pair_indices(rel: torch.Tensor):
    """Return (i_idx, j_idx) for rel in {0, 1} without the (N, N) mask.

    Returns None when rel is not binary so callers can fall back to
    ``torch.where(rel.unsqueeze(1) > rel.unsqueeze(0))``. The Cartesian-product
    ordering (positives outer, negatives inner) matches torch.where's
    row-major output, so callers see identical (i_idx, j_idx) tensors.
    Empty (n_pos == 0 or n_neg == 0) returns two zero-length long tensors on
    the same device, matching torch.where's empty-result shape.
    """
    is_one = rel == 1
    is_zero = rel == 0
    if not (is_zero | is_one).all():
        return None
    pos = torch.nonzero(is_one, as_tuple=False).squeeze(-1)
    neg = torch.nonzero(is_zero, as_tuple=False).squeeze(-1)
    n_pos = pos.numel()
    n_neg = neg.numel()
    if n_pos == 0 or n_neg == 0:
        empty = torch.empty(0, dtype=torch.long, device=rel.device)
        return empty, empty
    i_idx = pos.repeat_interleave(n_neg)
    j_idx = neg.repeat(n_pos)
    return i_idx, j_idx


def _ranknet_pair_cache_clear() -> None:
    """Test-only: clear the per-query pair-index cache between unit tests so
    state from a prior test can't bleed into a sibling assertion."""
    _ranknet_pair_cache.clear()


def ranknet_pairwise_loss(scores: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
    """RankNet pairwise loss for one query.

    Parameters
    ----------
    scores : (N,) tensor of model scores for the query's N docs.
    relevance : (N,) tensor of integer (or float) ground-truth relevance.

    Returns
    -------
    Scalar loss tensor. Returns 0.0 when no informative pair exists (all docs same relevance).

    Notes
    -----
    BCE-with-logits at target=1 reduces algebraically to softplus(-x) = log(1 + exp(-x)),
    so softplus is applied to the masked 1-D score-diff vector directly. Same gradients
    (verified to ~3e-8 max-abs) and numerical stability as the (N, N) BCE form.

    Queries larger than ``sqrt(_RANKNET_MAX_PAIRS_PER_QUERY)`` (~1414 docs) are
    randomly subsampled to that doc count for this loss call; the (N,N) pair
    tensor would otherwise allocate quadratically (10k docs ~> 400MB).
    """
    if scores.dim() != 1:
        scores = scores.view(-1)
    n = scores.shape[0]
    if n < 2:
        return scores.new_zeros(())

    # Cap quadratic blowup: 10k docs -> 100M pairs ~> 400MB float32 alloc.
    _max_n = int(_RANKNET_MAX_PAIRS_PER_QUERY**0.5)
    if n > _max_n:
        # torch.randperm picks a unique-index subsample on-device; uniform
        # over docs preserves the pair-distribution in expectation.
        idx = torch.randperm(n, device=scores.device)[:_max_n]
        scores = scores[idx]
        relevance = relevance[idx]
        n = _max_n

    rel = relevance.view(-1).to(scores.dtype)
    # Pairwise informative-pair indices via torch.where(rel_i > rel_j). The
    # (N, N) rel_diff mask is materialised once; the score_diff matrix is
    # NOT materialised. Old code did
    #     score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)  # (N, N)
    #     ... softplus(-score_diff[pair_mask]).mean()
    # which allocated a full (N, N) float32 tensor per call (~64KB at N=128,
    # called 90k times per train epoch on a typical LTR fuzz combo - 5.8GB
    # of throwaway tensor allocs amortised across the training loop, profile
    # 2026-05-20 attributed 64s self-time to ranknet_pairwise_loss). The
    # indexed form below allocates only a 1-D (n_pairs,) score_diff tensor
    # (~n_pairs * 4 bytes), strictly subset of the matrix that was being
    # indexed out anyway. Same gradient (verified analytically: the matrix
    # form's masked subset IS exactly scores[i_idx] - scores[j_idx]).
    #
    # Per-query cache: the (i_idx, j_idx) tensors depend only on the relevance
    # pattern. Queries repeat every epoch, so the same pattern hits the cache
    # ~n_epochs times. Bench shows ~20x per-call speedup at N=11. Skip caching
    # for N>_RANKNET_PAIR_CACHE_MAX_N (tuple hashing gets expensive) and never
    # cache on the subsampling path (torch.randperm produces a fresh pair
    # distribution per call).
    if n <= _RANKNET_PAIR_CACHE_MAX_N:
        # tuple(rel.tolist()) is a hashable digest of the relevance pattern;
        # cheap for small N (~5us at N=11). dtype + device.type included so a
        # CPU-built entry isn't accidentally indexed on GPU.
        cache_key = (n, rel.dtype, rel.device.type, tuple(rel.tolist()))
        cached = _ranknet_pair_cache.get(cache_key)
        if cached is None:
            i_idx, j_idx = torch.where(rel.unsqueeze(1) > rel.unsqueeze(0))
            if len(_ranknet_pair_cache) >= _RANKNET_PAIR_CACHE_SIZE:
                # FIFO eviction (Python 3.7+ dict preserves insertion order).
                _ranknet_pair_cache.pop(next(iter(_ranknet_pair_cache)))
            _ranknet_pair_cache[cache_key] = (i_idx, j_idx)
        else:
            i_idx, j_idx = cached
    else:
        # Large-N fall-through (N > _RANKNET_PAIR_CACHE_MAX_N). Here torch.where
        # materialises a (N, N) bool mask -> grows quadratically (N=1000 ->
        # 1M bytes, N=2000 -> 4M bytes per call). Binary-relevance Cartesian
        # shortcut avoids the mask entirely when applicable. Falls back to
        # torch.where for ordinal / multi-grade relevance.
        binary_pairs = _binary_pair_indices(rel)
        if binary_pairs is not None:
            i_idx, j_idx = binary_pairs
        else:
            i_idx, j_idx = torch.where(rel.unsqueeze(1) > rel.unsqueeze(0))
    if i_idx.numel() == 0:
        return scores.new_zeros(())
    score_diff_pairs = scores[i_idx] - scores[j_idx]
    # softplus(-x) = -log(sigmoid(x)) = BCE-w-logits(x, t=1) on informative (rel_i > rel_j) diffs.
    return cast(torch.Tensor, F.softplus(-score_diff_pairs).mean())


@torch.jit.script
def _ranknet_loss_precomputed_core(
    scores: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
) -> torch.Tensor:
    """torch.jit.script-compiled inner kernel for ranknet_pairwise_loss_precomputed.

    Fuses the four eager ops (scores[i] - scores[j] -> neg -> softplus -> mean)
    into a single TorchScript-traced graph that bypasses Python's per-op
    dispatch overhead. Bench at the c0105 query shape (cuda, n=10 / ~25 pairs):
    eager 197 us -> scripted 146 us (~26%). torch.compile(inductor) would fuse
    further via Triton but the Windows triton-windows DLL doesn't load on this
    box (ImportError: libtriton); compile(aot_eager) without inductor is
    actually SLOWER (~564 us) because it adds dispatch overhead with no
    fusion. TorchScript is the portable middle ground.

    None-handling and the i_idx.numel() == 0 short-circuit stay in the
    outer Python wrapper -- TorchScript Optional[Tensor] requires explicit
    annotations and the wrapping check is essentially free per-call.

    No ``typing.cast`` here (unlike the eager sibling functions) -- TorchScript's
    compiler rejects ``cast`` as "builtin cannot be used as a value" on some torch
    builds (confirmed on a 2.12.0.dev nightly). ``cast`` is a runtime no-op purely
    for mypy narrowing, so dropping it inside a scripted function changes nothing
    at runtime; ``F.softplus(...).mean()`` already returns ``torch.Tensor``.
    """
    score_diff_pairs = scores[i_idx] - scores[j_idx]
    return F.softplus(-score_diff_pairs).mean()


def ranknet_pairwise_loss_precomputed(
    scores: torch.Tensor,
    i_idx: torch.Tensor | None,
    j_idx: torch.Tensor | None,
) -> torch.Tensor:
    """RankNet loss with externally-precomputed pair indices.

    Skips the ``tuple(rel.tolist())`` cache-key build (which forces a GPU->CPU
    sync each batch) by accepting (i_idx, j_idx) already built CPU-side and
    moved to the scores' device by Lightning's batch transfer. ``None`` values
    signal a query with no informative pair -> loss is 0.

    Used by MLPRankerLightningModule when the batch sampler installed a
    per-query pair-index cache on the Dataset. Bit-exact equivalent of
    ``ranknet_pairwise_loss``; the only difference is who computes pair indices.

    bench-attempt-rejected 2026-05-23 (iter183): tried calling
    ``_ranknet_loss_precomputed_core`` directly from ``_compute_loss``,
    bypassing this wrapper's dim / None / numel checks
    (``GroupBatchSampler`` already guarantees informative pairs, so the
    checks are dead code on the precomputed path). Bench
    bench_ranker_loss_wrapper_overhead.py at cuda n_docs in (10,25,100):
    wrapper=152-167us core=146-162us; delta only 5-9us / call. On the
    c0027 540k-call shape that's just 3-5s saved against a 2782s wall
    (0.1-0.2%). Below the threshold to justify removing the defensive
    guard; the wrapper stays.
    """
    if scores.dim() != 1:
        scores = scores.view(-1)
    if i_idx is None or j_idx is None or i_idx.numel() == 0:
        return scores.new_zeros(())
    return cast(torch.Tensor, _ranknet_loss_precomputed_core(scores, i_idx, j_idx))


def listnet_top1_loss(scores: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
    """ListNet top-1 listwise loss for one query.

    Cross-entropy between the relevance-induced softmax distribution (true) and the
    predicted-score softmax distribution (Cao 2007). O(n) per query.
    """
    if scores.dim() != 1:
        scores = scores.view(-1)
    rel = relevance.view(-1).to(scores.dtype)
    n = scores.shape[0]
    if n < 2:
        return scores.new_zeros(())

    # All-equal relevance -> uniform target, KL not informative.
    if (rel == rel[0]).all():
        return scores.new_zeros(())

    true_p = F.softmax(rel, dim=0)
    pred_log_p = F.log_softmax(scores, dim=0)
    return -(true_p * pred_log_p).sum()
