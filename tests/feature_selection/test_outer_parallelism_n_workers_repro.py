"""Wave 9.1 loop-iter-18 regression: ``mi_direct(parallelism='outer')``
must be bit-exact across n_workers for the same ``base_seed``.

Pre-fix: ``parallel_mi`` used a single LCG state advanced sequentially
across iterations, seeded from ``base_seed * 2654435761 + worker_idx + 1``.
The random stream CONTENT (which shuffles each worker saw) was therefore
a function of n_workers because each worker got a different
base_seed-derived seed. Result: confidence varied across n_workers in
{1,2,4,8} for the same data and ``base_seed`` (4 distinct values),
breaking the "same seed -> identical output" reproducibility contract.

Concrete repro pre-fix:
  n_workers=1: conf=0.029126
  n_workers=2: conf=0.017500
  n_workers=4: conf=0.025000
  n_workers=8: conf=0.030000

The downstream consumer is the MRMR confidence gate that drops
candidates with ``confidence < min_nonzero_confidence``; 70% relative
spread across n_workers absolutely flipped ``support_`` membership
for the same frozen seed.

Fix: switch ``parallel_mi`` to per-iteration LCG seeding
(``state = base_seed * 2654435761 + (perm_offset + i + 1)``) matching
``parallel_mi_prange``'s scheme. Each worker receives its cumulative
``perm_offset`` from the dispatcher so worker w running perms
[offset, offset+count) consumes the SAME seeds that a single-worker
run would use at those same perm indices. The sequential else-branch
in ``mi_direct`` (which runs when ``n_workers == 1`` AND ``parallelism``
isn't explicitly set to 'outer'/'bc'/'inner') also switched to the
same per-iteration seeding for cross-path consistency.
"""
from __future__ import annotations

import numpy as np
import pytest


def _build_factors(seed: int = 12345, n: int = 600):
    rng = np.random.default_rng(int(seed))
    x = rng.integers(0, 4, n).astype(np.int32)
    y = np.where(rng.random(n) < 0.03, x % 2,
                  rng.integers(0, 2, n)).astype(np.int32)
    factors = np.column_stack([x, y]).astype(np.int32)
    nbins = np.array([4, 2], dtype=np.int32)
    return factors, nbins


@pytest.mark.parametrize("n_workers_pair", [(2, 4), (2, 8), (4, 8)])
def test_outer_bit_exact_across_n_workers_ge2(n_workers_pair):
    """Outer-pool branch (n_workers >= 2) must be bit-exact across
    n_workers for the same base_seed.
    """
    from mlframe.feature_selection.filters.permutation import mi_direct
    factors, nbins = _build_factors()
    nw_a, nw_b = n_workers_pair
    mi_a, conf_a = mi_direct(
        factors, x=(0,), y=(1,), factors_nbins=nbins,
        npermutations=400, min_nonzero_confidence=0.50,
        n_workers=nw_a, parallelism="outer", prefer_gpu=False,
        base_seed=12345,
    )
    mi_b, conf_b = mi_direct(
        factors, x=(0,), y=(1,), factors_nbins=nbins,
        npermutations=400, min_nonzero_confidence=0.50,
        n_workers=nw_b, parallelism="outer", prefer_gpu=False,
        base_seed=12345,
    )
    assert conf_a == conf_b, (
        f"outer-mode n_workers={nw_a} -> conf={conf_a}, "
        f"n_workers={nw_b} -> conf={conf_b}; same seed must give same output."
    )
    assert mi_a == mi_b


def test_inner_still_bit_exact_across_n_workers():
    """Negative control: inner mode (parallel_mi_prange) was already
    bit-exact pre-fix; the iter-18 changes must not regress it.
    """
    from mlframe.feature_selection.filters.permutation import mi_direct
    factors, nbins = _build_factors()
    confs = []
    for nw in (1, 2, 4, 8):
        _, conf = mi_direct(
            factors, x=(0,), y=(1,), factors_nbins=nbins,
            npermutations=400, min_nonzero_confidence=0.50,
            n_workers=nw, parallelism="inner", prefer_gpu=False,
            base_seed=12345,
        )
        confs.append(conf)
    assert len(set(confs)) == 1, (
        f"inner mode regressed - n_workers in {{1,2,4,8}} gave {confs}"
    )


def test_outer_reduces_to_full_inner_across_workers():
    """outer with n_workers=2 must match outer with n_workers=4 must
    match outer with n_workers=8 (the primary iter-18 contract).
    """
    from mlframe.feature_selection.filters.permutation import mi_direct
    factors, nbins = _build_factors(seed=42, n=800)
    out_values = []
    for nw in (2, 4, 8):
        _, conf = mi_direct(
            factors, x=(0,), y=(1,), factors_nbins=nbins,
            npermutations=800, min_nonzero_confidence=0.50,
            n_workers=nw, parallelism="outer", prefer_gpu=False,
            base_seed=42,
        )
        out_values.append(conf)
    assert len(set(out_values)) == 1, (
        f"outer-mode reproducibility broken: {out_values}"
    )
