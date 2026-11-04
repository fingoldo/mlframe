"""FixedSparseLinear breaks equal-importance ties with the seeded generator, not with argsort order."""
import torch

from mlframe.training.neural.fixed_sparse_linear import _build_importance_mask


def _kept(seed):
    """Indices kept for an all-equal importance vector under one seed."""
    g = torch.Generator().manual_seed(seed)
    mask = _build_importance_mask(torch.ones(10), out_features=2, in_features=10, sparsity=0.5, generator=g)
    return tuple(torch.nonzero(mask[0]).ravel().tolist())


def test_equal_importances_keep_different_inputs_for_different_seeds_and_the_same_for_one_seed():
    """A 1e-9 jitter vanished in float32 at importance 1.0, so the kept set was the same for every seed."""
    assert _kept(0) == _kept(0)
    assert len({_kept(s) for s in range(8)}) > 1
