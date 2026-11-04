"""In internal-batch training mode the rows that share a batch change every epoch, and every epoch still covers each row exactly once."""
import torch

from mlframe.training.neural.data import EpochBatchSampler, TorchDataset


def _epoch(dataset, sampler):
    """The batches (as sorted tuples of row ids) of one epoch."""
    return [tuple(sorted(dataset[i][0].ravel().tolist())) for i in sampler]


def test_batch_membership_changes_between_epochs_and_every_row_is_seen_once():
    """10 rows, batch size 4: batches were fixed slices [0..3], [4..7], [8, 9] in every epoch."""
    ds = TorchDataset(torch.arange(10, dtype=torch.float32).reshape(10, 1), labels=torch.zeros(10), batch_size=4, row_shuffle_seed=0)
    sampler = EpochBatchSampler(ds.num_batches, 0)
    e0, e1 = _epoch(ds, sampler), _epoch(ds, sampler)
    for epoch in (e0, e1):
        assert sorted(r for batch in epoch for r in batch) == [float(i) for i in range(10)]
        assert sorted(len(b) for b in epoch) == [2, 4, 4]
    assert {frozenset(b) for b in e0} != {frozenset(b) for b in e1}
