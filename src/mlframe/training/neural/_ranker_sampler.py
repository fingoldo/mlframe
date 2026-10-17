"""Group-aware batch sampler for the MLP ranker: one query per batch (carved out of ``ranker.py``)."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class GroupBatchSampler(Sampler):
    """Yield ONE query's row indices per batch.

    Skips queries with fewer than 2 docs (no pair possible) or a single distinct relevance
    value (no positive pair; such queries contribute 0 to the loss and cause numerical
    issues in some Lightning callbacks).

    Deterministic when ``shuffle=False``; with ``shuffle=True`` yields queries in per-epoch
    random order (within-query doc order is preserved).
    """

    def __init__(
        self,
        group_ids: np.ndarray,
        relevance: np.ndarray,
        shuffle: bool = True,
        seed: int = 0,
        queries_per_batch: int = 1,
    ):
        # Forward to torch Sampler base. Newer torch (2.x) ``Sampler.__init__``
        # is the bare object inheritance (no ``data_source`` kwarg) -- passing
        # ``data_source=None`` falls through to ``object.__init__(data_source=None)``
        # which raises ``TypeError: takes exactly one argument``. Older torch
        # accepted the kwarg as a no-op; the bare call is correct under both.
        # Older torch's Sampler stub declares data_source as required -- the ignore is only
        # "needed" against that stub; see the warn_unused_ignores override for this module.
        super().__init__()  # type: ignore[call-arg]
        self.group_ids = np.asarray(group_ids)
        self.relevance = np.asarray(relevance)
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        # OPT-7: when > 1, pack ``queries_per_batch`` queries
        # into ONE batch -- the concatenated row indices flow through the
        # DataLoader as one batch, the Dataset attaches the offset-corrected
        # concatenated (i_idx, j_idx) pair tensors built from per-query
        # cache, and Lightning runs ONE forward + ONE backward + ONE
        # optimizer.step per packed batch. iter216 measured 162000 / epoch
        # = 800s of optimizer wall on c0098 LTR; packing 32 queries drops
        # that to ~5000 / epoch.
        self.queries_per_batch = max(1, int(queries_per_batch))
        # Per-batch partition info: tuple(concat_indices) ->
        # list of (per_query_cache_key, query_start_offset_in_batch). The
        # Dataset reads this to compose per-query pair indices into a
        # single offset-corrected concat pair tensor.
        # Cleared at epoch start (set_epoch) -- prefetch overlap relies on
        # entries surviving from yield to __getitems__, but not beyond.
        self._batch_partition: dict[tuple, list[tuple]] = {}

        sort_idx = np.argsort(self.group_ids, kind="stable")
        sorted_groups = self.group_ids[sort_idx]
        boundaries = np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1
        starts = np.concatenate(([0], boundaries, [len(sorted_groups)]))
        self._query_slices: list[np.ndarray] = []
        for i in range(len(starts) - 1):
            s, e = starts[i], starts[i + 1]
            indices = sort_idx[s:e]
            if len(indices) < 2:
                continue
            rel_slice = self.relevance[indices]
            if rel_slice.ndim > 1:
                # Multilabel relevance: coerce to 1-D by summing labels (any non-zero = relevant).
                rel_slice = rel_slice.sum(axis=1)
            if len(np.unique(rel_slice)) < 2:
                continue
            self._query_slices.append(indices)

    def set_epoch(self, epoch: int) -> None:
        """Advance the sampler's internal epoch counter so ``shuffle=True`` reshuffles queries per epoch.

        Lightning only auto-calls ``set_epoch`` on torch's ``DistributedSampler``; this sampler is advanced
        explicitly by ``_SamplerSetEpochCallback.on_train_epoch_start``. Without it, every epoch would replay
        the SAME random query order (seeded once at construction), which stalls training in a local optimum.
        Also clears the OPT-7 multi-query ``_batch_partition`` cache since a new shuffle order invalidates it.
        """
        self._epoch = int(epoch)
        # OPT-7: drop last epoch's partition entries -- yields fresh order so
        # batch composition changes.
        if self.queries_per_batch > 1:
            self._batch_partition.clear()

    def __iter__(self):
        order = np.arange(len(self._query_slices))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self._epoch)
            rng.shuffle(order)
        if self.queries_per_batch == 1:
            # Legacy per-query path -- bit-exact prior behaviour.
            for i in order:
                yield self._query_slices[i].tolist()
            return
        # OPT-7 multi-query path: pack queries_per_batch queries / yield.
        qpb = self.queries_per_batch
        for batch_start in range(0, len(order), qpb):
            batch_q_ids = order[batch_start : batch_start + qpb]
            slices = [self._query_slices[q] for q in batch_q_ids]
            concat = np.concatenate(slices)
            # Per-query starts in concat positions (offsets for pair indices).
            offsets = [0]
            for s in slices:
                offsets.append(offsets[-1] + len(s))
            # Build per-query (cache_key, batch_offset) list. The Dataset's
            # ``install_pair_index_cache`` keys per-query (i_idx, j_idx) by
            # ``tuple(query_indices.tolist())``; mirror that here so the
            # Dataset can look them up.
            partition = [(tuple(s.tolist()), offsets[i]) for i, s in enumerate(slices)]
            indices_list = concat.tolist()
            key = tuple(indices_list)
            self._batch_partition[key] = partition
            yield indices_list

    def __len__(self) -> int:
        # Legacy: one batch = one query. OPT-7 batched: ceil(n_queries / qpb).
        if self.queries_per_batch <= 1:
            return len(self._query_slices)
        n = len(self._query_slices)
        return (n + self.queries_per_batch - 1) // self.queries_per_batch
