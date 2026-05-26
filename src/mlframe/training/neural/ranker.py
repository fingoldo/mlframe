"""MLP-based learning-to-rank (LTR) model with RankNet / ListNet pairwise/listwise loss.

sklearn-shaped (``fit(X, y, group_ids=...)`` + ``predict(X)``) so it slots into
``mlframe.training.ranking::fit_ranker`` alongside CatBoostRanker, XGBRanker, LGBMRanker.

Losses:
- RankNet (Burges 2005): pairwise BCEWithLogitsLoss on score differences. For every pair
  (i, j) within the same query where rel_i > rel_j, minimise -log sigmoid(score_i - score_j)
  (equivalently BCE(s_i - s_j, target=1.0)). Numerically stable - log(sigmoid) would overflow
  for large |s_diff|.
- ListNet (Cao 2007): listwise softmax cross-entropy between predicted top-1 distribution
  and the relevance-induced one. O(n) per query.

Group-aware batching via ``GroupBatchSampler`` yields ONE query per batch (skipping queries
with <2 docs or single-class relevance - no positive pairs possible).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader, Dataset, Sampler

from mlframe.training.neural.flat import generate_mlp

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Losses + pair-cache carved out to sibling _ranker_losses.py to drop
# this file below the 1k-LOC monolith threshold. Re-exported here so existing
# from mlframe.training.neural.ranker import ranknet_pairwise_loss callers
# keep working unchanged.
# ---------------------------------------------------------------------------
from ._ranker_losses import (  # noqa: F401, E402
    _RANKNET_MAX_PAIRS_PER_QUERY,
    _RANKNET_PAIR_CACHE_MAX_N,
    _RANKNET_PAIR_CACHE_SIZE,
    _ranknet_pair_cache,
    _ranknet_pair_cache_clear,
    _ranknet_loss_precomputed_core,
    ranknet_pairwise_loss,
    ranknet_pairwise_loss_precomputed,
    listnet_top1_loss,
)



# ----------------------------------------------------------------------------------
# Group-aware batching
# ----------------------------------------------------------------------------------


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
        # Wave 56 (2026-05-20): forward to torch Sampler base. Currently passing
        # data_source=None is a no-op (newer torch silently accepts it); explicit
        # super call reserves the hook for future torch state.
        super().__init__(data_source=None)
        self.group_ids = np.asarray(group_ids)
        self.relevance = np.asarray(relevance)
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        # OPT-7 (2026-05-23): when > 1, pack ``queries_per_batch`` queries
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


class _RankerDataset(Dataset):
    """Row-level dataset for (X, y); the GroupBatchSampler controls query grouping."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Wave 56 (2026-05-20): forward to torch Dataset base for forward-compat.
        super().__init__()
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        # Per-query precomputed (i_idx, j_idx) pair-index tensors, keyed by the
        # bytes view of the sorted row-index array yielded by GroupBatchSampler.
        # Populated by ``install_pair_index_cache`` BEFORE the DataLoader starts;
        # leaving this attribute as None falls back to the in-loss runtime cache
        # (tuple(rel.tolist()) key) which needs a per-call GPU->CPU sync.
        self._pair_idx_by_query: dict[bytes, tuple] | None = None
        # iter357 (2026-05-26): per-query precomputed batch tuple keyed by the
        # same indices_key as ``_pair_idx_by_query``. Each entry is the EXACT
        # return value of ``__getitems__`` for that query -- either a 4-tuple
        # (X_slice, y_slice, i_idx, j_idx) for queries with informative pairs
        # or a 2-tuple (X_slice, y_slice) for queries without. Indices yielded
        # by GroupBatchSampler in queries_per_batch=1 mode repeat verbatim
        # across epochs, so caching the (X, y) slices once shifts ~28k * (2 x
        # tensor-index + as_tensor + dict.get) into a single dict lookup. On
        # c0079 1M-row LTR profile, __getitems__ tottime drops 19s -> ~2s.
        # OPT-7 multi-query mode skips this cache; its batch key is a packed
        # multi-query union that doesn't repeat exactly across epochs.
        self._batch_by_query: dict[tuple, tuple] | None = None
        # OPT-7 (2026-05-23): optional reference to the active batch sampler so
        # ``__getitems__`` can read multi-query batch partition info to compose
        # offset-corrected concatenated pair tensors. Set via
        # ``attach_sampler``; None means legacy single-query path only.
        self._sampler: Optional["GroupBatchSampler"] = None

    def attach_sampler(self, sampler: "GroupBatchSampler") -> None:
        """OPT-7: register the active batch sampler so multi-query batch
        partition info is reachable from ``__getitems__``. Idempotent."""
        self._sampler = sampler

    def install_pair_index_cache(self, query_slices: list[np.ndarray]) -> None:
        """Build (i_idx, j_idx) once per query on CPU; key by ``tuple(indices)``
        so __getitems__ can attach them at near-zero per-call cost. Avoids the
        ``tuple(rel.tolist())`` device sync that ranknet_pairwise_loss
        otherwise pays each batch (~80us per cuda call on n=10 queries; 540k
        calls/30-epoch fit -> 43s saved on c0083).

        Key choice: ``tuple(indices)`` beats ``np.asarray(indices).tobytes()``
        by ~4.5x at the microbench (0.26us vs 1.17us per __getitems__ call;
        ~0.9s saved on a 540k-call fit). Python ints fit in CPython's small-
        int cache for typical row-indices so tuple hashing is just N pointer
        compares -- no allocation per call.
        """
        cache: dict[tuple, tuple] = {}
        batch_cache: dict[tuple, tuple] = {}
        for indices in query_slices:
            idx_tensor = torch.as_tensor(indices, dtype=torch.long)
            rel = self.y[idx_tensor]
            if rel.dim() > 1:
                # Multilabel y -> sum across labels matches what GBS uses to
                # decide query inclusion. Not used by ranknet directly.
                continue
            i_idx, j_idx = torch.where(rel.unsqueeze(1) > rel.unsqueeze(0))
            # ``tuple(indices.tolist())`` is 6.3x faster than the prior
            # ``tuple(int(v) for v in indices)`` (0.45us vs 2.85us / call at
            # the 10-row query shape; 78ms saved on c0149 32400-call install).
            # Same Python-int tuple shape so ``__getitems__`` ``indices_key
            # = tuple(indices)`` hash-matches identically; .tolist() returns
            # Python ints from numpy int64 via the C buffer protocol.
            key = tuple(indices.tolist())
            # iter357: pre-build X slice once per query. Same fancy-indexing
            # copy that __getitems__ would do every batch, but amortised
            # across all epochs (~10x on 10-epoch fits).
            x_slice = self.X[idx_tensor]
            if i_idx.numel() == 0:
                # No informative pair; the loss returns 0 without needing
                # pair indices. Mark with sentinel so __getitems__ can skip
                # the runtime cache build for these too.
                cache[key] = (None, None)
                batch_cache[key] = (x_slice, rel)
                continue
            i_long = i_idx.to(torch.long)
            j_long = j_idx.to(torch.long)
            cache[key] = (i_long, j_long)
            batch_cache[key] = (x_slice, rel, i_long, j_long)
        self._pair_idx_by_query = cache
        self._batch_by_query = batch_cache

    def install_batch_cache_no_pairs(self, query_slices: list[np.ndarray]) -> None:
        """iter364: listnet-side counterpart to install_pair_index_cache. The
        listnet loss doesn't need (i_idx, j_idx) pair tensors -- it only reads
        batch[1] (relevance) -- so we cache the (X_slice, y_slice) 2-tuple
        without paying the torch.where(rel_i > rel_j) cost. Cache shape
        matches the no-pair sentinel branch in install_pair_index_cache so
        __getitems__ can hit the same fast path. On c0059 1M-row LTR listnet
        the legacy __getitems__ ran 20.07s tottime / 28140 batches (711us per
        call doing torch.as_tensor + X[idx] + y[idx]); the cache collapses
        each batch to a single dict.get + return."""
        batch_cache: dict[tuple, tuple] = {}
        for indices in query_slices:
            idx_tensor = torch.as_tensor(indices, dtype=torch.long)
            rel = self.y[idx_tensor]
            if rel.dim() > 1:
                continue
            key = tuple(indices.tolist())
            x_slice = self.X[idx_tensor]
            batch_cache[key] = (x_slice, rel)
        self._batch_by_query = batch_cache

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __getitems__(self, indices):
        # PyTorch >= 1.13 calls __getitems__(list_of_indices) when present
        # instead of running [dataset[i] for i in indices] one row at a
        # time. One batched tensor index call replaces ~11 single-row slices
        # per query batch; ~3-3.5x faster end-to-end vs per-row + default
        # collate (microbench 2026-05-20, batch=11x32 floats). The matching
        # `_ranker_passthrough_collate` below unwraps the 1-element list so
        # the DataLoader doesn't re-stack an already-stacked batch.
        # iter357: legacy queries_per_batch=1 hot path -- check the precomputed
        # batch cache FIRST so the torch.as_tensor(indices, ...) + X[idx] +
        # y[idx] work is skipped entirely on hits. On c0079 1M-row LTR the
        # cache hits 100% of batches (indices repeat exactly across epochs),
        # collapsing __getitems__ to a single dict.get + indices->tuple.
        # ``tuple(indices)`` is ~4.5x faster than
        # ``np.asarray(indices, dtype=np.int64).tobytes()`` at the typical
        # ~10-row query shape (0.26us vs 1.17us / call) and matches the key
        # built by install_pair_index_cache via ``tuple(indices.tolist())``.
        if (
            self._batch_by_query is not None
            and not torch.is_tensor(indices)
        ):
            indices_key = tuple(indices)
            cached_batch = self._batch_by_query.get(indices_key)
            if cached_batch is not None:
                # OPT-7 multi-query path falls through; that path uses
                # _sampler._batch_partition which is keyed by a packed
                # multi-query union, not present in _batch_by_query.
                if (
                    self._sampler is None
                    or self._sampler.queries_per_batch <= 1
                ):
                    return [cached_batch]
        else:
            indices_key = None
        if torch.is_tensor(indices):
            idx = indices
            indices_key = None  # tensor input rare; fall through to in-loss cache
        else:
            idx = torch.as_tensor(indices, dtype=torch.long)
            if indices_key is None:
                indices_key = tuple(indices)
        # OPT-7 (2026-05-23): multi-query batch -- sampler stamped a
        # partition table when it yielded this batch. Compose per-query
        # pair tensors into one offset-corrected concat by adding each
        # query's start-offset to its (i_idx, j_idx). After concat, the
        # standard loss kernel
        # ``softplus(-(scores[i_idx] - scores[j_idx])).mean()`` computes
        # the cross-query loss in ONE forward+backward; each (i, j) pair
        # is intra-query by construction so no cross-query contamination.
        if (
            self._sampler is not None
            and self._sampler.queries_per_batch > 1
            and indices_key is not None
            and self._pair_idx_by_query is not None
        ):
            partition = self._sampler._batch_partition.get(indices_key)
            if partition is not None:
                concat_i_parts: List[torch.Tensor] = []
                concat_j_parts: List[torch.Tensor] = []
                for qkey, qstart in partition:
                    pair = self._pair_idx_by_query.get(qkey)
                    if pair is None:
                        continue
                    i_idx, j_idx = pair
                    if i_idx is None or j_idx is None:
                        # No informative pair sentinel from install_pair_index_cache.
                        continue
                    if qstart == 0:
                        concat_i_parts.append(i_idx)
                        concat_j_parts.append(j_idx)
                    else:
                        concat_i_parts.append(i_idx + qstart)
                        concat_j_parts.append(j_idx + qstart)
                if concat_i_parts:
                    cat_i = (concat_i_parts[0] if len(concat_i_parts) == 1
                             else torch.cat(concat_i_parts))
                    cat_j = (concat_j_parts[0] if len(concat_j_parts) == 1
                             else torch.cat(concat_j_parts))
                    return [(self.X[idx], self.y[idx], cat_i, cat_j)]
                # All queries in batch had zero informative pairs (rare):
                # fall through to 2-tuple shape so loss returns 0.
                return [(self.X[idx], self.y[idx])]
        # Legacy single-query path: attach precomputed pair indices if available.
        # The 4-tuple shape is recognised by _ranker_passthrough_collate +
        # training_step.
        if self._pair_idx_by_query is not None and indices_key is not None:
            pair = self._pair_idx_by_query.get(indices_key)
            if pair is not None:
                return [(self.X[idx], self.y[idx], pair[0], pair[1])]
        return [(self.X[idx], self.y[idx])]


def _ranker_passthrough_collate(batch):
    """Collate that unwraps the singleton produced by ``_RankerDataset.__getitems__``.

    When ``__getitems__`` returns ``[(X_batch, y_batch)]``, default_collate would
    iterate it as a 1-element list of rows and stack each tensor into shape
    ``(1, B, F)`` / ``(1, B)`` — wrong. This collate detects the marker shape
    (1-element list whose sole element is a pair of >=1-D tensors with matching
    first dim) and passes the pre-stacked batch through unchanged. Falls back to
    default_collate for the per-row __getitem__ path used by older PyTorch.
    """
    if (
        len(batch) == 1
        and isinstance(batch[0], tuple)
        and len(batch[0]) in (2, 4)
        and torch.is_tensor(batch[0][0])
        and torch.is_tensor(batch[0][1])
        and batch[0][0].dim() >= 2
    ):
        return batch[0]
    from torch.utils.data._utils.collate import default_collate
    return default_collate(batch)


# ----------------------------------------------------------------------------------
# Lightning module
# ----------------------------------------------------------------------------------


def _import_lightning():
    try:
        import lightning.pytorch as L
        return L
    except ImportError:
        import pytorch_lightning as L  # legacy package name
        return L


# Must be module-level (NOT nested inside _make_lightning_module) so PyTorch Lightning
# can pickle the class for save_hyperparameters / ModelCheckpoint. Closure-bound classes
# raise _pickle.PicklingError: Can't pickle ... it's not found as ...<locals>...
_L_MODULE = _import_lightning()

# Module-top EarlyStopping resolution: the fit-time try/except ImportError was repeated
# on every fit call and obscured the hot-loop signal.
try:
    from lightning.pytorch.callbacks import EarlyStopping as _EarlyStopping
except ImportError:
    try:
        from pytorch_lightning.callbacks import EarlyStopping as _EarlyStopping  # type: ignore
    except ImportError:
        _EarlyStopping = None  # type: ignore


class MLPRankerLightningModule(_L_MODULE.LightningModule):
    """LightningModule for MLPRanker.

    Module-level so it pickles cleanly for Lightning's checkpointing and
    save_hyperparameters. ``network`` is excluded from save_hyperparameters (large nn.Module,
    held via direct attribute assignment instead).
    """

    def __init__(self, network, loss_name: str, learning_rate: float):
        super().__init__()
        self.network = network
        self.loss_name = loss_name
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=["network"])

    def forward(self, x):
        return self.network(x).view(-1)

    def _compute_loss(self, batch, scores):
        # 4-tuple shape ``(X, y, i_idx, j_idx)`` is the precomputed-pairs fast
        # path installed by ``_RankerDataset.install_pair_index_cache``; the
        # 2-tuple ``(X, y)`` shape stays valid for the legacy in-loss cache.
        if self.loss_name == "ranknet":
            if len(batch) == 4:
                _, _, i_idx, j_idx = batch
                return ranknet_pairwise_loss_precomputed(scores, i_idx, j_idx)
            return ranknet_pairwise_loss(scores, batch[1])
        if self.loss_name == "listnet":
            return listnet_top1_loss(scores, batch[1])
        raise ValueError(f"unknown loss_name {self.loss_name!r}")

    def training_step(self, batch, batch_idx):
        x = batch[0]
        scores = self.forward(x)
        loss = self._compute_loss(batch, scores)
        # train_loss intentionally NOT logged: val_loss drives EarlyStopping and ranking
        # metrics are computed in ranker_suite.py from per-row predict scores. self.log
        # dominated ~10-15% of trainer.fit wall on tiny queries in profiling.
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        with torch.no_grad():
            scores = self.forward(x)
            loss = self._compute_loss(batch, scores)
        self.log("val_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def predict_step(self, batch, batch_idx):
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        with torch.no_grad():
            return self.forward(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


def _make_lightning_module(network: nn.Module, loss_name: str, learning_rate: float):
    """Back-compat factory for existing MLPRanker.fit call sites."""
    return MLPRankerLightningModule(network, loss_name, learning_rate)


class _SamplerSetEpochCallback(_L_MODULE.Callback):
    """Call set_epoch(current_epoch) on a custom train batch_sampler.

    PyTorch Lightning only auto-calls ``set_epoch`` on torch's DistributedSampler;
    custom samplers (here: ``GroupBatchSampler``) must be advanced explicitly per
    epoch so shuffle=True actually produces a different order across epochs. Without
    this, every epoch processed queries in the SAME random order (seeded once at
    sampler construction) and training stalled in a local optimum.
    """

    def __init__(self, sampler: GroupBatchSampler) -> None:
        super().__init__()
        self._sampler = sampler

    def on_train_epoch_start(self, trainer, pl_module) -> None:  # noqa: D401
        if hasattr(self._sampler, "set_epoch"):
            self._sampler.set_epoch(trainer.current_epoch)


# ----------------------------------------------------------------------------------
# sklearn-shaped wrapper
# ----------------------------------------------------------------------------------


class MLPRanker(BaseEstimator, RegressorMixin):
    """sklearn-shaped MLP ranker. Returns per-row 1-D scores via ``predict``.

    Hyperparameters:
        loss_fn      : "ranknet" (default) or "listnet"
        n_estimators : epochs (default 100)
        learning_rate: AdamW lr (default 1e-3)
        hidden_layers: tuple of hidden-layer sizes (default (64, 64))
        dropout      : (default 0.1)
        early_stopping_patience: stop when val_loss doesn't improve (default 10);
                                 None disables.
        enable_checkpointing: forwarded to Lightning Trainer (default False). Lightning's
                              default ModelCheckpoint writes last.ckpt per epoch; pure
                              overhead in the ranker_suite flow (joblib-serialised at end
                              of fit). Set True for resume-capable checkpoints.
    """

    def __init__(
        self,
        loss_fn: str = "ranknet",
        n_estimators: int = 100,
        learning_rate: float = 1e-3,
        hidden_layers: tuple[int, ...] = (64, 64),
        dropout: float = 0.1,
        early_stopping_patience: int | None = 10,
        seed: int = 42,
        verbose: int = 0,
        enable_checkpointing: bool = False,
        accumulate_grad_batches: int = 1,
        queries_per_batch: int = 32,
    ):
        self.loss_fn = loss_fn
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        self.verbose = verbose
        self.enable_checkpointing = enable_checkpointing
        # OPT-7 (2026-05-23): TRUE LTR batch packing. GroupBatchSampler packs
        # N=``queries_per_batch`` queries into each yielded batch; the Dataset
        # composes per-query (i_idx, j_idx) pair tensors with offset
        # corrections into one concat. Net: ONE forward + ONE backward +
        # ONE optimizer.step per packed batch instead of one per query.
        # iter216 measured 162000 batches / epoch on c0098 LTR (~800s in
        # optimizer.step alone); qpb=32 drops to ~5000 batches with same
        # gradient information. SUPERSEDES OPT-6 (accumulate_grad_batches)
        # which only amortised optimizer.step but kept forward+backward
        # per-query -- now both are amortised. Default reverted from 32 to 1
        # for ``accumulate_grad_batches`` since OPT-7 is the active path.
        # Set ``queries_per_batch=1`` to revert to the legacy per-query
        # path (e.g., for bit-exact gradient reproduction).
        self.queries_per_batch = max(1, int(queries_per_batch))
        # OPT-6 (2026-05-23): accumulate gradients across N "batches" (queries)
        # before each optimizer.step(). The Lightning-native equivalent of
        # explicit batch-packing across queries: each forward+backward is
        # cheap per-query, but optimizer.step() at ~5ms/call is the dominant
        # cost on the LTR-MLP tiny-batch (one-query-per-batch) path. iter183/
        # 216 measured 162000 optimizer.step() calls on c0098 LTR -> 800s of
        # 1150s wall (75%). With accumulate=32 the call count drops to ~5000
        # for the same epochs; effective batch size becomes 32 queries
        # (~10-50 rows/query x 32 = ~300-1600 rows per gradient update),
        # which is well within stable-optimizer territory for AdamW + LayerNorm.
        # accumulate_grad_batches=1 reverts to the legacy per-query update
        # path. Set via constructor for callers that need bit-exact gradient
        # flow vs prior behaviour (e.g., test reproducibility on a tiny
        # query set where the 32x averaging shifts the loss surface).
        self.accumulate_grad_batches = max(1, int(accumulate_grad_batches))

    def _x_to_array(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            # Defence-in-depth: caller should already have removed qid/target columns.
            X = X.select_dtypes(include=[np.number])
            return X.to_numpy(dtype=np.float32, copy=False)
        return np.asarray(X, dtype=np.float32)

    def _fit_imputer(self, X_arr: np.ndarray) -> None:
        # Tree rankers (CB/XGB/LGB) handle NaN/Inf natively; MLP doesn't. Without
        # this, even one NaN in the input scrambles the forward pass to NaN, the
        # loss becomes NaN, Lightning's NaN-guard halts training after epoch 0,
        # and EarlyStopping never reports the failure (observed 2026-05-21 on
        # fuzz combo c0063 with fillna_value_cfg=None: val_loss=nan, total
        # ranknet calls dropped from expected n_epochs*n_queries to one pass).
        # Per-column mean imputation matches sklearn SimpleImputer semantics;
        # all-NaN columns fall back to zero so the model still trains on the
        # remaining feature axes.
        finite_mask = np.isfinite(X_arr)
        col_has_any = finite_mask.any(axis=0)
        sums = np.where(finite_mask, X_arr, 0.0).sum(axis=0)
        counts = np.maximum(finite_mask.sum(axis=0), 1)
        means = np.where(col_has_any, sums / counts, 0.0)
        self._impute_means_ = means.astype(np.float32, copy=False)

    def _apply_imputer(self, X_arr: np.ndarray) -> np.ndarray:
        means = getattr(self, "_impute_means_", None)
        if means is None:
            return X_arr
        bad = ~np.isfinite(X_arr)
        if not bad.any():
            return X_arr
        # Wave 80 (2026-05-21): always copy. The prior inverse logic
        # `X_arr.copy() if not X_arr.flags.writeable else X_arr` silently
        # mutated the caller's writable buffer (the most common case), leaking
        # NaN/Inf replacements back into the user's frame. Caller-aliasing is
        # the bug class flagged by wave-38; do not propagate it here.
        X_out = X_arr.copy()
        # Broadcast the (n_features,) mean vector across rows; only NaN/Inf cells
        # are replaced so legitimate finite values pass through untouched.
        X_out[bad] = np.broadcast_to(means, X_arr.shape)[bad]
        return X_out

    def _fit_scaler(self, X_arr: np.ndarray) -> None:
        # NeuralNetStrategy.requires_scaling=True for a reason: AdamW + softplus
        # on raw-magnitude features (one binary 0/1 column next to a float column
        # spanning [1e3, 1e6]) bounces gradients across orders of magnitude per
        # step, never converges, and val_loss plateaus at ln(2) for ranknet
        # (observed 2026-05-21 on fuzz combo c0063 after the imputer fix:
        # 30 epochs ran but val_loss stayed 0.6931 = random-init baseline).
        # The ranker_suite caller never threads scaler= through to _fit_mlp_ranker,
        # so MLP gets unscaled features in production paths too. Standardise here
        # so the contract matches CB/XGB/LGB ("hand us any numeric tabular X").
        mean = X_arr.mean(axis=0)
        std = X_arr.std(axis=0)
        # Constant columns -> std=0; substitute 1.0 so the divide is a no-op
        # rather than producing inf or nan. The column already carries no signal.
        std = np.where(std > 0, std, 1.0)
        self._scaler_mean_ = mean.astype(np.float32, copy=False)
        self._scaler_std_ = std.astype(np.float32, copy=False)

    def _apply_scaler(self, X_arr: np.ndarray) -> np.ndarray:
        mean = getattr(self, "_scaler_mean_", None)
        std = getattr(self, "_scaler_std_", None)
        if mean is None or std is None:
            return X_arr
        return ((X_arr - mean) / std).astype(np.float32, copy=False)

    def fit(
        self,
        X,
        y,
        group_ids,
        X_val=None,
        y_val=None,
        group_ids_val=None,
        cat_features=None,
    ):
        """Fit the ranker.

        ``cat_features`` is accepted for signature symmetry with other rankers; categorical
        columns must already be numeric-encoded by the caller (MLPRanker doesn't support
        raw categoricals).
        """
        # _L_MODULE is the module-level resolved lightning package; the prior per-fit
        # ``_import_lightning()`` call re-resolved it on every fit (cheap but tarnishes
        # the audit signal for "lazy import inside hot loop"). Reuse the module top result.
        L = _L_MODULE

        # Wave 49 (2026-05-20): drop global RNG mutations -- they silently
        # overwrote caller's torch/numpy stream and broke reproducibility for any
        # sibling code in the same process. Downstream consumers already use
        # local generators (lines 224 + 488/509: np.random.default_rng(self.seed)
        # and the per-sampler seed kwarg); the only remaining torch RNG surface
        # is model init / DataLoader shuffle, which honour torch.Generator passed
        # through Lightning's `seed_everything(workers=False)`-equivalent contract.
        # If the caller wants global torch determinism they call torch.manual_seed
        # themselves before calling fit.

        X_arr = self._x_to_array(X)
        y_arr = np.asarray(y, dtype=np.float32).ravel()
        n_features = X_arr.shape[1]
        # Fit imputer + scaler on train statistics before applying to train/val/predict;
        # the same column means and z-score parameters are reused across all splits so
        # OOF / OOS see the same transform. Order matters: impute first (NaN -> mean)
        # so the scaler's std isn't poisoned by skipped NaN cells.
        self._fit_imputer(X_arr)
        X_arr = self._apply_imputer(X_arr)
        self._fit_scaler(X_arr)
        X_arr = self._apply_scaler(X_arr)

        network = generate_mlp(
            num_features=n_features,
            num_classes=1,
            nlayers=max(1, len(self.hidden_layers)),
            first_layer_num_neurons=self.hidden_layers[0] if self.hidden_layers else max(8, n_features),
            min_layer_neurons=1,
            dropout_prob=self.dropout,
            inputs_dropout_prob=0.0,
            use_layernorm=True,
            use_batchnorm=False,  # group-batching breaks BN stats; LN is safe
            verbose=0,
        )

        train_ds = _RankerDataset(X_arr, y_arr)
        # OPT-7: pack queries_per_batch queries per gradient update. ranknet
        # loss needs precomputed per-query (i_idx, j_idx); the multi-query
        # path needs both. listnet falls back to single-query.
        _train_qpb = self.queries_per_batch if (self.loss_fn == "ranknet" and y_arr.ndim == 1) else 1
        train_sampler = GroupBatchSampler(
            group_ids=np.asarray(group_ids), relevance=y_arr,
            shuffle=True, seed=self.seed,
            queries_per_batch=_train_qpb,
        )
        if len(train_sampler) == 0:
            raise ValueError(
                "MLPRanker.fit: zero usable training queries (every query has <2 docs OR "
                "single-class relevance). Increase rows per query OR ensure y has graded values."
            )
        # Precompute per-query pair-index tensors on CPU so __getitems__ can
        # attach them to each batch and the loss skips the tuple(rel.tolist())
        # device sync on every call. Only enabled for ranknet (listnet doesn't
        # need pair indices); skips when relevance is multilabel since the loss
        # falls back to the legacy path on those.
        if self.loss_fn == "ranknet" and y_arr.ndim == 1:
            train_ds.install_pair_index_cache(train_sampler._query_slices)
            if _train_qpb > 1:
                train_ds.attach_sampler(train_sampler)
        elif self.loss_fn == "listnet" and y_arr.ndim == 1:
            # iter364: listnet doesn't need pair indices but still benefits
            # from the (X_slice, y_slice) batch cache that collapses
            # __getitems__ to one dict.get per epoch.
            train_ds.install_batch_cache_no_pairs(train_sampler._query_slices)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            num_workers=0,
            collate_fn=_ranker_passthrough_collate,
        )

        val_loader = None
        if X_val is not None and y_val is not None and group_ids_val is not None:
            X_val_arr = self._apply_scaler(self._apply_imputer(self._x_to_array(X_val)))
            y_val_arr = np.asarray(y_val, dtype=np.float32).ravel()
            val_ds = _RankerDataset(X_val_arr, y_val_arr)
            _val_qpb = self.queries_per_batch if (self.loss_fn == "ranknet" and y_val_arr.ndim == 1) else 1
            val_sampler = GroupBatchSampler(
                group_ids=np.asarray(group_ids_val), relevance=y_val_arr,
                shuffle=False, seed=self.seed,
                queries_per_batch=_val_qpb,
            )
            if self.loss_fn == "ranknet" and y_val_arr.ndim == 1:
                val_ds.install_pair_index_cache(val_sampler._query_slices)
                if _val_qpb > 1:
                    val_ds.attach_sampler(val_sampler)
            elif self.loss_fn == "listnet" and y_val_arr.ndim == 1:
                val_ds.install_batch_cache_no_pairs(val_sampler._query_slices)
            if len(val_sampler) > 0:
                val_loader = DataLoader(
                    val_ds, batch_sampler=val_sampler, num_workers=0,
                    collate_fn=_ranker_passthrough_collate,
                )

        self.module_ = _make_lightning_module(
            network, self.loss_fn, self.learning_rate,
        )

        callbacks: list = [_SamplerSetEpochCallback(train_sampler)]
        if self.early_stopping_patience is not None and val_loader is not None and _EarlyStopping is not None:
            callbacks.append(
                _EarlyStopping(
                    monitor="val_loss", patience=self.early_stopping_patience,
                    mode="min", verbose=False,
                )
            )

        trainer = L.Trainer(
            max_epochs=self.n_estimators,
            enable_model_summary=False,
            enable_progress_bar=bool(self.verbose),
            logger=False,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=1,
            # ModelCheckpoint default is dead weight in the suite (joblib-serialised) and
            # emits "Checkpoint directory exists and is not empty" UserWarning across LTR runs.
            enable_checkpointing=self.enable_checkpointing,
            # OPT-6 (2026-05-23): see __init__ docstring -- amortizes
            # optimizer.step() cost across N batches (queries). 1 = legacy
            # per-query update; 32 = batched gradient update.
            accumulate_grad_batches=self.accumulate_grad_batches,
        )
        trainer.fit(
            model=self.module_,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        # iter357: the per-query batch caches built by install_pair_index_cache
        # are only useful during fit. Lightning's Trainer retains a reference
        # to train_ds via the DataLoader; without explicit cleanup those
        # ~18k pair tensors + X-slice tensors get joblib.dump'd downstream
        # (+114s on c0079 1M-row LTR profile). Nulling the dict releases the
        # tensors for GC and reduces the pickled MLPRanker to its essential
        # state (network weights + hparams).
        train_ds._pair_idx_by_query = None
        train_ds._batch_by_query = None
        if val_loader is not None:
            val_ds._pair_idx_by_query = None
            val_ds._batch_by_query = None
        self.trainer_ = trainer
        self.n_features_in_ = n_features
        return self

    def predict(self, X) -> np.ndarray:
        """Return per-row scores (1-D, higher=more relevant)."""
        X_arr = self._apply_scaler(self._apply_imputer(self._x_to_array(X)))
        self.module_.eval()
        device = next(self.module_.parameters()).device
        with torch.no_grad():
            x_t = torch.as_tensor(X_arr, dtype=torch.float32, device=device)
            scores = self.module_.forward(x_t).cpu().numpy()
        return scores.ravel()


__all__ = [
    "MLPRanker",
    "GroupBatchSampler",
    "ranknet_pairwise_loss",
    "ranknet_pairwise_loss_precomputed",
    "listnet_top1_loss",
]
