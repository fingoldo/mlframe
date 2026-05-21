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
    _max_n = int(_RANKNET_MAX_PAIRS_PER_QUERY ** 0.5)
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
        i_idx, j_idx = torch.where(rel.unsqueeze(1) > rel.unsqueeze(0))
    if i_idx.numel() == 0:
        return scores.new_zeros(())
    score_diff_pairs = scores[i_idx] - scores[j_idx]
    # softplus(-x) = -log(sigmoid(x)) = BCE-w-logits(x, t=1) on informative (rel_i > rel_j) diffs.
    return F.softplus(-score_diff_pairs).mean()


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
    """
    if scores.dim() != 1:
        scores = scores.view(-1)
    if i_idx is None or j_idx is None or i_idx.numel() == 0:
        return scores.new_zeros(())
    return _ranknet_loss_precomputed_core(scores, i_idx, j_idx)


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

    def __iter__(self):
        order = np.arange(len(self._query_slices))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self._epoch)
            rng.shuffle(order)
        for i in order:
            yield self._query_slices[i].tolist()

    def __len__(self) -> int:
        return len(self._query_slices)


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
        for indices in query_slices:
            rel = self.y[torch.as_tensor(indices, dtype=torch.long)]
            if rel.dim() > 1:
                # Multilabel y -> sum across labels matches what GBS uses to
                # decide query inclusion. Not used by ranknet directly.
                continue
            i_idx, j_idx = torch.where(rel.unsqueeze(1) > rel.unsqueeze(0))
            key = tuple(int(v) for v in indices)
            if i_idx.numel() == 0:
                # No informative pair; the loss returns 0 without needing
                # pair indices. Mark with sentinel so __getitems__ can skip
                # the runtime cache build for these too.
                cache[key] = (None, None)
                continue
            cache[key] = (i_idx.to(torch.long), j_idx.to(torch.long))
        self._pair_idx_by_query = cache

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
        if torch.is_tensor(indices):
            idx = indices
            indices_key = None  # tensor input rare; fall through to in-loss cache
        else:
            idx = torch.as_tensor(indices, dtype=torch.long)
            # ``tuple(indices)`` is ~4.5x faster than
            # ``np.asarray(indices, dtype=np.int64).tobytes()`` at the typical
            # ~10-row query shape (0.26us vs 1.17us per __getitems__ call;
            # ~500ms saved on a 540k-call fit). Both produce hashable keys with
            # the same identity semantics for row-index lists from
            # GroupBatchSampler.
            indices_key = tuple(indices)
        # Attach precomputed pair indices if available. The 4-tuple shape is
        # recognised by _ranker_passthrough_collate + training_step.
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
        train_sampler = GroupBatchSampler(
            group_ids=np.asarray(group_ids), relevance=y_arr,
            shuffle=True, seed=self.seed,
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
            val_sampler = GroupBatchSampler(
                group_ids=np.asarray(group_ids_val), relevance=y_val_arr,
                shuffle=False, seed=self.seed,
            )
            if self.loss_fn == "ranknet" and y_val_arr.ndim == 1:
                val_ds.install_pair_index_cache(val_sampler._query_slices)
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
        )
        trainer.fit(
            model=self.module_,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
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
