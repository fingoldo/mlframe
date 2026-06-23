"""Reusable container for the projected K-bank, hnswlib indices, target vector and standardiser state.

Two lifetimes:

- Mode A (OOF on train): rebuilt per fold inside ``_oof._kfold_attention_loop`` - the bank holds projections of one fold's train subset only, so val rows attend
  exclusively to non-self training rows.
- Mode B (inference on val/test/OOS/holdout): built once from the full train set in ``compute_row_attention`` and re-used for all subsequent query batches.
  Optionally GPU-resident: ``to_device()`` uploads the projected bank to cupy memory so the per-query stage-4 dispatch skips the H2D round-trip.

Disk caching uses a sha256 fingerprint of (X_train bytes + seed + n_heads + head_dim + metric + index params). Hits trigger a hnswlib ``load_index`` (~1-2 s for
the 10M, d=8 case) instead of the 10-30 minute rebuild. Cache misses rebuild and write the new artifacts atomically. The key includes the projection seed so a
seed change correctly invalidates.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from mlframe.utils.safe_pickle import safe_dump, safe_load

logger = logging.getLogger(__name__)


@dataclass
class KeyBank:
    """Holds everything an ``attend()`` call needs to score a batch of query rows.

    Fields:
        ``projections``    - (n_heads, d_input, head_dim)   random projection matrices, seed-derived
        ``k_proj``         - (n_heads, n_train, head_dim)   projected K-bank (host), L2-normalised
        ``y_train``        - (n_train,)                     targets in float32 (downcast at construction for consistent kernel signature)
        ``ann_indices``    - list[Any]                      one hnswlib Index per head, built on ``k_proj[h]``
        ``standardiser``   - Optional[RobustScaler-like]    fit on train; the same object is applied at query time
        ``head_dim``, ``n_heads``, ``n_train``, ``d_input`` - cached scalar metadata (for validation)
        ``k_proj_device``  - per-head cupy arrays if ``to_device()`` was called, else None (Mode B GPU residency)
        ``y_train_device`` - cupy array if device-resident, else None
        ``seed``           - the projection seed (used in cache fingerprint)
    """
    projections: np.ndarray
    k_proj: np.ndarray
    y_train: np.ndarray
    ann_indices: list[Any] = field(default_factory=list)
    standardiser: Any | None = None
    head_dim: int = 0
    n_heads: int = 0
    n_train: int = 0
    d_input: int = 0
    k_proj_device: list[Any] | None = None
    y_train_device: Any | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        # Cross-check shapes; field defaults from caller should already be consistent but assertions catch silent slicing mistakes.
        n_heads_p, d_input_p, head_dim_p = self.projections.shape
        n_heads_k, n_train_k, head_dim_k = self.k_proj.shape
        if (n_heads_p, head_dim_p) != (n_heads_k, head_dim_k):
            raise ValueError(
                f"KeyBank: projections vs k_proj inconsistent: projections={self.projections.shape} k_proj={self.k_proj.shape}."
            )
        if self.y_train.shape != (n_train_k,):
            raise ValueError(f"KeyBank: y_train shape {self.y_train.shape} != ({n_train_k},).")
        self.n_heads = n_heads_p
        self.d_input = d_input_p
        self.head_dim = head_dim_p
        self.n_train = n_train_k

    def to_device(self) -> None:
        """Upload ``k_proj`` and ``y_train`` to cupy device memory. Used by Mode B inference to amortise H2D across many query batches.

        After calling this, the device-resident arrays are passed to ``row_attention_stage4_cupy`` via its ``k_proj_device`` / ``y_train_device`` parameters so the
        per-call upload is skipped. Idempotent: re-calling is a no-op once arrays are resident.
        """
        if self.k_proj_device is not None:
            return
        from ._utils import is_gpu_available
        if not is_gpu_available():
            raise RuntimeError("KeyBank.to_device() called but GPU is not available.")
        import cupy as cp
        self.k_proj_device = [cp.asarray(self.k_proj[h], dtype=cp.float32) for h in range(self.n_heads)]
        self.y_train_device = cp.asarray(self.y_train, dtype=cp.float32)

    def free_device(self) -> None:
        """Release device-resident arrays. Call when finishing a Mode B inference session.

        Critical for downstream GPU operators: cupy holds device memory in its pool, so an unreleased KeyBank prevents the next GPU op from claiming the bytes.
        """
        if self.k_proj_device is not None:
            self.k_proj_device = None  # cupy refcount drops -> pool reclaims
            self.y_train_device = None


def _key_bank_fingerprint(
    X_train: np.ndarray,
    seed: int,
    n_heads: int,
    head_dim: int,
    metric: str,
    standardize: bool,
    ann_M: int,
    ann_ef_construction: int,
) -> str:
    """Produce a deterministic content-addressed cache key for the (projections, k_proj, ann_indices) artefacts.

    Hash inputs (in order, with delimiters so a length collision in any component cannot accidentally collide):
        - X_train.tobytes()         (the actual data; bytes hashing avoids casting issues)
        - X_train.dtype, X_train.shape
        - seed, n_heads, head_dim, metric, standardize, ann_M, ann_ef_construction

    Bytes-level hashing of large X is the right call for correctness despite the cost (~3 GB/s sha256 on modern CPUs ~= 1-3 s for 10M, d=64). The alternative
    (hash a downsampled fingerprint) silently collides on data that differs only outside the sample - the kind of bug that surfaces only when somebody adds new
    rows but the cache returns stale state.
    """
    h = hashlib.sha256()
    h.update(b"X_train|")
    h.update(X_train.tobytes())
    h.update(b"|dtype|")
    h.update(str(X_train.dtype).encode())
    h.update(b"|shape|")
    h.update(str(X_train.shape).encode())
    h.update(b"|seed|")
    h.update(str(int(seed)).encode())
    h.update(b"|n_heads|")
    h.update(str(int(n_heads)).encode())
    h.update(b"|head_dim|")
    h.update(str(int(head_dim)).encode())
    h.update(b"|metric|")
    h.update(metric.encode())
    h.update(b"|standardize|")
    h.update(str(bool(standardize)).encode())
    h.update(b"|ann_M|")
    h.update(str(int(ann_M)).encode())
    h.update(b"|ann_ef_construction|")
    h.update(str(int(ann_ef_construction)).encode())
    return h.hexdigest()


def try_load_key_bank(
    cache_dir: Path,
    fingerprint: str,
) -> Optional[KeyBank]:
    """Return a ``KeyBank`` loaded from ``cache_dir / <fingerprint>/`` if all artefacts exist, else None.

    Per-fingerprint layout:
        cache_dir/<fp>/
            projections.npy
            k_proj.npy
            y_train.npy
            metadata.pkl       (seed, standardiser, ann_backend, fold info)
            ann_h{h}.pkl       (pickled ANN index per head; pynndescent serialises cleanly via pickle, hnswlib uses its own .save_index API)

    Returns None on any missing file - cache is opportunistic, not authoritative. A partial cache (e.g. crash during write) is treated as a miss so the rebuild
    overwrites cleanly.
    """
    bank_dir = cache_dir / fingerprint
    if not bank_dir.exists():
        return None
    projections_path = bank_dir / "projections.npy"
    k_proj_path = bank_dir / "k_proj.npy"
    y_train_path = bank_dir / "y_train.npy"
    metadata_path = bank_dir / "metadata.pkl"
    for p in (projections_path, k_proj_path, y_train_path, metadata_path):
        if not p.exists():
            return None
    try:
        projections = np.load(projections_path)
        k_proj = np.load(k_proj_path)
        y_train = np.load(y_train_path)
        metadata = safe_load(str(metadata_path))
        bank = KeyBank(
            projections=projections,
            k_proj=k_proj,
            y_train=y_train,
            standardiser=metadata.get("standardiser"),
            seed=metadata.get("seed", 0),
        )
        ann_backend = metadata.get("ann_backend", "pynndescent")
        ann_indices: list[Any] = []
        for h in range(bank.n_heads):
            idx_path = bank_dir / f"ann_h{h}.pkl"
            if not idx_path.exists():
                # Legacy hnswlib cache may have used .bin; check that path too for backward compat.
                legacy_bin = bank_dir / f"ann_h{h}.bin"
                if not legacy_bin.exists():
                    logger.info("ann index for head %d missing in cache; treating as miss.", h)
                    return None
                idx_path = legacy_bin
            from ._row_attention_ann import _AnnIndex
            try:
                if str(idx_path).endswith(".pkl"):
                    ann_indices.append(safe_load(str(idx_path)))
                else:
                    # Legacy hnswlib path.
                    import hnswlib
                    idx = hnswlib.Index(space=metadata.get("ann_space", "cosine"), dim=bank.head_dim)
                    idx.load_index(str(idx_path))
                    ann_indices.append(_AnnIndex(backend="hnswlib", obj=idx, metric=metadata.get("ann_space", "cosine"), head_dim=bank.head_dim))
            except Exception as exc:
                logger.info("Failed to load ann index for head %d (%s: %s); treating as miss.", h, type(exc).__name__, exc)
                return None
        bank.ann_indices = ann_indices
        logger.info("KeyBank loaded from cache %s (backend=%s).", bank_dir, ann_backend)
        return bank
    except Exception as exc:  # pragma: no cover - corrupt cache is recoverable
        logger.info("KeyBank cache load failed (%s: %s); treating as miss.", type(exc).__name__, exc)
        return None


def save_key_bank(
    bank: KeyBank,
    cache_dir: Path,
    fingerprint: str,
    *,
    ann_space: str = "cosine",
    ann_backend: str = "pynndescent",
) -> None:
    """Persist ``bank`` to ``cache_dir / <fingerprint>/``. Atomic via tmp-dir rename.

    pynndescent indices pickle cleanly so we just dump each via ``pickle``; legacy hnswlib indices use ``save_index`` to a ``.bin`` file (still supported on load
    for back-compat). The rename is atomic on same-volume filesystems; cross-volume falls back to non-atomic copy + remove, which is fine for the cache use case
    (a partial cache is treated as a miss by ``try_load_key_bank``).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Wave 48 (2026-05-20): use a UUID-stamped tmp dir so two parallel workers
    # writing the same fingerprint don't race on the shared "<fingerprint>.tmp"
    # path (each gets its own scratch). Final rename onto the canonical dir is
    # still racy but harmless (content-addressable: both workers have the same
    # bytes); the loser's tmp_dir is left for the next save call to clean up.
    import uuid as _uuid
    tmp_dir = cache_dir / (fingerprint + ".tmp." + _uuid.uuid4().hex[:8])
    if tmp_dir.exists():
        # Leftover from a previous failed write; wipe and retry.
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Any failure mid-write (disk full, pickling error, save_index raising) must not leave the multi-hundred-MB tmp dir
    # orphaned -- it would never be reclaimed (the next save uses a fresh UUID dir). Mirror io.atomic_write_bytes: wipe
    # the scratch dir on any error before re-raising. The happy path is unchanged.
    try:
        np.save(tmp_dir / "projections.npy", bank.projections)
        np.save(tmp_dir / "k_proj.npy", bank.k_proj)
        np.save(tmp_dir / "y_train.npy", bank.y_train)
        metadata = {
            "seed": bank.seed,
            "standardiser": bank.standardiser,
            "ann_space": ann_space,
            "ann_backend": ann_backend,
        }
        # sha256-sidecar the pickles so the loader's safe_load gate accepts them; a tampered cache file is then refused.
        safe_dump(metadata, str(tmp_dir / "metadata.pkl"))
        for h, idx in enumerate(bank.ann_indices):
            # _AnnIndex wraps either a pynndescent NNDescent (pickle-friendly) or an hnswlib Index (needs .save_index). Detect by the .backend attribute.
            backend = getattr(idx, "backend", "pynndescent")
            if backend == "hnswlib":
                inner = getattr(idx, "obj", idx)
                inner.save_index(str(tmp_dir / f"ann_h{h}.bin"))
            else:
                safe_dump(idx, str(tmp_dir / f"ann_h{h}.pkl"))
    except BaseException:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    final_dir = cache_dir / fingerprint
    # Wave 48 (2026-05-20): rmtree+rename was a TOCTOU race -- two writers could
    # both wipe final_dir and the loser's rename would fail. Wrap both in
    # try/except OSError: caches are content-addressable so the loser silently
    # discards its tmp_dir (the winner's bytes are equivalent).
    if final_dir.exists():
        import shutil
        try:
            shutil.rmtree(final_dir, ignore_errors=True)
        except OSError:
            pass
    try:
        tmp_dir.rename(final_dir)
    except OSError as _rn_err:
        # Race: a sibling worker won the rename. Clean up our tmp and move on.
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug("KeyBank rename(%s -> %s) lost the race: %s", tmp_dir, final_dir, _rn_err)
        return
    logger.info("KeyBank saved to cache %s.", final_dir)
