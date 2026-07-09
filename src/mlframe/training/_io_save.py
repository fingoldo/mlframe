"""Save-side orchestration for mlframe model bundles.

Carved verbatim from ``training/io.py`` (2026-06-22) to bring the parent under
the 1000-LOC ceiling. Holds only :func:`save_mlframe_model` (atomic-write +
sidecar + version-stamp orchestration); ``load_mlframe_model`` + the
``_SafeUnpickler`` trust boundary stay in the parent ``io.py``.

``logger`` is created module-locally; the remaining parent-defined helpers
(``_LEAN_STRIP_FIELDS``, ``atomic_write_bytes``, ``_write_save_meta_sidecar``)
are imported LAZILY inside ``save_mlframe_model`` rather than at module top, so
the static import graph has no ``_io_save -> io -> _io_save`` cycle. The parent
imports ``save_mlframe_model`` from here at the BOTTOM of ``io.py``.
"""

from __future__ import annotations

import os
import logging
from types import SimpleNamespace
from typing import Optional, Dict, Any

import dill  # nosec B403 - pickle used only for trusted same-process/dev-local round-trips, see call sites in this file
import zstandard as zstd

logger = logging.getLogger("mlframe.training.io")


def save_mlframe_model(
    model: object,
    file: str,
    zstd_kwargs: Optional[Dict[str, Any]] = None,
    verbose: int = 1,
    lean: bool = False,
    durable: bool = False,
    auto_lean_retry: bool = True,
    auto_lean_pre_check_mb: float = 50.0,
) -> bool:
    """
    Save an mlframe model to a compressed file.

    Uses zstandard compression and dill serialization to handle complex
    Python objects including lambdas and closures.

    Args:
        model: The model object to save (typically a SimpleNamespace).
        file: Path to the output file.
        zstd_kwargs: Optional compression parameters for zstandard.
            Defaults to level=4, with checksum and content size.
        verbose: Verbosity level. If > 0, logs the file size.
        durable: Default ``False`` (flipped 2026-05-20 per accuracy/perf-
            over-legacy policy). When True, the underlying
            ``atomic_write_bytes`` issues a per-file ``os.fsync`` so the
            saved bundle survives a power loss BEFORE the OS page-cache
            commits. On Windows this costs ~400ms per file
            (FlushFileBuffers waits for the disk WRITE CACHE to commit);
            on the 2026-05-19 multi-model fuzz profile this was 86% of
            save wall (6.09s of 7.11s = 5x speedup by skipping). Atomic
            ``write-tmp-then-rename`` semantics hold either way --
            concurrent readers never see a partial file -- only the
            post-rename DURABILITY window is shortened.
            For ML model bundles the worst case on power loss is
            "re-train" (recoverable), so the default favours speed.
            Pass ``durable=True`` explicitly when writing irreplaceable
            state.
        lean: When True, strip inference-irrelevant heavy fields
            (train_preds/val_preds/test_preds/probs/target, train_od_idx,
            val_od_idx, trainset_features_stats) from a SHALLOW COPY of the
            namespace before serialization. This is the same field set
            handled by :func:`clean_mlframe_model`. The caller's object is
            not mutated. Default False preserves the historical
            "save-everything" behaviour for forensics/round-trip parity;
            callers that only need inference-ready bundles (the harness, prod
            serving) can flip it to skip the dill descent through the
            heaviest numpy attrs (observed 30x save speedup on cb+xgb
            multi-model bundles).
    Snapshot semantics:
        ``pickle.dumps(model)`` deep-walks the model graph and writes a
        FROZEN snapshot to disk. Subsequent in-memory mutation of the
        caller's ``model`` (or of any mutable bundle slot like
        ``composite_target_specs`` / ``dummy_baselines`` / fitted
        estimators) does NOT affect the saved file.

        Wave-19 P2 audit (2026-05-20) raised a concern that "callers
        commonly mutate model.composite_target_specs ... between two
        suite calls; the on-disk dump diverges silently". Re-verified
        2026-05-20: the audit framing was off. Pickle's snapshot
        guarantee is enough -- two distinct saves at different points
        in the caller's mutation timeline correctly produce two
        distinct snapshots (which is exactly what the operator wants).

        The genuine cross-suite-call mutation hazard (one suite mutating
        a shared mutable dict that the next suite reads) is a SEPARATE
        problem already handled by ``main.py``'s ``_copy.deepcopy``
        assignment site at the precomputed-bundle boundary (wave 11
        cluster, commit ``1959016``). No additional copy-on-save logic
        needed here.

    Returns:
        True if save was successful, False otherwise.
    """
    # lazy: parent-defined helpers, imported here to avoid the
    # _io_save <-> io module-level import cycle.
    from mlframe.training.io import (
        _LEAN_STRIP_FIELDS,
        atomic_write_bytes,
        _write_save_meta_sidecar,
    )

    if zstd_kwargs is None:
        zstd_kwargs = dict(
            level=4,
            write_checksum=True,
            write_content_size=True,
            threads=-1,
        )
    # E2.2 (2026-05-22): pre-pickle size pre-check via ``pympler.asizeof``.
    # Walking the in-memory object graph is ~300x faster than the fat save
    # itself (bench: asizeof=0.5ms vs save=160ms at N=5M). When the estimate
    # exceeds ``auto_lean_pre_check_mb`` AND the payload is a SimpleNamespace
    # (lean is a no-op otherwise) AND the caller didn't already pin lean,
    # flip lean=True upfront -- saves the fat-then-lean retry double-dump.
    # Threshold defaults to 100 MB in-memory (zstd-level-4 lean dumps land
    # ~50 MB on disk for similar shapes; the 2x margin avoids tripping on
    # borderline payloads that would still fit comfortably).
    #
    # bench-attempt-rejected (_benchmarks/bench_save_asizeof_precheck.py + bench_save_load_profile.py): the asizeof
    # graph walk is already cheap relative to the save -- ~0.02 ms on a shallow ndarray bundle and ~5 ms on a deep
    # fitted-RandomForest graph (300 trees), vs ~250 ms full save; ~3% of save wall, the rest is irreducible
    # pickle.dumps + zstd. No optimization warranted. NOTE asizeof grossly under-estimates Cython/numpy-buffer-backed
    # models (RF: est 0.4 MB vs 155 MB serialized), so it is correctly used ONLY for the SimpleNamespace eager/lean
    # flip and must NOT be swapped for a "cheaper" estimate that would change the gate decision.
    if not lean and auto_lean_retry and auto_lean_pre_check_mb > 0.0 and isinstance(model, SimpleNamespace):
        try:
            from pympler import asizeof as _pa
            _est_bytes = _pa.asizeof(model)
            _est_mb = _est_bytes / (1024 * 1024)
            if _est_mb > auto_lean_pre_check_mb:
                if verbose > 0:
                    logger.warning(
                        "[save-size-precheck] %s: in-memory payload ~%.1f MB "
                        "(> %.0f MB threshold) -- flipping lean=True BEFORE the "
                        "fat pickle to skip the auto-retry double-dump. Pass "
                        "``auto_lean_pre_check_mb=0`` to disable this pre-flip.",
                        file, _est_mb, auto_lean_pre_check_mb,
                    )
                lean = True
        except ImportError:
            # pympler is a hard dep per pyproject.toml; if it's somehow not
            # importable here we fall through to the post-save sensor retry.
            pass
        except Exception as _pa_err:
            # ``asizeof`` can stack-overflow on deeply-recursive objects with
            # ill-defined __dict__ traversal; never let it block a save.
            logger.debug(
                "[save-size-precheck] pympler.asizeof raised %s; falling through " "to post-save sensor retry.",
                _pa_err,
            )
    if lean and isinstance(model, SimpleNamespace):
        _lean = SimpleNamespace(**{k: v for k, v in vars(model).items() if k not in _LEAN_STRIP_FIELDS})
        _payload: object = _lean
    else:
        _payload = model

    # torch.compile produces an OptimizedModule wrapper that carries a non-picklable
    # ConfigModuleInstance closure (torch._dynamo). dill.dump raises TypeError
    # "cannot pickle 'ConfigModuleInstance' object". Walk the payload, locate any
    # attribute whose value carries ``_orig_mod`` (the un-compiled underlying
    # nn.Module that torch.compile preserves), temp-swap to the original for the
    # duration of dump, then restore. This keeps the caller's compile state intact
    # after save returns and avoids invasive changes to the model class hierarchy.
    #
    # 2026-05-21: also strips the Lightning bloat attrs. LightningModule._trainer
    # and PytorchLightningEstimator.prediction_datamodule both hold DataLoader
    # references that carry the whole training dataset. On 4M-row regression
    # this inflated MLP dumps to 311 MB for a network with 14k parameters.
    # Neither is needed for inference. Previously each walk ran independently
    # (2x recursion / seen-set churn for ~2400 mixed entries on c0024 / 1k LTR);
    # merged into one pass below.
    _compile_swaps: list = []
    _bloat_strips: list = []
    _walk_seen: set = set()
    # Name-based strip for the canonical attrs. Broadened beyond the original
    # ("_trainer", "prediction_datamodule") because a 2.4 GB MLP dump survived
    # the name-only strip: a Lightning Trainer / DataModule / DataLoader that
    # carries the whole 4M-row training dataset was held under a DIFFERENT
    # attr name (e.g. the Lightning ``.trainer`` backref, a retained
    # ``datamodule`` / ``*_dataloader``). The TYPE-based check below is the
    # robust catch-all; these names are the cheap fast path.
    _BLOAT_ATTR_NAMES = (
        "_trainer", "trainer", "prediction_datamodule", "datamodule",
        "_datamodule", "train_dataloader", "val_dataloader",
        "test_dataloader", "predict_dataloader",
    )

    def _looks_like_training_bloat(v: object) -> bool:
        """True for objects that carry the training dataset but are never
        needed for INFERENCE: a Lightning Trainer / *DataModule, or a torch
        DataLoader / Dataset. Duck-typed by module + class name so io.py does
        not hard-import torch / lightning. The fitted network weights live on
        the LightningModule (NOT matched here), so nulling these is safe."""
        try:
            cls = type(v)
            mod = getattr(cls, "__module__", "") or ""
            name = cls.__name__
        except Exception:
            return False
        if mod.startswith(("lightning", "pytorch_lightning")):
            # endswith (not ==) so Trainer subclasses / custom DataModules match.
            if name.endswith("Trainer") or name.endswith("DataModule"):
                return True
        if mod.startswith("torch.utils.data"):
            if name.endswith("DataLoader") or name.endswith("Dataset"):
                return True
        return False

    def _collect_pre_dump_swaps(obj: object) -> None:
        """Recursively walk ``obj.__dict__`` (cycle-guarded via ``_walk_seen``), temporarily stripping FS-internal suite markers, swapping ``torch.compile``-wrapped modules for their original, and nulling heavy Lightning training-only attributes so none of them reach the pickle; each mutation is recorded for restoration in the caller's ``finally`` block."""
        if id(obj) in _walk_seen:
            return
        _walk_seen.add(id(obj))
        _d = getattr(obj, "__dict__", None)
        if not isinstance(_d, dict):
            return
        for _k, _v in list(_d.items()):
            # FS suite-runtime markers: ``_build_pre_pipelines`` stamps selectors with private ``_mlframe_*`` attributes (selector-kind dispatch tag, weight-aware fit flag, the cross-target identity-cache override) purely so the in-process training loop can route report-build / weight-forwarding / cache reuse. None is needed for INFERENCE, so strip them from the pickle (temp-remove + restore-after-dump, same mechanism as the Lightning bloat strip) so the saved model carries no FS-internal suite state; restored on the caller's in-memory object in the finally block.
            if _k.startswith("_mlframe_") and (
                _k.endswith("_selector_kind_") or _k.endswith("_use_sample_weights_in_fs_") or _k.endswith("_identity_cache_override_")
            ):
                _bloat_strips.append((obj, _k, _v))
                del _d[_k]
                continue
            # torch.compile: swap to the un-compiled underlying nn.Module.
            _orig = getattr(_v, "_orig_mod", None)
            if _orig is not None and _orig is not _v:
                _compile_swaps.append((obj, _k, _v))
                _d[_k] = _orig
                # Recurse INTO the swapped-in original; the wrapper itself isn't
                # part of the dumped graph any more.
                _collect_pre_dump_swaps(_orig)
                continue
            # Lightning bloat strip: temp-null the heavy training-only objects
            # (by canonical name OR by type) so the 4M-row dataset they hold
            # never reaches the pickle. Don't recurse INTO them -- the whole
            # subtree is about to be discarded for the pickle pass.
            if _v is not None and (_k in _BLOAT_ATTR_NAMES or _looks_like_training_bloat(_v)):
                _bloat_strips.append((obj, _k, _v))
                _d[_k] = None
                continue
            _collect_pre_dump_swaps(_v)

    try:
        _collect_pre_dump_swaps(_payload)
    except Exception:
        # Defensive: a malformed payload should still attempt the dump; the
        # wrapping try/except below catches dump-time errors. Reset the swap
        # lists so the restore loop at the bottom is a no-op.
        _compile_swaps = []
        _bloat_strips = []

    try:
        # Atomic write prevents corruption when two train runs save to
        # the same file concurrently (2026-04-19 probe finding).
        #
        # Serialize to bytes BEFORE writing to disk so a pickle-vs-dill
        # fallback decision is made cleanly: dumping straight to the
        # zstd stream would leave the stream half-written if pickle.dump
        # raises mid-graph on an unpicklable closure/lambda, and the
        # subsequent dill.dump would corrupt the output. Worst case the
        # whole bundle lives briefly in RAM at ~payload-size; the dill
        # path is rare so this is a small extra alloc for the typical
        # pickle-fast path.
        import pickle as _pickle  # nosec B403 - pickle used only for trusted same-process/dev-local round-trips, see call sites in this file
        try:
            _payload_bytes = _pickle.dumps(_payload, protocol=_pickle.HIGHEST_PROTOCOL)
        except (TypeError, AttributeError, _pickle.PicklingError) as _pickle_err:
            # Non-picklable object in the graph -- dill handles closures /
            # lambdas / generators that vanilla pickle rejects.
            logger.info(
                "save_mlframe_model: pickle rejected the payload " "(%s); falling back to dill.dumps for %s",
                type(_pickle_err).__name__,
                file,
            )
            _payload_bytes = dill.dumps(_payload)

        def _writer(f):
            """Write the pre-serialized payload bytes through a zstd stream compressor into the atomic-write temp file handle."""
            compressor = zstd.ZstdCompressor(**zstd_kwargs)
            # closefd=False: stream_writer.__exit__ would otherwise close the wrapped
            # file (deterministic on Windows when threads=-1 hands the descriptor to
            # a background flush thread). atomic_write_bytes still needs the fd open
            # for its post-write f.flush() / os.fsync(fileno()) -- the durability
            # invariant established for the same-FS atomic rename.
            #
            # BufferedWriter wrapping (64KB/256KB/1MB/4MB) was benchmarked
            # 2026-04-14 on a fitted RandomForest + 1M-element ndarray payload --
            # all sizes landed within +/-5% of the direct write (high variance).
            # Direct write retained.
            with compressor.stream_writer(f, closefd=False) as zf:
                zf.write(_payload_bytes)

        atomic_write_bytes(file, _writer, fsync=durable)
        # Wave 19 P0 #1: write a sidecar .meta.json next to the .dump bundle
        # so the load side can refuse / WARN on version skew. Sidecar approach
        # (vs payload-wrapping) is non-invasive: old loaders simply ignore the
        # extra file; new loaders read it before unpickling.
        try:
            _write_save_meta_sidecar(file, durable=durable)
        except Exception as _meta_err:
            # Sidecar is best-effort: a failure here MUST NOT block the save
            # (the .dump is already on disk). Log at WARN so the operator
            # sees the version-stamp gap.
            logger.warning(
                "save_mlframe_model: failed to write .meta.json sidecar for "
                "%s: %s. Bundle saved; load-time version validation will "
                "fall through to back-compat path.",
                file, _meta_err,
            )
        size_mb = os.path.getsize(file) / (1024 * 1024)
        if verbose > 0:
            logger.info("Model saved successfully to %s. Size: %.2f Mb", file, size_mb)
        # 2026-05-21: suspicious-size sensor. Tabular ML model bundles
        # (CB / XGB / LGB / MLP / Linear) post-zstd should typically
        # land at <50 MB even on million-row training. Anything above
        # the threshold below is almost always an unstripped DataLoader /
        # trainer / optimizer state OR a forgotten OOF blob -- one
        # observed prod MLP dump was 311 MB because
        # ``LightningModule._trainer`` + ``prediction_datamodule`` held
        # refs to the 4M-row training frame (the strip step now nullifies
        # them during pickle).
        # Emit a HARD WARNING so operators see oversized dumps in their
        # run log instead of having to ``ls -lh`` the artefact dir.
        _SIZE_SUSPICIOUS_MB = 50.0
        if size_mb > _SIZE_SUSPICIOUS_MB:
            logger.warning(
                "[save-size-sensor] %s: dump is %.1f MB (> %.0f MB suspicious "
                "threshold). Tabular ML bundles should be <50 MB even on million-"
                "row training. Likely cause: a heavy attribute slipped past the "
                "strip layer (e.g. LightningModule.trainer / "
                "prediction_datamodule / fitted OOF preds / trainset_features_stats). "
                "If MLP: confirm save_mlframe_model nullified ``_trainer`` and "
                "``prediction_datamodule`` (added 2026-05-21). If tree booster: "
                "check that ``compute_trainset_metrics`` isn't holding the train "
                "frame on the fitted estimator. If you're saving forensics on "
                "purpose pass ``lean=False`` and ignore this warning.",
                file, size_mb, _SIZE_SUSPICIOUS_MB,
            )
            # E2.1 (2026-05-21): auto-retry with lean=True when the sensor fires on a
            # non-lean save. The default lean=False preserves forensic round-trip
            # parity but lets large per-split arrays leak (~16-32 MB each on 4M
            # rows). When auto_lean_retry=True (default) AND the payload is a
            # SimpleNamespace (lean is a no-op otherwise) AND lean wasn't already on,
            # rewrite the dump in lean form. Caller can disable per-call to keep the
            # original behaviour (e.g. forensic snapshot intentionally kept fat).
            if auto_lean_retry and not lean and isinstance(model, SimpleNamespace):
                if verbose > 0:
                    logger.warning(
                        "[save-size-sensor] %s: auto-retrying with lean=True to "
                        "strip per-split + OOF + trainset_features_stats fields. "
                        "Pass ``auto_lean_retry=False`` if you need the full bundle.",
                        file,
                    )
                # Note: we call ourselves recursively; auto_lean_retry=False guards
                # against an infinite loop (the lean save can't trip the sensor again
                # under a strip set that's correctly maintained).
                return save_mlframe_model(
                    model, file,
                    zstd_kwargs=zstd_kwargs, verbose=verbose,
                    lean=True, durable=durable,
                    auto_lean_retry=False,
                )
        return True
    except Exception:
        # Wave 41 (2026-05-20): caller sees only a False return; without the traceback,
        # production triage of "save returned False" is impossible (pickle / disk-full /
        # torch-compile errors all look identical).
        logger.exception("Could not save model to file %s", file)
        return False
    finally:
        # Restore torch.compile wrappers on the caller's payload so subsequent
        # predict / fit reuses keep the optimized graph rather than the unwrapped
        # eager fallback.
        for _parent, _k, _orig_v in _compile_swaps:
            try:
                _parent.__dict__[_k] = _orig_v
            except Exception:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
                # If something mutated the parent during dump (rare), there's
                # nothing useful to restore - log at debug only.
                logger.debug("save_mlframe_model: could not restore compile-wrapped attr %r", _k)
        # Restore Lightning bloat-strip attrs (``_trainer`` /
        # ``prediction_datamodule``) on the caller's payload. The save
        # path nulled them on the IN-MEMORY object during pickle to
        # produce a slim dump (cuts MLP file size 30-60x by dropping
        # DataLoader refs to the training dataset); restore so the
        # caller can still e.g. predict / continue training without
        # re-fitting.
        for _parent, _k, _orig_v in _bloat_strips:
            try:
                _parent.__dict__[_k] = _orig_v
            except Exception:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
                logger.debug(
                    "save_mlframe_model: could not restore Lightning bloat attr %r", _k,
                )
