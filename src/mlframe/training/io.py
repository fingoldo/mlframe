"""
Model serialization and I/O utilities for mlframe.

Provides functions for saving, loading, and cleaning mlframe models using
zstandard compression and dill serialization.

Threat model
------------
Pickle / dill deserialization is a code-execution vector. The default
``load_mlframe_model(file, safe=True)`` routes loads through
:class:`_SafeUnpickler`, which restricts ``find_class`` to a small allowlist
of module prefixes (numpy, pandas, polars, sklearn, torch, catboost,
lightgbm, xgboost, builtins, collections, datetime, dataclasses, types,
dill, scipy, mlframe, category_encoders) plus an explicit ``_SAFE_SPECIFIC``
set for a handful of ``typing`` markers. Loads of attacker-controlled files
should keep ``safe=True``; the ``safe=False`` opt-out emits a
``UserWarning`` and runs the vanilla ``dill.load`` (intended for trusted
in-process artefacts only). The atomic-write path in
:func:`atomic_write_bytes` provides crash-safety on the save side and is
unrelated to the unpickler trust boundary.
"""

from __future__ import annotations


import itertools
import logging
import os
import tempfile
import uuid
import warnings
from types import SimpleNamespace
from typing import Callable, Optional, Dict, Any

import dill
import zstandard as zstd

logger = logging.getLogger(__name__)


# Monotonic counter for the atomic-write temp-file name suffix. Combined
# with PID + 8-byte uuid hex this gives a unique name without paying
# mkstemp's O_EXCL retry loop (which is ~30x slower than direct os.open
# on Windows due to Defender / FS-filter intercepts).
_ATOMIC_WRITE_COUNTER = itertools.count(0)


def _atomic_write_counter() -> int:
    return next(_ATOMIC_WRITE_COUNTER)


_LEAN_STRIP_FIELDS = frozenset({
    "test_preds",
    "test_probs",
    "test_target",
    "val_preds",
    "val_probs",
    "val_target",
    "train_preds",
    "train_probs",
    "train_target",
    "train_od_idx",
    "val_od_idx",
    "trainset_features_stats",
    # 2026-05-21 P0 #2 follow-up: OOF preds/probs stamped on the model entry
    # at trainer.py:955 when ``oof_n_splits >= 2``. On 4M-row regression each
    # OOF array is ~16 MB (float32); inference-irrelevant (only consumed at
    # training time for level-1 stacking + OOF-based calibration). Without
    # them in the strip set, lean saves still leak 16-32 MB per model
    # whenever the suite stamps OOF -- the prod log's lean=True missed this
    # because oof_n_splits=0 was the active default, but it'll regress as
    # soon as any caller flips oof_n_splits>=2.
    "oof_preds",
    "oof_probs",
})


def atomic_write_bytes(target_path: str, writer_fn: Callable[[Any], None], *, fsync: bool = False) -> None:
    """Atomically write to ``target_path`` via write-tmp-then-rename.

    Previous implementation (2026-04-19 probe finding): callers used
    ``with open(target_path, "wb") as f: ...`` directly. Two parallel
    training runs writing to the same metadata.joblib / model file
    could truncate each other mid-write — subsequent load raised an
    opaque ``UnpicklingError`` / ``EOFError``, with no way to tell
    corruption from a legitimate version mismatch.

    This helper:
      1. Creates a temp file in the SAME directory (``os.replace``
         across filesystems isn't atomic; same-FS is).
      2. Invokes ``writer_fn(fileobj)`` — the caller owns the bytes.
      3. ``os.replace()`` atomically renames tmp → target (works on
         both POSIX and Windows since Python 3.3; ``os.rename`` on
         Windows would fail when target exists).
      4. Cleans up the tmp file on any exception so a failed write
         doesn't leak a ``metadata.joblib.xyz.tmp`` alongside the
         real file.

    The atomicity guarantee: a concurrent reader either sees the
    complete pre-write file or the complete post-write file, never
    a partial one. Concurrent writers still race (last writer wins),
    but neither produces corruption.

    Parameters
    ----------
    fsync : bool, default False
        When True, calls ``f.flush()`` + ``os.fsync(fd)`` before the
        rename so the new contents survive a power loss BEFORE the OS
        page-cache commits. The fsync is the dominant cost on Windows:
        ``nt.FlushFileBuffers`` blocks until the disk WRITE CACHE is
        committed (~400ms per call on commodity SSDs even for ~1MB
        files). Surfaced by the 2026-05-19 multi-model profile: 15
        model saves x 406ms fsync = 6.09s of 7.11s save wall (86%).

        The default was flipped from True to False on 2026-05-20 (user
        explicit OK + accuracy/perf-over-legacy policy). The atomic
        ``write-tmp-then-rename`` semantics still hold WITHOUT fsync --
        concurrent readers never see a partial file -- only the
        post-rename DURABILITY window is shortened (the OS flushes the
        page cache to physical disk within a few seconds; power loss
        in that window MAY leave a freshly-renamed file with its
        contents only on RAM-side pages). For ML model bundles this
        trade-off is favourable: the worst case is "re-train the model"
        (recoverable), not "data corruption" (unrecoverable). Pass
        ``fsync=True`` explicitly when writing irreplaceable state.
    """
    target_dir = os.path.dirname(target_path) or "."
    # mkstemp on Windows is ~30x slower than direct os.open because Microsoft
    # Defender / file-system filter drivers intercept the O_EXCL probe + the
    # secure-name generation. Surfaced by the 2026-05-19 multi-model fuzz
    # profile after the fsync skip + pickle-first land: nt.open was 33% of
    # save wall (15 mkstemp calls x 29ms each = 0.44s of 1.32s). We don't
    # need mkstemp's "secure unique name" guarantee for an internal temp file
    # inside our own directory -- a pid + counter + 8-byte uuid suffix avoids
    # collisions cheaply.
    _tmp_basename = (
        f"{os.path.basename(target_path)}.tmp."
        f"{os.getpid()}.{_atomic_write_counter():d}.{uuid.uuid4().hex[:8]}"
    )
    tmp_path = os.path.join(target_dir, _tmp_basename)
    fd = os.open(tmp_path, os.O_CREAT | os.O_WRONLY | os.O_EXCL)
    # ``fd`` ownership: passed to ``os.fdopen`` -> the resulting BufferedWriter takes
    # ownership and closes on context exit. But if ``os.fdopen`` itself raises (rare:
    # MemoryError during buffer alloc, or invalid-mode TypeError after a future
    # refactor), the raw fd is never adopted and Python leaks it. Under sustained
    # fuzz / cache-write pressure (composite_cache writes thousands of files per
    # suite call) this exhausts the process fd ceiling. Track adoption via a flag
    # and explicitly close in the except branch when adoption never happened.
    _fd_adopted = False
    try:
        with os.fdopen(fd, "wb") as f:
            _fd_adopted = True  # BufferedWriter now owns fd; on with-exit it closes.
            writer_fn(f)
            if fsync:
                # fsync inside the with-block: pickle.dump / dill.dump / numpy.save
                # only flush their own buffers, not the OS page cache. Without an
                # explicit fsync, a power loss between rename and writeback can
                # publish a visible filename whose contents are still dirty pages.
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp_path, target_path)
    except Exception:
        if not _fd_adopted:
            # os.fdopen raised before adopting; close fd manually so it doesn't leak.
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


# Allowlist of module prefixes for safe unpickling.
_SAFE_MODULE_PREFIXES: tuple = (
    "numpy",
    "pandas",
    "polars",
    "sklearn",
    "pytorch_lightning",
    "torch",
    "catboost",
    "lightgbm",
    "xgboost",
    "builtins",
    "collections",
    "datetime",
    "dataclasses",
    "types",
    "dill",
    "scipy",
    # Fix date 2026-04-15 (bug A): persisted CatBoost models reference assorted
    # mlframe.* helpers (metrics.ICE, training.helpers.*, etc.) inside their pickled
    # state; without this the cb model is silently dropped at load time.
    "mlframe",
    # Fix date 2026-04-22 (fuzz): Linear pre_pipelines include CatBoostEncoder /
    # OrdinalEncoder / OneHotEncoder from the category_encoders package. Blocking
    # them forced every cached-linear reload to fall back to retraining — the
    # "Unsafe class blocked by _SafeUnpickler allowlist:
    # category_encoders.cat_boost.CatBoostEncoder" WARN was noisy and defeated
    # the schema-hash caching mechanism for any suite that includes linear
    # models. Package is a widely-used sklearn-family transformer, safe to allow.
    "category_encoders",
    # Fix date 2026-05-20: recent pandas builds (>=2.0 with the pyarrow extension
    # backend, or 2.2+ where pyarrow is preferred for object-dtype Series)
    # pickle DataFrames through pyarrow.lib._restore_array. Without this entry
    # round-tripping any DataFrame through save_mlframe_model + load_mlframe_model
    # fails with "Unsafe class blocked by _SafeUnpickler allowlist:
    # pyarrow.lib._restore_array" and load_mlframe_model returns None
    # (observed 2026-05-20 in test_roundtrip_complex_nested_object on S:).
    # pyarrow is a first-party Apache project, safe to allow.
    "pyarrow",
)

# Specific safe names. `typing.TypeAlias` lands in pickled MLPRegressor
# attribute graphs since pytorch-lightning 2.x (its train/val dataloaders
# carry typing-annotated dataclasses); it's a no-op marker symbol with no
# code-execution surface.
_SAFE_SPECIFIC: frozenset = frozenset({
    ("types", "SimpleNamespace"),
    ("typing", "TypeAlias"),
    ("typing", "Any"),
    ("typing", "Optional"),
    ("typing", "Union"),
    ("typing", "List"),
    ("typing", "Dict"),
    ("typing", "Tuple"),
    ("typing", "Sequence"),
    ("typing", "Callable"),
    ("typing", "ClassVar"),
})


class _SafeUnpickler(dill.Unpickler):
    """Restricted unpickler that only allows a conservative allowlist of modules."""

    def find_class(self, module: str, name: str):
        # Allow exact specific pairs.
        if (module, name) in _SAFE_SPECIFIC:
            return super().find_class(module, name)
        # Allow by module prefix (module == prefix or module startswith prefix + ".").
        for prefix in _SAFE_MODULE_PREFIXES:
            if module == prefix or module.startswith(prefix + "."):
                return super().find_class(module, name)
        raise dill.UnpicklingError(
            f"Unsafe class blocked by _SafeUnpickler allowlist: {module}.{name}"
        )


_SIDECAR_META_VERSION = 1


def _collect_lib_versions() -> Dict[str, str]:
    """Snapshot the booster + serialization library versions at save time.

    Used by the .meta.json sidecar so the load-side can flag skew. Only the
    libraries whose internals appear in mlframe payloads are recorded; this
    keeps the sidecar small and the skew-check focused. Unavailable libraries
    are silently skipped (their absence is itself signal).
    """
    out: Dict[str, str] = {}
    for _name in (
        "mlframe", "numpy", "pandas", "polars", "scipy", "sklearn",
        "lightgbm", "xgboost", "catboost", "pyarrow", "dill",
    ):
        try:
            _mod = __import__(_name)
            _ver = getattr(_mod, "__version__", None)
            if _ver is not None:
                out[_name] = str(_ver)
        except (ImportError, AttributeError):
            continue
    return out


def _meta_sidecar_path(bundle_path: str) -> str:
    """Sibling path for the version-envelope sidecar JSON."""
    return bundle_path + ".meta.json"


def _write_save_meta_sidecar(bundle_path: str, *, durable: bool = False) -> None:
    """Write the .meta.json sidecar next to the just-written bundle.

    Schema (versioned by ``_SIDECAR_META_VERSION``):
        {
          "sidecar_version": 1,
          "saved_at_utc": "2026-05-20T12:34:56Z",
          "lib_versions": {"mlframe": "...", "lightgbm": "...", ...},
          "bundle_sha256": "...",  # not yet -- placeholder for future
        }

    Atomic-written via the same ``atomic_write_bytes`` helper as the
    bundle itself so a crash mid-write doesn't leave a half-meta file.
    """
    import json
    import datetime as _dt
    payload = {
        "sidecar_version": _SIDECAR_META_VERSION,
        "saved_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "lib_versions": _collect_lib_versions(),
    }
    meta_bytes = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")

    def _writer(f):
        f.write(meta_bytes)

    atomic_write_bytes(_meta_sidecar_path(bundle_path), _writer, fsync=durable)


def load_save_meta_sidecar(bundle_path: str) -> Optional[Dict[str, Any]]:
    """Read the .meta.json sidecar for a bundle. Returns None if no sidecar
    exists (legacy bundle pre-2026-05-20). Returns parsed dict otherwise.

    Errors during parse (corrupt JSON, missing fields) return None + WARN;
    callers should fall through to back-compat semantics rather than fail
    the whole load.
    """
    import json
    sidecar = _meta_sidecar_path(bundle_path)
    # Wave 48 (2026-05-20): the prior exists-then-open was a redundant TOCTOU check;
    # the except below already handles missing sidecar. Drop the precheck so the
    # race window collapses to zero.
    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(
                "load_save_meta_sidecar: %s is not a JSON object; ignoring.",
                sidecar,
            )
            return None
        return data
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as _e:
        logger.warning(
            "load_save_meta_sidecar: failed to read %s: %s. Falling back "
            "to back-compat (no version validation).", sidecar, _e,
        )
        return None


def validate_load_meta_sidecar(
    bundle_path: str,
    *,
    strict: bool = False,
) -> Optional[Dict[str, Any]]:
    """Load + validate the sidecar. Returns the parsed payload on success
    or None when no sidecar / unreadable. WARN-logs library-version drift
    between the saved snapshot and the live environment.

    ``strict=True`` raises ValueError on any drift; default False just WARNs
    (booster libraries are typically forward-compatible for minor versions
    but operators should see the skew when debugging metric regressions).
    """
    meta = load_save_meta_sidecar(bundle_path)
    if meta is None:
        return None
    saved_libs = meta.get("lib_versions") or {}
    live_libs = _collect_lib_versions()
    drift: list[str] = []
    for lib, saved_ver in saved_libs.items():
        live_ver = live_libs.get(lib)
        if live_ver is None:
            drift.append(f"{lib}: saved={saved_ver!r}, live=NOT-INSTALLED")
            continue
        if live_ver != saved_ver:
            drift.append(f"{lib}: saved={saved_ver!r}, live={live_ver!r}")
    if drift:
        msg = (
            f"load_mlframe_model: library-version drift detected for "
            f"bundle {bundle_path!r}:\n  " + "\n  ".join(drift) +
            "\nBooster libraries are typically forward-compatible for minor "
            "versions; if you see unexplained metric regression after a "
            "library upgrade, retrain on the live environment."
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
    return meta


def save_mlframe_model(
    model: object,
    file: str,
    zstd_kwargs: Optional[Dict[str, Any]] = None,
    verbose: int = 1,
    lean: bool = False,
    durable: bool = False,
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
    if zstd_kwargs is None:
        zstd_kwargs = dict(
            level=4,
            write_checksum=True,
            write_content_size=True,
            threads=-1,
        )
    if lean and isinstance(model, SimpleNamespace):
        _lean = SimpleNamespace(**{
            k: v for k, v in vars(model).items()
            if k not in _LEAN_STRIP_FIELDS
        })
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
    # references that carry the whole training dataset. On 4M-row TVT regression
    # this inflated MLP dumps to 311 MB for a network with 14k parameters.
    # Neither is needed for inference. Previously each walk ran independently
    # (2x recursion / seen-set churn for ~2400 mixed entries on c0024 / 1k LTR);
    # merged into one pass below.
    _compile_swaps: list = []
    _bloat_strips: list = []
    _walk_seen: set = set()
    _BLOAT_ATTR_NAMES = ("_trainer", "prediction_datamodule")

    def _collect_pre_dump_swaps(obj: object) -> None:
        if id(obj) in _walk_seen:
            return
        _walk_seen.add(id(obj))
        _d = getattr(obj, "__dict__", None)
        if not isinstance(_d, dict):
            return
        for _k, _v in list(_d.items()):
            # torch.compile: swap to the un-compiled underlying nn.Module.
            _orig = getattr(_v, "_orig_mod", None)
            if _orig is not None and _orig is not _v:
                _compile_swaps.append((obj, _k, _v))
                _d[_k] = _orig
                # Recurse INTO the swapped-in original; the wrapper itself isn't
                # part of the dumped graph any more.
                _collect_pre_dump_swaps(_orig)
                continue
            # Lightning bloat strip: temp-null _trainer / prediction_datamodule
            # at the matching attr names. Don't recurse INTO them -- the whole
            # subtree is about to be discarded for the pickle pass.
            if _k in _BLOAT_ATTR_NAMES and _v is not None:
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
        import pickle as _pickle
        try:
            _payload_bytes = _pickle.dumps(_payload, protocol=_pickle.HIGHEST_PROTOCOL)
        except (TypeError, AttributeError, _pickle.PicklingError) as _pickle_err:
            # Non-picklable object in the graph -- dill handles closures /
            # lambdas / generators that vanilla pickle rejects.
            logger.info(
                "save_mlframe_model: pickle rejected the payload "
                "(%s); falling back to dill.dumps for %s",
                type(_pickle_err).__name__, file,
            )
            _payload_bytes = dill.dumps(_payload)

        def _writer(f):
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
        # trainer / optimizer state OR a forgotten OOF blob -- the
        # TVT 2026-05-21 production MLP dump was 311 MB because
        # ``LightningModule._trainer`` + ``prediction_datamodule`` held
        # refs to the 4M-row training frame (cleaned up in this same
        # commit; the strip step nullifies them during pickle).
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
            except Exception:
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
            except Exception:
                logger.debug(
                    "save_mlframe_model: could not restore Lightning bloat attr %r", _k,
                )


def load_mlframe_model(file: str, safe: bool = True, strict_version: bool = False) -> Optional[object]:
    """
    Load an mlframe model from a compressed file.

    Args:
        file: Path to the model file.
        safe: If True (default), use _SafeUnpickler with a conservative allowlist.
            If False, use vanilla dill.load (unsafe — RCE risk from untrusted sources).
        strict_version: If True, raise on lib-version drift recorded in the
            .meta.json sidecar (post 2026-05-20 bundles). Default False just
            WARN-logs the drift -- booster libs are typically forward-
            compatible for minor versions.

    Returns:
        The loaded model object, or None if loading failed.
    """
    # Wave 19 P0 #1: validate the .meta.json sidecar BEFORE unpickling so the
    # operator sees library-version drift (catboost / lightgbm minor upgrades
    # silently change booster internals) instead of chasing a cryptic
    # AttributeError deep in predict(). Returns None on legacy bundles
    # (pre-2026-05-20) with no sidecar; that's a back-compat path with a
    # one-line INFO message in the helper.
    try:
        validate_load_meta_sidecar(file, strict=strict_version)
    except ValueError:
        # strict_version=True propagates the failure to the caller verbatim.
        raise
    except Exception as _meta_e:
        # Other (unexpected) failures in the sidecar reader are non-fatal:
        # the bundle itself may still be loadable. Logger.warning was already
        # called inside the helper for the known cases.
        logger.debug(
            "load_mlframe_model: sidecar validation raised unexpectedly "
            "(%s); proceeding with bundle load.", _meta_e,
        )
    try:
        with open(file, "rb") as f:
            decompressor = zstd.ZstdDecompressor()
            with decompressor.stream_reader(f) as zf:
                if safe:
                    model = _SafeUnpickler(zf).load()
                else:
                    warnings.warn(
                        "Loading without allowlist — trust source",
                        UserWarning,
                        stacklevel=2,
                    )
                    model = dill.load(zf)
        return model
    except Exception as e:
        # logger.exception captures the traceback automatically so the
        # operator can see the unpickler / zstd error stack rather than
        # only the str(e) summary. Previous .error() obscured the
        # underlying cause for any non-trivial load failure.
        logger.exception("Could not load model from file %s: %s", file, e)
        return None


def clean_mlframe_model(model: SimpleNamespace) -> SimpleNamespace:
    """
    Remove extra fields from a model's namespace to reduce RAM usage.

    Removes prediction arrays, target arrays, and outlier detection indices
    that are typically not needed after training.

    Args:
        model: The model namespace to clean.

    Returns:
        The cleaned model namespace (modified in place).
    """
    fields_to_remove = _LEAN_STRIP_FIELDS
    for field in fields_to_remove:
        if hasattr(model, field):
            delattr(model, field)
    return model


__all__ = [
    "save_mlframe_model",
    "load_mlframe_model",
    "clean_mlframe_model",
    "_SafeUnpickler",
]
