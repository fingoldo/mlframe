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
import uuid
from collections import OrderedDict
import warnings
from types import SimpleNamespace
from typing import Callable, Optional, Dict, Any

import dill  # nosec B403 - pickle used only for trusted same-process/dev-local round-trips, see call sites in this file
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
    _tmp_basename = f"{os.path.basename(target_path)}.tmp." f"{os.getpid()}.{_atomic_write_counter():d}.{uuid.uuid4().hex[:8]}"
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
    # 2026-05-30 (F-22 audit): modern Lightning installs expose the
    # umbrella ``lightning.*`` namespace (e.g. ``lightning.fabric.utilities.data.AttributeDict``
    # used by Lightning's hparams machinery) alongside the standalone
    # ``lightning_fabric.*`` package. Without these prefixes a fitted
    # PytorchLightningEstimator round-tripped through save_mlframe_model
    # / load_mlframe_model raises ``Unsafe class blocked by _SafeUnpickler``
    # for AttributeDict and load returns None. Both Lightning packages
    # are first-party (Lightning AI / pytorch-lightning project), safe
    # to allow alongside the original ``pytorch_lightning`` entry.
    "lightning",
    "lightning_fabric",
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
    # Fitted model state routinely carries functools.partial (neural weights_init_fcn, a parametrized
    # metric slot). Without this prefix such a bundle saves fine but safe-load blocks functools.partial
    # and returns None SILENTLY. Safe: a partial only STORES its func+args; the wrapped func is
    # re-resolved through find_class on unpickle and stays blocked if dangerous.
    "functools",
    "_functools",
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


# Code-execution primitives that live in the (allowlisted) ``builtins`` module. The prefix allow
# is needed for data containers (dict/list/set/tuple/bytes/...), but these names are a direct RCE
# gadget -- a legit model bundle never references them, an attacker payload uses exactly these
# (``(eval, ("__import__('os').system(...)",))``). Denied even though their module is allowlisted.
# ``__builtin__`` is the Python-2 / dill spelling of the same module.
_UNSAFE_BUILTINS: frozenset = frozenset({
    "eval", "exec", "execfile", "compile", "__import__", "import_module",
    "delattr", "globals", "locals", "vars",
    "open", "input", "breakpoint", "memoryview", "help",
})

# ``getattr`` is NOT in the blanket denylist: legitimate model bundles need it -- CatBoost's own
# ``__reduce__`` reconstructs via ``getattr(<catboost module>, "_setattr")``, so a blanket block makes
# the framework unable to load its OWN CatBoost dumps. Its operand is always an object that already
# passed ``find_class`` (the module/class allowlist), so the classic ``getattr(os, "system")`` gadget is
# unreachable -- ``os`` never lands on the stack. The residual risk is introspection escalation through
# dunder/code attributes (``__globals__`` -> module dict -> arbitrary callable); those attribute NAMES
# are denied below while ordinary attribute access (incl. underscore-prefixed helpers like ``_setattr``)
# is permitted via the restricted reconstructor.
_DANGEROUS_GETATTR_ATTRS: frozenset = frozenset({
    "__globals__", "__code__", "__closure__", "__func__", "__builtins__",
    "__subclasses__", "__bases__", "__mro__", "__dict__", "__getattribute__",
    "__reduce__", "__reduce_ex__", "__class__", "func_globals", "gi_frame",
    "cr_frame", "f_globals", "f_locals", "f_builtins",
})


def _safe_getattr(obj, name, *default):
    """Restricted ``getattr`` reconstructor for ``_SafeUnpickler``: refuses introspection-escalation
    attribute names (``__globals__`` / ``__code__`` / ``__subclasses__`` / ...) that could walk from an
    allowlisted object to an arbitrary callable. Plain attribute access (including underscore-prefixed
    library helpers) is allowed because the operand itself already passed the module/class allowlist."""
    if not isinstance(name, str) or name in _DANGEROUS_GETATTR_ATTRS:
        raise dill.UnpicklingError(f"Unsafe getattr blocked by _SafeUnpickler allowlist: getattr({type(obj).__name__}, {name!r})")
    return getattr(obj, name, *default)


def _safe_setattr(obj, name, value):
    """Restricted ``setattr`` reconstructor for ``_SafeUnpickler``. CatBoost's ``__reduce__`` restores
    estimator state via ``builtins.setattr``; like ``getattr`` its target object already passed the
    allowlist, so the only residual risk is type-confusion through dunder attributes (``__class__`` /
    ``__dict__`` / ``__bases__`` / ...). Those names are refused; ordinary attribute restoration is allowed."""
    if not isinstance(name, str) or name in _DANGEROUS_GETATTR_ATTRS:
        raise dill.UnpicklingError(f"Unsafe setattr blocked by _SafeUnpickler allowlist: setattr({type(obj).__name__}, {name!r})")
    setattr(obj, name, value)


class _SafeUnpickler(dill.Unpickler):
    """Restricted unpickler that only allows a conservative allowlist of modules."""

    def find_class(self, module: str, name: str):
        # Block code-exec builtins even though ``builtins`` is allowlisted for data containers.
        if module in ("builtins", "__builtin__") and name in _UNSAFE_BUILTINS:
            raise dill.UnpicklingError(f"Unsafe builtin blocked by _SafeUnpickler allowlist: {module}.{name}")
        # ``getattr`` / ``setattr`` are allowed but via restricted reconstructors (dangerous attr names denied).
        if module in ("builtins", "__builtin__") and name == "getattr":
            return _safe_getattr
        if module in ("builtins", "__builtin__") and name == "setattr":
            return _safe_setattr
        # Allow exact specific pairs.
        if (module, name) in _SAFE_SPECIFIC:
            return super().find_class(module, name)
        # Allow by module prefix (module == prefix or module startswith prefix + ".").
        for prefix in _SAFE_MODULE_PREFIXES:
            if module == prefix or module.startswith(prefix + "."):
                return super().find_class(module, name)
        raise dill.UnpicklingError(f"Unsafe class blocked by _SafeUnpickler allowlist: {module}.{name}")


_SIDECAR_META_VERSION = 1


# Libraries whose versions the .meta.json sidecar records, mapped from
# import-name -> PyPI distribution-name (only differ where listed; same name
# otherwise). cupy ships under arch-suffixed dists (cupy-cuda12x, ...) so its
# metadata lookup is unreliable -- the sys.modules fallback covers it.
_LIB_VERSION_DISTS: tuple = (
    ("mlframe", "mlframe"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("polars", "polars"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("lightgbm", "lightgbm"),
    ("xgboost", "xgboost"),
    ("catboost", "catboost"),
    ("pyarrow", "pyarrow"),
    ("dill", "dill"),
    ("cupy", "cupy"),
    ("numba", "numba"),
    ("pydantic", "pydantic"),
    ("lightning", "lightning"),
    ("torch", "torch"),
)


# Per-process memo of the live library-version snapshot, keyed by the identity of the ``_LIB_VERSION_DISTS`` tuple it was computed from.
# Installed-package metadata is immutable for a process lifetime, so re-parsing the RFC822 METADATA file for ~15 dists on every save AND every
# load is pure waste -- ``importlib.metadata.version`` over the list cost ~10 ms/call and dominated BOTH legs (~40% of save, ~83% of load on a
# small RF bundle: the actual unpickle was ~2 ms, the version walk ~10 ms). Keying on ``id(_LIB_VERSION_DISTS)`` keeps the cache correct under
# tests that ``monkeypatch.setattr`` a different dist tuple (new identity -> recompute) without an explicit clear.
_LIB_VERSIONS_CACHE: "Dict[int, Dict[str, str]]" = {}


def _collect_lib_versions() -> Dict[str, str]:
    """Snapshot the booster + serialization library versions at save time.

    Used by the .meta.json sidecar so the load-side can flag skew. Only the
    libraries whose internals appear in mlframe payloads are recorded; this
    keeps the sidecar small and the skew-check focused.

    CRITICAL: a version sidecar must NEVER force-load a heavy optional dep.
    Versions come from ``importlib.metadata.version`` (reads installed-package
    metadata WITHOUT importing the package) with a fallback to an
    already-imported module's ``__version__``. We never ``__import__`` the
    listed libs: on a lean tree-only / CPU-serving process the old
    ``__import__`` path paid ~14s of cold imports and pulled the whole neural
    stack (torch/transformers) resident in RAM purely to stamp version strings.
    A lib that is neither installed-with-metadata nor already imported is
    omitted -- its absence is itself correct signal for the skew-check.

    Result is memoised per ``_LIB_VERSION_DISTS`` identity: installed metadata
    does not change within a process, so save/load avoid re-parsing ~15 METADATA
    files every call. The cache copies out so callers can mutate freely.
    """
    import sys
    from importlib import metadata as _md

    _cached = _LIB_VERSIONS_CACHE.get(id(_LIB_VERSION_DISTS))
    if _cached is not None:
        return dict(_cached)

    out: Dict[str, str] = {}
    for _name, _dist in _LIB_VERSION_DISTS:
        _ver: Optional[str] = None
        try:
            _ver = _md.version(_dist)
        except _md.PackageNotFoundError:
            _ver = None
        except Exception:
            _ver = None
        if _ver is None:
            # Fallback for libs whose import-name != dist-name lookup failed
            # but that ARE already imported (e.g. arch-suffixed cupy dists).
            _mod = sys.modules.get(_name)
            if _mod is not None:
                _mv = getattr(_mod, "__version__", None)
                if _mv is not None:
                    _ver = str(_mv)
        if _ver is not None:
            out[_name] = str(_ver)
    _LIB_VERSIONS_CACHE[id(_LIB_VERSION_DISTS)] = out
    return dict(out)


def _lib_versions_cache_clear() -> None:
    """Reset the per-process library-version memo. For tests that mutate the live environment's installed metadata."""
    _LIB_VERSIONS_CACHE.clear()


def _meta_sidecar_path(bundle_path: str) -> str:
    """Sibling path for the version-envelope sidecar JSON."""
    return bundle_path + ".meta.json"


def _bundle_sha256(bundle_path: str, chunk: int = 1 << 20) -> Optional[str]:
    """Compute SHA-256 of the bundle file at save-side. Returns None when the bundle file is not yet readable (caller logs a non-fatal WARN -- the load-side sidecar check is independent and remains the primary RCE guard).

    Re-reads the just-written bundle rather than hashing the compressed bytes inline.
    bench-attempt-rejected (_benchmarks/bench_bundle_sha256_reopen_vs_inline.py): inline hashing saves only the
    reopen-read (~0.83ms@0.5MB / ~1.7ms@4MB / ~9.9ms@32MB) but has no FREE bit-identical capture -- one-shot
    ``compress()`` yields DIFFERENT on-disk bytes (changes the bundle + digest); the only bit-identical capture is a
    HashTee around the ``stream_writer`` fd, which entangles the hash with ``threads=-1`` background flush + the
    atomic-write ``fsync(fileno)`` durability invariant. Sub-1% of save wall (pickle+compress dominate); keep the
    reopen. Revisit only if 30MB+ fat-bundle hashing surfaces in a full-pipeline profile.
    """
    import hashlib
    try:
        h = hashlib.sha256()
        with open(bundle_path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def _write_save_meta_sidecar(bundle_path: str, *, durable: bool = False) -> None:
    """Write the .meta.json sidecar next to the just-written bundle.

    Schema (versioned by ``_SIDECAR_META_VERSION``):
        {
          "sidecar_version": 1,
          "saved_at_utc": "2026-05-20T12:34:56Z",
          "lib_versions": {"mlframe": "...", "lightgbm": "...", ...},
          "bundle_sha256": "<64-char hex>"   # SHA-256 of the bundle file.
        }

    Atomic-written via the same ``atomic_write_bytes`` helper as the
    bundle itself so a crash mid-write doesn't leave a half-meta file.

    ``bundle_sha256`` carries the actual SHA-256 of the just-written bundle so
    operators can audit payload integrity against the meta sidecar (independent
    of the inference-side ``.sha256`` sidecar check). When the bundle file is
    not yet readable the field is omitted rather than written as a placeholder
    -- callers downstream can detect absence vs. mismatch unambiguously.
    """
    # stdlib json (not orjson): this .meta.json sidecar is human-readable and
    # written with indent=2 for operator inspection, which orjson does not support.
    import json
    import datetime as _dt
    payload: Dict[str, Any] = {
        "sidecar_version": _SIDECAR_META_VERSION,
        "saved_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "lib_versions": _collect_lib_versions(),
    }
    digest = _bundle_sha256(bundle_path)
    if digest is not None:
        payload["bundle_sha256"] = digest
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
        with open(sidecar, encoding="utf-8") as f:
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
            f"bundle {bundle_path!r}:\n  " + "\n  ".join(drift) + "\nBooster libraries are typically forward-compatible for minor "
            "versions; if you see unexplained metric regression after a "
            "library upgrade, retrain on the live environment."
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
    return meta


# Inference-side warm cache for ``load_mlframe_model``. Keyed by (absolute file path, mtime_ns)
# so a model overwrite invalidates automatically; ``MLFRAME_LOAD_MODEL_CACHE_MAX`` controls capacity
# (default 32 -- a typical inference service holds <30 models in memory at once). The cache is
# byte-size-gated per CLAUDE.md: any single model whose on-disk compressed size already exceeds
# ``MLFRAME_LOAD_MODEL_CACHE_MAX_MB`` (default 2048) is loaded but NOT cached, because the
# in-memory unpickled form is typically larger still and a 4 GB CatBoost model would exhaust
# the inference host's RAM if cached.
_LOAD_MODEL_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
import threading as _threading  # noqa: E402
# Guards every mutation of _LOAD_MODEL_CACHE (get+move_to_end / store / popitem). A long-lived inference
# service calls load_mlframe_model concurrently from request threads; without this, concurrent move_to_end /
# popitem can raise "OrderedDict mutated during iteration" or corrupt the LRU order.
_LOAD_MODEL_CACHE_LOCK = _threading.Lock()


def _load_model_cache_clear() -> None:
    """Reset the inference-side ``load_mlframe_model`` cache. Useful in tests + when an operator wants to force a fresh load (e.g. after a model dir mass-update)."""
    with _LOAD_MODEL_CACHE_LOCK:
        _LOAD_MODEL_CACHE.clear()


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
    # Warm cache for repeat-load callers (typical of a long-lived inference service running
    # ``predict_mlframe_models_suite`` per request). Key on (abspath, mtime_ns) so a model
    # overwrite invalidates automatically. Skip cache for >2 GB on-disk bundles (CatBoost on a
    # rich suite can exceed this); the in-memory form is larger still and caching it would
    # break the host's RAM budget.
    _cache_key: Optional[tuple] = None
    try:
        _abs = os.path.abspath(file)
        _st = os.stat(_abs)
        _size_mb_env = os.environ.get("MLFRAME_LOAD_MODEL_CACHE_MAX_MB", "2048")
        try:
            _max_mb = float(_size_mb_env)
        except ValueError:
            _max_mb = 2048.0
        if _st.st_size <= _max_mb * (1024**2):
            _cache_key = (_abs, _st.st_mtime_ns, bool(safe))
            with _LOAD_MODEL_CACHE_LOCK:
                _hit = _LOAD_MODEL_CACHE.get(_cache_key)
                if _hit is not None:
                    _LOAD_MODEL_CACHE.move_to_end(_cache_key)
            if _hit is not None:
                # NOTE: this is the SHARED cached object (not a copy). Callers MUST NOT mutate it in place --
                # e.g. clean_mlframe_model() strips fields in place and would poison every later cache hit for
                # this (path, mtime). Copy first, or call _load_model_cache_clear(), if you need to mutate.
                return _hit
    except OSError:
        # Path doesn't exist or stat refused; fall through to the real loader which will surface the real error.
        pass
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
            "load_mlframe_model: sidecar validation raised unexpectedly " "(%s); proceeding with bundle load.",
            _meta_e,
        )
    try:
        with open(file, "rb") as f:
            decompressor = zstd.ZstdDecompressor()
            with decompressor.stream_reader(f) as zf:
                if safe:
                    model = _SafeUnpickler(zf).load()
                else:
                    warnings.warn(
                        "Loading without allowlist -- trust source",
                        UserWarning,
                        stacklevel=2,
                    )
                    model = dill.load(zf)  # nosec B301 - safe=False is caller opt-in (default safe=True uses _SafeUnpickler); warned above
        if _cache_key is not None:
            _max_count_env = os.environ.get("MLFRAME_LOAD_MODEL_CACHE_MAX", "32")
            try:
                _max_count = max(1, int(_max_count_env))
            except ValueError:
                _max_count = 32
            with _LOAD_MODEL_CACHE_LOCK:
                _LOAD_MODEL_CACHE[_cache_key] = model
                while len(_LOAD_MODEL_CACHE) > _max_count:
                    _LOAD_MODEL_CACHE.popitem(last=False)
        return model  # type: ignore[no-any-return]  # dill/pickle-loaded model is Any at the type level; declared Optional[object] is the honest external contract
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

    WARNING (audit5): mutates ``model`` IN PLACE. Do NOT call it on an object returned by
    ``load_mlframe_model`` that may be served again from the warm cache -- the stripped fields would then be
    missing for every later cache hit of the same (path, mtime). Clean a fresh copy, or call
    ``_load_model_cache_clear()`` first.

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
    "_load_model_cache_clear",
]


# Bottom-of-module re-export of the carved save orchestration. ``_io_save`` imports the shared
# helpers (logger / _LEAN_STRIP_FIELDS / atomic_write_bytes / _write_save_meta_sidecar) from this
# module at its top; importing it HERE (after those helpers are defined) keeps the public
# ``mlframe.training.io.save_mlframe_model`` API byte-for-byte unchanged without an import cycle.
from mlframe.training._io_save import save_mlframe_model  # noqa: E402
