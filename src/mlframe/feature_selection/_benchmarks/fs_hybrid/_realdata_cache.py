"""Local, checksummed cache of the real OpenML classification beds used by the round-4 real-data bench.

The real leg of this harness never produced a result because `fetch_openml` is a run-time network dependency:
one unreachable host turns the whole externally-valid comparison into "INCONCLUSIVE". This module splits the
two concerns apart. `fill_cache` is the only function that touches the network; it downloads each dataset once
and writes it as a compressed `.npz` plus a sidecar JSON carrying the SHA-256 of the payload. `load_cached` is
pure local I/O: it verifies the checksum and refuses to hand back a file that does not match what was recorded,
so a bench run is reproducible and offline.

Cache layout, one pair of files per dataset::

    <cache_dir>/<name>.npz     X (2-D float array), y (integer codes), columns (unicode array)
    <cache_dir>/<name>.json    {name, openml_name, version, n_rows, n_cols, n_classes, positive_rate,
                                sha256, fetched_utc, sklearn_version}

The cache directory is `$MLFRAME_REALDATA_CACHE` when set, else `<repo root>/.cache/realdata`, and is gitignored.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["OPENML_SETS", "cache_dir_default", "fill_cache", "load_cached", "available"]

# (display_name, fetch_openml kwargs) -- mirrors round4_broad_realdata_bench.OPENML_SETS, kept here so the cache
# filler has no import-time dependency on the bench module (which pulls in LightGBM and the whole selector stack).
OPENML_SETS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("madelon", dict(name="madelon", version=1)),
    ("gina_agnostic", dict(name="gina_agnostic", version=1)),
    ("gisette", dict(name="gisette", version=1)),
    ("scene", dict(name="scene", version=1)),
    ("Bioresponse", dict(name="Bioresponse", version=1)),
    ("hill-valley", dict(name="hill-valley", version=1)),
    ("isolet", dict(name="isolet", version=1)),
    ("arcene", dict(name="arcene", version=1)),
)

_ENV_VAR = "MLFRAME_REALDATA_CACHE"

PathLike = Union[str, "os.PathLike[str]"]


def cache_dir_default() -> Path:
    """Resolve the cache directory: `$MLFRAME_REALDATA_CACHE` when set, else `<repo root>/.cache/realdata`."""
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env)
    # this file lives at <repo>/src/mlframe/feature_selection/_benchmarks/fs_hybrid/_realdata_cache.py
    return Path(__file__).resolve().parents[5] / ".cache" / "realdata"


def _resolve(cache_dir: Optional[PathLike]) -> Path:
    """Return `cache_dir` as a Path, falling back to `cache_dir_default()` when it is None."""
    return cache_dir_default() if cache_dir is None else Path(cache_dir)


def _sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in 1 MiB chunks so a wide bed need not fit in memory twice."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_dense_frame(data: Any) -> pd.DataFrame:
    """Normalise whatever `fetch_openml` returned (frame, ndarray or scipy sparse matrix) to a dense DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data
    if hasattr(data, "toarray"):  # scipy sparse, from the as_frame=False sparse-ARFF path
        data = data.toarray()
    return pd.DataFrame(np.asarray(data))


def _to_float_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Coerce a fetched feature frame to a 2-D float array, downcasting to float32 only when that is lossless."""
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    arr = np.asarray(numeric.to_numpy(dtype=np.float64))
    arr = np.asarray(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)
    as32 = arr.astype(np.float32)
    # a float32 store halves the cache and every later read; only take it when the round trip is exact
    return as32 if np.array_equal(as32.astype(np.float64), arr) else arr


def _encode_target(target: Any) -> Tuple[np.ndarray, int]:
    """Factorize a fetched target into compact integer codes; returns (codes, n_classes)."""
    codes = pd.factorize(np.asarray(target).ravel())[0]
    n_classes = int(codes.max()) + 1 if codes.size else 0
    dtype = np.int8 if n_classes <= 127 else np.int32
    return codes.astype(dtype), n_classes


def _positive_rate(codes: np.ndarray) -> float:
    """Share of rows in the majority class -- the binarisation the bench itself applies in `_clean_xy`."""
    if codes.size == 0:
        return float("nan")
    counts = np.bincount(codes.astype(np.int64))
    return float(counts.max() / codes.size)


def fill_cache(
    sets: Sequence[Tuple[str, Dict[str, Any]]] = OPENML_SETS,
    cache_dir: Optional[PathLike] = None,
    overwrite: bool = False,
) -> List[Dict[str, Any]]:
    """Download each dataset via `fetch_openml` and write it to the local cache. The only networked entry point.

    Returns one status dict per requested dataset: `{name, ok, ...}` -- on success also the sidecar metadata plus
    `bytes`; on failure `error` with the exception type and message. A single failure never aborts the rest.
    """
    import sklearn
    from sklearn.datasets import fetch_openml

    root = _resolve(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    report: List[Dict[str, Any]] = []

    for name, kwargs in sets:
        npz_path = root / f"{name}.npz"
        json_path = root / f"{name}.json"
        if npz_path.exists() and json_path.exists() and not overwrite:
            cached_meta = json.loads(json_path.read_text(encoding="utf-8"))
            logger.info("cache hit for %s, skipping download (pass overwrite=True to refetch)", name)
            report.append(dict(cached_meta, ok=True, skipped=True, bytes=npz_path.stat().st_size))
            continue
        try:
            try:
                bundle = fetch_openml(as_frame=True, parser="auto", **kwargs)
            except ValueError as frame_exc:
                # sparse-ARFF beds (gisette v1) reject as_frame=True outright; they are dense enough to densify here
                if "sparse arff" not in str(frame_exc).lower():
                    raise
                logger.info("%s is a sparse ARFF bed, refetching with as_frame=False and densifying", name)
                bundle = fetch_openml(as_frame=False, parser="auto", **kwargs)
            frame = _as_dense_frame(bundle.data)
            matrix = _to_float_matrix(frame)
            codes, n_classes = _encode_target(bundle.target)
            columns = np.asarray([str(c) for c in frame.columns], dtype=np.str_)
            staging = npz_path.with_name(f"{name}.staging.npz")  # np.savez_compressed appends .npz to a name lacking it
            np.savez_compressed(staging, X=matrix, y=codes, columns=columns)
            os.replace(staging, npz_path)
            meta = dict(
                name=name,
                openml_name=str(kwargs.get("name", name)),
                version=int(kwargs.get("version", 0)),
                n_rows=int(matrix.shape[0]),
                n_cols=int(matrix.shape[1]),
                n_classes=n_classes,
                positive_rate=round(_positive_rate(codes), 6),
                sha256=_sha256_of(npz_path),
                fetched_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                sklearn_version=sklearn.__version__,
            )
            json_path.write_bytes(json.dumps(meta, indent=2, sort_keys=True).encode("utf-8"))
            logger.info("cached %s: shape=%s bytes=%s", name, matrix.shape, npz_path.stat().st_size)
            report.append(dict(meta, ok=True, skipped=False, bytes=npz_path.stat().st_size))
        except Exception as exc:
            logger.warning("failed to cache %s: %s: %s", name, type(exc).__name__, exc)
            report.append(dict(name=name, ok=False, error=f"{type(exc).__name__}: {exc}"))
    return report


def load_cached(name: str, cache_dir: Optional[PathLike] = None) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """Load one cached dataset as `(X, y, meta)`, verifying its SHA-256 first. Never touches the network.

    Raises `FileNotFoundError` when the pair of cache files is absent and `ValueError` when the payload no longer
    matches the recorded digest; both messages name `fill_cache` as the remedy.
    """
    root = _resolve(cache_dir)
    npz_path = root / f"{name}.npz"
    json_path = root / f"{name}.json"
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"real-data cache miss for {name!r} in {root} (expected {npz_path.name} + {json_path.name}); "
            "run mlframe.feature_selection._benchmarks.fs_hybrid._realdata_cache.fill_cache() once, with network access"
        )
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    actual = _sha256_of(npz_path)
    expected = str(meta.get("sha256", ""))
    if actual != expected:
        raise ValueError(
            f"real-data cache for {name!r} is corrupt: sha256 of {npz_path} is {actual} but {json_path.name} records "
            f"{expected}; delete both files and re-run fill_cache()"
        )
    with np.load(npz_path, allow_pickle=False) as payload:
        matrix = payload["X"]
        y = payload["y"]
        columns = [str(c) for c in payload["columns"]]
    return pd.DataFrame(matrix, columns=columns), np.asarray(y), meta


def available(cache_dir: Optional[PathLike] = None) -> List[Dict[str, Any]]:
    """List the datasets present in the cache: one sidecar-metadata dict per complete `.npz` + `.json` pair."""
    root = _resolve(cache_dir)
    if not root.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for json_path in sorted(root.glob("*.json")):
        npz_path = json_path.with_suffix(".npz")
        if not npz_path.exists():
            continue
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("unreadable cache sidecar %s: %s: %s", json_path, type(exc).__name__, exc)
            continue
        meta["bytes"] = npz_path.stat().st_size
        out.append(meta)
    return out


def _main(argv: Optional[Iterable[str]] = None) -> int:
    """CLI entry point: `python _realdata_cache.py [name ...]` fills the cache and prints one report line per set."""
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    names = list(argv) if argv is not None else sys.argv[1:]
    sets = tuple((nm, kw) for nm, kw in OPENML_SETS if not names or nm in names)
    for row in fill_cache(sets=sets):
        if row.get("ok"):
            print(
                f"OK   {row['name']:<14} rows={row.get('n_rows')} cols={row.get('n_cols')} "
                f"classes={row.get('n_classes')} pos_rate={row.get('positive_rate')} bytes={row.get('bytes')}",
                flush=True,
            )
        else:
            print(f"FAIL {row['name']:<14} {row.get('error')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
