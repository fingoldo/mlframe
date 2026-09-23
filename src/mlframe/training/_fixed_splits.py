"""Exact, reusable train/val/test splits keyed on a stable row id or on date windows.

Two halves:

* ``record_split_membership`` writes ``split_ids.parquet`` ([id_column, "split"]) + a ``split_ids.json`` sidecar for
  every run with ``TrainingSplitConfig.id_column`` set, so a later run can replay the exact holdouts.
* ``pinned_train_val_test_split`` builds a split where some splits are fixed by that file (``split_ids_path`` +
  ``reuse_splits``) or by half-open date windows (``test_start``/``test_end``/``val_start``/``val_end``). Whatever is
  not pinned is carved from the remaining rows by the regular ``make_train_test_split`` via index mapping, so the
  fraction machinery (shuffle_val, val_placement, calib_size, groups, stratification) is reused, never duplicated.

Rules (also documented on the config fields):

* ids must be unique and non-null in the frame, and unique in the membership file -> ValueError otherwise;
* ids listed for a reused split but absent from the frame -> WARNING with the per-split count; a reused split that
  matches zero rows -> ValueError (nothing would be pinned, almost always a wrong file / id dtype);
* leakage guard (``train_before_holdout``): pool rows with timestamp >= the earliest pinned holdout start are
  excluded from train. A file-pinned split's start is its SEQUENTIAL block start recorded in the sidecar (rows of
  the split newer than every earlier-split row), so a shuffled val's random part never moves the cutoff; without a
  sidecar the pinned test's min timestamp is used. A window's start is the window start (or the min pinned
  timestamp when only an end is given);
* a split pinned by both a file and a window, or a row claimed by two pinned splits -> ValueError.
"""

from __future__ import annotations

import logging
import os
from os.path import join
from typing import Any, Callable, Optional, cast

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

SPLIT_CODES = {"train": 0, "val": 1, "test": 2, "calib": 3}
SPLIT_IDS_FILENAME = "split_ids.parquet"
SPLIT_IDS_SIDECAR = "split_ids.json"
_WINDOW_FIELDS = ("test_start", "test_end", "val_start", "val_end")


def has_pinned_splits(split_config: Any) -> bool:
    """True when the config pins at least one split by an id file or a date window."""
    if split_config is None:
        return False
    if getattr(split_config, "split_ids_path", None):
        return True
    return any(getattr(split_config, f, None) is not None for f in _WINDOW_FIELDS)


def split_id_columns_from_metadata(metadata: Any) -> list:
    """Id column recorded at train time, to drop it from a predict frame (it is never a model feature)."""
    if not isinstance(metadata, dict):
        return []
    _m = metadata.get("split_membership")
    _c = _m.get("id_column") if isinstance(_m, dict) else None
    return [_c] if _c else []


def extract_row_ids(df: Any, id_column: str) -> np.ndarray:
    """Read the row-key column from ``df`` and validate it is a usable key (present, non-null, unique)."""
    _cols = list(df.columns)
    if id_column not in _cols:
        raise ValueError(
            f"TrainingSplitConfig.id_column={id_column!r} is not a column of the frame after the features/targets "
            f"extractor ({len(_cols)} columns). Keep the id column in the extractor output (it is dropped from the "
            "features automatically) or fix the name."
        )
    ids = df[id_column].to_numpy()
    _s = pd.Series(ids)
    _n_null = int(_s.isna().sum())
    if _n_null:
        raise ValueError(f"id_column {id_column!r} has {_n_null} null value(s); a split key must identify every row.")
    _dup = _s.duplicated(keep=False)
    if bool(_dup.any()):
        _examples = _s[_dup].drop_duplicates().head(5).tolist()
        raise ValueError(
            f"id_column {id_column!r} is not unique: {int(_dup.sum())} rows share {int(_s[_dup].nunique())} ids "
            f"(e.g. {_examples}). A duplicated key cannot pin one row to one split; deduplicate or use a finer key."
        )
    return cast(np.ndarray, ids)


def _ts_series(timestamps: Any) -> Optional[pd.Series]:
    """Positional pandas Series view of the suite timestamps (index reset so .iloc/.to_numpy align with row idx)."""
    if timestamps is None:
        return None
    if pl is not None and isinstance(timestamps, pl.Series):
        timestamps = timestamps.to_pandas()
    if isinstance(timestamps, pd.Series):
        return timestamps.reset_index(drop=True)
    return pd.Series(timestamps)


def _is_datetime(ts: pd.Series) -> bool:
    """True when ``ts`` has a datetime64 dtype (tz-aware or naive)."""
    return cast(bool, pd.api.types.is_datetime64_any_dtype(ts.dtype))


def _align_bound(bound: Any, ts: pd.Series) -> Any:
    """Make a window bound comparable with ``ts`` (tz-aware vs naive)."""
    if bound is None:
        return None
    bound = pd.Timestamp(bound)
    _tz = getattr(ts.dt, "tz", None)
    if _tz is not None and bound.tz is None:
        return bound.tz_localize(_tz)
    if _tz is None and bound.tz is not None:
        return bound.tz_convert(None)
    return bound


def _to_jsonable(v: Any) -> Any:
    """Convert a scalar (timestamp, numpy number, NaN) into a JSON-serialisable value; missing becomes None."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(v).isoformat()
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v


def _from_jsonable(v: Any, ts: Optional[pd.Series]) -> Any:
    """Inverse of ``_to_jsonable``: parse a stored bound back into a timestamp aligned with ``ts`` when it is datetime."""
    if v is None:
        return None
    if ts is not None and _is_datetime(ts):
        return _align_bound(pd.Timestamp(v), ts)
    return v


def _sequential_start(ts: Optional[pd.Series], split_idx: np.ndarray, earlier_idx: np.ndarray) -> Any:
    """First timestamp of the split's newest contiguous block: its rows newer than every row of the earlier splits."""
    if ts is None or len(split_idx) == 0:
        return None
    _vals = ts.iloc[split_idx]
    if len(earlier_idx):
        _m = ts.iloc[earlier_idx].max()
        if not pd.isna(_m):
            _vals = _vals[_vals > _m]
    if len(_vals) == 0:
        return None
    _v = _vals.min()
    return None if pd.isna(_v) else _v


def _write_json(path: str, payload: dict) -> None:
    """Write ``payload`` as sorted, indented JSON, via orjson when installed else the stdlib."""
    try:
        import orjson

        with open(path, "wb") as f:
            f.write(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2))
    except ImportError:
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True, indent=2)


def _read_json(path: str) -> Optional[dict]:
    """Read a JSON sidecar, returning None when the file does not exist."""
    if not os.path.exists(path):
        return None
    try:
        import orjson

        with open(path, "rb") as f:
            return cast(Optional[dict], orjson.loads(f.read()))
    except ImportError:
        import json

        with open(path, "r", encoding="utf-8") as f:
            return cast(Optional[dict], json.load(f))


def split_dir_for(data_dir: Optional[str], models_dir: Optional[str], target_name: str, model_name: str) -> Optional[str]:
    """Same per-(target, model) directory ``save_split_artifacts`` writes the other split artifacts to."""
    if not data_dir or not models_dir:
        return None
    from pyutilz.strings import slugify

    return join(data_dir, models_dir, slugify(target_name), slugify(model_name))


def sidecar_path_for(parquet_path: str) -> str:
    """``split_ids.json`` next to a membership parquet (same stem)."""
    _root, _ = os.path.splitext(parquet_path)
    return _root + ".json"


def record_split_membership(
    *,
    row_ids: np.ndarray,
    id_column: str,
    train_idx: Any,
    val_idx: Any,
    test_idx: Any,
    calib_idx: Any,
    timestamps: Any,
    split_dir: Optional[str],
) -> dict:
    """Persist the realised split membership and return the ``metadata["split_membership"]`` entry.

    Only the path + counts go into metadata (not the id arrays: millions of ids would bloat every pickled metadata);
    without ``split_dir`` nothing is written and ``path`` is None.
    """
    ts = _ts_series(timestamps)
    _parts = {k: (np.asarray(v, dtype=np.int64) if v is not None else np.array([], dtype=np.int64)) for k, v in
              (("train", train_idx), ("val", val_idx), ("test", test_idx), ("calib", calib_idx))}
    counts = {k: len(v) for k, v in _parts.items()}
    _train_like = np.concatenate([_parts["train"], _parts["calib"]])
    holdout_starts = {
        "val": _to_jsonable(_sequential_start(ts, _parts["val"], _train_like)),
        "test": _to_jsonable(_sequential_start(ts, _parts["test"], np.concatenate([_train_like, _parts["val"]]))),
    }
    entry = {"id_column": id_column, "path": None, "counts": counts, "holdout_starts": holdout_starts}
    if not split_dir:
        logger.warning(
            "TrainingSplitConfig.id_column=%r is set but no data_dir was given, so split_ids.parquet was not written; "
            "pass data_dir to make this split reusable.", id_column,
        )
        return entry
    os.makedirs(split_dir, exist_ok=True)
    _pos = np.concatenate(list(_parts.values()))
    _lab = np.concatenate([np.full(len(v), k, dtype=object) for k, v in _parts.items()])
    _order = np.argsort(_pos, kind="stable")
    _frame = pd.DataFrame({id_column: np.asarray(row_ids)[_pos[_order]], "split": _lab[_order]})
    path = join(split_dir, SPLIT_IDS_FILENAME)
    _frame.to_parquet(path, index=False, compression="zstd")
    _write_json(sidecar_path_for(path), {"id_column": id_column, "counts": counts, "holdout_starts": holdout_starts})
    entry["path"] = path
    logger.info("Split membership (%s) written to %s: %s.", id_column, path, counts)
    return entry


def load_split_ids(path: str, id_column: str) -> tuple:
    """Read a membership file -> (ids ndarray, split-label ndarray, sidecar dict or None); validates its content."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"split_ids_path {path!r} does not exist.")
    _df = pd.read_parquet(path)
    for _c in (id_column, "split"):
        if _c not in _df.columns:
            raise ValueError(f"split_ids_path {path!r} has no column {_c!r} (columns: {list(_df.columns)}).")
    _bad = sorted(set(_df["split"].astype(str).unique()) - set(SPLIT_CODES))
    if _bad:
        raise ValueError(f"split_ids_path {path!r} has unknown split labels {_bad}.")
    _dup = _df[id_column].duplicated()
    if bool(_dup.any()):
        raise ValueError(f"split_ids_path {path!r} lists {int(_dup.sum())} id(s) more than once; each id must belong to one split.")
    return _df[id_column].to_numpy(), _df["split"].astype(str).to_numpy(), _read_json(sidecar_path_for(path))


def _details(ts: Optional[pd.Series], idx: np.ndarray, tag: str) -> str:
    """Human-readable date-range summary of rows ``idx`` of ``ts``, suffixed with ``[tag]`` when given."""
    from ._splitting_helpers import _build_details

    _d = _build_details(ts, idx, None, 0, "") if ts is not None else ""
    return f"{_d} [{tag}]" if tag else _d


def _pool_relative_sizes(test_size: float, val_size: float, calib_size, *, n_rows: int, n_pool: int, n_pinned_test: int):
    """Configured whole-frame split fractions, rescaled to the unpinned pool the splitter actually receives.

    The splitter reads ``test_size`` / ``calib_size`` as fractions of the frame it is given and ``val_size`` as a fraction
    of what is left after test. Handing it the configured values with only the unpinned pool made every carve a share of
    that pool: with test pinned to 20% of the rows, ``val_size=0.2`` produced a 16% val, and nothing reported it. The
    returned sizes reproduce the configured whole-frame counts. When the pool is too small for that they cannot, and the
    pool-relative values are kept with a warning.
    """
    if n_pool <= 0 or n_pool == n_rows:
        return test_size, val_size, calib_size
    scale = n_rows / n_pool
    t = test_size * scale
    n_test_total = n_pinned_test + test_size * n_rows
    left_after_test = n_pool * (1.0 - t)
    v = val_size * (n_rows - n_test_total) / left_after_test if left_after_test > 0 else 1.0
    c = None if calib_size is None else calib_size * scale
    if t + (c or 0.0) >= 1.0 or v >= 1.0:
        logger.warning(
            "Pinned split: the %d unpinned rows cannot hold test_size=%s, val_size=%s, calib_size=%s of all %d rows; those "
            "fractions are applied to the unpinned rows instead.", n_pool, test_size, val_size, calib_size, n_rows,
        )
        return test_size, val_size, calib_size
    logger.info(
        "Pinned split: configured test/val/calib fractions %s/%s/%s of all %d rows are %.4f/%.4f/%s of the %d unpinned rows.",
        test_size, val_size, calib_size, n_rows, t, v, "None" if c is None else f"{c:.4f}", n_pool,
    )
    return t, v, c


def pinned_train_val_test_split(
    *,
    n_rows: int,
    row_ids: Optional[np.ndarray],
    timestamps: Any,
    split_config: Any,
    stratify_y: Any,
    groups: Any,
    splitter: Callable,
    splitter_kwargs: dict,
) -> tuple:
    """Split with some splits pinned by ids/windows; returns make_train_test_split's 8-tuple plus an info dict."""
    ts = _ts_series(timestamps)
    labels = np.full(n_rows, -1, dtype=np.int8)
    source: dict = {}
    starts: dict = {}
    info: dict = {"missing_ids": {}}

    _path = getattr(split_config, "split_ids_path", None)
    if _path:
        _id_col = split_config.id_column
        if row_ids is None:
            raise ValueError("split_ids_path is set but the row ids were not extracted (id_column missing from the frame).")
        _fids, _fsplits, _sidecar = load_split_ids(_path, _id_col)
        _row_ids = pd.Series(row_ids)
        _notes: list = []  # one WARNING after the loop instead of one per reused split
        for _s in split_config.reuse_splits:
            _sel = _fids[_fsplits == _s]
            if len(_sel) == 0:
                _notes.append(f"it lists no {_s!r} rows, so {_s!r} is carved normally instead")
                continue
            _mask = _row_ids.isin(pd.Index(_sel)).to_numpy()
            _matched = int(_mask.sum())
            if _matched == 0:
                raise ValueError(
                    f"None of the {len(_sel)} {_s!r} ids in {_path!r} occur in the frame's {_id_col!r} column "
                    f"(frame dtype {_row_ids.dtype}, file dtype {_sel.dtype}); wrong file or id dtype mismatch."
                )
            if _matched < len(_sel):
                info["missing_ids"][_s] = len(_sel) - _matched
                _notes.append(f"{len(_sel) - _matched} of {len(_sel)} {_s!r} ids are absent from the current frame; the pinned {_s} set has {_matched} rows")
            labels[_mask] = SPLIT_CODES[_s]
            source[_s] = "file"
            if _s in ("val", "test") and ts is not None:
                _recorded = ((_sidecar or {}).get("holdout_starts") or {}).get(_s) if _sidecar else None
                if _sidecar is not None:
                    starts[_s] = _from_jsonable(_recorded, ts)
                elif _s == "test":
                    starts[_s] = ts[_mask].min()
                    _notes.append(f"it has no {SPLIT_IDS_SIDECAR} sidecar, so the pinned test min timestamp {starts[_s]} is the train cutoff")
        if _notes:
            logger.warning("split_ids_path %s: %s.", _path, "; ".join(_notes))

    for _s in ("test", "val"):
        _lo, _hi = getattr(split_config, f"{_s}_start", None), getattr(split_config, f"{_s}_end", None)
        if _lo is None and _hi is None:
            continue
        if _s in source:
            raise ValueError(f"split {_s!r} is pinned both by split_ids_path and by a date window.")
        if ts is None:
            raise ValueError(f"{_s}_start/{_s}_end need timestamps, but the features/targets extractor produced none.")
        if not _is_datetime(ts):
            raise ValueError(f"{_s}_start/{_s}_end need datetime timestamps; got dtype {ts.dtype}.")
        _lo, _hi = _align_bound(_lo, ts), _align_bound(_hi, ts)
        _mask = ts.notna().to_numpy().copy()
        if _lo is not None:
            _mask &= (ts >= _lo).to_numpy()
        if _hi is not None:
            _mask &= (ts < _hi).to_numpy()
        if not _mask.any():
            raise ValueError(f"{_s} window [{_lo}, {_hi}) selects 0 rows (timestamps span {ts.min()} .. {ts.max()}).")
        _clash = _mask & (labels != -1)
        if _clash.any():
            raise ValueError(f"{int(_clash.sum())} rows in the {_s} window are already pinned to another split by split_ids_path.")
        labels[_mask] = SPLIT_CODES[_s]
        source[_s] = "window"
        starts[_s] = _lo if _lo is not None else ts[_mask].min()

    unassigned = labels == -1
    _starts = [v for v in starts.values() if v is not None and not pd.isna(v)]
    cutoff = min(_starts) if _starts else None
    info["cutoff"] = _to_jsonable(cutoff)
    n_newer = 0
    if cutoff is not None and getattr(split_config, "train_before_holdout", True):
        _newer = unassigned & (ts >= cutoff).to_numpy()
        n_newer = int(_newer.sum())
        if n_newer:
            logger.info("Pinned split: %d unpinned rows with timestamp >= %s (earliest pinned holdout start) excluded from "
                        "train to avoid training on the holdout period (train_before_holdout=True).", n_newer, cutoff)
        unassigned &= ~_newer
    info["n_excluded_after_cutoff"] = n_newer
    pool = np.flatnonzero(unassigned)

    _test_sz = 0.0 if "test" in source else float(split_config.test_size)
    _val_sz = 0.0 if "val" in source else float(split_config.val_size)
    _calib_sz = None if ("calib" in source or "train" in source) else getattr(split_config, "calib_size", None)
    details = {"train": "", "val": "", "test": "", "calib": ""}
    if len(pool) and (_test_sz > 0 or _val_sz > 0 or (_calib_sz or 0) > 0):
        _kw = dict(splitter_kwargs)
        _kw.update(zip(("test_size", "val_size", "calib_size"), _pool_relative_sizes(_test_sz, _val_sz, _calib_sz, n_rows=n_rows, n_pool=len(pool), n_pinned_test=int((labels == SPLIT_CODES["test"]).sum()))))
        _sub = lambda a: None if a is None else np.asarray(a)[pool]  # noqa: E731
        _tr, _va, _te, details["train"], details["val"], details["test"], _ca, details["calib"] = splitter(
            df=pd.DataFrame(index=pd.RangeIndex(len(pool))),
            timestamps=None if ts is None else ts.iloc[pool].reset_index(drop=True),
            stratify_y=_sub(stratify_y),
            groups=_sub(groups),
            return_calib=True,
            **_kw,
        )
        for _code, _local in ((0, _tr), (1, _va), (2, _te), (3, _ca)):
            if _local is not None and len(_local):
                labels[pool[np.asarray(_local)]] = _code
    elif len(pool):
        labels[pool] = 0
    if "train" in source:
        # The pinned train is exact: pool rows the splitter left in train are not added to it.
        _extra = unassigned & (labels == 0)
        if _extra.any():
            logger.info("Pinned split: train is pinned by split_ids_path; %d unpinned rows excluded.", int(_extra.sum()))
            labels[_extra] = -1
    idx = {k: np.flatnonzero(labels == c) for k, c in SPLIT_CODES.items()}
    if "train" in source and "calib" not in source and (getattr(split_config, "calib_size", None) or 0) > 0:
        from ._split_helpers import _carve_calib_from_train

        _tr, _ca = _carve_calib_from_train(idx["train"], float(split_config.calib_size), n_total=n_rows,
                                           timestamps=ts, groups=groups, rng=np.random.default_rng(split_config.random_seed))
        idx["train"], idx["calib"] = np.sort(_tr), np.sort(_ca)
        details["calib"] = f"{len(_ca)} calib rows"
    if len(idx["train"]) == 0:
        raise ValueError(f"Pinned split left 0 train rows (pinned: {source}, excluded after cutoff: {n_newer}).")

    for _s, _src in source.items():
        details[_s] = _details(ts, idx[_s], f"pinned:{_src}")
    if not details["train"] and ts is not None:
        details["train"] = _details(ts, idx["train"], "")

    if groups is not None:
        _g = np.asarray(groups)
        _tr_groups = np.unique(_g[np.concatenate([idx["train"], idx["calib"]])])
        for _s in ("val", "test"):
            if _s in source and len(idx[_s]):
                _n_span = len(np.intersect1d(_tr_groups, np.unique(_g[idx[_s]])))
                if _n_span:
                    logger.warning("Pinned split: %d group(s) have rows in both train and the pinned %s set; rows were NOT moved "
                                   "(the pinned membership wins). Expect group leakage in %s metrics.", _n_span, _s, _s)
                info.setdefault("groups_spanning", {})[_s] = _n_span

    info["sources"] = dict(source)
    logger.info(
        "%d train rows %s, %d val rows %s, %d test rows %s%s (split pinned: %s).",
        len(idx["train"]), details["train"], len(idx["val"]), details["val"], len(idx["test"]), details["test"],
        f", {len(idx['calib'])} calib rows" if len(idx["calib"]) else "",
        ", ".join(f"{k}={v}" for k, v in source.items()),
    )
    return (idx["train"], idx["val"], idx["test"], details["train"], details["val"], details["test"], idx["calib"], details["calib"], info)


__all__ = [
    "SPLIT_IDS_FILENAME",
    "extract_row_ids",
    "has_pinned_splits",
    "load_split_ids",
    "pinned_train_val_test_split",
    "record_split_membership",
    "split_id_columns_from_metadata",
]
