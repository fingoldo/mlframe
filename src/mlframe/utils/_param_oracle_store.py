"""Append-only parquet store + canonical-JSON helpers for ParamOracle.

Carved from ``_param_oracle.py`` so the parent stays under the LOC ceiling.
The store mirrors KernelTuningCache's concurrency-safe merge-on-write; the
sort-keys JSON serialiser keys oracle rows. Re-exported from the parent.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Sequence

import orjson

from .log_throttle import log_throttle

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


_STORE_COLUMNS = (
    "schema_version", "fn_name", "host", "fp_bucket_json",
    "param_combo_json", "objective_json", "n_obs", "ts",
)


def _stable_json(obj: Any) -> str:
    """Canonical sort-keys JSON encoding of ``obj``."""
    return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS, default=str).decode("utf-8")


def stable_json(obj: Any) -> str:
    """Public alias for the canonical sort-keys JSON serialiser used to key oracle rows; cross-package consumers import this instead of reaching into the private name."""
    return _stable_json(obj)


class _ParquetStore:
    """Append-only parquet store with concurrency-safe writes.

    Mirrors ``KernelTuningCache._save``: writes go to a per-PROCESS temp
    shard then merge into the canonical store under a ``filelock`` advisory
    lock with an atomic ``os.replace``. If ``filelock`` is unavailable we
    still serialise within-process via a re-read-merge-write but accept the
    cross-process race (documented, same degradation policy as the kernel
    cache).

    Aggregation: rows with the same ``(fn_name, host, fp_bucket_json,
    param_combo_json)`` are folded to a single row whose ``objective_json``
    holds the per-key MEDIAN and ``n_obs`` the total observation count.
    """

    def __init__(self, store_path: str):
        """Bind the store to ``store_path``, creating its parent directory."""
        self._path = store_path
        os.makedirs(os.path.dirname(os.path.abspath(store_path)), exist_ok=True)

    # ---- low-level read ----

    def read_rows(self) -> list[dict]:
        """Read all rows from the parquet store, or ``[]`` if it doesn't exist yet or fails to parse."""
        if not os.path.isfile(self._path):
            return []
        try:
            import pyarrow.parquet as pq
            tbl = pq.read_table(self._path)
            return list(tbl.to_pylist())
        except Exception as e:
            logger.warning("param_oracle: failed to read %s: %s", self._path, e)
            return []

    # ---- low-level write ----

    def _write_rows(self, rows: list[dict], dest: str) -> None:
        """Atomically write ``rows`` to ``dest`` (temp file + ``os.replace``)."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        cols = {c: [r.get(c) for r in rows] for c in _STORE_COLUMNS}
        tbl = pa.table(cols)
        tmp = dest + f".{uuid.uuid4().hex}.tmp"
        pq.write_table(tbl, tmp)
        os.replace(tmp, dest)

    def append(self, rows: list[dict]) -> None:
        """Append observation rows, then re-aggregate, concurrency-safely."""
        if not rows:
            return
        lock_path = self._path + ".lock"
        try:
            from filelock import FileLock
            lock = FileLock(lock_path, timeout=30)
        except ImportError:
            lock = None

        def _do() -> None:
            """Read-aggregate-write cycle, run inside the cross-process file lock (when available) so concurrent appenders don't clobber each other's writes."""
            existing = self.read_rows()
            merged = self._aggregate(existing + rows)
            self._write_rows(merged, self._path)

        if lock is not None:
            with lock:
                _do()
        else:
            _do()

    # ---- aggregation ----

    @staticmethod
    def _aggregate(rows: list[dict]) -> list[dict]:
        """Fold rows on (fn_name, host, fp_bucket_json, param_combo_json),
        median-aggregating each objective metric and summing n_obs."""
        from collections import defaultdict
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            key = (
                r.get("fn_name"), r.get("host"),
                r.get("fp_bucket_json"), r.get("param_combo_json"),
            )
            groups[key].append(r)

        out: list[dict] = []
        for (fn_name, host, fpj, pcj), grp in groups.items():
            # Collect every objective metric across the group's observations.
            metric_vals: dict[str, list[tuple[float, int]]] = {}
            total_obs = 0
            latest_ts = ""
            for r in grp:
                total_obs += int(r.get("n_obs", 1) or 1)
                ts = str(r.get("ts", ""))
                if ts > latest_ts:
                    latest_ts = ts
                try:
                    obj = orjson.loads(r.get("objective_json") or "{}")
                except Exception as e:
                    log_throttle(
                        logger,
                        "param_oracle_corrupt_objective_json",
                        logging.WARNING,
                        "param_oracle: corrupt objective_json on row (fn_name=%s, host=%s, param_combo_json=%s): %s",
                        r.get("fn_name"), r.get("host"), r.get("param_combo_json"), e,
                    )
                    obj = {}
                w = int(r.get("n_obs", 1) or 1)
                for mk, mv in obj.items():
                    if isinstance(mv, (int, float)) and not isinstance(mv, bool):
                        # Carry the (value, weight) PAIR rather than replicating the value `n_obs` times.
                        # `n_obs` accumulates monotonically across every append and `_aggregate` runs over all
                        # existing rows on each one, so a row whose count had reached ~500k built a 500k-element
                        # Python float list per metric per row -- gigabytes across a few dozen keys, and a
                        # multi-second stall INSIDE the cross-process file lock, blocking every other process
                        # appending to the same store.
                        metric_vals.setdefault(mk, []).append((float(mv), max(1, w)))
            median_obj = {mk: _weighted_median(vs) for mk, vs in metric_vals.items()}
            out.append({
                "schema_version": SCHEMA_VERSION,
                "fn_name": fn_name,
                "host": host,
                "fp_bucket_json": fpj,
                "param_combo_json": pcj,
                "objective_json": _stable_json(median_obj),
                "n_obs": total_obs,
                "ts": latest_ts,
            })
        return out


def _weighted_median(pairs: "list[tuple[float, int]]") -> float:
    """Median of ``value`` weighted by integer ``weight``, computed from the pairs without expanding them.

    Same result as taking the median of the expanded list, in O(k log k) on the number of DISTINCT rows rather
    than O(sum of weights) time AND memory.
    """
    if not pairs:
        return float("nan")
    ordered = sorted(pairs, key=lambda t: t[0])
    total = sum(w for _, w in ordered)
    half = total / 2.0
    seen = 0
    for value, weight in ordered:
        seen += weight
        if seen >= half:
            return float(value)
    return float(ordered[-1][0])


def _median(vals: Sequence[float]) -> float:
    """Median of ``vals``, or ``nan`` if empty."""
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)
