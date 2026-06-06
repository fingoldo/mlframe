"""Append-only parquet store + canonical-JSON helpers for ParamOracle.

Carved from ``_param_oracle.py`` so the parent stays under the LOC ceiling.
The store mirrors KernelTuningCache's concurrency-safe merge-on-write; the
sort-keys JSON serialiser keys oracle rows. Re-exported from the parent.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Sequence

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


_STORE_COLUMNS = (
    "schema_version", "fn_name", "host", "fp_bucket_json",
    "param_combo_json", "objective_json", "n_obs", "ts",
)


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


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
        self._path = store_path
        os.makedirs(os.path.dirname(os.path.abspath(store_path)), exist_ok=True)

    # ---- low-level read ----

    def read_rows(self) -> list[dict]:
        if not os.path.isfile(self._path):
            return []
        try:
            import pyarrow.parquet as pq
            tbl = pq.read_table(self._path)
            return tbl.to_pylist()
        except Exception as e:
            logger.warning("param_oracle: failed to read %s: %s", self._path, e)
            return []

    # ---- low-level write ----

    def _write_rows(self, rows: list[dict], dest: str) -> None:
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
            metric_vals: dict[str, list[float]] = {}
            total_obs = 0
            latest_ts = ""
            for r in grp:
                total_obs += int(r.get("n_obs", 1) or 1)
                ts = str(r.get("ts", ""))
                if ts > latest_ts:
                    latest_ts = ts
                try:
                    obj = json.loads(r.get("objective_json") or "{}")
                except Exception:
                    obj = {}
                w = int(r.get("n_obs", 1) or 1)
                for mk, mv in obj.items():
                    if isinstance(mv, (int, float)) and not isinstance(mv, bool):
                        # Weight each pre-aggregated row by its obs count so
                        # median reflects underlying observations.
                        metric_vals.setdefault(mk, []).extend([float(mv)] * max(1, w))
            median_obj = {mk: _median(vs) for mk, vs in metric_vals.items()}
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


def _median(vals: Sequence[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)
