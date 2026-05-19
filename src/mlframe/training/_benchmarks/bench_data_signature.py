"""Benchmark for ``data_signature`` (composite_cache.py).

Audit D P0-1/P0-2 (2026-05-18) replaced the per-column full-frame ``to_numpy()`` materialisation
with a single polars ``select`` for min/max/null across all columns + ``col.gather(sample_idx)``
for the per-column sample bytes. This benchmark compares the legacy per-col-materialisation path
against the current ``data_signature`` implementation on a synthetic 200-column × 1M-row frame.

Run:

    python -m mlframe.training._benchmarks.bench_data_signature

The legacy path is re-implemented inline here (the production code no longer contains it) so the
benchmark can compare; the production implementation is imported from ``composite_cache``.
"""

from __future__ import annotations

import hashlib
import time
from typing import Sequence

import numpy as np

try:
    import polars as pl
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"polars unavailable: {exc}")

from mlframe.training.composite_cache import data_signature


def _legacy_data_signature(
    df: pl.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    *,
    sample_n: int = 1000,
    random_state: int = 42,
) -> str:
    """Pre-fix path: full-column ``to_numpy()`` per column, then index into the sample.

    Kept inline only for the speedup comparison; production code never used to call this directly
    -- the inline loop sat inside ``data_signature``. We re-create the byte format closely enough
    for an apples-to-apples wall-clock comparison; digest values are intentionally not asserted
    equal because that is precisely the perf regression being measured.
    """
    n_rows = len(df)
    rng = np.random.default_rng(random_state)
    sample_n_eff = min(n_rows, int(sample_n))
    sample_idx = np.sort(rng.choice(n_rows, size=sample_n_eff, replace=False))
    h = hashlib.blake2b(digest_size=16)
    h.update(b"nrows=")
    h.update(str(int(n_rows)).encode("utf-8"))
    h.update(target_col.encode("utf-8"))
    for c in feature_cols:
        h.update(b"|")
        h.update(str(c).encode("utf-8"))
    for c in [target_col] + [c for c in feature_cols if c != target_col]:
        if c not in df.columns:
            continue
        col = df.get_column(c)
        # ★ The expensive operation we eliminated: full-column to_numpy.
        full = col.to_numpy()
        h.update(str(df.schema[c]).encode("utf-8"))
        h.update(b"|stats=")
        kind = getattr(full.dtype, "kind", "")
        if kind == "f":
            isnan = ~np.isfinite(full)
            n_null = int(isnan.sum())
            finite = full[~isnan]
            if finite.size == 0:
                h.update(f"all_null:{n_null}".encode("utf-8"))
            else:
                h.update(
                    f"min={float(np.min(finite)):.12g};max={float(np.max(finite)):.12g};null={n_null}".encode("utf-8")
                )
        elif kind in ("i", "u", "b"):
            try:
                h.update(
                    f"intmin={int(np.min(full))};intmax={int(np.max(full))};nuniq={int(np.unique(full).size)}".encode("utf-8")
                )
            except Exception:
                h.update(b"int_opaque")
        else:
            try:
                u = np.unique(full.astype(str, copy=False))
                h.update(f"uniq={int(u.size)};first={u[0] if u.size else ''};last={u[-1] if u.size else ''}".encode("utf-8"))
            except Exception:
                h.update(b"opaque")
        sampled = full[sample_idx]
        h.update(np.ascontiguousarray(sampled).tobytes())
    return h.hexdigest()


def _make_frame(n_rows: int = 1_000_000, n_cols: int = 200, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {f"f{i}": rng.standard_normal(n_rows).astype(np.float64) for i in range(n_cols)}
    cols["target"] = rng.standard_normal(n_rows).astype(np.float64)
    return pl.DataFrame(cols)


def _time(fn, *args, **kwargs) -> float:
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0


def main() -> None:
    print("Building 200-col x 1,000,000-row float64 frame ...")
    df = _make_frame()
    feature_cols = [c for c in df.columns if c != "target"]

    # Warm up polars internals (lazy-cache init).
    _ = data_signature(df.head(100), "target", feature_cols[:5])

    print("Timing LEGACY (full-column to_numpy per col) ...")
    t_legacy = _time(_legacy_data_signature, df, "target", feature_cols)
    print(f"  legacy: {t_legacy:.3f}s")

    print("Timing CURRENT (single lazy select + gather) ...")
    t_fast = _time(data_signature, df, "target", feature_cols)
    print(f"  current: {t_fast:.3f}s")

    speedup = t_legacy / t_fast if t_fast > 0 else float("inf")
    print(f"\nSpeedup: {speedup:.1f}x  (legacy / current)")


if __name__ == "__main__":
    main()
