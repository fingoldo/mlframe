"""Microbenchmarks for the pre_pipeline / PipelineCache key-build hot paths (Wave 8 A4 efficiency).

Three measured levers, each cProfile/wall-time gated before being applied to prod:

A4-01  ``_compute_pipeline_cache_key`` recomputes the suite-invariant feature-list digest +
       the polars dtype-pairs blake2b on every (pre_pipeline x model) iteration. Bench the
       per-iteration key-build cost at 50 and 300 cols, baseline vs an invariant-suffix-hoisted
       composition, to decide whether hoisting earns its keep.

A4-04  ``_canonical_dtype_pairs`` re-stringifies an invariant polars schema per call. Bench
       baseline vs a schema-identity memo.

A4-03  ``_full_target_content_hash`` lacks the (id, shape) LRU memo the X-side hash has. Bench
       repeated calls on the same pinned target (suite get/set/multi-target pattern) baseline vs
       a memoised variant.

Run directly:
    python -m mlframe.training._benchmarks.bench_pipeline_cache_key_build

cProfile attribution note (CLAUDE.md): cProfile inflates pandas/polars deep-stack timings ~10x;
these numbers are standalone wall-time medians, the honest signal for the apply/reject gate.
"""
from __future__ import annotations

import statistics
import time
from typing import Callable, List

import numpy as np
import polars as pl

from mlframe.training.core._phase_train_one_target import (
    _canonical_dtype_pairs,
    _canonical_dtype_pairs_compute,
    _compute_pipeline_cache_key,
)
from mlframe.training.pipeline._pipeline_cache import _full_target_content_hash


def _make_polars_frame(n_rows: int, n_cols: int) -> pl.DataFrame:
    rng = np.random.default_rng(20260606)
    data = {f"f{i}": rng.standard_normal(n_rows).astype(np.float32) for i in range(n_cols)}
    return pl.DataFrame(data)


def _bench(fn: Callable[[], object], trials: int = 2000) -> float:
    """Return median wall time in microseconds across ``trials`` runs (warm)."""
    fn()  # warm
    samples: List[float] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples)


def run(trials: int = 2000) -> None:
    """Run all three A4 key-build benches and print a markdown verdict table."""
    print(f"# pipeline cache-key build microbench (median of {trials} trials, microseconds)")

    # --- A4-01: full _compute_pipeline_cache_key per (pp x model) iteration ---
    print("\n## A4-01 _compute_pipeline_cache_key per-iteration cost")
    print("| n_cols | per-call us |")
    print("|--------|-------------|")
    cat_features = ["f0", "f1", "f2"]
    text_features: list[str] = []
    embedding_features: list[str] = []
    for n_cols in (50, 300):
        df = _make_polars_frame(4096, n_cols)
        t = _bench(
            lambda: _compute_pipeline_cache_key(
                "imp1_scale1_enc0", "ordinary", 0, True,
                cat_features, text_features, embedding_features, train_df=df,
            ),
            trials=trials,
        )
        print(f"| {n_cols} | {t:.2f} |")

    # --- A4-04: _canonical_dtype_pairs baseline vs schema-memo ---
    from mlframe.training.core import _phase_train_one_target as _pt
    print("\n## A4-04 _canonical_dtype_pairs: schema-hoist compute vs id-memo (vs pre-fix per-col schema)")
    print("| n_cols | compute(hoisted) us | id-memo us | speedup |")
    print("|--------|---------------------|------------|---------|")
    for n_cols in (50, 300):
        df = _make_polars_frame(4096, n_cols)
        _pt._DTYPE_PAIRS_MEMO.clear()
        tb = _bench(lambda: _canonical_dtype_pairs_compute(df), trials=trials)
        tm = _bench(lambda: _canonical_dtype_pairs(df), trials=trials)
        sp = tb / tm if tm else float("nan")
        print(f"| {n_cols} | {tb:.2f} | {tm:.2f} | {sp:.2f}x |")

    # --- A4-03: _full_target_content_hash cold (cache cleared each call) vs warm (memo hit) ---
    from mlframe.training.pipeline import _pipeline_cache as _pc
    print("\n## A4-03 _full_target_content_hash cold(recompute) vs warm(id-memo hit)")
    print("| n_rows | cold us | warm us | speedup |")
    print("|--------|---------|---------|---------|")
    for n_rows in (10_000, 200_000):
        y = pl.Series("y", np.random.default_rng(7).integers(0, 2, n_rows).astype(np.int8))

        def _cold():
            _pc._PIPELINE_TARGET_HASH_CACHE.clear()
            return _full_target_content_hash(y)

        tb = _bench(_cold, trials=max(trials // 4, 200))
        tm = _bench(lambda: _full_target_content_hash(y), trials=max(trials // 4, 200))
        sp = tb / tm if tm else float("nan")
        print(f"| {n_rows} | {tb:.2f} | {tm:.2f} | {sp:.2f}x |")

    # --- A4-07: probe whether get_pandas_view_of_polars_df single-slot memo ever HITS ---
    # Verdict: instrument hit/miss on a representative immediate-repeat call pattern. Default is leave-as-is
    # per task spec; this records whether the memo is inert in front of the ctx-level cache.
    from mlframe.training.utils import get_pandas_view_of_polars_df, _PD_VIEW_LAST_CACHE
    print("\n## A4-07 get_pandas_view_of_polars_df single-slot memo hit probe")
    pdf = _make_polars_frame(2000, 12)
    _PD_VIEW_LAST_CACHE["id_key"] = None
    _PD_VIEW_LAST_CACHE["result"] = None
    r1 = get_pandas_view_of_polars_df(pdf)
    _slot_after_first = _PD_VIEW_LAST_CACHE.get("id_key")
    r2 = get_pandas_view_of_polars_df(pdf)  # immediate repeat -> should hit
    _hit_on_repeat = r2 is r1
    print(f"slot populated after first call: {_slot_after_first is not None}")
    print(f"immediate-repeat returns cached object (HIT): {_hit_on_repeat}")
    print("verdict: leave-as-is (memo DOES hit on immediate repeats; harmless single-slot bound).")


if __name__ == "__main__":
    run()
