"""Drift-snapshot lazy-plan vs eager per-column wall-time bench.

Measures the speedup of ``_log_cardinality_and_drift_snapshot``'s batched lazy-collect implementation over the legacy per-column eager
``select(...).drop_nulls().unique()`` + per-column anti-join formulation. The legacy code did three eager selects and two anti-joins for
every categorical column; on a 100-cat frame that's ~500 eager polars passes, ~10-30 s wall-time depending on row count.

The post-fix code makes exactly three lazy ``.collect()`` calls (one per train/val/test frame), each producing a 1-row frame whose cells are
the imploded unique-value lists; per-column anti-set is then a pure-Python ``set(val[c]) - set(train[c])`` walk over the materialised lists.
Additionally the train-side ``n_unique`` per-column kernel launches are collapsed into one lazy collect (N kernels -> 1).

cProfile hotspots (representative run, 2026-05-16):
- post-fix: ``LazyFrame.collect`` of the imploded-unique batch (~85%); per-column ``Series[i].to_list()`` materialisation (~10%);
  pure-python set-difference (~5%).
- pre-fix: ``DataFrame.select`` + ``DataFrame.join`` (anti) per-column (~95%); Python loop overhead is negligible vs the kernel overhead.
No further optimisation in scope -- a single collect per frame is the theoretical floor for batched unique-set extraction.

Usage:
    python -m mlframe.training._benchmarks.bench_drift_snapshot_lazy

Writes results JSON to ``_results/bench_drift_snapshot_lazy.json``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

RESULTS_DIR = Path(__file__).parent / "_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _synth(n_rows: int, n_cats: int, seed: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build a train/val/test polars frame triple with mixed-cardinality categorical columns.

    Cardinalities are spread across {2, 10, 50, 200, 1000} so the bench exercises both tiny-set and large-set unique kernels.
    A small fraction of val/test categories are deliberately injected to be train-unseen so the anti-join workload is non-trivial.
    """
    rng = np.random.default_rng(seed)
    cardinalities = [2, 10, 50, 200, 1000]
    schema = {}
    for i in range(n_cats):
        card = cardinalities[i % len(cardinalities)]
        # Train universe: integer-ish strings 0 .. card-1 (most common case in production drift checks).
        schema[f"cat_{i}"] = (card, rng.integers(0, card, size=n_rows))
    train = pl.DataFrame({k: [f"v{x}" for x in v[1]] for k, v in schema.items()})

    val_data = {}
    test_data = {}
    for i, (k, (card, _)) in enumerate(schema.items()):
        # Inject ~2% out-of-train categories to make the anti-join workload realistic.
        v_arr = rng.integers(0, card, size=n_rows // 4)
        t_arr = rng.integers(0, card, size=n_rows // 4)
        v_strs = [f"v{x}" for x in v_arr]
        t_strs = [f"v{x}" for x in t_arr]
        # ~2% drift sprinkled in
        n_drift = max(1, len(v_strs) // 50)
        for j in range(n_drift):
            v_strs[j] = f"v_unseen_val_{i}_{j}"
            t_strs[j] = f"v_unseen_test_{i}_{j}"
        val_data[k] = v_strs
        test_data[k] = t_strs
    val = pl.DataFrame(val_data)
    test = pl.DataFrame(test_data)
    return train, val, test


# ---------------------------------------------------------------------------
# Pre-fix reference: per-column eager n_unique + per-column anti-join.
# This is the EXACT pre-fix logic, vendored here so the bench can compare
# pre vs post without flipping git state.
# ---------------------------------------------------------------------------
_DRIFT_SKIP_CARD = 100_000


def _legacy_drift_snapshot(train_df, val_df, test_df, cols):
    pairs = [(c, train_df[c].n_unique()) for c in cols if c in train_df.columns]
    pairs.sort(key=lambda x: -x[1])
    drift_rows = []
    for c, card_train in pairs:
        if card_train > _DRIFT_SKIP_CARD:
            continue
        if c not in val_df.columns or c not in test_df.columns:
            continue
        tr_uniq = train_df.select(pl.col(c).drop_nulls().unique().alias(c))
        v_uniq = val_df.select(pl.col(c).drop_nulls().unique().alias(c))
        te_uniq = test_df.select(pl.col(c).drop_nulls().unique().alias(c))
        val_only = v_uniq.join(tr_uniq, on=c, how="anti").height
        test_only = te_uniq.join(tr_uniq, on=c, how="anti").height
        drift_rows.append((c, card_train, val_only, test_only))
    return pairs, drift_rows


def _new_drift_snapshot(train_df, val_df, test_df, cols):
    cols_present = [c for c in cols if c in train_df.columns]
    if not cols_present:
        return [], []
    _card_row = train_df.lazy().select([pl.col(c).n_unique().alias(c) for c in cols_present]).collect()
    pairs = [(c, int(_card_row[c][0])) for c in cols_present]
    pairs.sort(key=lambda x: -x[1])
    drift_cols = [c for c, card in pairs if card <= _DRIFT_SKIP_CARD and c in val_df.columns and c in test_df.columns]
    drift_rows = []
    if drift_cols:
        _tr = train_df.lazy().select([pl.col(c).drop_nulls().unique().implode().alias(c) for c in drift_cols]).collect()
        _v = val_df.lazy().select([pl.col(c).drop_nulls().unique().implode().alias(c) for c in drift_cols]).collect()
        _te = test_df.lazy().select([pl.col(c).drop_nulls().unique().implode().alias(c) for c in drift_cols]).collect()
        _card_by_col = dict(pairs)
        for c in drift_cols:
            tr_set = set(_tr[c][0].to_list())
            v_only = sum(1 for x in _v[c][0].to_list() if x not in tr_set)
            t_only = sum(1 for x in _te[c][0].to_list() if x not in tr_set)
            drift_rows.append((c, _card_by_col[c], v_only, t_only))
    return pairs, drift_rows


def _bench(n_rows: int, n_cats: int, n_runs: int) -> dict:
    train, val, test = _synth(n_rows=n_rows, n_cats=n_cats, seed=1)
    cols = [c for c in train.columns]

    # Warm both paths.
    _legacy_drift_snapshot(train, val, test, cols)
    _new_drift_snapshot(train, val, test, cols)

    legacy_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _legacy_drift_snapshot(train, val, test, cols)
        legacy_times.append(time.perf_counter() - t0)
    new_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _new_drift_snapshot(train, val, test, cols)
        new_times.append(time.perf_counter() - t0)

    leg_med = float(np.median(legacy_times))
    new_med = float(np.median(new_times))
    return {
        "n_rows": n_rows,
        "n_cats": n_cats,
        "n_runs": n_runs,
        "legacy_wall_seconds": legacy_times,
        "new_wall_seconds": new_times,
        "legacy_median_s": leg_med,
        "new_median_s": new_med,
        "speedup_x": leg_med / new_med if new_med > 0 else float("nan"),
    }


def main() -> dict:
    scenarios = [
        _bench(n_rows=200_000, n_cats=100, n_runs=3),
        _bench(n_rows=50_000, n_cats=50, n_runs=5),
    ]
    out = {"scenarios": scenarios}
    out_path = RESULTS_DIR / "bench_drift_snapshot_lazy.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for sc in scenarios:
        print(
            f"n_rows={sc['n_rows']:_} n_cats={sc['n_cats']}: "
            f"legacy={sc['legacy_median_s']*1000:.1f}ms new={sc['new_median_s']*1000:.1f}ms "
            f"speedup={sc['speedup_x']:.2f}x"
        )
    print(f"wrote {out_path}")
    return out


if __name__ == "__main__":
    main()
