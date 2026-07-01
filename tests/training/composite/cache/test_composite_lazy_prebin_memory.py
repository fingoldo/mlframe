"""Lazy polars prebinning: bit-identity, peak-RAM win, end-to-end fit parity.

Covers ``_prebin_feature_columns_lazy`` (screening.py) and its size-gated wiring
into discovery ``fit`` (_fit.py). The lazy path pulls + bins one polars column at
a time so the float32 (n, F) plane is never materialised -- bit-identical codes,
lower peak RAM. The eager path stays the default for ndarray / small / dedup-on /
knn inputs.
"""
from __future__ import annotations

import gc
import tracemalloc

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from mlframe.training.composite.discovery.screening import (
    _prebin_feature_columns,
    _prebin_feature_columns_lazy,
)
from mlframe.training.composite.discovery import CompositeTargetDiscovery


def _frame(n: int, f: int, seed: int = 0, nan_cols=(3,)):
    rng = np.random.default_rng(seed)
    data = {f"c{j}": rng.standard_normal(n).astype(np.float32) for j in range(f)}
    for j in nan_cols:
        if j < f:
            data[f"c{j}"][rng.integers(0, n, max(1, n // 50))] = np.nan
    return pl.DataFrame(data), list(data.keys())


def _eager_codes(df, cols, rows, nbins):
    from mlframe.training.composite.discovery.screening import _extract_column_array

    mat = np.column_stack([_extract_column_array(df, c, rows=rows) for c in cols])
    return _prebin_feature_columns(mat, nbins=nbins)


# ---------------------------------------------------------------------------
# Bit-identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nbins", [16, 64, 200])  # 200 crosses the int16->int32 dtype boundary.
def test_lazy_prebin_bit_identical_to_eager(nbins):
    df, cols = _frame(3000, 14, nan_cols=(3, 9))
    rng = np.random.default_rng(1)
    rows = np.sort(rng.choice(3000, 2500, replace=False))
    eager = _eager_codes(df, cols, rows, nbins)
    lazy = _prebin_feature_columns_lazy(df, cols, rows, nbins=nbins)
    assert eager.dtype == lazy.dtype
    assert eager.shape == lazy.shape
    assert np.array_equal(eager, lazy), "lazy prebin codes must be bit-identical to eager"


def test_lazy_prebin_too_few_rows_returns_all_sentinel():
    # Below 5*nbins rows both paths return the all -1 sentinel matrix.
    df, cols = _frame(60, 5, nan_cols=())
    rows = np.arange(60)
    lazy = _prebin_feature_columns_lazy(df, cols, rows, nbins=16)
    assert lazy.shape == (60, 5)
    assert (lazy == -1).all()


def test_lazy_prebin_all_nan_column_is_sentinel():
    df, cols = _frame(2000, 4, nan_cols=())
    # Force one column fully NaN -> both paths bin it to -1 everywhere.
    arr = df["c1"].to_numpy().copy()
    arr[:] = np.nan
    df = df.with_columns(pl.Series("c1", arr))
    rows = np.arange(2000)
    eager = _eager_codes(df, cols, rows, 16)
    lazy = _prebin_feature_columns_lazy(df, cols, rows, nbins=16)
    assert np.array_equal(eager, lazy)
    assert (lazy[:, cols.index("c1")] == -1).all()


# ---------------------------------------------------------------------------
# biz_value: peak-RAM win (the whole point of the lazy path)
# ---------------------------------------------------------------------------

def test_biz_val_lazy_prebin_peak_ram_below_float_plane():
    """Lazy peak alloc must be well under the eager float plane it avoids.

    On n=120k, F=80 the eager float32 plane is ~36.6 MB; the eager path peak
    includes that plane PLUS the column_stack source, so lazy (one column at a
    time) must come in clearly lower. Floor: lazy peak <= 60% of eager peak.
    Measured ~0.36x (bench_lazy_prebin_memory: 2.79x at n=200k/F=100). Margin
    wide enough that allocator noise does not trip it.
    """
    n, f, nbins = 120_000, 80, 16
    df, cols = _frame(n, f, nan_cols=(3, 17))
    rows = np.arange(n)
    # Warm: polars schema / numpy first-touch out of the measured region.
    _prebin_feature_columns_lazy(df, cols[:2], rows[:1000], nbins=nbins)

    gc.collect()
    tracemalloc.start()
    eager = _eager_codes(df, cols, rows, nbins)
    _cur, eager_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    gc.collect()
    tracemalloc.start()
    lazy = _prebin_feature_columns_lazy(df, cols, rows, nbins=nbins)
    _cur, lazy_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert np.array_equal(eager, lazy)
    float_plane = n * f * 4
    assert lazy_peak <= 0.60 * eager_peak, (
        f"lazy peak {lazy_peak/1e6:.1f} MB should be <=60% of eager "
        f"{eager_peak/1e6:.1f} MB (float plane {float_plane/1e6:.1f} MB)"
    )
    # Lazy must avoid materialising a full float plane's worth of transient.
    assert lazy_peak < float_plane, (
        f"lazy peak {lazy_peak/1e6:.1f} MB should be below the avoided "
        f"float plane {float_plane/1e6:.1f} MB"
    )


# ---------------------------------------------------------------------------
# End-to-end: lazy-gated fit == eager fit (same specs, same report)
# ---------------------------------------------------------------------------

def _fit_frame(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n).astype(np.float32)
    feats = {f"f{j}": rng.standard_normal(n).astype(np.float32) for j in range(6)}
    # Target with a learnable linear-residual structure on the base.
    y = (0.8 * base + 0.3 * feats["f0"] + 0.1 * rng.standard_normal(n)).astype(np.float32)
    data = {"y": y, "base": base, **feats}
    return pl.DataFrame(data), list(feats.keys()) + ["base"]


def _run_fit(force_lazy: str, monkeypatch):
    from mlframe.training.composite.discovery import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    monkeypatch.setenv("MLFRAME_DISCOVERY_LAZY_PREBIN", force_lazy)
    df, feature_cols = _fit_frame()
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True,
        mi_estimator="bin",
        screening="mi",
        mi_sample_n=None,
        base_candidates=["base"],
        dedup_x_remaining_for_mi_baseline=False,  # lazy gate requires dedup off.
        random_state=0,
    )
    disc = CompositeTargetDiscovery(config=cfg)
    train_idx = np.arange(len(df))
    disc.fit(df, "y", feature_cols, train_idx=train_idx)
    names = [s.name for s in disc.specs_]
    gains = {s.name: round(float(s.mi_gain), 10) for s in disc.specs_}
    return names, gains


def test_lazy_gated_fit_matches_eager_fit(monkeypatch):
    """Discovery fit with the lazy prebin forced ON yields IDENTICAL specs +
    mi_gain to the eager path forced OFF -- the gate is a pure RAM optimisation,
    never a numeric one."""
    eager_names, eager_gains = _run_fit("0", monkeypatch)
    lazy_names, lazy_gains = _run_fit("1", monkeypatch)
    assert eager_names == lazy_names, "lazy and eager fit must discover the same specs"
    assert eager_gains == lazy_gains, "lazy and eager mi_gain must be bit-identical"
    assert eager_names, "fixture should discover at least one spec"
