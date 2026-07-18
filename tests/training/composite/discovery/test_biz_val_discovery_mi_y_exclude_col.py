"""D10 regression: per-base ``mi_y`` baseline via ``exclude_col`` is bit-identical to ``np.delete`` AND allocates less.

``CompositeTargetDiscovery.fit`` derives each base candidate's baseline ``MI(y, X_without_base)`` by excluding that
base's feature column. The legacy path materialised a fresh ``np.delete(full_prebinned, base_idx, axis=1)`` (n, F-1)
matrix copy per base; the new path threads ``exclude_col`` into ``_mi_per_feature_prebinned`` /
``_mi_to_target_prebinned`` (and adds ``_aggregate_mi_per_feature_excluding``) so the per-feature loop SKIPS that one
column on the full matrix -- no per-base matrix copy.

These tests pin BOTH sides of the win:
* **bit-identity** -- the exclude-col MI equals the delete-the-column MI element-for-element, for mean AND sum, across
  interior / first / last drop indices, and on a matrix carrying NaN (-1 sentinel) columns. A future "just delete it"
  cannot silently change the numerics.
* **allocation reduction** -- a paired ``tracemalloc`` peak shows the exclude-col path allocates strictly less than the
  ``np.delete`` path over a B-base baseline sweep, so a revert that reintroduces the per-base copy trips this test.
"""

from __future__ import annotations

import tracemalloc
import warnings

import numpy as np
import pytest

from mlframe.training.composite.discovery.screening import (
    _aggregate_mi_per_feature,
    _aggregate_mi_per_feature_excluding,
    _mi_per_feature_prebinned,
    _mi_to_target_prebinned,
    _prebin_feature_columns,
)

warnings.filterwarnings("ignore")


def _make_prebinned(n: int = 8_000, f: int = 12, nbins: int = 12, seed: int = 0, *, nan_cols=()):
    """Make prebinned."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, f)).astype(np.float32)
    for c in nan_cols:
        # Scatter NaN into a column so prebin emits the -1 sentinel there.
        m = rng.random(n) < 0.15
        x[m, c] = np.nan
    # Signal from the first few columns (robust to small f, incl. f=1).
    coefs = np.array([0.6, 0.3, -0.2, 0.15], dtype=np.float64)[:f]
    y = x[:, : coefs.size].astype(np.float64) @ coefs
    y = np.where(np.isfinite(y), y, 0.0) + rng.standard_normal(n) * 0.5
    return _prebin_feature_columns(x, nbins=nbins), y, nbins


# ----------------------------------------------------------------------
# bit-identity: exclude_col == np.delete, both aggregations, all indices
# ----------------------------------------------------------------------


@pytest.mark.parametrize("aggregation", ["mean", "sum"])
@pytest.mark.parametrize("drop", [0, 5, 11])
def test_mi_to_target_prebinned_exclude_col_bit_identical(aggregation: str, drop: int) -> None:
    """Mi to target prebinned exclude col bit identical."""
    prebinned, y, nbins = _make_prebinned()
    ref = _mi_to_target_prebinned(
        np.delete(prebinned, drop, axis=1),
        y,
        nbins=nbins,
        aggregation=aggregation,
    )
    got = _mi_to_target_prebinned(
        prebinned,
        y,
        nbins=nbins,
        aggregation=aggregation,
        exclude_col=drop,
    )
    assert got == ref, f"exclude_col={drop} agg={aggregation}: {got!r} != delete {ref!r}"


@pytest.mark.parametrize("drop", [0, 4, 11])
def test_mi_per_feature_prebinned_exclude_col_vector_bit_identical(drop: int) -> None:
    """The returned per-feature VECTOR (not just its aggregate) matches the deleted-matrix vector exactly."""
    prebinned, y, nbins = _make_prebinned()
    ref_vec = _mi_per_feature_prebinned(np.delete(prebinned, drop, axis=1), y, nbins=nbins)
    got_vec = _mi_per_feature_prebinned(prebinned, y, nbins=nbins, exclude_col=drop)
    assert ref_vec is not None and got_vec is not None
    assert got_vec.shape == ref_vec.shape == (prebinned.shape[1] - 1,)
    assert np.array_equal(got_vec, ref_vec), "per-feature MI vector diverged from np.delete path"


@pytest.mark.parametrize("aggregation", ["mean", "sum"])
@pytest.mark.parametrize("drop", [0, 6, 11])
def test_aggregate_excluding_bit_identical(aggregation: str, drop: int) -> None:
    """Aggregate excluding bit identical."""
    rng = np.random.default_rng(7)
    v = rng.standard_normal(12).astype(np.float64)
    ref = _aggregate_mi_per_feature(np.delete(v, drop), aggregation)
    got = _aggregate_mi_per_feature_excluding(v, aggregation, drop)
    assert got == ref, f"agg-excluding drop={drop} agg={aggregation}: {got!r} != {ref!r}"


def test_exclude_col_bit_identical_with_nan_sentinel_columns() -> None:
    """A matrix carrying -1-sentinel (NaN) columns still matches the deleted-matrix baseline exactly."""
    prebinned, y, nbins = _make_prebinned(nan_cols=(2, 7, 9))
    for drop in (0, 2, 7, prebinned.shape[1] - 1):
        ref = _mi_to_target_prebinned(np.delete(prebinned, drop, axis=1), y, nbins=nbins, aggregation="mean")
        got = _mi_to_target_prebinned(prebinned, y, nbins=nbins, aggregation="mean", exclude_col=drop)
        assert got == ref, f"NaN-col exclude_col={drop}: {got!r} != delete {ref!r}"


def test_exclude_col_edges_and_degenerate() -> None:
    """Single-column matrix -> excluding the only column yields an empty feature set (0.0 baseline);
    out-of-range / None exclude_col is a no-op equal to the full aggregate."""
    prebinned, y, nbins = _make_prebinned(f=1)
    # Excluding the only column: empty feature set -> _mi_per_feature_prebinned returns None -> aggregate 0.0.
    assert _mi_to_target_prebinned(prebinned, y, nbins=nbins, exclude_col=0) == 0.0
    # None / out-of-range exclude is identical to the no-exclude full aggregate.
    full = _mi_to_target_prebinned(prebinned, y, nbins=nbins)
    assert _mi_to_target_prebinned(prebinned, y, nbins=nbins, exclude_col=None) == full
    assert _mi_to_target_prebinned(prebinned, y, nbins=nbins, exclude_col=99) == full
    v = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    assert _aggregate_mi_per_feature_excluding(v, "mean", 99) == _aggregate_mi_per_feature(v, "mean")


# ----------------------------------------------------------------------
# allocation reduction: exclude_col peak < np.delete peak over B bases
# ----------------------------------------------------------------------


def _peak_bytes(fn) -> int:
    """Peak bytes."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def test_exclude_col_peak_alloc_strictly_lower_than_delete() -> None:
    """biz_value (RAM): over a B-base baseline sweep the exclude_col path's tracemalloc peak is well below the
    np.delete path's, because it never materialises the per-base (n, F-1) matrix copy. Floor the win generously
    (>= 1.5x lower peak) so allocator noise never trips it but a revert that reintroduces np.delete does."""
    n, f, b, nbins = 50_000, 40, 5, 12
    prebinned, y, _ = _make_prebinned(n=n, f=f, nbins=nbins)
    idxs = list(range(b))

    def _old():
        """Old."""
        return [
            _aggregate_mi_per_feature(
                _mi_per_feature_prebinned(np.delete(prebinned, k, axis=1), y, nbins=nbins),
                "mean",
            )
            for k in idxs
        ]

    def _new():
        """New."""
        return [_mi_to_target_prebinned(prebinned, y, nbins=nbins, aggregation="mean", exclude_col=k) for k in idxs]

    # bit-identity over the whole sweep (belt-and-suspenders vs the parametrized cases above).
    assert _old() == _new()

    old_peak = min(_peak_bytes(_old) for _ in range(3))
    new_peak = min(_peak_bytes(_new) for _ in range(3))
    assert (
        new_peak * 1.5 <= old_peak
    ), f"exclude_col peak {new_peak / 1e6:.2f} MB not >=1.5x below np.delete peak {old_peak / 1e6:.2f} MB -- the per-base matrix copy may have crept back."


def test_discovery_fit_still_produces_specs_end_to_end() -> None:
    """Functional non-regression: a small discovery fit through the touched per-base baseline path still runs and
    yields specs (the exclude_col swap is internal + bit-identical, so discovery behaviour is unchanged)."""
    pd = pytest.importorskip("pandas")
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    rng = np.random.default_rng(3)
    n = 4_000
    base = np.cumsum(rng.standard_normal(n)).astype(np.float64)
    y = base + rng.standard_normal(n) * 0.3
    feats = {f"f{j}": rng.standard_normal(n) for j in range(6)}
    feats["lag_base"] = base
    df = pd.DataFrame({"y": y, **feats})
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True,
        screening="mi",
        mi_estimator="bin",
        base_candidates="auto",
        transforms=("diff", "linear_residual"),
    )
    disc = CompositeTargetDiscovery(config=cfg)
    disc.fit(df, "y", [c for c in df.columns if c != "y"], train_idx=np.arange(n))
    assert isinstance(disc.specs_, list)
    assert isinstance(disc.report_, list) and len(disc.report_) >= 1


def test_mi_per_feature_matrix_level_sentinel_gate_routes_correctly() -> None:
    """iter76: ``_mi_per_feature_prebinned`` detects the -1 non-finite sentinel ONCE for the whole matrix
    (``(fb < 0).any()``) and skips the per-column ``col_valid``/``.sum()`` scan when absent, instead of scanning
    every column. The gate must be bit-identical to the explicit per-column path AND route correctly:
    a sentinel-FREE matrix sends every column FULL-length to the MI kernel (fast path); a sentinel matrix sends
    the masked (shorter) subset for the affected columns (slow path preserved). The kernel-arg-length spy fails if
    a future change either masks unnecessarily on clean data or drops the sentinel masking on dirty data."""
    import mlframe.training.composite.discovery.screening as S

    n, f, nbins = 6_000, 20, 12
    pb_clean, y, _ = _make_prebinned(n=n, f=f, nbins=nbins, seed=7)
    pb_sent, _, _ = _make_prebinned(n=n, f=f, nbins=nbins, seed=7, nan_cols=(2, 9, 15))

    def _ref(fb, target):
        """Ref."""
        out = []
        finite = np.isfinite(target)
        t_f = target[finite] if finite.sum() != finite.shape[0] else target
        fb_f = fb[finite] if finite.sum() != finite.shape[0] else fb
        qs = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
        t_idx = np.searchsorted(np.nanquantile(t_f, qs), t_f, side="right").astype(np.int64)
        np.clip(t_idx, 0, nbins - 1, out=t_idx)
        for j in range(fb_f.shape[1]):
            cb = fb_f[:, j]
            cv = cb >= 0
            ncv = int(cv.sum())
            if ncv < 5 * nbins:
                out.append(0.0)
            elif ncv == cb.shape[0]:
                out.append(S._mi_from_binned_pair(cb, t_idx, nbins=nbins))
            else:
                out.append(S._mi_from_binned_pair(cb[cv], t_idx[cv], nbins=nbins))
        return np.asarray(out, dtype=np.float64)

    for pb in (pb_clean, pb_sent):
        got = _mi_per_feature_prebinned(pb, y, nbins=nbins)
        assert np.array_equal(got, _ref(pb, y)), "matrix-level gate diverged from per-column path"

    real_kernel = S._mi_from_binned_pair
    seen_lengths: list[int] = []

    def _spy(x_idx, y_idx, *, nbins):
        """Spy."""
        seen_lengths.append(int(x_idx.shape[0]))
        return real_kernel(x_idx, y_idx, nbins=nbins)

    S._mi_from_binned_pair = _spy
    try:
        seen_lengths.clear()
        _mi_per_feature_prebinned(pb_clean, y, nbins=nbins)
        assert seen_lengths and all(l == n for l in seen_lengths), "sentinel-free matrix must send full-length columns to the MI kernel (fast path)"
        seen_lengths.clear()
        _mi_per_feature_prebinned(pb_sent, y, nbins=nbins)
        assert any(l < n for l in seen_lengths), "sentinel matrix must send the masked (shorter) subset for non-finite columns (slow path)"
    finally:
        S._mi_from_binned_pair = real_kernel
