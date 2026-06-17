"""Bit-identity test for the BATCHED external-validation MI sweep in
``check_prospective_fe_pairs`` (2026-06-07 perf optimization).

The external-validation tie-break computes, per leader config,
``best_valid_mi = max over (external_factor x binary_func) of
mi_direct(discretize_array(binary_func(param_a, param_b)))``. The optimization
replaces that per-candidate ``discretize_array`` + ``mi_direct`` double loop
(the dominant serial FE hotspot at wide p) with ONE
``discretize_2d_quantile_batch`` + ONE ``_dispatch_batch_mi_with_noise_gate``,
then a ``max``. This test asserts the batched ``best_valid_mi`` is BIT-IDENTICAL
to the per-candidate reduction on the default FE path (``parallelism='outer'``,
``n_workers=1``, ``base_seed=0``, ``npermutations < 32`` -> CPU ``parallel_mi_prange``),
including on candidate columns carrying NaN/inf (from div-by-zero / log of
negatives / overflow), which the per-candidate ``discretize_array`` bins via
NaN-ignoring ``np.nanpercentile`` + ``searchsorted`` exactly as the batch does.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.feature_engineering import (
    discretize_array,
    discretize_2d_quantile_batch,
)
from mlframe.feature_selection.filters.permutation import mi_direct
from mlframe.feature_selection.filters.info_theory import (
    batch_mi_with_noise_gate,
    merge_vars,
)
from mlframe.feature_selection.filters._feature_engineering_pairs import (
    _dispatch_batch_mi_with_noise_gate,
    _materialise_extval_njit,
    _njit_binary_op_codes,
)
from mlframe.feature_selection.filters.feature_engineering import create_binary_transformations


def _make_y(n, n_classes, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n).astype(np.int32)
    classes_y, freqs_y, _ = merge_vars(
        factors_data=y.reshape(-1, 1),
        vars_indices=np.array([0], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=np.array([int(y.max()) + 1], dtype=np.int64),
        dtype=np.int32,
    )
    return classes_y, freqs_y


def _candidate_columns(n, K, seed):
    """K raw float candidate columns mimicking ``binary_func(param_a, param_b)``
    outputs: includes div-by-zero (inf), log-of-negative (nan), overflow, and
    plain informative/noise columns -- the exact value pathologies the FE
    external-validation sweep feeds into discretize_array."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, 1)).astype(np.float32)
    cols = []
    for k in range(K):
        b = rng.standard_normal(n).astype(np.float64)
        kind = k % 5
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if kind == 0:
                c = a[:, 0] * b                      # mul (clean)
            elif kind == 1:
                c = a[:, 0] / b                      # div -> inf where b~0
            elif kind == 2:
                c = np.log(b)                        # nan where b<0
            elif kind == 3:
                c = np.exp(a[:, 0] * 50.0) * b       # overflow -> inf
            else:
                c = a[:, 0] + b                      # add (clean)
        cols.append(np.asarray(c, dtype=np.float64))
    return np.column_stack(cols)


def _per_candidate_best(raw_cols, nbins, dtype, classes_y, classes_y_safe, freqs_y,
                        npermutations, min_nonzero_confidence):
    """The ORIGINAL per-candidate external-validation reduction."""
    best = -1.0
    K = raw_cols.shape[1]
    for k in range(K):
        disc = discretize_array(arr=raw_cols[:, k], n_bins=nbins, method="quantile", dtype=dtype)
        fe_mi, _ = mi_direct(
            disc.reshape(-1, 1),
            x=np.array([0], dtype=np.int64),
            y=None,
            factors_nbins=np.array([nbins], dtype=np.int64),
            classes_y=classes_y,
            classes_y_safe=classes_y_safe,
            freqs_y=freqs_y,
            min_nonzero_confidence=min_nonzero_confidence,
            npermutations=npermutations,
            prefer_gpu=False,
        )
        if fe_mi > best:
            best = fe_mi
    return best


def _batched_best(raw_cols, nbins, dtype, classes_y, classes_y_safe, freqs_y,
                  npermutations, min_nonzero_confidence):
    """The OPTIMIZED batched reduction (matches the in-code path)."""
    disc_2d = discretize_2d_quantile_batch(raw_cols, n_bins=nbins, dtype=dtype)
    mi = _dispatch_batch_mi_with_noise_gate(
        disc_2d=disc_2d,
        quantization_nbins=nbins,
        classes_y=classes_y,
        classes_y_safe=classes_y_safe,
        freqs_y=freqs_y,
        npermutations=npermutations,
        min_nonzero_confidence=min_nonzero_confidence,
        use_su=False,
        batch_mi_kernel=batch_mi_with_noise_gate,
    )
    return float(np.max(mi)) if (mi is not None and len(mi)) else -1.0


@pytest.mark.parametrize("n,K,nbins", [(200, 12, 5), (500, 30, 10), (800, 18, 8)])
@pytest.mark.parametrize("npermutations", [0, 3])
@pytest.mark.parametrize("min_nonzero_confidence", [0.99, 0.0])
def test_external_validation_batch_bit_identical(n, K, nbins, npermutations, min_nonzero_confidence):
    classes_y, freqs_y = _make_y(n, n_classes=3, seed=7 + n + K + nbins)
    classes_y_safe = classes_y.copy()
    raw_cols = _candidate_columns(n, K, seed=4242 + n + K)

    ref = _per_candidate_best(
        raw_cols, nbins, np.int32, classes_y, classes_y_safe, freqs_y,
        npermutations, float(min_nonzero_confidence),
    )
    got = _batched_best(
        raw_cols, nbins, np.int32, classes_y, classes_y_safe, freqs_y,
        npermutations, float(min_nonzero_confidence),
    )
    # EXACT equality -- the external-validation MI feeds the secondary tie-break,
    # so any drift can change which engineered form MRMR keeps.
    assert got == ref, (
        f"batched best_valid_mi != per-candidate: n={n} K={K} nbins={nbins} "
        f"nperm={npermutations} mnc={min_nonzero_confidence}: got={got!r} ref={ref!r}"
    )


def test_external_validation_batch_handles_nan_inf_columns():
    """Columns that are entirely inf/nan after a bin_func must bin + score
    identically in both paths (no nan_to_num is applied -- nanpercentile ignores
    NaN, searchsorted routes NaN to the rightmost bin, identically per-column)."""
    n = 400
    classes_y, freqs_y = _make_y(n, n_classes=2, seed=11)
    classes_y_safe = classes_y.copy()
    rng = np.random.default_rng(3)
    a = rng.standard_normal(n)
    cols = np.column_stack([
        a / np.zeros(n),                       # all +-inf
        np.log(-np.abs(a) - 1.0),              # all nan
        a * rng.standard_normal(n),            # clean informative-ish
    ]).astype(np.float64)
    ref = _per_candidate_best(cols, 10, np.int32, classes_y, classes_y_safe, freqs_y, 3, 0.99)
    got = _batched_best(cols, 10, np.int32, classes_y, classes_y_safe, freqs_y, 3, 0.99)
    assert got == ref, f"nan/inf column mismatch: got={got!r} ref={ref!r}"


# ---------------------------------------------------------------------------
# njit per-column searchsorted in discretize_2d_quantile_batch (2026-06-07).
# Must be byte-identical to the per-column ``discretize_array(method='quantile')``
# the FE pair-search historically called, including NaN/inf -> rightmost bin and
# float32 vs float64 buffer dtypes (the chunk path feeds float32, ext-val float64).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n,K,nbins", [(200, 7, 5), (500, 33, 10), (1000, 16, 8), (300, 4, 3)])
@pytest.mark.parametrize("buf_dtype", [np.float32, np.float64])
def test_discretize_2d_quantile_batch_matches_per_column(n, K, nbins, buf_dtype):
    raw = _candidate_columns(n, K, seed=99 + n + K + nbins).astype(buf_dtype)
    got = discretize_2d_quantile_batch(raw, n_bins=nbins, dtype=np.int32)
    ref = np.empty_like(got)
    for k in range(K):
        ref[:, k] = discretize_array(arr=raw[:, k], n_bins=nbins, method="quantile", dtype=np.int32)
    assert np.array_equal(got, ref), (
        f"discretize_2d_quantile_batch != per-column discretize_array: "
        f"n={n} K={K} nbins={nbins} dtype={buf_dtype}; "
        f"mismatched cols={np.where(~(got == ref).all(axis=0))[0][:10]}"
    )


def test_discretize_2d_quantile_batch_constant_and_tie_columns():
    """Constant columns (all edges equal) and tie-heavy columns must bin identically."""
    n = 400
    cols = np.column_stack([
        np.full(n, 3.0),                                  # constant
        np.where(np.arange(n) < n // 2, 0.0, 1.0),        # 2-value tie
        np.r_[np.zeros(n - 5), np.arange(5) + 100.0],     # near-constant + outliers
    ]).astype(np.float64)
    got = discretize_2d_quantile_batch(cols, n_bins=10, dtype=np.int32)
    ref = np.empty_like(got)
    for k in range(cols.shape[1]):
        ref[:, k] = discretize_array(arr=cols[:, k], n_bins=10, method="quantile", dtype=np.int32)
    assert np.array_equal(got, ref)


def test_discretize_2d_quantile_batch_no_full_buffer_float64_copy(monkeypatch):
    """REGRESSION (2026-06-07): the njit-searchsorted optimisation must NOT upcast the
    full-width float32 FE buffer to a float64 copy. The original optimisation did
    ``arr_c = np.ascontiguousarray(arr2d, dtype=np.float64)`` which DOUBLED a multi-GB
    float32 buffer into float64 and OOM-crashed ``MRMR.fit`` on the wide canonical
    fixture (20000 x ~19000 cols -> 2.9 GiB float64 alloc on top of numpy.percentile's
    own copy). The fix passes ``arr2d`` at its native float32 dtype to the kernel (numba
    promotes per-element against the float64 edges, byte-identically). This test pins
    that contract: discretising a float32 buffer must never request a float64-dtyped
    ``ascontiguousarray`` of an array as large as the buffer.

    We spy on ``np.ascontiguousarray`` inside the discretization module: a float64 cast
    of an array with >= n_rows*n_cols elements is the forbidden full-buffer upcast. (The
    small ``edges[1:-1]`` float64 contiguity call is allowed -- it is (n_bins-1) x n_cols,
    far smaller than the buffer.)
    """
    import mlframe.feature_selection.filters.discretization as _disc_mod

    n, K, nbins = 2000, 64, 10
    raw = _candidate_columns(n, K, seed=314).astype(np.float32)
    buffer_cells = n * K

    real_ascontig = np.ascontiguousarray
    forbidden = {"hit": False}

    def _spy_ascontiguousarray(a, dtype=None):
        arr = np.asarray(a)
        if (
            dtype is not None
            and np.dtype(dtype) == np.float64
            and arr.dtype == np.float32
            and arr.size >= buffer_cells
        ):
            forbidden["hit"] = True
        return real_ascontig(a, dtype=dtype) if dtype is not None else real_ascontig(a)

    monkeypatch.setattr(_disc_mod.np, "ascontiguousarray", _spy_ascontiguousarray)

    got = discretize_2d_quantile_batch(raw, n_bins=nbins, dtype=np.int32)

    assert not forbidden["hit"], (
        "discretize_2d_quantile_batch made a full-buffer float64 copy of the float32 "
        "input -- the OOM regression. Pass arr2d to the njit kernel at its native dtype."
    )
    # And it must still be code-identical to the per-column 1-D path (native float32 kernel).
    ref = np.empty_like(got)
    for k in range(K):
        ref[:, k] = discretize_array(arr=raw[:, k], n_bins=nbins, method="quantile", dtype=np.int32)
    assert np.array_equal(got, ref), "native-float32 njit searchsorted diverged from per-column reference"


# ---------------------------------------------------------------------------
# njit external-validation candidate materialisation (_materialise_extval_njit):
# ALL (external_factor x bin_func) columns in one nogil kernel. Must produce the
# DISCRETISED codes identical to the numpy bin_funcs the per-candidate path used
# (raw float64, no nan_to_num) -- signed-zero / last-bit raw differences are
# allowed only insofar as they vanish under discretisation, so we compare the
# CODES (what actually feeds MI), the contract that matters for selection.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n,n_ext", [(200, 5), (500, 12), (800, 3)])
@pytest.mark.parametrize("a_dtype", [np.float32, np.float64])
def test_materialise_extval_njit_codes_match_numpy(n, n_ext, a_dtype):
    bin_t = create_binary_transformations(preset="minimal")
    op_codes = _njit_binary_op_codes(bin_t)
    assert op_codes is not None  # minimal preset is fully njit-coded
    rng = np.random.default_rng(2024 + n + n_ext)
    # param_a includes zeros (div eps path) and extremes (overflow); param_b too.
    param_a = (rng.standard_normal(n) * rng.choice([1.0, 1e6, 0.0], size=n)).astype(a_dtype)
    pb_mat = np.empty((n, n_ext), dtype=np.float64)
    for e in range(n_ext):
        pb_mat[:, e] = rng.standard_normal(n) * rng.choice([1.0, 0.0, -1e-12, 1e8], size=n)

    n_ops = len(bin_t)
    out = np.empty((n, n_ext * n_ops), dtype=np.float64)
    _materialise_extval_njit(np.ascontiguousarray(param_a), pb_mat, op_codes, out)

    # numpy reference in the SAME ext-outer / op-inner column order.
    ref = np.empty((n, n_ext * n_ops), dtype=np.float64)
    col = 0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for e in range(n_ext):
            for fn in bin_t.values():
                ref[:, col] = fn(param_a, pb_mat[:, e])
                col += 1

    # Compare DISCRETISED codes (the MI-relevant contract). Quantile binning of
    # the two raw buffers must be byte-identical.
    got_codes = discretize_2d_quantile_batch(out, n_bins=10, dtype=np.int32)
    ref_codes = discretize_2d_quantile_batch(ref, n_bins=10, dtype=np.int32)
    assert np.array_equal(got_codes, ref_codes), (
        f"extval njit materialise codes != numpy: n={n} n_ext={n_ext} a_dtype={a_dtype}; "
        f"mismatched cols={np.where(~(got_codes == ref_codes).all(axis=0))[0][:10]}"
    )


# ---------------------------------------------------------------------------
# OPT4 (2026-06-07): _quantile_edges_2d_njit replaces np.percentile(axis=0) in
# discretize_2d_quantile_batch (the FE sweep's dominant numpy hotspot: ndarray.partition
# = ~20% of fit, re-partitioning the whole buffer once per quantile). Must be BIT-IDENTICAL
# to np.percentile(method='linear') for BOTH float32 and float64 (numpy sorts in the array
# dtype but lerps in float64), across ties / constant / heavy-tail columns and any nbins.
# ---------------------------------------------------------------------------
from mlframe.feature_selection.filters.discretization import _quantile_edges_2d_njit


def _kths_for(q: np.ndarray, n_rows: int) -> np.ndarray:
    """Order-statistic indices the kernel reads (lo / lo+1 per quantile, n-1 clamped) -- mirrors the caller."""
    _lo = np.floor((q / 100.0) * (n_rows - 1)).astype(np.int64)
    s = set()
    for _l in _lo.tolist():
        if _l >= n_rows - 1:
            s.add(n_rows - 1)
        else:
            s.add(int(_l)); s.add(int(_l) + 1)
    return np.array(sorted(s), dtype=np.int64)


@pytest.mark.parametrize("n,K", [(200, 7), (500, 33), (1500, 16), (300, 4), (2407, 11)])
@pytest.mark.parametrize("nbins", [3, 4, 7, 10, 20])
@pytest.mark.parametrize("dt", [np.float32, np.float64])
def test_quantile_edges_2d_njit_bit_identical_to_numpy(n, K, nbins, dt):
    """``_quantile_edges_2d_njit`` == ``np.percentile(arr, q, axis=0)`` to the BIT."""
    rng = np.random.default_rng(7 + n + K + nbins)
    arr = rng.standard_normal((n, K)).astype(dt)
    q = np.linspace(0, 100, nbins + 1)
    ref = np.percentile(arr, q, axis=0)
    out = np.empty((q.shape[0], K), dtype=np.float64)
    _quantile_edges_2d_njit(np.ascontiguousarray(arr), q, _kths_for(q, n), out)
    assert np.array_equal(ref, out), (
        f"quantile edges != np.percentile: n={n} K={K} nbins={nbins} dtype={dt}; "
        f"maxabs={np.abs(ref - out).max():.3e}"
    )


@pytest.mark.parametrize("dt", [np.float32, np.float64])
def test_quantile_edges_2d_njit_ties_constant_heavytail(dt):
    """Tie-heavy, constant, and heavy-tailed columns must match np.percentile bit-for-bit."""
    rng = np.random.default_rng(123)
    n = 1000
    arr = np.column_stack([
        rng.integers(0, 5, n).astype(dt),                 # tie-heavy
        np.full(n, 2.5, dtype=dt),                        # constant
        rng.standard_t(2.0, n).astype(dt),                # heavy tail
        np.r_[np.zeros(n - 3), [10.0, 20.0, 30.0]].astype(dt),  # near-constant + outliers
    ])
    for nbins in (4, 10, 16):
        q = np.linspace(0, 100, nbins + 1)
        ref = np.percentile(arr, q, axis=0)
        out = np.empty((q.shape[0], arr.shape[1]), dtype=np.float64)
        _quantile_edges_2d_njit(np.ascontiguousarray(arr), q, _kths_for(q, n), out)
        assert np.array_equal(ref, out), f"nbins={nbins} dtype={dt} maxabs={np.abs(ref-out).max():.3e}"
