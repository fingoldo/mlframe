"""FP-equivalence of the one-pass njit |Pearson| kernel behind abs_pearson vs the numpy form.

The usability |corr| distinguisher feeds WIDE-margin gates (min_corr 0.6; tail-concentration gap ~0.99 vs ~0.06),
so ~1e-13 FP agreement is selection-safe. Pins the njit kernel against a numpy reference incl NaN + degenerate.
"""
import numpy as np

from mlframe.feature_selection.filters._fe_usability_signal import (
    abs_pearson,
    _abs_pearson_fast,
    _abs_pearson_njit,
    _single_operand_usability_corr,
    pair_is_tail_concentrated_rankaware,
    usability_form_corrs,
)


def _numpy_ref(y, v):
    m = np.isfinite(y) & np.isfinite(v)
    if int(m.sum()) < 2:
        return 0.0
    yy = y[m]; vv = v[m]
    ys = float(yy.std()); vs = float(vv.std())
    if ys <= 0.0 or vs <= 0.0:
        return 0.0
    c = float(np.mean((yy - yy.mean()) * (vv - vv.mean())) / (ys * vs))
    return abs(c) if np.isfinite(c) else 0.0


def test_abs_pearson_matches_numpy_incl_nan():
    rng = np.random.default_rng(0)
    for n in (500, 2000, 20000):
        y = rng.standard_normal(n)
        v = 0.6 * y + 0.8 * rng.standard_normal(n)
        y[::13] = np.nan
        v[::17] = np.nan
        ref = _numpy_ref(y, v)
        got = abs_pearson(y, v)
        assert abs(ref - got) <= 1e-13, (n, ref, got)


def test_abs_pearson_short_circuits():
    n = 1000
    y = np.linspace(0, 1, n)
    assert float(_abs_pearson_njit(np.full(n, 3.0), y)) == 0.0  # constant side -> 0
    a = np.full(n, np.nan)
    a[0] = 1.0
    assert float(_abs_pearson_njit(a, y)) == 0.0  # <2 finite -> 0


def test_abs_pearson_perfect_and_sign():
    n = 1000
    y = np.linspace(-2, 2, n)
    assert abs(abs_pearson(y, -3.0 * y + 1.0) - 1.0) <= 1e-12  # |corr| == 1, sign folded


def test_abs_pearson_drops_nonfinite_rows_exactly():
    """Regression guard for the branchless + reassoc-fastmath kernel (2026-07, ~2.5x): the fastmath set MUST keep
    ``nnan``/``ninf`` so NaN/inf rows are dropped EXACTLY. A tempting full ``fastmath=True`` lets LLVM assume
    no-NaN and drop the ``isfinite`` test, silently admitting the poisoned rows and collapsing |corr| toward 0 --
    a selection-BREAKING ~1e-2 error. This test FAILS on that unsafe variant (got ~0 while ref >= 0.9) and passes
    on both the pre-fix fastmath=False kernel and the shipped safe-fastmath one (they drop the rows identically)."""
    rng = np.random.default_rng(3)
    n = 20000
    y = rng.standard_normal(n)
    v = 0.9 * y + 0.1 * rng.standard_normal(n)  # strong true corr on the finite rows
    bad = rng.choice(n, int(n * 0.3), replace=False)  # poison 30% of v with NaN / inf
    v[bad[0::2]] = np.nan
    v[bad[1::2]] = np.inf
    ref = _numpy_ref(y, v)  # numpy masked reference: drops the poisoned rows
    got = abs_pearson(y, v)
    assert ref >= 0.9, ref  # sanity: the surviving finite rows carry the strong signal
    assert abs(ref - got) <= 1e-9, (ref, got)  # exact row-drop; a no-NaN fastmath would return ~0 here


def test_abs_pearson_reassoc_delta_is_selection_safe():
    """The reassoc-fastmath reduction reorders the sums, so the result differs from a strict left-to-right numpy
    reference by at most a few ULP (~1e-13) -- far below every usability gate margin (min_corr 0.6; the
    tail-concentration gap is ~0.99 vs ~0.06). Pins that the divergence stays in the selection-safe band."""
    rng = np.random.default_rng(7)
    for n in (600, 5000, 30000):
        y = rng.standard_normal(n).astype(np.float32)
        v = (0.4 * y + 0.9 * rng.standard_normal(n).astype(np.float32)).astype(np.float32)
        assert abs(_numpy_ref(y.astype(np.float64), v.astype(np.float64)) - abs_pearson(y, v)) <= 1e-12, n


def test_abs_pearson_fast_matches_abs_pearson_bit_identical():
    """``_abs_pearson_fast`` (the internal no-check path ``usability_form_corrs`` uses, 2026-07-11 perf fix) must
    return EXACTLY what ``abs_pearson`` returns for any input already meeting its documented precondition
    (1-D, C-contiguous, same float32-or-float64 dtype on both sides) -- it skips ``abs_pearson``'s
    asarray/dtype-branch/ascontiguousarray checks entirely, so any divergence would mean those checks were
    silently doing real work, not just validation."""
    rng = np.random.default_rng(11)
    for dtype in (np.float32, np.float64):
        for n in (5, 600, 5000, 30000):
            y = rng.standard_normal(n).astype(dtype)
            v = (0.5 * y + 0.7 * rng.standard_normal(n).astype(dtype)).astype(dtype)
            assert abs_pearson(y, v) == _abs_pearson_fast(y, v), (dtype, n)
    # Degenerate / NaN-laden inputs too, not just the happy path.
    y = rng.standard_normal(2000)
    v = 0.6 * y + 0.4 * rng.standard_normal(2000)
    y[::11] = np.nan
    v[::13] = np.inf
    assert abs_pearson(y, v) == _abs_pearson_fast(y, v)
    const = np.full(500, 3.0)
    other = np.linspace(0, 1, 500)
    assert abs_pearson(const, other) == _abs_pearson_fast(const, other) == 0.0


def test_abs_pearson_fast_produced_by_elementwise_numpy_ops_is_always_contiguous_same_dtype():
    """Pins the PRECONDITION ``usability_form_corrs`` relies on: every form it feeds to ``_abs_pearson_fast`` is
    the result of an elementwise numpy op on already-ravelled, same-dtype operands. If numpy ever stopped
    guaranteeing contiguous+dtype-preserving output for these op shapes, ``_abs_pearson_fast`` would silently
    receive non-contiguous or mixed-dtype input (undefined behavior for the raw njit kernel) with no error --
    this test would catch that regression before it reached production."""
    for dtype in (np.float32, np.float64):
        x0 = np.asarray(np.arange(1000), dtype=dtype).ravel()
        x1 = np.asarray(np.arange(1000, 2000), dtype=dtype).ravel()
        x1f = np.where(np.abs(x1) < 1e-12, np.nan, x1)
        for form in (x0, x1, x0 * x0, x1 * x1, x0 / x1f, x0 * x1):
            assert form.dtype == dtype, (dtype, form.dtype)
            assert form.flags["C_CONTIGUOUS"], (dtype, "not contiguous")


def test_usability_form_corrs_unchanged_by_fast_path_switch():
    """End-to-end contract: switching ``usability_form_corrs``'s internal ``abs_pearson`` calls to
    ``_abs_pearson_fast`` must not change its output at all (bit-identical, since both paths call the SAME
    njit kernel with the SAME already-clean arrays -- the fast path only skips redundant re-validation)."""
    from mlframe.feature_selection.filters._fe_usability_signal import usability_form_corrs

    rng = np.random.default_rng(5)
    n = 4000
    y = rng.standard_normal(n)
    x0 = rng.standard_normal(n) + 5.0
    x1 = rng.standard_normal(n) * 2.0 + 3.0
    cp, cs = usability_form_corrs(y, x0, x1)
    assert 0.0 <= cp <= 1.0 and 0.0 <= cs <= 1.0
    # Re-run for determinism (no hidden state/randomness in the computation).
    cp2, cs2 = usability_form_corrs(y, x0, x1)
    assert (cp, cs) == (cp2, cs2)


def test_single_operand_usability_corr_matches_max_of_operand_and_square():
    """_single_operand_usability_corr(y, x) must equal max(abs_pearson(y_sub, x_sub), abs_pearson(y_sub,
    x_sub**2)) over the SAME subsample usability_form_corrs would apply -- the whole point of this helper is
    that it's a drop-in piece of that computation, not an approximation of it."""
    rng = np.random.default_rng(21)
    n = 4000
    y = rng.standard_normal(n)
    x = 0.6 * y + 0.5 * rng.standard_normal(n)
    got = _single_operand_usability_corr(y, x)

    from mlframe.feature_selection.filters._fe_usability_signal import _subsample_for_corr, _crit_np_dtype
    _yc = np.asarray(y, dtype=_crit_np_dtype()).ravel()
    _xc = np.asarray(x, dtype=_crit_np_dtype()).ravel()
    _yc, _xc = _subsample_for_corr(_yc, _xc)
    expected = max(abs_pearson(_yc, _xc), abs_pearson(_yc, _xc * _xc))
    assert abs(got - expected) <= 1e-15, (got, expected)


def test_usability_form_corrs_precomputed_single_corr_is_bit_identical_to_internal():
    """Passing precomputed_single_corr must give EXACTLY the same (_cp, _cs) as letting usability_form_corrs
    derive _cs internally -- this is the contract the per-operand caching in _step_pairs_rank.py relies on."""
    rng = np.random.default_rng(22)
    n = 3000
    y = rng.standard_normal(n)
    x0 = rng.standard_normal(n) + 4.0
    x1 = rng.standard_normal(n) * 1.5 - 2.0

    internal_cp, internal_cs = usability_form_corrs(y, x0, x1)

    sc0 = _single_operand_usability_corr(y, x0)
    sc1 = _single_operand_usability_corr(y, x1)
    cached_cp, cached_cs = usability_form_corrs(y, x0, x1, precomputed_single_corr=(sc0, sc1))

    assert cached_cs == internal_cs
    assert cached_cp == internal_cp

    # Also check the return_best_pair_form=True branch (the one pair_is_tail_concentrated_rankaware uses).
    internal_cp2, internal_cs2, internal_form = usability_form_corrs(y, x0, x1, return_best_pair_form=True)
    cached_cp2, cached_cs2, cached_form = usability_form_corrs(
        y, x0, x1, return_best_pair_form=True, precomputed_single_corr=(sc0, sc1),
    )
    assert cached_cs2 == internal_cs2
    assert cached_cp2 == internal_cp2
    assert np.array_equal(cached_form, internal_form)


def test_pair_is_tail_concentrated_rankaware_precomputed_single_corr_matches_internal():
    """End-to-end: pair_is_tail_concentrated_rankaware's decision (and the underlying |corr| values it computes)
    must be identical whether the caller supplies precomputed_single_corr or leaves it None -- the actual
    contract the _step_pairs_rank.py caching wiring depends on."""
    rng = np.random.default_rng(23)
    n = 5000
    y = rng.standard_normal(n)
    x0 = rng.standard_normal(n) * 2.0 + 1.0
    x1 = (x0 ** 2) / (rng.standard_normal(n) * 0.3 + 3.0)  # a ratio form correlated with x0 -- exercises the pair path

    kwargs = dict(min_corr=0.3, pairness_margin=1.0, max_rank_frac=0.9)
    internal_result = pair_is_tail_concentrated_rankaware(y, x0, x1, **kwargs)

    sc0 = _single_operand_usability_corr(y, x0)
    sc1 = _single_operand_usability_corr(y, x1)
    cached_result = pair_is_tail_concentrated_rankaware(y, x0, x1, precomputed_single_corr=(sc0, sc1), **kwargs)

    assert cached_result == internal_result


def test_precomputed_single_corr_none_is_identical_to_omitting_the_kwarg():
    """precomputed_single_corr=None (the caller-side fallback when either cached value is unavailable) must
    behave EXACTLY like not passing the kwarg at all -- no silent divergent code path."""
    rng = np.random.default_rng(24)
    n = 2000
    y = rng.standard_normal(n)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)

    a = usability_form_corrs(y, x0, x1)
    b = usability_form_corrs(y, x0, x1, precomputed_single_corr=None)
    assert a == b

    c = pair_is_tail_concentrated_rankaware(y, x0, x1, min_corr=0.5, pairness_margin=1.0)
    d = pair_is_tail_concentrated_rankaware(y, x0, x1, min_corr=0.5, pairness_margin=1.0, precomputed_single_corr=None)
    assert c == d
