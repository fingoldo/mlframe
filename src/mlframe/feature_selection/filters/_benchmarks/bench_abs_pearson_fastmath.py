"""A/B bench for the ``_fe_usability_signal._abs_pearson_njit`` fastmath+branchless optimisation (2026-07).

RESOLVED: branchless select + ``fastmath={'reassoc','contract','arcp','afn','nsz'}`` (nnan/ninf DELIBERATELY
excluded so the NaN-drop survives) is 2.1-2.5x over the old plain-branch ``fastmath=False`` kernel at every
n in 600..30000, selection-equivalent (diff <=~1e-16 single ULP, NaN rows dropped EXACTLY).

Also documents REJECTED alternatives so the negative results are reproducible:
  * batch-9-forms-into-1-njit-dispatch: 0.33x at n=30000 -- compute-bound not dispatch-bound; the f64 matrix
    materialisation + strided column writes cost more than the 9 saved dispatches.
  * full ``fastmath=True``: 4x BUT silently returns 0.0 on NaN data (LLVM drops the isfinite test) -> a
    selection-breaking ~1e-2 error. UNSAFE.
  * 2026-07-11 re-tried the batch-forms idea with a per-row-outer/per-form-inner loop layout (see
    ``_abs_pearson_batch_njit`` below): 1.5x SLOWER (205us/form vs 138us/form at n=30000, k=9) -- the scatter
    writes into k-sized accumulator arrays inside the row loop still defeat the flat branchless/reassoc
    vectorisation the single-form kernel gets. Independently confirms the original batching rejection above
    under a different implementation.
  * 2026-07-11 shifted one-pass formula (see ``_abs_pearson_onepass_njit`` below): 2.19x FASTER (33us vs 73us,
    n=30000) but reproduces the exact catastrophic-cancellation failure the two-pass mean-then-center design
    was written to fix -- a near-constant column (``1.0 + linspace(0, 1e-15, 1000)``) against random noise
    returns ``|corr|`` ~0.029 instead of the correct ~0.0. Shifting by a single sample point (``y[0]``) does
    not reliably keep the accumulated terms at the data's own spread scale the way full mean-centering does.
    UNSAFE, not shipped.
  * 2026-07-11 ACCEPTED: ``_abs_pearson_fast`` (skips ``abs_pearson``'s ``np.asarray``/dtype-branch/
    ``np.ascontiguousarray`` wrapper checks for callers -- ``usability_form_corrs`` -- that can guarantee
    already-contiguous same-dtype input) is bit-identical and a genuine but modest ~3.8% faster (58.2us vs
    60.5us/call at n=30000); most of the per-call cost is the kernel's own two-pass traversal, not the wrapper.

Run:  python path/to/_benchmarks/bench_abs_pearson_fastmath.py
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit

from mlframe.feature_selection.filters._fe_usability_signal import _abs_pearson_njit, abs_pearson  # noqa: F401


# The pre-fix baseline (plain branch, fastmath=False) reconstructed for the A/B OLD side.
@njit(cache=True, fastmath=False)
def _abs_pearson_baseline(y, v):
    n = 0
    sa = 0.0; sv = 0.0; saa = 0.0; svv = 0.0; sav = 0.0
    for i in range(y.shape[0]):
        a = np.float64(y[i]); b = np.float64(v[i])
        if np.isfinite(a) and np.isfinite(b):
            n += 1
            sa += a; sv += b; saa += a * a; svv += b * b; sav += a * b
    if n < 2:
        return 0.0
    inv = 1.0 / n
    va = saa - sa * sa * inv
    vv2 = svv - sv * sv * inv
    if va <= 0.0 or vv2 <= 0.0:
        return 0.0
    den = (va * vv2) ** 0.5
    if den <= 0.0:
        return 0.0
    c = (sav - sa * sv * inv) / den
    if not np.isfinite(c):
        return 0.0
    return -c if c < 0.0 else c


@njit(cache=True, fastmath={"reassoc", "contract", "arcp", "afn", "nsz"})
def _abs_pearson_batch_njit(y, forms):
    """REJECTED (2026-07-11): row-outer/form-inner batched kernel, k forms sharing the y[i] read per row. See
    module docstring for the measured 1.5x-slower result."""
    n_rows = y.shape[0]
    k = forms.shape[1]
    n_arr = np.zeros(k, dtype=np.int64)
    sa_arr = np.zeros(k, dtype=np.float64)
    sv_arr = np.zeros(k, dtype=np.float64)
    for i in range(n_rows):
        a = np.float64(y[i])
        a_finite = np.isfinite(a)
        for j in range(k):
            b = np.float64(forms[i, j])
            finite = a_finite and np.isfinite(b)
            av = a if finite else 0.0
            bv = b if finite else 0.0
            n_arr[j] += finite
            sa_arr[j] += av
            sv_arr[j] += bv
    ma_arr = np.zeros(k, dtype=np.float64)
    mv_arr = np.zeros(k, dtype=np.float64)
    for j in range(k):
        if n_arr[j] >= 2:
            inv = 1.0 / n_arr[j]
            ma_arr[j] = sa_arr[j] * inv
            mv_arr[j] = sv_arr[j] * inv
    saa_arr = np.zeros(k, dtype=np.float64)
    svv_arr = np.zeros(k, dtype=np.float64)
    sav_arr = np.zeros(k, dtype=np.float64)
    for i in range(n_rows):
        a = np.float64(y[i])
        a_finite = np.isfinite(a)
        for j in range(k):
            b = np.float64(forms[i, j])
            finite = a_finite and np.isfinite(b)
            da = (a - ma_arr[j]) if finite else 0.0
            db = (b - mv_arr[j]) if finite else 0.0
            saa_arr[j] += da * da
            svv_arr[j] += db * db
            sav_arr[j] += da * db
    result = np.zeros(k, dtype=np.float64)
    _cv2 = 1e-16
    for j in range(k):
        if n_arr[j] < 2:
            continue
        if saa_arr[j] <= n_arr[j] * _cv2 * ma_arr[j] * ma_arr[j] or svv_arr[j] <= n_arr[j] * _cv2 * mv_arr[j] * mv_arr[j]:
            continue
        den = (saa_arr[j] * svv_arr[j]) ** 0.5
        if den <= 0.0:
            continue
        c = sav_arr[j] / den
        if not np.isfinite(c):
            continue
        result[j] = -c if c < 0.0 else c
    return result


@njit(cache=True, fastmath={"reassoc", "contract", "arcp", "afn", "nsz"})
def _abs_pearson_onepass_njit(y, v):
    """REJECTED (2026-07-11): shifted one-pass formula (shift = y[0]/v[0], SIMD-friendly, no cross-iteration
    dependency unlike Welford). Faster but numerically unsafe -- see module docstring."""
    n = y.shape[0]
    ky = np.float64(y[0]) if np.isfinite(np.float64(y[0])) else 0.0
    kv = np.float64(v[0]) if np.isfinite(np.float64(v[0])) else 0.0
    cnt = 0
    sa = 0.0; sv = 0.0; saa = 0.0; svv = 0.0; sav = 0.0
    for i in range(n):
        a = np.float64(y[i]); b = np.float64(v[i])
        finite = np.isfinite(a) and np.isfinite(b)
        da = (a - ky) if finite else 0.0
        db = (b - kv) if finite else 0.0
        cnt += finite
        sa += da; sv += db
        saa += da * da; svv += db * db; sav += da * db
    if cnt < 2:
        return 0.0
    inv = 1.0 / cnt
    var_a = saa - sa * sa * inv
    var_v = svv - sv * sv * inv
    if var_a <= 0.0 or var_v <= 0.0:
        return 0.0
    den = (var_a * var_v) ** 0.5
    if den <= 0.0:
        return 0.0
    c = (sav - sa * sv * inv) / den
    if not np.isfinite(c):
        return 0.0
    return -c if c < 0.0 else c


def main() -> None:
    rng = np.random.default_rng(1)
    R = 3000
    print(f"{'n':>7} {'nan':>5} {'old_us':>9} {'new_us':>9} {'speedup':>8} {'maxdiff':>10}")
    for n in (600, 5000, 30000):
        for frac in (0.0, 0.1):
            y = rng.standard_normal(n).astype(np.float32)
            v = rng.standard_normal(n).astype(np.float32).copy()
            if frac > 0:
                v[rng.choice(n, int(n * frac), replace=False)] = np.nan
            old = _abs_pearson_baseline(y, v)
            new = _abs_pearson_njit(y, v)
            diff = abs(old - new)
            t = time.perf_counter()
            for _ in range(R):
                _abs_pearson_baseline(y, v)
            t_old = (time.perf_counter() - t) / R * 1e6
            t = time.perf_counter()
            for _ in range(R):
                _abs_pearson_njit(y, v)
            t_new = (time.perf_counter() - t) / R * 1e6
            print(f"{n:>7} {frac:>5.0%} {t_old:>9.1f} {t_new:>9.1f} {t_old / t_new:>7.2f}x {diff:>10.1e}")

    print("\n-- rejected alternatives (2026-07-11) --")
    n = 30000
    y = rng.standard_normal(n).astype(np.float32)
    v = (0.5 * y + 0.7 * rng.standard_normal(n).astype(np.float32)).astype(np.float32)
    k = 9
    forms = np.ascontiguousarray(rng.standard_normal((n, k)).astype(np.float32))
    forms_list = [np.ascontiguousarray(forms[:, j]) for j in range(k)]
    _abs_pearson_batch_njit(y, forms)  # warm
    _abs_pearson_onepass_njit(y, v)  # warm

    Rb = 3000
    t = time.perf_counter()
    for _ in range(Rb):
        _abs_pearson_batch_njit(y, forms)
    t_batch = (time.perf_counter() - t) / Rb / k * 1e6
    t = time.perf_counter()
    for _ in range(Rb):
        for f in forms_list:
            _abs_pearson_njit(y, f)
    t_loop = (time.perf_counter() - t) / Rb / k * 1e6
    print(f"batched-9-forms: {t_batch:.1f}us/form vs {t_loop:.1f}us/form separate ({t_loop / t_batch:.2f}x)")

    Ro = 20000
    t = time.perf_counter()
    for _ in range(Ro):
        _abs_pearson_onepass_njit(y, v)
    t_one = (time.perf_counter() - t) / Ro * 1e6
    t = time.perf_counter()
    for _ in range(Ro):
        _abs_pearson_njit(y, v)
    t_two = (time.perf_counter() - t) / Ro * 1e6
    print(f"one-pass shifted: {t_one:.1f}us/call vs two-pass {t_two:.1f}us/call ({t_two / t_one:.2f}x faster, UNSAFE)")

    x_const = (1.0 + np.linspace(0, 1e-15, 1000)).astype(np.float64)
    y_rand = rng.standard_normal(1000)
    print(
        f"near-constant-column safety check: onepass={_abs_pearson_onepass_njit(x_const, y_rand):.4f} "
        f"(want ~0) vs twopass={float(_abs_pearson_njit(x_const, y_rand)):.4f} (want ~0)"
    )


if __name__ == "__main__":
    main()
