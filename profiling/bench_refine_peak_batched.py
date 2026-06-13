"""Microbench: batched-periodogram refine-peak vs the scalar per-freq scan loop.

The frequency-detector's ``_refine_peak_freq`` evaluates periodogram power at ~20
scalar frequencies via repeated ``_power_centered`` calls, each doing its own
``np.sin``/``np.cos`` over the full train slice + two centered dot products. The
batched variant stacks all scan frequencies into one matrix sin/cos eval + a
single matmul against the (pre-centered) y. Both must return the SAME argmax freq.

Run: PYTHONPATH=src python profiling/bench_refine_peak_batched.py
"""
from __future__ import annotations
import time
import numpy as np


def _corr_sq_centered(v, y_centered, y_ss):
    vc = v - v.mean()
    v_ss = float(vc @ vc)
    if v_ss < 1e-24 or y_ss < 1e-24:
        return 0.0
    num = float(vc @ y_centered)
    return (num * num) / (v_ss * y_ss)


def _power_centered(z, yc, y_ss, freq):
    ang = 2.0 * np.pi * float(freq) * z
    return _corr_sq_centered(np.sin(ang), yc, y_ss) + _corr_sq_centered(np.cos(ang), yc, y_ss)


def _scan_scalar(z_tr, yc, y_ss, center, half_width, step):
    lo_r = max(0.05, center - half_width)
    hi_r = center + half_width
    n_steps = int(round((hi_r - lo_r) / step)) + 1
    best_f = center
    best_p = _power_centered(z_tr, yc, y_ss, center)
    for k in range(n_steps):
        f = lo_r + step * k
        p = _power_centered(z_tr, yc, y_ss, f)
        if p > best_p:
            best_p = p
            best_f = f
    return best_f, best_p


def _refine_scalar(z_tr, yc, y_ss, coarse_f):
    f1, _ = _scan_scalar(z_tr, yc, y_ss, coarse_f, 0.25, 0.05)
    f2, _ = _scan_scalar(z_tr, yc, y_ss, f1, 0.05, 0.0125)
    return float(f2)


def _batched_powers(z_tr, yc, y_ss, freqs):
    """Periodogram power at each freq, batched. freqs: 1D array length m.
    Returns power array length m -- bit-identical to _power_centered per freq."""
    if y_ss < 1e-24:
        return np.zeros(len(freqs), dtype=np.float64)
    ang = (2.0 * np.pi) * np.outer(np.asarray(freqs, dtype=np.float64), z_tr)  # (m, n)
    S = np.sin(ang)
    C = np.cos(ang)
    Sc = S - S.mean(axis=1, keepdims=True)
    Cc = C - C.mean(axis=1, keepdims=True)
    s_ss = np.einsum("ij,ij->i", Sc, Sc)
    c_ss = np.einsum("ij,ij->i", Cc, Cc)
    num_s = Sc @ yc
    num_c = Cc @ yc
    p = np.zeros(len(freqs), dtype=np.float64)
    ms = s_ss >= 1e-24
    mc = c_ss >= 1e-24
    p[ms] += (num_s[ms] * num_s[ms]) / (s_ss[ms] * y_ss)
    p[mc] += (num_c[mc] * num_c[mc]) / (c_ss[mc] * y_ss)
    return p


def _scan_batched(z_tr, yc, y_ss, center, half_width, step):
    lo_r = max(0.05, center - half_width)
    hi_r = center + half_width
    n_steps = int(round((hi_r - lo_r) / step)) + 1
    freqs = np.empty(n_steps + 1, dtype=np.float64)
    freqs[0] = center
    for k in range(n_steps):
        freqs[k + 1] = lo_r + step * k
    powers = _batched_powers(z_tr, yc, y_ss, freqs)
    # Match the scalar loop's tie-break: center is the initial best; a later
    # freq replaces it only on STRICT >. Walk in the same order.
    best_f = freqs[0]
    best_p = powers[0]
    for k in range(n_steps):
        if powers[k + 1] > best_p:
            best_p = powers[k + 1]
            best_f = freqs[k + 1]
    return best_f, best_p


def _refine_batched(z_tr, yc, y_ss, coarse_f):
    f1, _ = _scan_batched(z_tr, yc, y_ss, coarse_f, 0.25, 0.05)
    f2, _ = _scan_batched(z_tr, yc, y_ss, f1, 0.05, 0.0125)
    return float(f2)


def main():
    rng = np.random.default_rng(0)
    # scene train slice ~= 2500 * 2/3 ~= 1667
    for n in (800, 1667, 2500, 5000):
        z = np.sort(rng.random(n))
        yc_all = []
        # mix of genuine-oscillation + noise targets
        for trial in range(60):
            f_true = 0.3 + 6.0 * rng.random()
            y = np.sin(2 * np.pi * f_true * z) + 0.3 * rng.standard_normal(n)
            yc = y - y.mean()
            yc_all.append((yc, float(yc @ yc), 0.3 + 6.0 * rng.random()))
        # correctness
        max_diff = 0.0
        for yc, y_ss, cf in yc_all:
            a = _refine_scalar(z, yc, y_ss, cf)
            b = _refine_batched(z, yc, y_ss, cf)
            max_diff = max(max_diff, abs(a - b))
        # warm + time
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            for yc, y_ss, cf in yc_all:
                _refine_scalar(z, yc, y_ss, cf)
        t_sc = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(reps):
            for yc, y_ss, cf in yc_all:
                _refine_batched(z, yc, y_ss, cf)
        t_ba = time.perf_counter() - t0
        print(f"n={n:5d}  scalar={t_sc*1e3:8.2f}ms  batched={t_ba*1e3:8.2f}ms  "
              f"speedup={t_sc/t_ba:.2f}x  max|df|={max_diff:.2e}")


if __name__ == "__main__":
    main()
