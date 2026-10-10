"""Stress test of the zero-crossing / OLS / grid shift estimators across scenarios (monotone, non-monotone, multi-crossing, heavy tail, discrete u, no shift) and noise levels.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.exp8_zc_stress <reps> <A|B|C>  (A: monotone n x SNR sweep, B: bin-count sweep, C: other scenarios)``."""

import sys

import numpy as np

from .core import bins_of, mi_cnt, ols_t, qbin, rank01, zc_t

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
rng = np.random.default_rng(77)


def mi_of(f, yb):
    """MI of a feature against target codes (10 bins)."""
    return mi_cnt(bins_of(f, 10), yb, 10, 10, False)


def gen(scn, n, snr, tc):
    """Draw one scenario: returns (u, v, truth, y, true crossing)."""
    x = rng.random(n)
    z = rng.random(n)
    u = x
    v = z
    if scn == "mono":
        tr = (x - tc) * z
    elif scn == "nonmono_vid":
        tr = (x - tc) * np.cos(2 * np.pi * z)  # v=z identity (mismatched)
    elif scn == "nonmono_vright":
        tr = (x - tc) * np.cos(2 * np.pi * z)
        v = np.cos(2 * np.pi * z)
    elif scn == "multicross":
        tr = np.sin(3 * np.pi * x) * z
        tc = np.nan  # crossings at 1/3, 2/3: no single t
    elif scn == "heavytail":
        tr = (x - tc) * z
    elif scn == "discrete_u":
        x = rng.integers(0, 6, n) / 5.0
        u = x
        tr = (x - tc) * z
    elif scn == "noshift":
        tr = x * z
        tc = 0.0
    sd = tr.std() / np.sqrt(snr)
    noise = rng.standard_t(1.5, n) * sd / 2 if scn == "heavytail" else rng.standard_normal(n) * sd
    return u, v, tr, tr + noise, tc


def estimators(u, v, y, yb, yr, nbs):
    """Shift estimates of all estimators for one draw."""
    est = {"t0": 0.0, "median": -np.median(u)}
    for nb in nbs:
        est[f"zc{nb}"] = zc_t(u, v, yr, nb, 0.0, True, False)
        est[f"zc{nb}_sig+med"] = zc_t(u, v, yr, nb, 2.0, True, True)
        est[f"zc{nb}_mid"] = zc_t(u, v, yr, nb, 0.0, False, False)
    est["ols"] = ols_t(u, v, yr, False, False)
    est["ols_winsor"] = ols_t(u, v, yr, True, False)
    est["ols_huber+wins"] = ols_t(u, v, yr, True, True)
    us = np.sort(u)
    best = (-1, 0.0)
    for q in np.linspace(0.1, 0.9, 9):
        t = -us[int(q * (len(u) - 1))]
        m = mi_of((u + t) * v, yb)
        if m > best[0]:
            best = (m, t)
    est["grid9"] = best[1]
    for k_, t in est.items():
        if np.isnan(t):
            est[k_] = 0.0
    return est


def run(scn, n, snr, nbs, reps):
    """Regret, captured gain, failure rate and shift error per estimator over replicates."""
    acc = {}
    for r in range(reps):
        tc = [0.3, 0.5, 0.7][r % 3]
        u, v, tr, y, tcv = gen(scn, n, snr, tc)
        yb = qbin(y, 10)
        yr = rank01(y)
        m0 = mi_of(u * v, yb)
        mT = mi_of((u - tcv) * v, yb) if not np.isnan(tcv) else max(mi_of((u - c) * v, yb) for c in (0.2, 0.33, 0.5, 0.67, 0.8))
        for k_, t in estimators(u, v, y, yb, yr, nbs).items():
            m = mi_of((u + t) * v, yb)
            a = acc.setdefault(k_, dict(regret=[], cap=[], terr=[], fail=[]))
            a["regret"].append(mT - m)
            if mT - m0 > 0.003:
                c = float(np.clip((m - m0) / (mT - m0), -1, 1))
                a["cap"].append(c)
                a["fail"].append(c < 0.5)
            if not np.isnan(tcv):
                a["terr"].append(abs(t + tcv) / u.std())
    return acc


def show(title, acc, keys=None):
    """Print one summary table."""
    print(f"\n== {title}")
    for k_, a in acc.items():
        if keys and not any(k_.startswith(p) for p in keys):
            continue
        print(
            f"  {k_:16s} regret {np.mean(a['regret']):+.4f}  captured {np.mean(a['cap']) if a['cap'] else float('nan'):.2f}  fail {np.mean(a['fail']) if a['fail'] else float('nan'):.2f}  |t err|/sd(u) {np.mean(a['terr']) if a['terr'] else float('nan'):.2f}"
        )


mode = sys.argv[2] if len(sys.argv) > 2 else "A"
if mode == "A":  # mono, n x snr sweep, estimators except nb sweep (nb=8)
    for n in (2000, 5000, 30000, 100000):
        for snr in (3, 1, 0.3, 0.1):
            show(f"mono n={n} snr={snr} (reps={REPS})", run("mono", n, snr, (8,), REPS), keys=("t0", "median", "zc8", "ols", "grid"))
elif mode == "B":  # nb sweep at n=5k,30k snr 1,.3
    for n in (5000, 30000):
        for snr in (1, 0.3):
            show(f"mono nb sweep n={n} snr={snr}", run("mono", n, snr, (6, 8, 12, 16), REPS), keys=("zc",))
else:  # other scenarios
    for scn in ("nonmono_vid", "nonmono_vright", "multicross", "heavytail", "discrete_u", "noshift"):
        for n in (5000, 30000):
            for snr in (1, 0.3):
                show(f"{scn} n={n} snr={snr}", run(scn, n, snr, (8,), REPS), keys=("t0", "median", "zc8", "ols", "grid"))
