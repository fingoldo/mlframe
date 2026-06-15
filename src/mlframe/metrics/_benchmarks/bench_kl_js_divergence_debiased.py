"""Bench: Miller-Madow bias correction for binned KL / JS drift divergence (qual-6).

The binned plug-in KL(P||Q) and JS(P,Q) are positively biased in finite samples: the empirical histograms
over-resolve the two distributions, so both sit clearly above the true divergence even when P and Q are drawn from
the SAME distribution (true divergence 0). The bias is a Miller-Madow MI/entropy-bias floor:

    KL : (Kp-1)/(2 np) + (Kq-1)/(2 nq)   [Kp/Kq occupied bins, np/nq sample sizes]
    JS : (K-1)/(2 N)                      [K occupied pooled bins, N = np+nq]  (JS == I(label; bin))

This bench measures |estimate - ground-truth divergence| across multiple seeds AND scenarios (same-distribution +
shifted) at several (n, nbins) configs, comparing the plug-in (``bias_correction=False``) against the corrected
default (``bias_correction=True``). RESOLVED on a majority-of-cells win.

Run:  PYTHONPATH=src python src/mlframe/metrics/_benchmarks/bench_kl_js_divergence_debiased.py
"""
from __future__ import annotations

import numpy as np

from mlframe.metrics._drift import kl_divergence, js_divergence, _safe_quantile_bins, _bin_counts

SEEDS = list(range(8))
CONFIGS = [(200, 20), (200, 50), (1000, 20), (1000, 50)]
SHIFTS = {"same": 0.0, "shift0.5": 0.5, "shift1.0": 1.0}


def _truth(metric: str, mu: float, ngrid: int = 400, big: int = 400_000) -> float:
    if mu == 0.0:
        return 0.0
    rng = np.random.default_rng(7)
    a = rng.normal(mu, 1.0, big); b = rng.normal(0.0, 1.0, big)
    if metric == "kl":
        edges = _safe_quantile_bins(b, ngrid)
        p = _bin_counts(a, edges); p /= p.sum(); q = _bin_counts(b, edges); q /= q.sum()
    else:
        edges = _safe_quantile_bins(np.concatenate([a, b]), ngrid)
        p = _bin_counts(a, edges); p /= p.sum(); q = _bin_counts(b, edges); q /= q.sum()
    m = 0.5 * (p + q); eps = 1e-12
    def kl(u, v):
        s = 0.0
        for i in range(len(u)):
            if u[i] > 0:
                s += u[i] * np.log(u[i] / max(v[i], eps))
        return s
    return kl(p, q) if metric == "kl" else 0.5 * kl(p, m) + 0.5 * kl(q, m)


def run() -> dict:
    out = {}
    for metric, fn in (("kl", kl_divergence), ("js", js_divergence)):
        truth = {nm: _truth(metric, mu) for nm, mu in SHIFTS.items()}
        wins = total = 0
        rows = []
        for n, nbins in CONFIGS:
            for nm, mu in SHIFTS.items():
                pe = []; ce = []; w = 0
                for s in SEEDS:
                    rng = np.random.default_rng(s)
                    a = rng.normal(mu, 1.0, n); b = rng.normal(0.0, 1.0, n)
                    # reference=b (Q), target=a (P) per public signature.
                    plug = fn(b, a, nbins=nbins, bias_correction=False)
                    corr = fn(b, a, nbins=nbins, bias_correction=True)
                    ep = abs(plug - truth[nm]); ec = abs(corr - truth[nm])
                    pe.append(ep); ce.append(ec)
                    if ec < ep:
                        w += 1
                wins += w; total += len(SEEDS)
                rows.append((n, nbins, nm, truth[nm], float(np.mean(pe)), float(np.mean(ce)), w, len(SEEDS)))
        out[metric] = {"rows": rows, "wins": wins, "total": total}
    return out


if __name__ == "__main__":
    res = run()
    for metric in ("kl", "js"):
        d = res[metric]
        print(f"=== {metric.upper()} (Miller-Madow debias vs plug-in) ===")
        print(f"{'n':>5} {'nbins':>5} {'scenario':>9} {'truth':>7} {'plugin|b|':>10} {'corr|b|':>9} {'win':>6}")
        for n, nbins, nm, t, pb, cb, w, tot in d["rows"]:
            print(f"{n:>5} {nbins:>5} {nm:>9} {t:>7.4f} {pb:>10.5f} {cb:>9.5f} {w:>3}/{tot}")
        print(f"  corrected-wins = {d['wins']}/{d['total']}")
