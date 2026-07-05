"""PID XOR synergy bench (2026-05-30 Wave 8).

Quantifies the Williams-Beer + Ince I_ccs PID decomposition on the
canonical synergistic gate: ``y = x1 XOR x2``. Expected theoretical
decomposition under H0=full XOR:

    Unique(X1; Y) = 0
    Unique(X2; Y) = 0
    Redundant(X1, X2; Y) = 0
    Synergistic(X1, X2; Y) = H(Y) = ln(2) = 0.693 nats

Then noises the input gradually and tracks how synergy decays vs how a
naive MRMR / plug-in MI would score the same features. Establishes the
PID synergy threshold above which features should bypass the redundancy
gate.

Reference: Ince, R. (2017), "Measuring Multivariate Redundant Information
with Pointwise Common Change in Surprisal", Entropy 19(7):318.
arXiv:1602.05063

Run::

    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_pid_xor_synergy
"""
from __future__ import annotations

import math
import time

import numpy as np

from mlframe.feature_selection.filters._pid_decomposition import pid_decomposition
from mlframe.feature_selection.filters._adaptive_nbins import _plug_in_mi


def gen_xor(n: int, noise: float, seed: int = 0):
    """Generate ``y = x1 XOR x2`` with ``noise`` fraction of bit-flips on y."""
    rng = np.random.default_rng(int(seed))
    x1 = rng.integers(0, 2, n).astype(np.int64)
    x2 = rng.integers(0, 2, n).astype(np.int64)
    y_clean = x1 ^ x2
    flip = rng.random(n) < noise
    y = np.where(flip, 1 - y_clean, y_clean).astype(np.int64)
    return x1, x2, y


def naive_mi(x: np.ndarray, y: np.ndarray) -> float:
    """Plug-in I(X; Y) on integer-encoded arrays (no Miller-Madow)."""
    return float(_plug_in_mi(x.astype(np.int64), y.astype(np.int64), miller_madow=False))


def main():
    print("=" * 72)
    print("PID XOR synergy bench -- Williams-Beer + Ince I_ccs")
    print("=" * 72)
    print()
    print(f"{'noise':>6} {'I(X1;Y)':>10} {'I(X2;Y)':>10} {'I(X1X2;Y)':>10}" f" {'PID_red':>10} {'PID_unq1':>10} {'PID_unq2':>10}" f" {'PID_syn':>10}")
    print("-" * 92)
    n = 8000
    for noise in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        x1, x2, y = gen_xor(n, noise)
        mi_x1 = naive_mi(x1, y)
        mi_x2 = naive_mi(x2, y)
        # Joint I({X1,X2}; Y) via composite encoding.
        composite = (x1 * 2 + x2).astype(np.int64)
        mi_joint = naive_mi(composite, y)
        pid = pid_decomposition(x1, x2, y, 2, 2, 2)
        print(f"{noise:>6.2f} {mi_x1:>10.4f} {mi_x2:>10.4f} {mi_joint:>10.4f}"
              f" {pid['redundant']:>10.4f} {pid['unique_x1']:>10.4f}"
              f" {pid['unique_x2']:>10.4f} {pid['synergistic']:>10.4f}")

    print()
    print("Headline check on noise=0.0 XOR:")
    x1, x2, y = gen_xor(50_000, 0.0)
    pid = pid_decomposition(x1, x2, y, 2, 2, 2)
    truth_synergy = math.log(2.0)
    print(f"  Theoretical synergy: ln(2) = {truth_synergy:.4f}")
    print(f"  Measured synergy:    {pid['synergistic']:.4f}")
    print(f"  Error:               {abs(pid['synergistic'] - truth_synergy):.4f}")
    print()
    print("Per-X marginal MI on noiseless XOR are both 0 (each input independent of y);")
    print("only the JOINT carries the signal -- this is the canonical synergy case where")
    print("naive MRMR / Fleuret-CMIM REJECTS both features and PID-aware selection KEEPS them.")


if __name__ == "__main__":
    main()
