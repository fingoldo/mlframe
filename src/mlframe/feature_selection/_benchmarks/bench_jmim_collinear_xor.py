"""JMIM vs Fleuret-CMIM on XOR + collinear-noise reflection (Wave 8).

Designed per the MRMR research survey (docs/MRMR_RESEARCH.md):

    "Bench: synthetic XOR + 10 collinear noise reflections of one
    informative feature; current Fleuret will reject the second XOR
    component; JMI keeps it."

Test scenario:
  * x_a, x_b: two informative binary features whose XOR drives y.
  * z_1..z_10: 10 noisy reflections of x_a (high correlation, low MI to y
    individually because the signal lives in the (x_a, x_b) pair).
  * Noise features: rng.standard_normal columns that have no signal at all.

What we expect to see:
  * Fleuret CMIM (current default) -- ``min_k I(X_b; Y | Z_k)`` collapses
    because once x_a is picked, the 10 z_k near-duplicates make every
    candidate look 'redundant via at least one Z'. The second XOR
    component (x_b) gets rejected.
  * JMIM (Bennasar 2015) -- ``min_j I(X_b, X_j; Y)`` uses the JOINT
    information; the (x_b, x_a) joint carries the full XOR signal so
    JMIM keeps x_b.

The bench dumps the selected feature ranking under both modes and the
JMIM gain spread on the contested xor partner.

Reference: Brown, G. et al. (2012), "A Unifying Framework for Information
Theoretic Feature Selection", JMLR 13:27-66. Bennasar, M., Hua, Y.,
Setchi, R. (2015), "Feature selection using Joint Mutual Information
Maximisation", Expert Syst. Appl. 42(22):8520-8532.

Run::

    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_jmim_collinear_xor
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


@dataclass
class BenchResult:
    method: str
    selected: list
    n_features: int
    runtime_s: float


def make_xor_collinear_dataset(n: int = 4000, n_collinear: int = 10,
                                n_pure_noise: int = 10,
                                noise_level: float = 0.1,
                                seed: int = 0) -> tuple:
    """Synthesise the agent B test bed.

    Returns:
        X: pandas DataFrame with columns
            ``[x_a, x_b, z_1..z_n_collinear, n_1..n_pure_noise]``.
        y: pandas Series y = x_a XOR x_b (with a tiny noise floor).
        signal_features: list of the names that should ideally survive --
            x_a AND x_b (xor needs both).
    """
    rng = np.random.default_rng(int(seed))
    x_a = rng.integers(0, 2, n).astype(np.float64)
    x_b = rng.integers(0, 2, n).astype(np.float64)
    y_clean = (x_a.astype(np.int64) ^ x_b.astype(np.int64))
    # Tiny label flip so the binning has resolution.
    flip = rng.random(n) < noise_level * 0.05
    y = np.where(flip, 1 - y_clean, y_clean).astype(np.int64)

    cols = {
        "x_a": x_a + noise_level * rng.standard_normal(n),
        "x_b": x_b + noise_level * rng.standard_normal(n),
    }
    # 10 noisy reflections of x_a: highly correlated to x_a, individually
    # carry near-zero MI to y because y is the XOR (independent of x_a alone).
    for k in range(int(n_collinear)):
        cols[f"z_a_{k+1}"] = x_a + (0.3 + 0.05 * k) * rng.standard_normal(n)
    # Pure noise.
    for k in range(int(n_pure_noise)):
        cols[f"noise_{k+1}"] = rng.standard_normal(n)

    X = pd.DataFrame(cols)
    return X, pd.Series(y, name="y"), ["x_a", "x_b"]


def run_one(method_label: str, **mrmr_kwargs) -> BenchResult:
    X, y, signal_features = make_xor_collinear_dataset(seed=42)
    t0 = time.perf_counter()
    sel = MRMR(verbose=0, **mrmr_kwargs)
    sel.fit(X, y)
    rt = time.perf_counter() - t0
    chosen = list(sel.get_feature_names_out())
    return BenchResult(
        method=method_label, selected=chosen,
        n_features=len(chosen), runtime_s=rt,
    )


def main():
    print("=" * 78)
    print("JMIM vs Fleuret on XOR + 10 collinear z_a reflections")
    print("=" * 78)
    print()
    print("Synthetic dataset:")
    print("  n=4000 rows, n_collinear=10 reflections of x_a, n_pure_noise=10")
    print("  y = x_a XOR x_b (binary)")
    print("  Ground truth: BOTH x_a AND x_b are required to predict y.")
    print()

    results = []

    # 1) Legacy Fleuret CMIM.
    r1 = run_one("Fleuret CMIM (legacy)")
    results.append(r1)

    # 2) JMIM aggregator (Wave 8).
    r2 = run_one("JMIM (Bennasar 2015)", redundancy_aggregator="jmim")
    results.append(r2)

    # 3) JMIM + SU normalisation (the headline production combo).
    r3 = run_one(
        "JMIM + SU normalisation",
        redundancy_aggregator="jmim", mi_normalization="su",
    )
    results.append(r3)

    # 4) Fleuret + BUR bonus.
    r4 = run_one("Fleuret + BUR bonus", bur_lambda=0.5)
    results.append(r4)

    # 5) JMIM + BUR bonus.
    r5 = run_one(
        "JMIM + BUR (full Wave 8 combo)",
        redundancy_aggregator="jmim", bur_lambda=0.3,
    )
    results.append(r5)

    print(f"{'method':<35} {'rt_s':>7} {'n_sel':>6} {'x_a kept':>10}"
          f" {'x_b kept':>10}")
    print("-" * 78)
    for r in results:
        kept_a = "yes" if "x_a" in r.selected else "no"
        kept_b = "yes" if "x_b" in r.selected else "no"
        marker = "  <- BOTH kept!" if (kept_a == "yes" and kept_b == "yes") else ""
        print(f"{r.method:<35} {r.runtime_s:>7.2f} {r.n_features:>6}"
              f" {kept_a:>10} {kept_b:>10}{marker}")

    print()
    print("Detailed selections (top-10):")
    for r in results:
        print(f"  {r.method}:")
        for f in r.selected[:10]:
            print(f"    - {f}")
        print()


if __name__ == "__main__":
    main()
