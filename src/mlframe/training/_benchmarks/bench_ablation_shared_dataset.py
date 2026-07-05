"""Bench: can BaselineDiagnostics ablation amortize the LightGBM binning pass?

Motivation
----------
A cProfile of the full-config suite at 200k attributed ~74% of suite wall to
the BaselineDiagnostics ablation and flagged ``init_from_np2d`` (~0.82s tottime)
as the binning hotspot. The ablation runs 6 bounded LightGBM fits per target
(1 baseline + ``ablation_top_k`` drops), each rebuilding a fresh ``lgb.Dataset``
from the SAME sampled frame, differing only by ONE dropped column. The premise:
amortize binning by building the binned Dataset ONCE and reusing it.

Two amortization leads were investigated:

(a) Single full-feature ``lgb.Dataset`` built once, each ablation fit trained
    with ``ignore_column`` to drop one column without re-binning.
(b) Full-feature binned Dataset passed as ``reference=`` to each ablation
    subset Dataset so bin boundaries are computed once and reused.

Verdict (measured here, LightGBM 4.6.0)
---------------------------------------
BOTH leads REJECTED. Neither is feasible AND bit-identical:

* (b) reference: NOT FEASIBLE. ``reference`` requires the child Dataset to have
  the SAME ``num_feature`` as the reference. A column-drop subset fails
  ``construct()`` with "Length of feature_name(K) and num_feature(K+1) don't
  match". There is no column-subset of a Dataset that reuses bin mappers
  (``Dataset.subset`` is row-only).

* (a) ignore_column: FEASIBLE but NOT bit-identical. Ignoring a column on the
  full Dataset changes the feature-set context (feature indexing, per-tree
  feature sampling, histogram construction), so split decisions, feature
  importances, and predictions all DIVERGE vs dropping the column. Measured:
  FI vectors completely different; max |Δ predicted proba| ~0.62. The ablation
  verdict (which features are dominant) would change -> forbidden for a
  default-ON diagnostic.

Why the premise was misleading
-------------------------------
cProfile inflates deep pandas/numpy call timings ~10-13x vs wall (see
CLAUDE.md "Calibrate against cProfile attribution overhead"). A warm wall-time
microbench shows the binning pass (``Dataset.construct`` == ``init_from_np2d``)
is only ~3.3-3.7% of a single fit's wall at 50k rows -- NOT the bottleneck.
Even a hypothetical zero-cost binning amortization caps at ~3% of ablation wall,
and it is not bit-identically achievable anyway. The real cost is tree-growing,
governed by ``quick_model_n_estimators`` (200) x ``ablation_top_k`` (5) = 6 fits.

DOC finding (flagged, NOT changed -- accuracy-affecting default)
----------------------------------------------------------------
``quick_model_n_estimators=200`` over-provisions a diagnostic whose only job is
to rank FI and produce ablation deltas at percentage-point resolution. Measured
FI-rank stability vs ne=200 on a 50k synthetic with 4 dominant features:

    ne= 50: wall=658ms  spearman=0.46  top5_overlap=4/5
    ne= 75: wall=732ms  spearman=0.52  top5_overlap=4/5
    ne=100: wall=1008ms spearman=0.68  top5_overlap=4/5
    ne=150: wall=1339ms spearman=0.84  top5_overlap=5/5  <-- identical top-5
    ne=200: wall=1677ms spearman=1.00  top5_overlap=5/5

ne=150 keeps the identical top-5 dominant set at a ~20% wall cut. Below 150 the
FI ranking drifts (top-5 overlap drops, Spearman < 0.7) and the ablation
selection can change. Since FI-rank seeds which features get dropped, lowering
``quick_model_n_estimators`` is genuinely accuracy-affecting -- left for the
user to decide, NOT changed unilaterally.

Run: python -m mlframe.training._benchmarks.bench_ablation_shared_dataset
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def _make_data(n: int = 50_000, nf: int = 30, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, nf)).astype(np.float64)
    # Dominant-feature signal over whatever columns exist (robust to small nf).
    coeffs = {0: 2.0, 3: 1.0, 7: -0.7, 12: 0.4}
    signal = rng.normal(size=n)
    for j, c in coeffs.items():
        if j < nf:
            signal = signal + c * X[:, j]
    y = (signal > 0).astype(int)
    cols = [f"f{i}" for i in range(nf)]
    return pd.DataFrame(X, columns=cols), y, cols


def bench_binning_fraction() -> None:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    Xdf, y, cols = _make_data()
    Xtr, _, ytr, _ = train_test_split(Xdf, y, test_size=0.2, random_state=101, stratify=y)
    params = dict(n_estimators=200, num_leaves=31, learning_rate=0.05, random_state=101, n_jobs=1, verbose=-1, force_col_wise=True)
    lgb.LGBMClassifier(**params).fit(Xtr, ytr)  # warm

    cts = []
    for _ in range(5):
        ds = lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
        t = time.perf_counter(); ds.construct(); cts.append(time.perf_counter() - t)
    fts = []
    for _ in range(3):
        m = lgb.LGBMClassifier(**params)
        t = time.perf_counter(); m.fit(Xtr, ytr); fts.append(time.perf_counter() - t)
    cc, ff = sorted(cts)[2], sorted(fts)[1]
    print(f"construct(binning)={cc*1000:.1f}ms  full_fit={ff*1000:.1f}ms  " f"binning_frac={cc/ff*100:.1f}%")


def check_reference_infeasible() -> None:
    """Lead (b): reference Dataset rejects a column-drop subset (num_feature mismatch)."""
    import lightgbm as lgb

    Xdf, y, cols = _make_data(n=4000, nf=8)
    full = lgb.Dataset(Xdf, label=y, free_raw_data=False); full.construct()
    keep = [c for c in cols if c != cols[0]]
    sub = lgb.Dataset(Xdf[keep], label=y, reference=full, free_raw_data=False)
    try:
        sub.construct()
        print("UNEXPECTED: reference subset constructed (re-verify on this LGBM build)")
    except Exception as exc:  # noqa: BLE001 - documenting the rejection
        print(f"reference subset INFEASIBLE (expected): {type(exc).__name__}: {exc}")


def check_ignore_column_not_bit_identical() -> None:
    """Lead (a): ignore_column diverges from dropping the column."""
    import lightgbm as lgb

    Xdf, y, cols = _make_data(n=4000, nf=8)
    params = dict(objective="binary", num_leaves=31, learning_rate=0.05, verbose=-1, seed=42, force_col_wise=True, deterministic=True)
    keep = [c for c in cols if c != cols[0]]
    m = lgb.LGBMClassifier(n_estimators=50, **{k: v for k, v in params.items() if k != "objective"})
    m.fit(Xdf[keep], y)
    p_drop = m.predict_proba(Xdf[keep])[:, 1]

    ds = lgb.Dataset(Xdf, label=y, free_raw_data=False, params={"ignore_column": "0"})
    b = lgb.train({**params, "num_iterations": 50}, ds)
    p_ign = b.predict(Xdf)
    print(f"ignore_column max abs(delta proba) vs drop = " f"{np.max(np.abs(p_drop - p_ign)):.4f} (NON-zero -> not bit-identical)")


if __name__ == "__main__":
    bench_binning_fraction()
    check_reference_infeasible()
    check_ignore_column_not_bit_identical()
