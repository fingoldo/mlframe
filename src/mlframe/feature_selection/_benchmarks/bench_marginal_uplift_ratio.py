"""Gate-level benchmark for ``_FE_MARGINAL_UPLIFT_MIN_RATIO`` (the one HIGH module-constant
threshold-conversion candidate; the full-fit monkeypatch is cache-invalid -- see
MRMR_HARDCODED_THRESHOLDS_BENCH.md, "Module-constant items").

The marginal-uplift gate is an ALTERNATIVE recall path (_pairs_core.py ~1447): a pair the joint-
prevalence gate rejected is still admitted when its best engineered 1-D column beats the LARGER operand's
1-D marginal MI by >= 1.30x (paired with a two-tier joint-recovery floor). 1.30 is a hardcoded constant.

QUESTION: would a DATA-DERIVED uplift bar (Miller-Madow-debias both 1-D MIs before the ratio, mirroring
the shipped fe_min_pair_mi_prevalence "auto") separate GENUINE synergy pairs from cross-signal ARTEFACTS
better than the fixed 1.30 -- justifying a guarded "auto" conversion?

Unlike the joint-prevalence ratio (1-D / 2-D: the joint bias term is ~nbins x larger, so debiasing fixes a
structural depression), the uplift ratio is 1-D / 1-D with COMPARABLE occupied cardinality. Debiasing both
sides by (k-1)(k_y-1)/2n is NOT a clean cancellation -- (a-d)/(b-d) > a/b for a>b -- so it inflates ratios
above 1. This bench measures, across seeds, whether that inflation WIDENS the genuine-vs-artefact margin
(a real conversion case) or merely shifts both (no separation gain -> keep the constant).

Method: the documented GATE-LEVEL harness (bypasses the fit cache). Generate the canonical fixture
y = a**2/b + log(c)*sin(d) + noise; build the genuine engineered columns (a**2/b, log(c)*sin(d)) and the
documented cross-signal artefact (sub(exp(a), invcbrt(c))); discretise everything with the project's
discretiser; compute 1-D plug-in MI vs binned y + occupied-k for each; report raw vs MM-debiased uplift
ratio for genuine vs artefact, and the SEPARATION MARGIN (min genuine ratio - max artefact ratio) under
each. A positive, larger margin under debiasing would justify the conversion.

Run:
  PYTHONPATH=<repo>/src;<pyutilz>/src CUDA_VISIBLE_DEVICES="" MLFRAME_DISABLE_HNSW=1 \
    python -m mlframe.feature_selection._benchmarks.bench_marginal_uplift_ratio
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.discretization import discretize_array
from mlframe.feature_selection.filters.info_theory._entropy_kernels import mi_miller_madow_correct

_NBINS = 10


def _occupied_k(codes: np.ndarray) -> int:
    return int(np.unique(np.asarray(codes)).size)


def _plugin_mi_1d(x_codes: np.ndarray, y_codes: np.ndarray) -> float:
    """Plug-in MI of two integer-coded 1-D arrays (nats), consistent estimator for both ratio sides."""
    n = x_codes.size
    if n == 0:
        return 0.0
    kx = int(x_codes.max()) + 1
    ky = int(y_codes.max()) + 1
    joint = np.zeros((kx, ky), dtype=np.float64)
    np.add.at(joint, (x_codes, y_codes), 1.0)
    joint /= n
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log(joint[nz] / (px @ py)[nz])))


def _disc(arr: np.ndarray) -> np.ndarray:
    d = discretize_array(arr=np.asarray(arr, dtype=np.float64), n_bins=_NBINS, method="quantile", dtype=np.int32)
    return np.asarray(d, dtype=np.int64).ravel()


def _ratios(eng_col, op_a, op_b, y_codes, ky, n):
    """raw and MM-debiased uplift ratio = eng_MI / max(operand marginal MI)."""
    eng_c = _disc(eng_col)
    mi_eng = _plugin_mi_1d(eng_c, y_codes); k_eng = _occupied_k(eng_c)
    a_c, b_c = _disc(op_a), _disc(op_b)
    mi_a = _plugin_mi_1d(a_c, y_codes); mi_b = _plugin_mi_1d(b_c, y_codes)
    if mi_a >= mi_b:
        mi_op, k_op = mi_a, _occupied_k(a_c)
    else:
        mi_op, k_op = mi_b, _occupied_k(b_c)
    if mi_op <= 0.0:
        return None
    raw = mi_eng / mi_op
    num = max(0.0, mi_miller_madow_correct(mi_eng, k_eng, ky, n))
    den = mi_miller_madow_correct(mi_op, k_op, ky, n)
    deb = (num / den) if den > 1e-9 else raw
    return raw, deb


def main():
    raw_g, deb_g, raw_x, deb_x = [], [], [], []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        n = 2500
        a = rng.uniform(1.0, 5.0, n); b = rng.uniform(1.0, 5.0, n)
        c = rng.uniform(1.0, 5.0, n); d = rng.uniform(0.0, 2.0 * np.pi, n)
        noise = rng.normal(0.0, 1.0, n)
        y = a ** 2 / b + np.log(c) * np.sin(d) + noise / 5.0
        y_codes = _disc(y); ky = _occupied_k(y_codes)
        with np.errstate(all="ignore"):
            # GENUINE synergy pairs
            g1 = _ratios(a**2 / b, a, b, y_codes, ky, n)  # (a,b) -> a^2/b
            g2 = _ratios(np.log(c) * np.sin(d), c, d, y_codes, ky, n)  # (c,d) -> log(c)sin(d)
            # CROSS-SIGNAL ARTEFACT (documented): sub(exp(a), invcbrt(c)) over operands a (signal1) + c (signal2)
            art = _ratios(np.exp(np.clip(a, None, 20)) - np.sign(c) * np.abs(c) ** (-1.0 / 3.0), a, c, y_codes, ky, n)
        for r, lst_raw, lst_deb in [(g1, raw_g, deb_g), (g2, raw_g, deb_g), (art, raw_x, deb_x)]:
            if r is not None:
                lst_raw.append(r[0]); lst_deb.append(r[1])

    raw_g, deb_g, raw_x, deb_x = map(np.array, (raw_g, deb_g, raw_x, deb_x))
    print(f"GENUINE  raw uplift: mean={raw_g.mean():.3f} min={raw_g.min():.3f}  | MM-debiased: mean={deb_g.mean():.3f} min={deb_g.min():.3f}")
    print(f"ARTEFACT raw uplift: mean={raw_x.mean():.3f} max={raw_x.max():.3f}  | MM-debiased: mean={deb_x.mean():.3f} max={deb_x.max():.3f}")
    print(f"BAR=1.30: raw genuine-pass={100*(raw_g>=1.30).mean():.0f}% artefact-pass={100*(raw_x>=1.30).mean():.0f}%")
    sep_raw = raw_g.min() - raw_x.max()
    sep_deb = deb_g.min() - deb_x.max()
    print(f"SEPARATION MARGIN (min genuine - max artefact): raw={sep_raw:+.3f}  MM-debiased={sep_deb:+.3f}")
    print(f"VERDICT: debiasing {'WIDENS' if sep_deb > sep_raw + 1e-6 else 'does NOT widen'} the separation "
          f"(delta={sep_deb - sep_raw:+.3f}) -> {'CONVERT (guarded auto)' if sep_deb > sep_raw + 0.05 else 'KEEP fixed 1.30'}")


if __name__ == "__main__":
    main()
