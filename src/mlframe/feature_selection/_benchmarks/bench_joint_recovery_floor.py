"""Gate-level benchmark for the marginal-uplift TWO-TIER JOINT-RECOVERY floor
(`_FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO` 0.82 / `_FE_MARGINAL_UPLIFT_STRICT_JOINT_RATIO` 0.84) -- the LOW
module-constant follow-up to bench_marginal_uplift_ratio.py (see MRMR_HARDCODED_THRESHOLDS_BENCH.md).

WHY this axis (not the uplift ratio): the prior iteration showed the uplift ratio (1-D / 1-D) does NOT
bind the documented HARD cross-signal artefact (uplift 1.441 > 1.30 passes the bar). What rejects that
artefact is the JOINT-RECOVERY floor: its joint_ratio = best_engineered_1D_MI / pair_2D_joint_MI = 0.814
sits below the 0.84 strict floor while the genuine pairs sit at 0.829 / 0.849. So the joint floor is the
axis that actually binds -- the right place to ask whether a data-derived value beats the constant.

WHY it might convert (unlike the uplift ratio): the joint ratio is 1-D / 2-D. The 2-D joint denominator's
Miller-Madow bias term (~k_joint ~ nbins^2 occupied cells) is ~nbins x LARGER than the 1-D numerator's, so
the RAW ratio is structurally DEPRESSED below 1.0 -- exactly the asymmetry the shipped
`fe_min_pair_mi_prevalence` "auto" debiases. `mm_debiased_prevalence_ratio` already implements this exact
1-D/2-D MM-debiased ratio. QUESTION: does debiasing the joint ratio WIDEN the genuine-vs-artefact margin
(min genuine - max artefact) beyond the fixed-floor margin -- and by more than the ~0.006 cross-HW MI noise
the fixed two-tier floor was calibrated against? If yes -> guarded "auto"; if no -> KEEP the fixed floor.

Method: gate-level (bypasses the fit cache). Canonical fixture y = a**2/b + log(c)*sin(d) + noise; build the
genuine engineered summaries (a**2/b, log(c)*sin(d)) and the documented cross-signal artefact
(sub(exp(a), invcbrt(c))); for each, eng 1-D MI + occupied-k and the pair 2-D joint MI + occupied-k via the
project discretiser; report raw vs MM-debiased joint_ratio for genuine vs artefact and the separation margin.

Run:
  PYTHONPATH=<repo>/src;<pyutilz>/src CUDA_VISIBLE_DEVICES="" MLFRAME_DISABLE_HNSW=1 \
    python -m mlframe.feature_selection._benchmarks.bench_joint_recovery_floor
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.discretization import discretize_array
from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_gates import mm_debiased_prevalence_ratio

_NBINS = 10


def _occupied_k(codes):
    return int(np.unique(np.asarray(codes)).size)


def _disc(arr):
    d = discretize_array(arr=np.asarray(arr, dtype=np.float64), n_bins=_NBINS, method="quantile", dtype=np.int32)
    return np.asarray(d, dtype=np.int64).ravel()


def _plugin_mi(x_codes, y_codes):
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


def _joint_codes(a_codes, b_codes):
    """Dense 2-D joint code of two binned operands (the pair's joint contingency)."""
    kb = int(b_codes.max()) + 1
    jc = a_codes * kb + b_codes
    # renumber to contiguous occupied codes (so occupied-k is exact)
    _, inv = np.unique(jc, return_inverse=True)
    return inv.astype(np.int64)


def _joint_ratio(eng_col, op_a, op_b, y_codes, ky, n):
    """raw and MM-debiased joint_ratio = eng_1D_MI / pair_2D_joint_MI."""
    eng_c = _disc(eng_col)
    mi_eng = _plugin_mi(eng_c, y_codes); k_eng = _occupied_k(eng_c)
    jc = _joint_codes(_disc(op_a), _disc(op_b))
    mi_joint = _plugin_mi(jc, y_codes); k_joint = _occupied_k(jc)
    if mi_joint <= 0.0:
        return None
    raw = mi_eng / mi_joint
    deb = mm_debiased_prevalence_ratio(mi_eng, mi_joint, k_eng=k_eng, k_joint=k_joint, k_y=ky, n=n)
    return raw, deb


def main():
    raw_g, deb_g, raw_x, deb_x = [], [], [], []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        n = 2500
        a = rng.uniform(1.0, 5.0, n); b = rng.uniform(1.0, 5.0, n)
        c = rng.uniform(1.0, 5.0, n); d = rng.uniform(0.0, 2.0 * np.pi, n)
        y = a ** 2 / b + np.log(c) * np.sin(d) + rng.normal(0.0, 1.0, n) / 5.0
        y_codes = _disc(y); ky = _occupied_k(y_codes)
        with np.errstate(all="ignore"):
            g1 = _joint_ratio(a ** 2 / b, a, b, y_codes, ky, n)
            g2 = _joint_ratio(np.log(c) * np.sin(d), c, d, y_codes, ky, n)
            art = _joint_ratio(np.exp(np.clip(a, None, 20)) - np.sign(c) * np.abs(c) ** (-1.0 / 3.0), a, c, y_codes, ky, n)
        for r, lr, ld in [(g1, raw_g, deb_g), (g2, raw_g, deb_g), (art, raw_x, deb_x)]:
            if r is not None:
                lr.append(r[0]); ld.append(r[1])

    raw_g, deb_g, raw_x, deb_x = map(np.array, (raw_g, deb_g, raw_x, deb_x))
    print(f"GENUINE  raw joint_ratio: mean={raw_g.mean():.3f} min={raw_g.min():.3f}  | MM-debiased: mean={deb_g.mean():.3f} min={deb_g.min():.3f}")
    print(f"ARTEFACT raw joint_ratio: mean={raw_x.mean():.3f} max={raw_x.max():.3f}  | MM-debiased: mean={deb_x.mean():.3f} max={deb_x.max():.3f}")
    print(f"FIXED floor 0.82/0.84: raw genuine-pass(>=0.84)={100*(raw_g>=0.84).mean():.0f}% artefact-pass(>=0.84)={100*(raw_x>=0.84).mean():.0f}%")
    sep_raw = raw_g.min() - raw_x.max()
    sep_deb = deb_g.min() - deb_x.max()
    print(f"SEPARATION MARGIN (min genuine - max artefact): raw={sep_raw:+.3f}  MM-debiased={sep_deb:+.3f}  delta={sep_deb-sep_raw:+.3f}")

    # HARNESS-FAITHFULNESS GUARD. The REAL gate (validated, HW-calibrated) places genuine pairs ABOVE the
    # artefact on joint-recovery (documented: genuine 0.829 / 0.849 > artefact 0.814). If THIS harness shows
    # the artefact >= genuine (negative raw separation), it does NOT reproduce the gate's ordering and ANY
    # conversion verdict from it is invalid -- do not trust the "widens" delta. The infidelity is expected:
    # this harness uses the RAW generating column as the "engineered summary" (the real gate searches the
    # whole elementary-op library for the BEST summary, which recovers MORE of the joint) and a raw plug-in
    # MI (the gate uses mi_direct with permutation-debias / min_nonzero_confidence), so the genuine eng MI is
    # understated and the 2-D joint over-binning-inflated -- enough to FLIP the ordering.
    faithful = sep_raw > 0.0
    if not faithful:
        print("HARNESS INVALID: raw separation is NEGATIVE -> this naive reconstruction INVERTS the gate's "
              "validated genuine>artefact ordering (genuine joint-recovery here < artefact). The 'widens' "
              "delta is meaningless on an inverted ranking. A faithful assessment needs the real op-library "
              "search for the best engineered summary + the gate's mi_direct kernel, not this raw plug-in.")
        print("VERDICT: KEEP fixed two-tier floor -- conversion CANNOT be assessed by this harness "
              "(infidelity > any debias effect); the fixed floor is calibrated on the real-pipeline numbers.")
    else:
        print(f"VERDICT: debiasing {'WIDENS' if sep_deb > sep_raw + 0.05 else 'does NOT materially widen'} the "
              f"genuine-vs-artefact margin -> {'CONVERT (guarded auto)' if sep_deb > sep_raw + 0.05 else 'KEEP fixed two-tier floor'}")


if __name__ == "__main__":
    main()
