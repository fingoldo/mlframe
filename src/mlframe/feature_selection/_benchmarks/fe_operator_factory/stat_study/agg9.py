"""Aggregate the exp9 false-accept / power runs (``fa_<target>_<n>_*.json``): margins calibrated on the noise y-permutation null and the acceptance rate of each decision rule per target.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.agg9``."""

import json

import numpy as np

from ..common._paths import data_dir

ALPHA = 0.05
names = ["preset", "grid1", "ols1", "ols2", "grid2"]
S = names[1:]


def load(tgt, n):
    """Concatenate the replicate lists of all runs of one target and size."""
    reps = []
    for f in sorted(data_dir("stat_study").glob(f"fa_{tgt}_{n}_*.json")):
        reps += json.loads(f.read_text())["reps"]
    return reps


def stats(reps):
    """Stack the observed, null, conditional-null and held-out arrays of the replicates."""
    obs = np.array([r["obs"] for r in reps])
    null = np.array([r["null"] for r in reps])
    cnull = np.array([r["cnull"] for r in reps])
    hobs = np.array([r["hobs"] for r in reps])
    hnull = np.array([r["hnull"] for r in reps])
    return obs, null, cnull, hobs, hnull


for n in (2000, 5000):
    noise = load("noise", n)
    if not noise:
        continue
    obs0, null0, cnull0, hobs0, hnull0 = stats(noise)
    # calibrated margins = q95 of pooled null gain (delta = family MI - preset MI) from the NOISE y-permutation nulls
    dn = null0[:, :, 1:] - null0[:, :, [0]]  # reps x B x 4
    marg = np.quantile(dn.reshape(-1, 4), 0.95, axis=0)
    hn = hnull0.reshape(-1, 3)
    hmarg = np.quantile(hn, 0.95, axis=0)  # held-out gain margins for grid1, ols1, ols2
    print(
        f"\n##### n={n}: noise reps={len(noise)}  margin q95 (b): "
        + " ".join(f"{s}={m:.5f}" for s, m in zip(S, marg))
        + "   held-out margin: "
        + " ".join(f"{s}={m:.5f}" for s, m in zip(["grid1", "ols1", "ols2"], hmarg))
    )
    print("mean null gain (selection inflation, y-perm): " + " ".join(f"{s}={m:.5f}" for s, m in zip(S, dn.reshape(-1, 4).mean(0))))
    print("target".ljust(9) + " rule".ljust(24) + "".join(s.rjust(8) for s in S) + "   (rate of acceptance; noise & noshift = false-accept, others = power)")
    for tgt in ("noise", "noshift", "case2", "half", "q25", "two"):
        reps = load(tgt, n)
        if not reps:
            continue
        obs, null, cnull, hobs, hnull = stats(reps)
        R = len(reps)
        d_obs = obs[:, 1:] - obs[:, [0]]
        # (a) absolute association p-value of the family's own MI vs y-perm null of the same statistic
        pa = np.array([[(1 + (null[r, :, 1 + j] >= obs[r, 1 + j]).sum()) / (null.shape[1] + 1) for j in range(4)] for r in range(R)])
        # (b) margin
        accb = d_obs > marg
        # (d) conditional-permutation null on the gain
        cd = cnull[:, :, 1:] - cnull[:, :, [0]]
        pd = np.array([[(1 + (cd[r, :, j] >= d_obs[r, j]).sum()) / (cd.shape[1] + 1) for j in range(4)] for r in range(R)])
        # (c) held-out
        accc = hobs > hmarg

        def row(lab, a):
            """Print one acceptance-rate row."""
            return print(tgt.ljust(9) + lab.ljust(24) + "".join(f"{a[:, j].mean():8.2f}" if a.shape[1] == 4 else "" for j in range(a.shape[1])))

        pa0 = np.array([[(null[r, :, 1 + j] >= obs[r, 1 + j]).sum() == 0 for j in range(4)] for r in range(R)])
        row(f" (a) abs perm max-stat [R={R}]", pa0)
        row(" (b) margin q95(noise)", accb)
        pd0 = np.array([[(cd[r, :, j] >= d_obs[r, j]).sum() == 0 for j in range(4)] for r in range(R)])
        row(" (d) cond-perm max-stat", pd0)
        print(tgt.ljust(9) + " (c) held-out>margin".ljust(24) + "   grid1 ols1 ols2: " + " ".join(f"{accc[:, j].mean():.2f}" for j in range(3)))
        print(tgt.ljust(9) + " mean gain (obs)".ljust(24) + "".join(f"{d_obs[:, j].mean():8.4f}" for j in range(4)) + f"   preset MI {obs[:, 0].mean():.4f}")
