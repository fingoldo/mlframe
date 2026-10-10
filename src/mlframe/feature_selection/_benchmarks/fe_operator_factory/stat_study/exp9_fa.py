"""False-accept / power study.  usage: exp9_fa.py n target R B seed0
Family per selector = best over the K=5 base pairs (top plain-mul MI) of: 1-D grid (G=9), closed-form 1-param (ols), closed-form 2-param (ols2), 2-D grid (7x7).
Delta_s = MI(best shifted candidate) - MI(best preset combo).  Nulls: y-permutation (whole selection re-run) and conditional permutation (within strata of preset-best bin)."""

import json
import sys
import time

import numpy as np

from ..common._paths import scratch_dir
from .core import bins_batch, bins_of, family_eval, get_presets, mi_bins, mi_cnt, mi_feats, preset_cands, qbin, rank01, unary_mat
from .core2 import family_eval2

n = int(sys.argv[1])
tgt = sys.argv[2]
R = int(sys.argv[3])
B = int(sys.argv[4])
seed0 = int(sys.argv[5]) if len(sys.argv) > 5 else 0
UN, BI = get_presets("minimal")
nu = len(UN)
NP = 2 * nu * nu
K = 5
G1 = 9
G2 = 7
NAMES = ["preset", "grid1", "ols1", "ols2", "grid2"]


def target(x, z, rng):
    """Target for the chosen scenario: noise or a noisy truth."""
    if tgt == "noise":
        return rng.standard_normal(len(x))
    tr = {
        "noshift": np.log(x) * np.sin(z),
        "add": x**2 + z,
        "case2": np.log(2 * x) * np.sin(z / 3),
        "half": (x - 0.5) * z,
        "q25": (x - 0.25) * z,
        "two": (x - 0.3) * (z - 0.7),
        "mix": 3 * x * z + 0.5 * z + 1.5 * x,
    }[tgt]
    return tr + rng.standard_normal(len(x)) * tr.std()


def pair_uv(UX, UZ, p):
    """The (u, v) operand pair of pair index ``p`` (role, unary, unary)."""
    role, rem = divmod(p, nu * nu)
    iu, iv = divmod(rem, nu)
    return (UX[iu], UZ[iv]) if role == 0 else (UZ[iu], UX[iv])


def prep(x, z):
    """Precompute the binned candidate banks: preset, plain product, 1-D grid and 2-D grid."""
    UX, UZ = unary_mat(UN, x), unary_mat(UN, z)
    rows = []
    for _ids, M in preset_cands(UX, UZ, BI):
        rows.append(M)
    F = np.ascontiguousarray(np.concatenate(rows))
    plain = np.empty((NP, n))
    g1 = np.empty((NP * G1, n))
    g2 = np.empty((NP * G2 * G2, n))
    q1 = np.linspace(0.1, 0.9, G1)
    q2 = np.linspace(0.1, 0.9, G2)
    for p in range(NP):
        u, v = pair_uv(UX, UZ, p)
        us = np.sort(u)
        vs = np.sort(v)
        plain[p] = u * v
        for g, q in enumerate(q1):
            g1[p * G1 + g] = (u - us[int(q * (n - 1))]) * v
        for g, q in enumerate(q2):
            for h, q_ in enumerate(q2):
                g2[(p * G2 + g) * G2 + h] = (u - us[int(q * (n - 1))]) * (v - vs[int(q_ * (n - 1))])
    return dict(UX=UX, UZ=UZ, F=F, Bp=bins_batch(F, 10), Bplain=bins_batch(plain, 10), Bg1=bins_batch(g1, 10), Bg2=bins_batch(g2, 10))


def evaluate(D, yb, yr):
    """Family maxima (preset, 1-D grid, OLS 1-param, OLS 2-param, 2-D grid) for one target; returns (maxima, preset-best index)."""
    mp = mi_bins(D["Bp"], yb, 10, 10, False)
    P = mp.max()
    Kidx = np.argsort(-mi_bins(D["Bplain"], yb, 10, 10, False))[:K]
    g1 = mi_bins(D["Bg1"], yb, 10, 10, False).reshape(NP, G1).max(1)[Kidx].max()
    g2 = mi_bins(D["Bg2"], yb, 10, 10, False).reshape(NP, G2 * G2).max(1)[Kidx].max()
    o1 = family_eval(D["UX"], D["UZ"], yb, yr, 3, 10, 10, False, 8, 0.0, True, False, False, False, 9)[0][Kidx].max()
    o2 = family_eval2(D["UX"], D["UZ"], yb, yr, 6, 10, 10, False, False, False, 9)[0][Kidx].max()
    return np.array([P, g1, o1, o2, g2]), int(mp.argmax())


def heldout(D, yb, yr, A, Bm, FA, FB, UXA, UZA, UXB, UZB):
    """Held-out gain over the preset of each family selected on half A and scored on half B."""
    ybA, ybB = yb[A], yb[Bm]
    ib = int(mi_feats(FA, ybA, 10, 10, False).argmax())
    pB = mi_cnt(bins_of(FB[ib], 10), ybB, 10, 10, False)
    out = []
    for kind in ("grid1", "ols1", "ols2"):
        if kind == "grid1":
            mis, ts = family_eval(UXA, UZA, ybA, yr[A], 4, 10, 10, False, 8, 0.0, True, False, False, False, 9)
            ss = np.zeros_like(ts)
        elif kind == "ols1":
            mis, ts = family_eval(UXA, UZA, ybA, yr[A], 3, 10, 10, False, 8, 0.0, True, False, False, False, 9)
            ss = np.zeros_like(ts)
        else:
            mis, ss, ts = family_eval2(UXA, UZA, ybA, yr[A], 6, 10, 10, False, False, False, 9)
        j = int(mis.argmax())
        u, v = pair_uv(UXB, UZB, j)
        # kind ols2/ss shifts u (ss), v (ts); 1-param kinds: ts shifts u
        f = (u + ts[j]) * v if kind != "ols2" else (u + ss[j]) * (v + ts[j])
        out.append(mi_cnt(bins_of(f, 10), ybB, 10, 10, False) - pB)
    return np.array(out)


res = []
for r in range(R):
    t0 = time.time()
    rng = np.random.default_rng(5000 + seed0 * 1000 + r)
    x, z = rng.random(n), rng.random(n)
    y = target(x, z, rng)
    yb = qbin(y, 10)
    yr = rank01(y)
    D = prep(x, z)
    obs, ib = evaluate(D, yb, yr)
    strata = bins_of(D["F"][ib], 10)
    null = []
    cnull = []
    for _b in range(B):
        pi = rng.permutation(n)
        null.append(evaluate(D, yb[pi], yr[pi])[0])
        yc = yb.copy()
        yrc = yr.copy()
        for s_ in range(10):
            idx = np.where(strata == s_)[0]
            pi = rng.permutation(idx)
            yc[idx] = yb[pi]
            yrc[idx] = yr[pi]
        cnull.append(evaluate(D, yc, yrc)[0])
    perm = rng.permutation(n)
    A = np.zeros(n, bool)
    A[perm[: n // 2]] = True
    Bm = ~A
    FA = np.ascontiguousarray(D["F"][:, A])
    FB = np.ascontiguousarray(D["F"][:, Bm])
    UXA, UZA, UXB, UZB = [np.ascontiguousarray(M[:, m]) for M, m in ((D["UX"], A), (D["UZ"], A), (D["UX"], Bm), (D["UZ"], Bm))]
    hobs = heldout(D, yb, yr, A, Bm, FA, FB, UXA, UZA, UXB, UZB)
    hnull = []
    for _b in range(B):
        pi = rng.permutation(n)
        hnull.append(heldout(D, yb[pi], yr[pi], A, Bm, FA, FB, UXA, UZA, UXB, UZB))
    res.append(dict(obs=obs.tolist(), null=np.array(null).tolist(), cnull=np.array(cnull).tolist(), hobs=hobs.tolist(), hnull=np.array(hnull).tolist()))
    print(f"rep {r} {time.time() - t0:.1f}s", flush=True)
    (scratch_dir("stat_study") / f"fa_{tgt}_{n}_{seed0}.json").write_text(json.dumps(dict(n=n, target=tgt, names=NAMES, reps=res)))
