"""Correctness check of tt_b5_cost: the fast one-pass criteria against explicit float64 re-implementations that REFIT every ridge / recompute every transform.
Run: python tt_b5_cost_check.py   (prints max abs differences; small n so the explicit refit loop is cheap)."""
import math, time, warnings
import numpy as np
warnings.simplefilter("ignore")
import tt_b5_cost as C


def ref_transforms(y, K, positive=True):
    names, codes, lams = C.cand_arrays(K, positive)
    n = len(y)
    Z = np.empty((n, K)); J = np.empty((n, K))
    floor = y.min() if y.min() > 0 else 1e-12
    for k, (c, lam) in enumerate(zip(codes, lams)):
        ay = np.abs(y); lay = np.log(np.maximum(ay, 1e-6)); yp = np.maximum(y, floor); ly = np.log(yp)
        if c == 0: z, j = y, 0 * y
        elif c == 1: z, j = np.cbrt(y), -math.log(3) - 2 / 3 * lay
        elif c == 2: z, j = np.sign(y) * np.sqrt(ay), -math.log(2) - 0.5 * lay
        elif c == 3: z, j = y * ay, math.log(2) + lay
        elif c == 4: z, j = ly, -ly
        elif c == 5: z, j = -1 / yp, -2 * ly
        elif c == 6: z, j = np.arcsinh(y), -0.5 * np.log1p(y * y)
        elif c == 7:
            pos = y >= 0
            l1 = np.where(pos, np.log1p(np.where(pos, y, 0)), np.log1p(np.where(pos, 0, -y)))
            z = np.where(pos, np.expm1(lam * l1) / lam, -np.expm1((2 - lam) * l1) / (2 - lam))
            j = np.where(pos, (lam - 1) * l1, (1 - lam) * l1)
        else: z, j = np.expm1(lam * ly) / lam, (lam - 1) * ly
        Z[:, k], J[:, k] = z, j
    return names, Z, J


def ref_target(Xt, y, K, B, seed, chunk_rows, lam=1e-6, sel=None, alpha=0.05):
    """Explicit: refit the ridge on A for every candidate and every sign-flip draw (float64, np.linalg)."""
    n = len(y); X = Xt.astype(np.float64).T
    if sel is not None: X = X[:, sel]
    D = np.hstack([np.ones((n, 1)), X])
    nchunks = (n + chunk_rows - 1) // chunk_rows
    nA = sum(min(n, (i + 1) * chunk_rows) - i * chunk_rows for i in range(nchunks // 2))
    A = slice(0, nA); Bh = slice(nA, n)
    positive = y.min() > 0
    names, Z, J = ref_transforms(y, K, positive)
    def ll(Z, J):
        DA, DB = D[A], D[Bh]
        G = DA.T @ DA
        reg = lam * np.trace(G) / G.shape[0]
        out = np.empty(Z.shape[1])
        for k in range(Z.shape[1]):
            W = np.linalg.solve(G + reg * np.eye(G.shape[0]), DA.T @ Z[A, k])
            out[k] = -0.5 * np.log(np.mean((Z[Bh, k] - DB @ W) ** 2)) + J[Bh, k].mean()
        return out
    LL = ll(Z, J)
    Gall = D.T @ D
    beta = np.linalg.solve(Gall + lam * np.trace(Gall) / Gall.shape[0] * np.eye(Gall.shape[0]), D.T @ y)
    yhat = D @ beta; r = y - yhat
    S = np.vstack([C.chunk_signs(seed, i, min(n, (i + 1) * chunk_rows) - i * chunk_rows, B) for i in range(nchunks)]).astype(np.float64)
    T = np.empty(B)
    for b in range(B):
        ys = yhat + S[:, b] * r
        if positive: ys = np.maximum(ys, y.min())
        _, Zs, Js = ref_transforms(ys, K, positive)
        L = ll(Zs, Js)
        T[b] = (L[1:] - L[0]).max()
    return dict(LL=LL, dLL=LL - LL[0], T=T, tau=float(np.quantile(T, 1 - alpha)))


def ref_feature(Xt, y, K, B, seed, null_rows=None, chunk_rows=5000, beta=None):
    p, n = Xt.shape
    y0 = y[:min(chunk_rows, n)]
    r = (y - y0.mean()).astype(np.float32)
    X = Xt.astype(np.float64)
    x0 = X[:, :min(chunk_rows, n)]
    mu = x0.mean(1); sd = x0.std(1) + 1e-12; eps = 1e-3 * sd
    def cands(j):
        x = X[j]; ax = np.abs(x); d = (x - mu[j]) / sd[j]
        zs = [x, np.log(ax + eps[j]), np.sign(x) * np.sqrt(ax), x * ax, np.cbrt(x), np.abs(x - mu[j]), d * d, np.tanh(d), np.exp(np.clip(d, -8, 8)), np.sign(x) / (ax + eps[j])]
        return np.array(zs[:K])
    if null_rows is None:
        rows_idx = np.arange(n)
    else:
        nchunks = (n + chunk_rows - 1) // chunk_rows
        rows_idx = np.concatenate([np.arange(i * chunk_rows, min(n, i * chunk_rows + null_rows)) for i in range(nchunks)])
    R = C.perm_matrix(r[rows_idx], B, seed).astype(np.float64)
    beta = np.zeros(p) if beta is None else beta
    def rho2(Zk, Rm, xj, bj):
        Rm = Rm + bj * xj[:, None]
        Zc = Zk - Zk.mean(1, keepdims=True); Rc = Rm - Rm.mean(0, keepdims=True)
        return (Zc @ Rc) ** 2 / ((Zc ** 2).sum(1)[:, None] * (Rc ** 2).sum(0)[None, :])
    s_obs = np.empty((p, K)); T = np.empty((p, B))
    for j in range(p):
        Zk = cands(j)
        s_obs[j] = rho2(Zk, r.astype(np.float64)[:, None], X[j], beta[j])[:, 0]
        Zs = Zk[:, rows_idx]
        sn = rho2(Zs, R, X[j][rows_idx], beta[j])
        T[j] = (sn[1:, 1:] - sn[0:1, 1:]).max(0) * (1.0 if null_rows is None else len(rows_idx) / n)
    return s_obs, T


if __name__ == "__main__":
    n, p, K, B = 20000, 12, 9, 12
    prov = C.GenProvider(n, p, seed=3, chunk_rows=2000)
    Xt = np.concatenate([prov.chunk(i)[0] for i in range(prov.nchunks)], axis=1)
    y = prov.target()
    for cap in (None, 6):
        t = time.time()
        f = C.target_criterion(prov, K=K, B=B, cap=cap, seed=5)
        ref = ref_target(Xt, y, K, B, 5, 2000, sel=f["sel"])
        print(f"TARGET cap={cap}: max|dLL fast-ref| = {np.abs(f['dLL'] - ref['dLL']).max():.2e}  max|LL fast-ref| = {np.abs(f['LL'] - ref['LL']).max():.2e}  "
              f"max|T fast-ref| = {np.abs(f['null_max'] - ref['T']).max():.2e}  tau fast {f['tau']:.3e} ref {ref['tau']:.3e}  (scale of dLL: {np.abs(ref['dLL']).max():.3f}, of T: {ref['T'].mean():.2e})  best {f['best']}  {time.time()-t:.1f}s", flush=True)
    beta = np.random.default_rng(1).standard_normal(p) * 0.2
    for nr in (None, 400):
        f = C.feature_criterion(prov, K=10, B=B, null_rows=nr, seed=5, row_block=700, beta=beta)
        s_ref, T_ref = ref_feature(Xt, y, 10, B, 5, nr, chunk_rows=2000, beta=beta)
        print(f"FEATURE null_rows={nr}: max|s fast-ref| = {np.abs(f['s'] - s_ref).max():.2e} (scale {s_ref.max():.3f})  max|T fast-ref| = {np.abs(f['null_max'] - T_ref).max():.2e} (scale {T_ref.mean():.2e})", flush=True)
