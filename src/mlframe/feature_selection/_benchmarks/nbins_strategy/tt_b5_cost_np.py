"""B5: the FEATURE criterion of tt_b5_cost rewritten on numpy float32 SIMD ufuncs (log, exp, tanh, sqrt have AVX2 float32 loops in numpy 1.26; cbrt is replaced by
sign * exp(log|x| / 3) because numpy's cbrt is a scalar libm call) over cache-sized blocks, the projections on r and on the permuted residuals done by one sgemm per block,
and the per-column moments by einsum on the same block.  Same statistics as ``tt_b5_cost.feature_criterion`` (checked in tt_b5_cost_np_check.py), same API.
A small thread pool over column blocks uses ``workers`` threads (numpy ufuncs and BLAS release the GIL)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

import tt_b5_cost as C
from tt_b5_cost import _Stage, perm_matrix, _rho2, SubProvider


def _block(xb, mu, isd, eps, K, Zv):
    """Fill Zv (nb, K, rs) float32 with the K candidates of the block xb (nb, rs) float32; mu, isd, eps are (nb, 1) float32.  np.copysign is a slow scalar loop, so the sign is
    applied by one multiplication with np.sign(xb)."""
    ax = np.abs(xb)
    sg = np.sign(xb)
    Zv[:, 0, :] = xb
    np.log(ax + eps, out=Zv[:, 1, :])
    np.multiply(np.sqrt(ax), sg, out=Zv[:, 2, :])
    np.multiply(xb, ax, out=Zv[:, 3, :])
    t = np.log(ax + np.float32(1e-30))
    t *= np.float32(1.0 / 3.0)
    np.exp(t, out=t)
    np.multiply(t, sg, out=Zv[:, 4, :])
    c = xb - mu
    np.abs(c, out=Zv[:, 5, :])
    d = c * isd
    np.multiply(d, d, out=Zv[:, 6, :])
    np.tanh(d, out=Zv[:, 7, :])
    if K > 8:
        np.exp(np.clip(d, -8.0, 8.0), out=Zv[:, 8, :])
    if K > 9:
        np.multiply(np.float32(1.0) / (ax + eps), sg, out=Zv[:, 9, :])


def feature_criterion_np(prov, K=9, B=40, null_rows=None, seed=0, alpha=0.05, col_block=16, row_block=2048, workers=2, timing=None, beta=None):
    tm = {} if timing is None else timing
    p = prov.p
    beta = np.zeros(p) if beta is None else np.asarray(beta, dtype=np.float64)
    with _Stage(tm, "pilot", prov):
        Xt0, y0 = prov.chunk(0)
        mu = Xt0.mean(1, dtype=np.float64)
        sd = Xt0.std(1, dtype=np.float64) + 1e-12
        mu32, isd32, eps32 = mu.astype(np.float32)[:, None], (1.0 / sd).astype(np.float32)[:, None], (1e-3 * sd).astype(np.float32)[:, None]
        yall = prov.target()
        r_all = (yall - yall[: len(y0)].mean()).astype(np.float32)
    n = prov.n
    sub = SubProvider(prov, null_rows) if null_rows is not None else prov
    n_u = sub.n
    with _Stage(tm, "null_setup", prov):
        if null_rows is None:
            r_u = r_all
        else:
            r_u = np.concatenate([r_all[prov.bounds(i)[0]: prov.bounds(i)[0] + (sub.bounds(i)[1] - sub.bounds(i)[0])] for i in range(prov.nchunks)])
        R = perm_matrix(r_u, B, seed) if B > 0 else r_u[:, None].astype(np.float32)
    Bc = R.shape[1]
    accn = np.zeros((p * K, Bc))
    mom = np.zeros((p, K, 3))
    nbk = [(j0, min(col_block, p - j0)) for j0 in range(0, p, col_block)]
    tls = {}

    def work(args):
        Xt, i0, rs, Rb, j0, nb = args
        import threading
        buf = tls.get(threading.get_ident())
        if buf is None or buf.shape[0] < col_block * K:
            buf = tls[threading.get_ident()] = np.empty((col_block * K, row_block), dtype=np.float32)
        xb = Xt[j0:j0 + nb, i0:i0 + rs]
        Zt = buf[: nb * K, :rs]
        Zv = Zt.reshape(nb, K, rs)
        _block(xb, mu32[j0:j0 + nb], isd32[j0:j0 + nb], eps32[j0:j0 + nb], K, Zv)
        accn[j0 * K:(j0 + nb) * K] += Zt @ Rb
        mom[j0:j0 + nb, :, 0] += Zv.sum(axis=2)                       # float32 block sums (2048 terms), accumulated in float64 across blocks
        mom[j0:j0 + nb, :, 1] += np.einsum("jkr,jkr->jk", Zv, Zv)
        mom[j0:j0 + nb, :, 2] += np.einsum("jkr,jr->jk", Zv, xb)

    with _Stage(tm, "stream", prov), ThreadPoolExecutor(max_workers=workers) as ex:
        off = 0
        for i in range(sub.nchunks):
            Xt, _ = sub.chunk(i)
            rows = Xt.shape[1]
            for i0 in range(0, rows, row_block):
                rs = min(row_block, rows - i0)
                Rb = R[off + i0: off + i0 + rs]
                list(ex.map(work, [(Xt, i0, rs, Rb, j0, nb) for j0, nb in nbk]))
            off += rows
    with _Stage(tm, "finish", prov):
        Sr = float(r_u.sum(dtype=np.float64)); Srr = float((r_u.astype(np.float64) ** 2).sum())
        sn = _rho2(mom[:, :, 0], mom[:, :, 1], mom[:, :, 2], accn.reshape(p, K, Bc), Sr, Srr, beta, n_u)
        out = dict(K=K, p=p, names=C.FEATURE_CANDS[:K], s=sn[:, :, 0])
        if null_rows is not None and B > 0:
            # the observed statistic must use every row: the subsample null is only for the threshold (this variant reports the subsample observed value; use feature_criterion for all rows)
            out["observed_on_subsample"] = True
        gain = sn[:, 1:, 0] - sn[:, 0:1, 0]
        kb = gain.argmax(1) + 1
        best_gain = gain[np.arange(p), kb - 1]
        out.update(best_k=kb, best_gain=best_gain)
        if B > 0:
            T = (sn[:, 1:, 1:] - sn[:, 0:1, 1:]).max(1)
            if null_rows is not None:
                T = T * (n_u / n)
            tau = np.quantile(T, 1 - alpha, axis=1)
            out.update(tau=tau, null_max=T, accepted=best_gain > np.maximum(tau, 0.0))
    out["timing"] = dict(tm)
    return out


def feature_criterion_np_sub(prov, K=9, B=40, null_rows=2500, seed=0, alpha=0.05, timing=None, **kw):
    """Observed statistic on ALL rows (B = 0 pass), the null on the first ``null_rows`` rows of every chunk, threshold rescaled by n_sub / n."""
    tm = {} if timing is None else timing
    obs = feature_criterion_np(prov, K=K, B=0, seed=seed, alpha=alpha, timing=tm, **kw)
    nul = feature_criterion_np(prov, K=K, B=B, null_rows=null_rows, seed=seed, alpha=alpha, timing=tm, **kw)
    obs.update(tau=nul["tau"], null_max=nul["null_max"], accepted=obs["best_gain"] > np.maximum(nul["tau"], 0.0), timing=dict(tm))
    return obs
