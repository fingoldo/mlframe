"""Measurement harness for the MRMR monotone-warp linear-usability tie-break.

Binned MI / SU are monotone-invariant, so a strictly-monotone warp ``g = exp(4 f)`` of an informative ``f`` ties ``f`` exactly on relevance AND redundancy; the redundancy gate keeps one of
``{f, g}`` and -- pre-fix -- the survivor is decided purely by column order (whichever the greedy reached first becomes the DCD anchor). ``f`` and ``g`` are equivalent for trees, but ``f`` is
strictly more linearly usable: a downstream linear model that sees ``g`` instead of ``f`` (especially alongside a second linear signal whose scale the ``exp`` tail dominates after
standardization) loses AUC. The tie-break prefers the more linearly-usable raw member on the exact MI tie (``|corr(col, rank(col))|`` proxy, measured on the RAW pre-binning values).

This harness MEASURES the gate the fix is shipped on:

  1. WARP RECOVERY -- on the warp fixture (f, g=exp(4f), a second linear signal h, noise), across both column orders and >=5 seeds, post-fix MRMR keeps f (not g) on a MAJORITY of seeds and the
     downstream LogisticRegression AUC improves vs pre-fix by a measured margin. Pre-fix flip-flops with column order.

  2. NON-TIE BIT-IDENTITY -- on two ordinary signal+noise fixtures (no monotone-warp duplicate; one with a merely-CORRELATED-but-not-monotone column to prove the gate does not over-fire on
     ordinary collinearity), the selected SET is byte-identical pre-fix vs post-fix; the tie-break must not fire off the band.

Pre-fix is reproduced in-process via ``warp_tiebreak_prefer_linear=False`` -- the production default is ``True``, and the entire new code path is gated behind that flag, so flag-off is the exact
pre-fix code path by construction (no destructive git stash needed in this shared worktree).

Run::

    PYTHONPATH=src CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 python -m mlframe.feature_selection._benchmarks.bench_warp_linear_tiebreak

Measured 2026-06-12 (store py3.14, CPU): reverse (g-first) order f-recovery 0/5 -> 5/5 seeds, downstream AUC 0.9708 -> 0.9836 (delta +0.0128); natural (f-first) order unchanged (already f);
non-tie selection byte-identical across both fixtures x5 seeds. Raw-only (no engineered-hinge rescue) the warp costs a linear model +0.179 AUC at n=3000 -- the FE hinge stage masks part of that
end-to-end, so +0.0128 is the conservative end-to-end figure.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WARP_SEEDS = (0, 1, 2, 3, 4)


def _mrmr(flag: bool, seed: int):
    from mlframe.feature_selection.filters.mrmr import MRMR

    return MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0, random_seed=seed, warp_tiebreak_prefer_linear=flag)


def _make_warp(seed: int, n: int = 3000, reverse: bool = False):
    """f ~ N(0,1); g = exp(4 f) (rank-identical to f -> MI(f;y)==MI(g;y)); a second genuine linear signal h whose scale the exp tail of g dominates after standardization; y from f + h."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n)
    g = np.exp(4.0 * f)
    h = rng.standard_normal(n)
    y = ((f + 0.8 * h + 0.3 * rng.standard_normal(n)) > 0).astype(np.int64)
    cols = {"f": f, "g": g, "h": h}
    for i in range(4):
        cols[f"noise{i}"] = rng.standard_normal(n)
    df = pd.DataFrame(cols)
    if reverse:
        df = df[["g", "f", "h"] + [f"noise{i}" for i in range(4)]]
    return df, pd.Series(y, name="y")


def _make_signal_noise(seed: int, n: int = 2000):
    """Ordinary multi-signal + noise; NO monotone-warp duplicate (the tie-break must not fire)."""
    rng = np.random.default_rng(seed)
    x1, x2, x3 = (rng.standard_normal(n) for _ in range(3))
    y = ((1.5 * x1 - 1.0 * x2 + 0.5 * x3 + 0.4 * rng.standard_normal(n)) > 0).astype(np.int64)
    cols = {"x1": x1, "x2": x2, "x3": x3}
    for i in range(6):
        cols[f"noise{i}"] = rng.standard_normal(n)
    return pd.DataFrame(cols), pd.Series(y, name="y")


def _make_correlated_noise(seed: int, n: int = 2500):
    """A column c = 0.6 a + noise is CORRELATED with a but NOT a monotone duplicate (rank-corr ~0.6, below the 0.99 band) -- proves the gate ignores ordinary collinearity."""
    rng = np.random.default_rng(seed)
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    c = 0.6 * a + 0.4 * rng.standard_normal(n)
    y = ((1.2 * a + 0.8 * b + 0.3 * rng.standard_normal(n)) > 0.2).astype(np.int64)
    cols = {"a": a, "b": b, "c": c}
    for i in range(7):
        cols[f"z{i}"] = rng.standard_normal(n)
    return pd.DataFrame(cols), pd.Series(y, name="y")


def _downstream_auc(sel, df, y, seed: int) -> float:
    Xt = sel.transform(df.copy())
    if hasattr(Xt, "to_numpy"):
        Xt = Xt.to_numpy()
    Xt = np.asarray(Xt, dtype=float)
    Xtr, Xte, ytr, yte = train_test_split(Xt, y.to_numpy(), test_size=0.4, random_state=seed, stratify=y.to_numpy())
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xtr, ytr)
    return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])


def _f_won(names) -> bool:
    joined = " ".join(names)
    raws = joined.replace("noise", "")
    return ("f" in raws) and ("g" not in raws)


def run() -> dict:
    out = {}
    warnings.simplefilter("ignore")
    for reverse in (False, True):
        pre_auc, post_auc, fpre, fpost = [], [], 0, 0
        for s in WARP_SEEDS:
            df, y = _make_warp(s, reverse=reverse)
            so = _mrmr(False, s).fit(df.copy(), y)
            sn = _mrmr(True, s).fit(df.copy(), y)
            fpre += _f_won(list(so.get_feature_names_out()))
            fpost += _f_won(list(sn.get_feature_names_out()))
            pre_auc.append(_downstream_auc(so, df, y, s))
            post_auc.append(_downstream_auc(sn, df, y, s))
        out[f"warp_reverse_{reverse}"] = dict(
            f_recovery_pre=fpre, f_recovery_post=fpost, n_seeds=len(WARP_SEEDS),
            auc_pre=float(np.mean(pre_auc)), auc_post=float(np.mean(post_auc)),
            auc_delta=float(np.mean(post_auc) - np.mean(pre_auc)),
        )
    for name, fx in (("signal_noise", _make_signal_noise), ("correlated_noise", _make_correlated_noise)):
        identical = True
        diffs = []
        for s in range(5):
            df, y = fx(s)
            off = tuple(_mrmr(False, s).fit(df.copy(), y).get_feature_names_out())
            on = tuple(_mrmr(True, s).fit(df.copy(), y).get_feature_names_out())
            if off != on:
                identical = False
                diffs.append((s, off, on))
        out[f"nontie_{name}"] = dict(byte_identical=identical, diffs=diffs)
    return out


if __name__ == "__main__":
    import json

    results = run()
    print(json.dumps(results, indent=2))
    for reverse in (False, True):
        r = results[f"warp_reverse_{reverse}"]
        print(f"WARP reverse={reverse}: f-recovery {r['f_recovery_pre']}/{r['n_seeds']} -> {r['f_recovery_post']}/{r['n_seeds']} | AUC {r['auc_pre']:.4f} -> {r['auc_post']:.4f} ({r['auc_delta']:+.4f})")
    for name in ("signal_noise", "correlated_noise"):
        print(f"NON-TIE {name}: byte-identical={results[f'nontie_{name}']['byte_identical']}")
