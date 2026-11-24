"""Real-data validation (round-3 follow-up): does the hybrid's advantage + fe_strict hold OUTSIDE synthetics?

Loads a REAL classification dataset (madelon -- the standard FS benchmark with interactions + many noise features;
falls back to OpenML 'gina_agnostic' / sklearn covtype-binary / breast_cancer if offline). Compares the FS
strategies on honest held-out AUC, AND tests mrmr_fe with the round-3 fe_strict prevalence gates vs the looser
pre-round-3 defaults -- to check fe_strict generalizes before flipping the shared MRMR class default (the
parsimony_tol lesson: don't flip a shared-infra default on one synthetic).
"""
from __future__ import annotations
from typing import Any
import logging

logger = logging.getLogger(__name__)

import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_selector import HybridSelector
import fs_selectors as S
from _downstream_shared import mrmr_sel_transform


def load_real():
    """Return (X df, y series, name). Try real FS datasets in order; fall back to a real sklearn dataset offline."""
    from sklearn.datasets import fetch_openml, load_breast_cancer
    for name, kw in [("madelon", dict(name="madelon", version=1)), ("gina_agnostic", dict(name="gina_agnostic", version=1))]:
        try:
            d = fetch_openml(as_frame=True, parser="auto", **kw)
            X = d.data.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            X.columns = [f"f{i}" for i in range(X.shape[1])]
            y = pd.Series(pd.factorize(d.target)[0]); y = (y == y.value_counts().idxmax()).astype(int).reset_index(drop=True)
            if X.shape[1] >= 20:
                return X.reset_index(drop=True), y, name
        except Exception as e:
            logger.debug("dataset %s skipped: %s: %s", name, type(e).__name__, e)
            print(f"  (skip {name}: {type(e).__name__})", flush=True)
    try:
        from sklearn.datasets import fetch_covtype
        d = fetch_covtype(as_frame=True)
        X = d.frame.iloc[:, :-1].sample(n=6000, random_state=0)
        X.columns = [f"f{i}" for i in range(X.shape[1])]
        y = (d.frame["Cover_Type"].loc[X.index] == 2).astype(int).reset_index(drop=True)
        return X.reset_index(drop=True), y, "covtype_binary"
    except Exception as e:
        logger.debug("covtype dataset load failed, falling back to breast_cancer: %s", e)
        d = load_breast_cancer(as_frame=True)
        X = d.data.copy(); X.columns = [f"f{i}" for i in range(X.shape[1])]
        return X.reset_index(drop=True), d.target.reset_index(drop=True), "breast_cancer"


def downstream(Xtr, Xte, ytr, yte):
    o = {}
    o["lgbm"] = roc_auc_score(yte, np.asarray(lgb.LGBMClassifier(n_estimators=300, verbose=-1).fit(Xtr, ytr).predict_proba(Xte))[:, 1])
    o["logit"] = roc_auc_score(yte, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    o["knn"] = roc_auc_score(yte, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    return {k: round(float(v), 4) for k, v in o.items()}


def mrmr_fe_variant(fe_strict):
    from mlframe.feature_selection.filters import MRMR
    import re
    _SAFE = re.compile(r"^[A-Za-z0-9_]+$")
    class _Sel:
        def fit(self, X, y):
            kw: dict[str, Any] = dict(fe_synergy_min_prevalence=1.5, fe_min_engineered_mi_prevalence=0.97) if fe_strict else {}
            self.m_ = MRMR(verbose=0, fe_max_steps=1, n_jobs=-1, random_seed=0, **kw); self.m_.fit(X, y)
            out = list(self.m_.transform(X.iloc[:5]).columns)
            self.ren_ = {c: (c if _SAFE.match(str(c)) else f"eng_{i}") for i, c in enumerate(out)}
            self.raw_selected_ = [c for c in out if _SAFE.match(str(c)) and c in X.columns]
            return self
        transform = mrmr_sel_transform
    return _Sel()


FLIP_SPLIT_SEEDS = 5


def flip_verdict(deltas, tol=0.003):
    """(holds, mean, se) for paired strict-minus-default deltas; holds only when the lower 95% bound of the mean is at least -tol.

    A single split with a fixed tolerance (12-F33) could not tell a real 0.002 gap from noise.
    """
    d = np.asarray(deltas, dtype=np.float64)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
    return bool(mean - 1.96 * se >= -tol), mean, se


def main():
    X, y, name = load_real()
    print(f"REAL dataset: {name}  shape={X.shape}  pos_rate={round(float(y.mean()),3)}", flush=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    rows = []
    strategies = {
        "all": lambda: None,
        "mrmr_fe_default": lambda: mrmr_fe_variant(fe_strict=False),
        "mrmr_fe_strict": lambda: mrmr_fe_variant(fe_strict=True),
        "rfecv_perm": lambda: S.RFECVSel("lgbm_perm"),
        "boruta_fe": lambda: S.Cascade("boruta_fe", S.MRMRSel(fe=True), S.BorutaSel()),
        "H2_mrmrfe_rfecvlogit": lambda: S.Cascade("H2", S.MRMRSel(fe=True), S.RFECVSel("logit")),
        "hybrid": lambda: HybridSelector(vote=1, use_fe=True),
        "hybrid_nofe": lambda: HybridSelector(vote=1, use_fe=False),
    }
    for nm, mk in strategies.items():
        t0 = time.time()
        sel = mk()
        if sel is None:
            Ztr, Zte = Xtr, Xte
        else:
            sel.fit(Xtr, ytr); Ztr, Zte = sel.transform(Xtr), sel.transform(Xte)
        a = downstream(Ztr, Zte, ytr, yte); am = round(float(np.nanmean(list(a.values()))), 4)
        rows.append(dict(strategy=nm, n=Ztr.shape[1], fit_s=round(time.time() - t0, 1), auc_mean=am, **a))
        print(f"{nm:22s} n={Ztr.shape[1]:3d} {rows[-1]['fit_s']:6.1f}s mean={am} auc={a}", flush=True)
    df = pd.DataFrame(rows).sort_values("auc_mean", ascending=False)
    print("\n=== ranked (real data) ===")
    print(df.to_string(index=False))
    # 12-F33: one split with a fixed tolerance decided the flip. Re-run both variants over several splits and judge the paired delta.
    deltas = []
    for split_seed in range(FLIP_SPLIT_SEEDS):
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.4, random_state=split_seed, stratify=y)
        pair = {}
        for tag, strict in (("default", False), ("strict", True)):
            sel = mrmr_fe_variant(fe_strict=strict)
            sel.fit(Xa, ya)
            Za, Zb = sel.transform(Xa), sel.transform(Xb)
            pair[tag] = float(np.nanmean(list(downstream(Za, Zb, ya, yb).values())))
        deltas.append(pair["strict"] - pair["default"])
    holds, mean_d, se_d = flip_verdict(deltas)
    print(f"\nfe_strict vs default over {FLIP_SPLIT_SEEDS} splits: mean delta {mean_d:+.4f} (se {se_d:.4f})"
          f" -> {'fe_strict holds/helps -> safe to flip MRMR default' if holds else 'fe_strict NOT shown to hold -> do NOT flip global default'}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning); warnings.filterwarnings("ignore", category=DeprecationWarning)  # silenced for the bench run only, not at import
    main()
