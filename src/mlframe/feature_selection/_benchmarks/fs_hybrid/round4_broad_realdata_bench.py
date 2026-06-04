"""Round-4 BROAD real-data generalisation test for this round's 7 shipped FS wins.

The 7 wins shipped this round were validated on madelon + 2 synthetics only. This bench asks the honest
generalisation question: do the production HybridSelector (the 3 default-on hybrid wins -- tree member,
mrmr_synergy_cap=250, tree_rich_ops) and MRMRTreeRescued (the gated tree-importance rescue) hold an
advantage over plain mrmr_fe / all-features ACROSS several REAL classification datasets, not just madelon?

It loads several real FS-relevant datasets via fetch_openml (try, skip on failure), and on an honest
60/40 stratified split compares downstream (lgbm / logit / knn) held-out AUC of:
  all          -- all features, no selection (the do-nothing baseline)
  mrmr_fe      -- MRMR + feature engineering (fe_max_steps=1), the incumbent FS baseline
  hybrid       -- HybridSelector(vote=1, use_fe=True): the 3 shipped hybrid wins, all default-on
  mrmr_tree    -- MRMRTreeRescued(fe_max_steps=1): the gated tree-importance rescue win

Reports n_selected + per-model AUC + mean AUC for every (dataset x strategy), then a per-win generalisation
verdict that explicitly flags any dataset where the hybrid REGRESSES vs mrmr_fe (that is the point of the test).

READ + BENCH ONLY: no production file is edited; this is a new bench file. Memory-frugal for wide data
(gisette = 5000 feats): rows subsampled to <= ROW_BUDGET and features capped to <= FEAT_BUDGET (noted in the
report). On OOM / paging / "Unable to allocate" / LightGBM "model format" the dataset is retried ONCE after a
90s sleep with halved row/feature budgets. stdout -> D:/Temp/broad_val_bench_stdout.txt, checkpoints ->
D:/Temp/broad_val_progress.txt, results table -> D:/Temp/broad_val_results.md.
"""
from __future__ import annotations
import os, sys, time, gc, traceback

os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round3_realdata_bench import downstream  # lgbm/logit/knn held-out AUC, returns {model: auc}
from hybrid_selector import HybridSelector

import re
_SAFE = re.compile(r"^[A-Za-z0-9_]+$")

PROGRESS = r"D:/Temp/broad_val_progress.txt"
RESULTS = r"D:/Temp/broad_val_results.md"

# Wide-data budgets (memory frugality). gisette has 5000 feats; the hybrid runs n_jobs=-1 internally and the
# permutation-FI pass + boruta are O(rows*feats), so cap both. Halved on the retry path.
ROW_BUDGET = 3000
FEAT_BUDGET = 1200


def _chk(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(PROGRESS, "a", encoding="ascii", errors="replace") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------- dataset loading
# Each entry: (display_name, fetch_openml kwargs). Tried in order; any failure is skipped (logged), not fatal.
# A spread of FS-relevant real classification beds: madelon (the anchor / interaction+noise benchmark), gina_agnostic
# (round-3 fallback), gisette (5000-feat wide stress test), plus several more real ones. sklearn breast_cancer is the
# guaranteed-offline fallback so the bench always produces at least one real comparison.
OPENML_SETS = [
    ("madelon",       dict(name="madelon", version=1)),
    ("gina_agnostic", dict(name="gina_agnostic", version=1)),
    ("gisette",       dict(name="gisette", version=1)),
    ("scene",         dict(name="scene", version=1)),
    ("Bioresponse",   dict(name="Bioresponse", version=1)),
    ("hill-valley",   dict(name="hill-valley", version=1)),
    ("isolet",        dict(name="isolet", version=1)),
    ("arcene",        dict(name="arcene", version=1)),
]


def _clean_xy(d, name, feat_budget):
    """Coerce a fetched OpenML bundle to (numeric X df with safe f-names, binary y series, used_feat_count)."""
    X = d.data
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # drop zero-variance columns (pure noise / constant) -- cheap and helps the wide beds
    nunique = X.nunique()
    X = X.loc[:, nunique[nunique > 1].index]
    # cap features for wide beds: keep the highest-variance feat_budget columns (variance is a y-blind, leak-free proxy)
    capped = False
    if X.shape[1] > feat_budget:
        var = X.var(axis=0).sort_values(ascending=False)
        X = X[var.index[:feat_budget]]
        capped = True
    X.columns = [f"f{i}" for i in range(X.shape[1])]
    # binary target: factorize, then "is the majority class" (consistent with round3 load_real)
    y = pd.Series(pd.factorize(np.asarray(d.target).ravel())[0])
    y = (y == y.value_counts().idxmax()).astype(int).reset_index(drop=True)
    return X.reset_index(drop=True), y, capped


def load_one(name, kw, row_budget, feat_budget):
    """Fetch + clean + subsample one OpenML dataset. Returns (X, y, note) or raises (caller skips)."""
    from sklearn.datasets import fetch_openml
    d = fetch_openml(as_frame=True, parser="auto", **kw)
    X, y, capped = _clean_xy(d, name, feat_budget)
    notes = []
    if capped:
        notes.append(f"feat-capped to {X.shape[1]} (top-var of original)")
    # row subsample for memory frugality (stratified) on big beds
    if X.shape[0] > row_budget:
        idx, _ = train_test_split(np.arange(X.shape[0]), train_size=row_budget, random_state=0, stratify=y)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
        notes.append(f"rows subsampled to {row_budget}")
    if X.shape[1] < 5 or X.shape[0] < 100 or y.nunique() < 2:
        raise ValueError(f"degenerate after clean: shape={X.shape} y_nunique={y.nunique()}")
    return X, y, ("; ".join(notes) if notes else "full")


def fallback_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True)
    X = d.data.copy(); X.columns = [f"f{i}" for i in range(X.shape[1])]
    return X.reset_index(drop=True), d.target.reset_index(drop=True), "full (sklearn offline fallback)"


# ---------------------------------------------------------------------------- strategies
def _safe_rename(out_cols, X_cols):
    """Map an MRMR/transform output column list to LightGBM-safe names; non-ASCII engineered recipe names -> eng_N."""
    ren, k = {}, 0
    for c in out_cols:
        if _SAFE.match(str(c)) and c in X_cols:
            ren[c] = c
        else:
            ren[c] = f"eng_{k}"; k += 1
    return ren


def make_mrmr_fe():
    """Plain MRMR + FE (fe_max_steps=1), the incumbent FS baseline. Returns a fit/transform adapter."""
    from mlframe.feature_selection.filters import MRMR

    class _Sel:
        def fit(self, X, y):
            self.m_ = MRMR(verbose=0, fe_max_steps=1, n_jobs=-1, random_seed=0)
            self.m_.fit(X, y)
            out = list(self.m_.transform(X.iloc[:5]).columns)
            self.ren_ = _safe_rename(out, set(X.columns))
            return self

        def transform(self, X):
            df = self.m_.transform(X).copy(); df.columns = [self.ren_[c] for c in df.columns]; return df
    return _Sel()


def make_mrmr_tree():
    """MRMRTreeRescued(fe_max_steps=1): MRMR + FE + the gated tree-importance rescue (the shipped rescue win)."""
    from mlframe.feature_selection.filters import MRMRTreeRescued

    class _Sel:
        def fit(self, X, y):
            self.m_ = MRMRTreeRescued(verbose=0, fe_max_steps=1, n_jobs=-1, random_seed=0)
            self.m_.fit(X, y)
            out = list(self.m_.transform(X.iloc[:5]).columns)
            self.ren_ = _safe_rename(out, set(X.columns))
            return self

        def transform(self, X):
            df = self.m_.transform(X).copy(); df.columns = [self.ren_[c] for c in df.columns]; return df
    return _Sel()


STRATEGIES = {
    "all":       lambda: None,                              # no selection (do-nothing baseline)
    "mrmr_fe":   make_mrmr_fe,                              # incumbent FS baseline
    "hybrid":    lambda: HybridSelector(vote=1, use_fe=True),  # the 3 shipped hybrid wins (default-on)
    "mrmr_tree": make_mrmr_tree,                            # the shipped MRMRTreeRescued rescue win
}


def eval_strategy(nm, mk, Xtr, Xte, ytr, yte):
    """Fit one strategy + score downstream AUC. Returns a result dict (or an error dict on failure)."""
    t0 = time.time()
    sel = mk()
    if sel is None:
        Ztr, Zte = Xtr, Xte
    else:
        sel.fit(Xtr, ytr)
        Ztr, Zte = sel.transform(Xtr), sel.transform(Xte)
    # align test columns to train (engineered name order is deterministic, but guard anyway)
    common = [c for c in Ztr.columns if c in Zte.columns]
    Ztr, Zte = Ztr[common], Zte[common]
    a = downstream(Ztr, Zte, ytr, yte)
    am = round(float(np.nanmean(list(a.values()))), 4)
    return dict(strategy=nm, n=int(Ztr.shape[1]), fit_s=round(time.time() - t0, 1), auc_mean=am, **a)


_OOM_MARKERS = ("unable to allocate", "out of memory", "memoryerror", "paging file",
                "bad allocation", "model format", "cannot allocate")


def _is_oom(exc) -> bool:
    s = (str(exc) + " " + type(exc).__name__).lower()
    return isinstance(exc, MemoryError) or any(m in s for m in _OOM_MARKERS)


def run_dataset(name, X, y, note):
    """Run all strategies on one dataset (honest 60/40 stratified split). Returns list of result rows."""
    _chk(f"DATASET {name}: shape={X.shape} pos_rate={round(float(y.mean()),3)} note=[{note}]")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    rows = []
    for nm, mk in STRATEGIES.items():
        try:
            r = eval_strategy(nm, mk, Xtr, Xte, ytr, yte)
            r["dataset"] = name
            rows.append(r)
            _chk(f"  {name}/{nm:10s} n={r['n']:4d} {r['fit_s']:6.1f}s mean={r['auc_mean']} "
                 f"lgbm={r.get('lgbm')} logit={r.get('logit')} knn={r.get('knn')}")
        except Exception as e:
            _chk(f"  {name}/{nm:10s} STRATEGY FAILED: {type(e).__name__}: {e}")
            rows.append(dict(dataset=name, strategy=nm, n=-1, fit_s=-1.0, auc_mean=float("nan"),
                             lgbm=float("nan"), logit=float("nan"), knn=float("nan"),
                             error=f"{type(e).__name__}: {e}"))
        gc.collect()
    return rows


def load_with_retry(name, kw):
    """Load + run a dataset; on OOM-ish failure sleep 90s and retry ONCE with halved budgets. Returns rows or []."""
    for attempt, (rb, fb) in enumerate([(ROW_BUDGET, FEAT_BUDGET), (ROW_BUDGET // 2, FEAT_BUDGET // 2)]):
        try:
            X, y, note = load_one(name, kw, rb, fb)
            if attempt == 1:
                note = note + " | RETRY (reduced budget after OOM)"
            return run_dataset(name, X, y, note)
        except Exception as e:
            if _is_oom(e) and attempt == 0:
                _chk(f"  (OOM-ish on {name}: {type(e).__name__}: {e}; sleeping 90s then retrying with reduced budget)")
                gc.collect(); time.sleep(90); continue
            _chk(f"  (skip {name}: {type(e).__name__}: {e})")
            if attempt == 0 and not _is_oom(e):
                return []  # genuine load failure (dataset absent / network) -> skip, do not retry
    return []


# ---------------------------------------------------------------------------- report
def write_results(df):
    if df.empty:
        with open(RESULTS, "w", encoding="ascii", errors="replace") as f:
            f.write("# Broad real-data FS validation\n\nNo datasets loaded.\n")
        return "No datasets loaded."

    lines = ["# Broad real-data FS validation (round-4)\n",
             "Honest 60/40 stratified split; downstream held-out AUC (lgbm / logit / knn) + mean.",
             "Strategies: all (no selection) | mrmr_fe (incumbent) | hybrid (3 shipped hybrid wins) | "
             "mrmr_tree (MRMRTreeRescued rescue win).\n"]

    # full table
    lines.append("## Full dataset x strategy table\n")
    lines.append("| dataset | strategy | n_sel | fit_s | lgbm | logit | knn | mean |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ds in df["dataset"].drop_duplicates():
        sub = df[df.dataset == ds]
        for _, r in sub.iterrows():
            def g(k):
                v = r.get(k)
                return "" if (v is None or (isinstance(v, float) and np.isnan(v))) else (f"{v:.4f}" if isinstance(v, float) else v)
            lines.append(f"| {ds} | {r['strategy']} | {int(r['n']) if r['n']==r['n'] else ''} | "
                         f"{r['fit_s']} | {g('lgbm')} | {g('logit')} | {g('knn')} | {g('auc_mean')} |")
        lines.append("| | | | | | | | |")

    # per-dataset deltas vs mrmr_fe and vs all
    lines.append("\n## Per-dataset deltas (mean AUC)\n")
    lines.append("| dataset | all | mrmr_fe | hybrid | mrmr_tree | hybrid-mrmr_fe | mrmr_tree-mrmr_fe | hybrid-all |")
    lines.append("|---|---|---|---|---|---|---|---|")
    deltas = []
    for ds in df["dataset"].drop_duplicates():
        sub = df[df.dataset == ds].set_index("strategy")
        def m(s):
            return float(sub.loc[s, "auc_mean"]) if s in sub.index else float("nan")
        a_all, a_fe, a_hy, a_tr = m("all"), m("mrmr_fe"), m("hybrid"), m("mrmr_tree")
        d_hy_fe = a_hy - a_fe
        d_tr_fe = a_tr - a_fe
        d_hy_all = a_hy - a_all
        deltas.append(dict(dataset=ds, hy_fe=d_hy_fe, tr_fe=d_tr_fe, hy_all=d_hy_all))
        def f(v):
            return "" if (v != v) else f"{v:.4f}"
        lines.append(f"| {ds} | {f(a_all)} | {f(a_fe)} | {f(a_hy)} | {f(a_tr)} | "
                     f"{f(d_hy_fe):>+} | {f(d_tr_fe):>+} | {f(d_hy_all):>+} |")

    # verdicts
    dd = pd.DataFrame(deltas)
    def summ(col, label, baseline):
        v = dd[col].dropna()
        if v.empty:
            return f"- {label}: no comparable datasets."
        wins = int((v > 0.002).sum()); ties = int((v.abs() <= 0.002).sum()); regress = int((v < -0.002).sum())
        worst = v.min(); worst_ds = dd.loc[v.idxmin(), "dataset"] if not v.empty else "?"
        best = v.max(); best_ds = dd.loc[v.idxmax(), "dataset"] if not v.empty else "?"
        regset = ", ".join(f"{r.dataset}({r[col]:+.4f})" for _, r in dd.iterrows() if r[col] == r[col] and r[col] < -0.002)
        out = (f"- {label} (vs {baseline}, n={len(v)}): mean delta {v.mean():+.4f}, median {v.median():+.4f}; "
               f"{wins} win / {ties} tie / {regress} regress (|>0.002|). "
               f"best {best_ds} {best:+.4f}; worst {worst_ds} {worst:+.4f}.")
        if regset:
            out += f" REGRESSIONS: {regset}."
        return out

    lines.append("\n## Generalisation verdict\n")
    lines.append("Threshold: |delta| > 0.002 mean-AUC counts as a win/regression; within is a tie.\n")
    lines.append(summ("hy_fe", "hybrid", "mrmr_fe"))
    lines.append(summ("tr_fe", "mrmr_tree", "mrmr_fe"))
    lines.append(summ("hy_all", "hybrid", "all-features"))

    text = "\n".join(lines) + "\n"
    with open(RESULTS, "w", encoding="ascii", errors="replace") as f:
        f.write(text)
    return text


def main():
    for p in (PROGRESS, RESULTS):
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
        except Exception:
            pass
    open(PROGRESS, "w", encoding="ascii").close()
    _chk("=== broad real-data FS validation START ===")
    _chk(f"budgets: ROW_BUDGET={ROW_BUDGET} FEAT_BUDGET={FEAT_BUDGET}")

    allrows = []
    loaded_any = False
    for name, kw in OPENML_SETS:
        try:
            rows = load_with_retry(name, kw)
        except Exception as e:
            _chk(f"  (hard-skip {name}: {type(e).__name__}: {e})")
            rows = []
        if rows:
            loaded_any = True
            allrows += rows
            # incremental results write so partial progress survives a later OOM
            try:
                write_results(pd.DataFrame(allrows))
            except Exception as e:
                _chk(f"  (results write failed mid-run: {type(e).__name__}: {e})")

    if not loaded_any:
        _chk("No OpenML datasets loaded -> falling back to sklearn breast_cancer")
        X, y, note = fallback_breast_cancer()
        allrows += run_dataset("breast_cancer", X, y, note)

    df = pd.DataFrame(allrows)
    _chk("=== ALL RESULTS ===")
    show_cols = [c for c in ["dataset", "strategy", "n", "fit_s", "lgbm", "logit", "knn", "auc_mean"] if c in df.columns]
    print(df[show_cols].to_string(index=False), flush=True)
    text = write_results(df)
    _chk(f"=== wrote results to {RESULTS} ===")
    print("\n" + text, flush=True)
    _chk("=== broad real-data FS validation DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _chk("FATAL:\n" + traceback.format_exc())
        raise
