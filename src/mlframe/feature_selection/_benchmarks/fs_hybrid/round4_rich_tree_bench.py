"""Round-4 follow-up: extend the tree member to engineer RICH operators (not just a*b products) per co-occurrence pair.

Agent E (A3-5) measured that products + rich operators (absdiff |a-b|, signed sign(a)|b|, ratio a/(|b|+1)) beats
products-only on madelon (+0.026 mean, +0.054 logit) and hard_synth (logit +0.031), and a capacity control proved
the gain is the OPERATOR CLASS not column count. The hybrid's tree member currently folds only a*b products. This
prototype (subclass, NO production edit) folds rich operators of the SAME co-occurrence pairs, all synergy-gated
(FI[col] > max(FI[operands])) so only the regime-useful ones survive. Measure vs the shipped tree-member hybrid.
PASS: madelon/hard_synth >= +0.01 mean WITHOUT regressing synth > 0.005.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import train_test_split
from round3_realdata_bench import load_real, downstream
from synth import make_dataset
from hard_synth import make_hard_dataset
from hybrid_selector import HybridSelector

_OPS = {
    "mul": lambda a, b: a * b,
    "absd": lambda a, b: np.abs(a - b),
    "sign": lambda a, b: np.sign(a) * np.abs(b),
    "rat": lambda a, b: a / (np.abs(b) + 1.0),
}


class RichTreeHybrid(HybridSelector):
    """Tree member that engineers a small RICH-operator set per co-occurrence pair (mul/absd/sign/rat), each folded
    into the shared frame and synergy-gated like the base product columns."""

    def __init__(self, *a, rich_ops=("mul", "absd", "sign", "rat"), **kw):
        super().__init__(*a, **kw)
        self.rich_ops = tuple(rich_ops)

    def _tree_signals(self, X, y):
        super()._tree_signals(X, y)  # ranking + base a*b pairs/names (FE-gated)
        base_pairs = list(getattr(self, "_tree_prod_pairs_", []))
        self._tree_op_ = {}
        pairs, names = [], []
        for i, (aa, bb) in enumerate(base_pairs):
            for op in self.rich_ops:
                nm = f"t{op}_{i}"
                pairs.append((aa, bb)); names.append(nm); self._tree_op_[nm] = (aa, bb, op)
        self._tree_prod_pairs_, self._tree_prod_names_ = pairs, names

    def _augment(self, X):
        # build MRMR eng cols via the base path with tree products temporarily disabled, then add rich-op cols
        saved_pairs, saved_names = self._tree_prod_pairs_, self._tree_prod_names_
        self._tree_prod_pairs_, self._tree_prod_names_ = [], []
        base = super()._augment(X)  # [raw | MRMR eng]
        self._tree_prod_pairs_, self._tree_prod_names_ = saved_pairs, saved_names
        cols = {}
        for nm in self._tree_prod_names_:
            aa, bb, op = self._tree_op_[nm]
            if aa in X.columns and bb in X.columns:
                v = _OPS[op](X[aa].values.astype(float), X[bb].values.astype(float))
                cols[nm] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if not cols:
            return base
        extra = pd.DataFrame(cols, index=base.reset_index(drop=True).index)
        aug = pd.concat([base.reset_index(drop=True), extra], axis=1)
        return aug.loc[:, ~aug.columns.duplicated()]


def run(name, X, y, seeds=(0, 1, 2)):
    rows = []
    for sd in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)
        for tag, cls in (("baseline", HybridSelector), ("richtree", RichTreeHybrid)):
            t0 = time.time()
            h = cls(vote=1, use_fe=True, random_state=sd).fit(Xtr, ytr)
            a = downstream(h.transform(Xtr), h.transform(Xte), ytr, yte); am = round(float(np.nanmean(list(a.values()))), 4)
            rows.append(dict(bed=name, seed=sd, variant=tag, n=int(h.transform(Xte).shape[1]), fit_s=round(time.time()-t0,1), auc_mean=am, **a))
            print(f"  [{name}] sd{sd} {tag:9s} n={rows[-1]['n']:3d} {rows[-1]['fit_s']:6.1f}s mean={am} {a}", flush=True)
    return rows


def main():
    allrows = []
    Xh, yh, _ = make_hard_dataset(n_samples=5000, seed=0); print(f"=== hard_synth {Xh.shape} ===", flush=True)
    allrows += run("hard_synth", Xh, yh)
    Xr, yr, rname = load_real(); print(f"=== {rname} {Xr.shape} ===", flush=True)
    allrows += run(rname, Xr, yr)
    Xs, ys, _ = make_dataset(n_samples=5000, seed=0); print(f"=== synth {Xs.shape} ===", flush=True)
    allrows += run("synth", Xs, ys)
    df = pd.DataFrame(allrows)
    print("\n=== mean over seeds ===")
    print(df.groupby(["bed", "variant"]).agg(auc=("auc_mean", "mean"), std=("auc_mean", "std"), n=("n", "mean")).round(4).to_string())
    print("\n=== verdict ===")
    for bed in df.bed.unique():
        b0 = df[(df.bed == bed) & (df.variant == "baseline")]["auc_mean"].mean()
        b1 = df[(df.bed == bed) & (df.variant == "richtree")]["auc_mean"].mean()
        print(f"  {bed:12s} richtree {round(b1,4)} vs baseline {round(b0,4)}  d={round(b1-b0,4):+}")
    df.to_csv("D:/Temp/round4_rich_tree_rows.csv", index=False)


if __name__ == "__main__":
    main()
