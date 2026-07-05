"""Round-4 idea A3-5 -- RICHER FE OPERATOR CLASS (conditional / threshold / ratio / signed) vs PRODUCTS.

mlframe's FE binary operators are products / sums / div / min-max / hypot / atan2 -- it LACKS
conditional/threshold/ratio terms. Some real signal (incl. parts of madelon's structure) is
threshold/conditional, which PRODUCTS cannot linearize. This bench MEASURES whether a richer
operator CLASS recovers signal that products miss.

Two parts (all bench-local, NO production edit; operators are LEAK-SAFE: thresholds computed on
TRAIN only, ratios use eps):

  (1) DESIGNED beds -- y is a NON-product interaction. For each, does engineering the RIGHT operator
      class recover it (downstream AUC > 0.80) where products-only (a*b) FAIL (< 0.70)? Isolates the
      mechanism. Beds: gated y=(a>0)*b ; proximity y=(|a-b|<0.5) ; ratio y=a/(|b|+1) ; signed
      y=sign(a)*|b| ; threshold-and y=(a>0)&(b>0).

  (2) REAL beds -- madelon (load_real) + hard_synth. Pick the top tree-co-occurrence pairs via
      shallow_tree_signals (reused from round4_tree_seed_bench), engineer EACH new operator class for
      those pairs, append to the raw frame, compare downstream AUC vs the products-only (a*b) version
      (== round4 tree_top25+cooccur). Do threshold/ratio operators ADD over products on real data?

VERDICT printed at the end: (a) designed beds confirm recovery of non-product signal? (b) real-data
gain over products-only? Recommend the operator set to add to the FE registry only if (b) holds.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round3_realdata_bench import load_real, downstream
from hard_synth import make_hard_dataset
from round4_tree_seed_bench import shallow_tree_signals

EPS = 1e-9
N_JOBS = 4  # machine under concurrent load


# ------------------------------------------------------------------ leak-safe operators
# Each operator is fit on TRAIN (computes any threshold/median from train only) and applied to both
# train and test with the SAME stored parameters. column-wise; vectorised numpy.

def _fit_params(Xtr, pairs):
    """Per-pair train-only parameters: thresholds (0 and train-median) for each operand, used by ops."""
    p = {}
    for a, b in pairs:
        va, vb = Xtr[a].values, Xtr[b].values
        p[(a, b)] = dict(
            med_a=float(np.median(va)), med_b=float(np.median(vb)),
            # proximity scale: train median |a-b| -> a data-driven "close" threshold
            med_absdiff=float(np.median(np.abs(va - vb))),
            sd_a=float(np.std(va) + EPS), sd_b=float(np.std(vb) + EPS),
        )
    return p


def _apply_op(op, X, pairs, params):
    """Return a DataFrame of engineered columns for `op` over `pairs`, using train-fit `params`."""
    new = {}
    for i, (a, b) in enumerate(pairs):
        if a not in X.columns or b not in X.columns:
            continue
        va, vb = X[a].values.astype(float), X[b].values.astype(float)
        pr = params[(a, b)]
        if op == "product":  # baseline: a*b
            col = va * vb
        elif op == "gated0":  # (a>0)*b   gated interaction, threshold 0
            col = (va > 0.0).astype(float) * vb
        elif op == "gated_med":  # (a>median_a)*b  gated interaction, train median
            col = (va > pr["med_a"]).astype(float) * vb
        elif op == "proximity":  # |a-b| < train-median(|a-b|)  -> proximity indicator
            col = (np.abs(va - vb) < pr["med_absdiff"]).astype(float)
        elif op == "absdiff":  # |a-b|  (continuous proximity, no threshold)
            col = np.abs(va - vb)
        elif op == "ratio":  # a / (|b| + 1)   standardized ratio
            col = va / (np.abs(vb) + 1.0)
        elif op == "ratio_eps":  # a / (|b| + eps) std ratio, small eps (clipped)
            col = np.clip(va / (np.abs(vb) + EPS), -1e6, 1e6)
        elif op == "signed":  # sign(a) * |b|   signed magnitude
            col = np.sign(va) * np.abs(vb)
        elif op == "thr_and":  # (a>0) & (b>0)   threshold-AND (logical)
            col = ((va > 0.0) & (vb > 0.0)).astype(float)
        elif op == "thr_and_med":  # (a>med_a) & (b>med_b)
            col = ((va > pr["med_a"]) & (vb > pr["med_b"])).astype(float)
        else:
            raise ValueError(op)
        new[f"{op}_{i}"] = col
    return pd.DataFrame(new, index=X.index)


# operator classes to evaluate as FE on real data (each appended to the raw frame)
REAL_OPS = ["product", "gated0", "gated_med", "proximity", "absdiff", "ratio", "signed", "thr_and", "thr_and_med"]


# ------------------------------------------------------------------ downstream (frugal lgbm)
def downstream_frugal(Xtr, Xte, ytr, yte):
    """downstream() but with capped lgbm n_jobs + modest n_estimators (concurrent-load friendly)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb
    o = {}
    o["lgbm"] = roc_auc_score(yte, lgb.LGBMClassifier(n_estimators=200, n_jobs=N_JOBS, verbose=-1, random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    o["logit"] = roc_auc_score(yte, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    o["knn"] = roc_auc_score(yte, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    return {k: round(float(v), 4) for k, v in o.items()}


def _checkpoint(msg):
    with open(r"D:/Temp/fe_ops_progress.txt", "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


# ================================================================== PART 1: DESIGNED BEDS
def make_designed(kind, n=6000, seed=0, n_noise=20):
    """Frames where y is a SPECIFIC non-product interaction of two operands a,b, + noise pool.

    Operands a,b are raw columns; downstream must ENGINEER the right operator to linearize.
    We keep the marginal of a and b ~uninformative so pure selection / products are stressed.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    if kind == "gated":  # y depends on (a>0)*b -- a PRODUCT of an indicator and b
        s = (a > 0).astype(float) * b
    elif kind == "proximity":  # y depends on |a-b| being small
        s = -np.abs(a - b)  # closer -> higher logit
    elif kind == "ratio":  # y depends on a/(|b|+1) -- |b| acts as a denominator, NOT linear
        s = a / (np.abs(b) + 1.0)
    elif kind == "signed":  # y depends on sign(a)*|b|
        s = np.sign(a) * np.abs(b)
    elif kind == "thr_and":  # y depends on (a>0)&(b>0)
        s = ((a > 0) & (b > 0)).astype(float)
    else:
        raise ValueError(kind)
    s = (s - s.mean()) / (s.std() + EPS)
    logit = 2.2 * s  # strong, but Bayes < 1 (noise added)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    cols = {"a": a, "b": b}
    for i in range(n_noise):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    order = list(X.columns); rng.shuffle(order); X = X[order]
    return X, pd.Series(y, name="target")


def run_designed():
    print("\n########## PART 1: DESIGNED BEDS (non-product interactions) ##########", flush=True)
    _checkpoint("[ckpt] PART1 designed beds start")
    # which operator class is the "right" one for each bed (for reading the table)
    right_op = {"gated": "gated0", "proximity": "proximity/absdiff", "ratio": "ratio", "signed": "signed", "thr_and": "thr_and"}
    rows = []
    pairs = [("a", "b")]
    for kind in ["gated", "proximity", "ratio", "signed", "thr_and"]:
        X, y = make_designed(kind, seed=0)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
        params = _fit_params(Xtr, pairs)
        # raw-only baseline (no engineered op): can the model get it from a,b alone?
        variants = {"raw_only": None}
        for op in REAL_OPS:
            variants[op] = op
        print(f"\n--- designed bed: {kind}  (right operator: {right_op[kind]}) ---", flush=True)
        for tag, op in variants.items():
            t0 = time.time()
            if op is None:
                Ztr, Zte = Xtr, Xte
            else:
                etr = _apply_op(op, Xtr, pairs, params)
                ete = _apply_op(op, Xte, pairs, params)
                Ztr = pd.concat([Xtr, etr], axis=1)
                Zte = pd.concat([Xte, ete], axis=1)
            a = downstream_frugal(Ztr, Zte, ytr, yte)
            am = round(float(np.nanmean(list(a.values()))), 4)
            rows.append(dict(bed=f"designed_{kind}", variant=tag, n=int(Ztr.shape[1]), fit_s=round(time.time() - t0, 1), auc_mean=am, **a))
            print(f"  {tag:14s} n={int(Ztr.shape[1]):3d} {rows[-1]['fit_s']:5.1f}s mean={am} {a}", flush=True)
    _checkpoint("[ckpt] PART1 done")
    return rows


# ================================================================== PART 2: REAL BEDS
def run_real_bed(name, X, y, seed=0, top_pairs=12):
    print(f"\n########## PART 2 bed: {name}  shape={X.shape} ##########", flush=True)
    _checkpoint(f"[ckpt] PART2 {name} start")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    rows = []

    def emit(tag, Ztr, Zte, t0):
        a = downstream_frugal(Ztr, Zte, ytr, yte)
        am = round(float(np.nanmean(list(a.values()))), 4)
        rows.append(dict(bed=name, variant=tag, n=int(Ztr.shape[1]), fit_s=round(time.time() - t0, 1), auc_mean=am, **a))
        print(f"  {tag:22s} n={int(Ztr.shape[1]):3d} {rows[-1]['fit_s']:6.1f}s mean={am} {a}", flush=True)

    # all-raw baseline
    t0 = time.time(); emit("all_raw", Xtr, Xte, t0)

    # shallow-tree co-occurrence pairs (fit on TRAIN only -- leak-safe)
    t0 = time.time()
    ranked, pairs = shallow_tree_signals(Xtr, ytr, top_pairs=top_pairs)
    tree_s = round(time.time() - t0, 1)
    base = ranked[:25]
    print(f"  shallow-tree: {len(ranked)} nonzero feats, {len(pairs)} pairs, top {pairs[:4]} ({tree_s}s)", flush=True)

    # params from TRAIN only (over the chosen pairs)
    params = _fit_params(Xtr, pairs)

    # base = tree_top25 raw (the round4 reference substrate). Each op appended to THIS base.
    Btr, Bte = Xtr[base], Xte[base]
    t0 = time.time(); emit("tree_top25_raw", Btr, Bte, t0)

    for op in REAL_OPS:
        t0 = time.time()
        etr = _apply_op(op, Xtr, pairs, params)  # operands come from FULL frame, appended to base
        ete = _apply_op(op, Xte, pairs, params)
        Ztr = pd.concat([Btr, etr], axis=1)
        Zte = pd.concat([Bte, ete], axis=1)
        emit(f"tree_top25+{op}", Ztr, Zte, t0)

    # combined "all rich ops" (every non-product op together) appended to base -- does the union win?
    t0 = time.time()
    etr = pd.concat([_apply_op(op, Xtr, pairs, params) for op in REAL_OPS if op != "product"], axis=1)
    ete = pd.concat([_apply_op(op, Xte, pairs, params) for op in REAL_OPS if op != "product"], axis=1)
    emit("tree_top25+ALLrich", pd.concat([Btr, etr], axis=1), pd.concat([Bte, ete], axis=1), t0)

    _checkpoint(f"[ckpt] PART2 {name} done")
    return rows


def main():
    allrows = []
    allrows += run_designed()

    # real beds
    Xr, yr, rname = load_real()
    allrows += run_real_bed(rname, Xr, yr, seed=0)
    Xh, yh, _ = make_hard_dataset(n_samples=5000, seed=0)
    allrows += run_real_bed("hard_synth", Xh, yh, seed=0)

    df = pd.DataFrame(allrows)
    print("\n=================== ALL RESULTS ===================")
    print(df.to_string(index=False), flush=True)

    # ---- PART 1 verdict: did the right operator recover where products fail?
    # The discriminating model is LOGIT (linear): it CANNOT represent threshold/ratio/conditional
    # terms without the engineered operator, and products cannot linearize them either. Trees (lgbm)
    # recover thresholds natively from raw splits, so we report lgbm/mean too but judge on logit.
    print("\n=================== PART 1 VERDICT (designed; recovery judged on LOGIT) ===================")
    print("  (recovery := right-operator logit > 0.80 AND products-only logit < 0.70)")
    for kind in ["gated", "proximity", "ratio", "signed", "thr_and"]:
        sub = df[df.bed == f"designed_{kind}"].set_index("variant")
        for metric in ["logit", "auc_mean", "lgbm"]:
            col = sub[metric]
            prod = float(col.get("product", float("nan")))
            raw = float(col.get("raw_only", float("nan")))
            rich = {op: float(col[op]) for op in REAL_OPS if op != "product" and op in col.index}
            best_rich_op = max(rich, key=rich.get)
            best_rich = rich[best_rich_op]
            tag = ""
            if metric == "logit":
                recovered = best_rich > 0.80
                prod_fails = prod < 0.70
                v = "CONFIRMS" if (recovered and prod_fails) else ("operator-helps" if best_rich - prod > 0.03 else "no-gain")
                tag = f"  -> {v} [rich>0.80={recovered}, prod<0.70={prod_fails}]"
            print(f"  {kind:10s} [{metric:8s}] raw={raw:.4f} product={prod:.4f} " f"best_rich={best_rich:.4f}({best_rich_op}){tag}")

    # ---- PART 2 verdict: do rich ops ADD over products-only on real data?
    print("\n=================== PART 2 VERDICT (real: add over products?) ===================")
    for bed in [b for b in df.bed.unique() if not b.startswith("designed_")]:
        sub = df[df.bed == bed].set_index("variant")
        for metric in ["auc_mean", "logit", "lgbm"]:
            b = sub[metric]
            prod = float(b.get("tree_top25+product", float("nan")))
            base = float(b.get("tree_top25_raw", float("nan")))
            allraw = float(b.get("all_raw", float("nan")))
            rich_ops = {op: float(b[f"tree_top25+{op}"]) for op in REAL_OPS if op != "product" and f"tree_top25+{op}" in b.index}
            allrich = float(b.get("tree_top25+ALLrich", float("nan")))
            best_rich_op = max(rich_ops, key=rich_ops.get)
            best_rich = rich_ops[best_rich_op]
            print(f"\n  [{bed}/{metric}] all_raw={allraw:.4f} tree25_raw={base:.4f} products={prod:.4f}")
            for op, v in sorted(rich_ops.items(), key=lambda kv: -kv[1]):
                print(f"      +{op:12s} {v:.4f}  (d vs products {v - prod:+.4f})")
            print(f"      +ALLrich     {allrich:.4f}  (d vs products {allrich - prod:+.4f})")
            gain = best_rich - prod
            print(f"    BEST rich = +{best_rich_op} {best_rich:.4f}  vs products {prod:.4f}  -> "
                  f"{'ADDS' if gain > 0.005 else ('neutral' if gain > -0.005 else 'WORSE')} ({gain:+.4f})")


if __name__ == "__main__":
    main()
