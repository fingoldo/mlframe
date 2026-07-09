"""Fairness/attribution control for the ALLrich gain found in fe_richops_bench.

Main bench result: on madelon, tree_top25+ALLrich (n=121) beat tree_top25+product (n=37) by ~+0.020
mean AUC. But ALLrich adds 96 engineered columns vs products' 12 -> the gain could be feature-COUNT
capacity, not the operator class. This control disentangles it on the two real beds:

  base               = tree_top25 raw (25)  -- substrate
  +product           = base + a*b for the 12 pairs        (37)  -- products-only reference
  +product+ALLrich   = base + products + all rich ops     (133)  -- do rich ADD on top of products?
  +product+noiseN    = base + products + N random-noise cols  -- CAPACITY control (N=96, ALLrich size)
  +ALLrich           = base + all rich ops (no products)  (121)  -- rich instead of products (main-bench dup)
  +product12x2       = base + a*b for 24 pairs (more pairs, same op)  -- "just more pairs" control

If +product+ALLrich >> +product AND >> +product+noise(96) -> rich operators genuinely ADD signal
products miss (recommend). If +product+ALLrich ~ +product+noise(96) -> the win was capacity, not class.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round3_realdata_bench import load_real
from hard_synth import make_hard_dataset
from round4_tree_seed_bench import shallow_tree_signals
from fe_richops_bench import _fit_params, _apply_op, REAL_OPS, downstream_frugal


def _checkpoint(msg):
    with open(r"D:/Temp/fe_ops_progress.txt", "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def run_bed(name, X, y, seed=0):
    print(f"\n########## CONTROL bed: {name}  shape={X.shape} ##########", flush=True)
    _checkpoint(f"[ckpt] CONTROL {name} start")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    rng = np.random.default_rng(123)
    rows = []

    def emit(tag, Ztr, Zte, t0):
        a = downstream_frugal(Ztr, Zte, ytr, yte)
        am = round(float(np.nanmean(list(a.values()))), 4)
        rows.append(dict(bed=name, variant=tag, n=int(Ztr.shape[1]), fit_s=round(time.time() - t0, 1), auc_mean=am, **a))
        print(f"  {tag:24s} n={int(Ztr.shape[1]):3d} {rows[-1]['fit_s']:6.1f}s mean={am} {a}", flush=True)

    # 24 pairs so product12x2 has a same-class-but-more-pairs control
    ranked, pairs = shallow_tree_signals(Xtr, ytr, top_pairs=24)
    base_cols = ranked[:25]
    Btr, Bte = Xtr[base_cols], Xte[base_cols]
    pairs12 = pairs[:12]
    params = _fit_params(Xtr, pairs)  # params for all 24 pairs

    def eng(op, frame, prs):
        return _apply_op(op, frame, prs, params)

    prod12_tr = eng("product", Xtr, pairs12); prod12_te = eng("product", Xte, pairs12)
    prod24_tr = eng("product", Xtr, pairs);   prod24_te = eng("product", Xte, pairs)
    rich_ops = [o for o in REAL_OPS if o != "product"]
    rich12_tr = pd.concat([eng(o, Xtr, pairs12) for o in rich_ops], axis=1)
    rich12_te = pd.concat([eng(o, Xte, pairs12) for o in rich_ops], axis=1)
    n_rich = rich12_tr.shape[1]

    # capacity control: same #cols as ALLrich, pure gaussian noise
    noise_tr = pd.DataFrame(rng.standard_normal((Btr.shape[0], n_rich)), columns=[f"noi_{i}" for i in range(n_rich)], index=Btr.index)
    noise_te = pd.DataFrame(rng.standard_normal((Bte.shape[0], n_rich)), columns=[f"noi_{i}" for i in range(n_rich)], index=Bte.index)

    t0 = time.time()
    emit("base_raw25", Btr, Bte, t0)
    t0 = time.time()
    emit("+product12", pd.concat([Btr, prod12_tr], axis=1), pd.concat([Bte, prod12_te], axis=1), t0)
    t0 = time.time()
    emit("+product24_morepairs", pd.concat([Btr, prod24_tr], axis=1), pd.concat([Bte, prod24_te], axis=1), t0)
    t0 = time.time()
    emit("+product12+noise_cap", pd.concat([Btr, prod12_tr, noise_tr], axis=1), pd.concat([Bte, prod12_te, noise_te], axis=1), t0)
    t0 = time.time()
    emit("+ALLrich", pd.concat([Btr, rich12_tr], axis=1), pd.concat([Bte, rich12_te], axis=1), t0)
    t0 = time.time()
    emit("+product12+ALLrich", pd.concat([Btr, prod12_tr, rich12_tr], axis=1), pd.concat([Bte, prod12_te, rich12_te], axis=1), t0)
    _checkpoint(f"[ckpt] CONTROL {name} done")
    return rows


def main():
    allrows = []
    Xr, yr, rname = load_real()
    allrows += run_bed(rname, Xr, yr, seed=0)
    Xh, yh, _ = make_hard_dataset(n_samples=5000, seed=0)
    allrows += run_bed("hard_synth", Xh, yh, seed=0)

    df = pd.DataFrame(allrows)
    print("\n=================== CONTROL RESULTS ===================")
    print(df.to_string(index=False), flush=True)

    print("\n=================== CONTROL VERDICT ===================")
    for bed in df.bed.unique():
        for metric in ["auc_mean", "logit", "lgbm"]:
            b = df[df.bed == bed].set_index("variant")[metric]
            prod = float(b.get("+product12", float("nan")))
            cap = float(b.get("+product12+noise_cap", float("nan")))
            both = float(b.get("+product12+ALLrich", float("nan")))
            morepairs = float(b.get("+product24_morepairs", float("nan")))
            # does rich ADD over products, beyond what equal-count noise would?
            add_over_prod = both - prod
            add_over_cap = both - cap
            verdict = ("REAL-ADD" if add_over_cap > 0.005 and add_over_prod > 0.005
                       else "capacity-only" if add_over_prod > 0.005 else "no-add")
            print(f"  [{bed}/{metric}] product={prod:.4f} +noise_cap={cap:.4f} +morepairs={morepairs:.4f} "
                  f"product+ALLrich={both:.4f}  | d(vs prod)={add_over_prod:+.4f} d(vs noise_cap)={add_over_cap:+.4f} -> {verdict}")


if __name__ == "__main__":
    main()
