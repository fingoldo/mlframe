"""Bayes shrinkage: Micci-Barreca classical (per-cat within) vs sklearn (pooled within).

Both implementations in pure Python so we can compare formulas without rebuilding Rust.

formula = lambda * cat_mean + (1 - lambda) * global_mean
classical:  lambda_i = var_y * n_i / (var_y * n_i + (ssd_i / n_i))    # per-category within
sklearn:    lambda_i = var_y * n_i / (var_y * n_i + pooled_within)     # pooled across all cats
            pooled_within = mean over all rows of (y - cat_mean(y))^2

Run: D:/ProgramData/anaconda3/python.exe bench_bayes_formula.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import make_high_card_cat, train_test_split_frame


def encode_bayes_classical(values: np.ndarray, target: np.ndarray, global_mean: float, global_var: float) -> dict[str, float]:
    """Per-category within-variance shrinkage."""
    cats = {}
    for v, t in zip(values, target):
        if v not in cats:
            cats[v] = []
        cats[v].append(t)
    out = {}
    for k, ts in cats.items():
        n = len(ts)
        if n <= 1:
            out[k] = global_mean
            continue
        mean = sum(ts) / n
        ssd = sum((t - mean) ** 2 for t in ts)
        within_var = ssd / n
        denom = global_var * n + within_var
        if denom == 0:
            out[k] = global_mean
        else:
            lam = global_var * n / denom
            out[k] = lam * mean + (1 - lam) * global_mean
    return out


def encode_bayes_pooled(values: np.ndarray, target: np.ndarray, global_mean: float, global_var: float) -> dict[str, float]:
    """sklearn-style: pooled within-variance across all categories."""
    cats: dict = {}
    for v, t in zip(values, target):
        cats.setdefault(v, []).append(t)
    # First compute per-cat means
    cat_means = {k: sum(ts) / len(ts) for k, ts in cats.items()}
    # Pooled within: mean of (y - cat_mean)^2 over all rows
    sum_sq = 0.0
    for v, t in zip(values, target):
        diff = t - cat_means[v]
        sum_sq += diff * diff
    pooled_within = sum_sq / len(values)
    out = {}
    for k, ts in cats.items():
        n = len(ts)
        if n <= 1:
            out[k] = global_mean
            continue
        denom = global_var * n + pooled_within
        if denom == 0:
            out[k] = global_mean
        else:
            lam = global_var * n / denom
            out[k] = lam * cat_means[k] + (1 - lam) * global_mean
    return out


def oof_encode(df: pl.DataFrame, cat_cols: list[str], target: str, cv: int, encoder_fn, seed: int = 42):
    """OOF encoding using given Bayes encoder function."""
    n = len(df)
    out_data = {c: np.full(n, np.nan) for c in cat_cols}
    y = df[target].to_numpy()
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    for train_idx, val_idx in skf.split(np.zeros(n), y):
        y_tr = y[train_idx]
        global_mean = float(y_tr.mean())
        global_var = float(y_tr.var())
        for c in cat_cols:
            vals_tr = df[c].to_numpy()[train_idx]
            vals_val = df[c].to_numpy()[val_idx]
            mapping = encoder_fn(vals_tr, y_tr, global_mean, global_var)
            for j, idx in enumerate(val_idx):
                out_data[c][idx] = mapping.get(vals_val[j], global_mean)
    new_df = df.clone()
    for c in cat_cols:
        new_df = new_df.with_columns(pl.Series(c + "_enc", out_data[c]))
    return new_df


def full_encode(df_train: pl.DataFrame, df_test: pl.DataFrame, cat_cols: list[str], target: str, encoder_fn):
    """Full-train mapping applied to test."""
    y = df_train[target].to_numpy()
    global_mean = float(y.mean())
    global_var = float(y.var())
    out_test = {}
    for c in cat_cols:
        vals_tr = df_train[c].to_numpy()
        vals_te = df_test[c].to_numpy()
        mapping = encoder_fn(vals_tr, y, global_mean, global_var)
        out_test[c + "_enc"] = np.array([mapping.get(v, global_mean) for v in vals_te])
    test_enc = df_test.clone()
    for c in cat_cols:
        test_enc = test_enc.with_columns(pl.Series(c + "_enc", out_test[c + "_enc"]))
    return test_enc


def evaluate(formula_name: str, encoder_fn, tr: pl.DataFrame, te: pl.DataFrame,
             cat_cols: list[str], num_cols: list[str], cv: int = 5):
    t0 = time.perf_counter()
    tr_enc = oof_encode(tr, cat_cols, "y", cv, encoder_fn)
    te_enc = full_encode(tr, te, cat_cols, "y", encoder_fn)
    elapsed = time.perf_counter() - t0
    enc_cols = [c + "_enc" for c in cat_cols]
    X_tr = np.hstack([tr_enc[c].to_numpy().reshape(-1, 1) for c in enc_cols + num_cols])
    X_te = np.hstack([te_enc[c].to_numpy().reshape(-1, 1) for c in enc_cols + num_cols])
    X_tr = np.nan_to_num(X_tr, nan=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0)
    y_tr = tr["y"].to_numpy()
    y_te = te["y"].to_numpy()
    m = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
    return {
        "train_auc": float(roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1])),
        "test_auc": float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])),
        "elapsed_s": elapsed,
    }


def main():
    print("=" * 80)
    print("Bayes shrinkage formula comparison: classical vs pooled (sklearn-style)")
    print("=" * 80)
    n_repeats = 5
    sizes = [(20000, 500), (50000, 1000)]
    all_results = []
    for n, card in sizes:
        print(f"\n--- n={n}, cardinality={card}, {n_repeats} repeats ---")
        print(f"{'formula':<25} {'train AUC':>12} {'test AUC':>11} {'gap':>9} {'time s':>8}")
        for formula, fn in [("classical_per_cat", encode_bayes_classical),
                            ("pooled_sklearn", encode_bayes_pooled)]:
            train_aucs, test_aucs, times = [], [], []
            for seed in range(n_repeats):
                df = make_high_card_cat(n=n, n_cat_cols=3, cardinality=card,
                                         signal_strength=0.3, seed=seed)
                tr, te = train_test_split_frame(df, frac=0.7, seed=seed)
                cat_cols = [c for c in tr.columns if c.startswith("c")]
                num_cols = [c for c in tr.columns if c.startswith("n")]
                r = evaluate(formula, fn, tr, te, cat_cols, num_cols)
                train_aucs.append(r["train_auc"])
                test_aucs.append(r["test_auc"])
                times.append(r["elapsed_s"])
            train_med = float(np.median(train_aucs))
            test_med = float(np.median(test_aucs))
            time_med = float(np.median(times))
            gap = train_med - test_med
            print(f"{formula:<25} {train_med:>12.4f} {test_med:>11.4f} {gap:>9.4f} {time_med:>8.2f}")
            all_results.append({
                "n": n, "card": card, "formula": formula,
                "train_med": train_med, "test_med": test_med, "gap": gap, "time_med": time_med,
                "test_std": float(np.std(test_aucs)),
            })
    print("=" * 80)
    print("\nVerdict (higher test AUC + smaller gap = better):")
    for n, card in sizes:
        rows = [r for r in all_results if r["n"] == n and r["card"] == card]
        winner = max(rows, key=lambda x: x["test_med"])
        print(f"  n={n}, card={card}: {winner['formula']} "
              f"(test={winner['test_med']:.4f}, gap={winner['gap']:.4f})")


if __name__ == "__main__":
    main()
