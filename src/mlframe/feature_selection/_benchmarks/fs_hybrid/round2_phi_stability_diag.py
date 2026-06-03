"""Round-2 R2s-6 diagnostic: is per-feature cross-fold importance-stability a usable DROP-PROTECTION signal, or do
redundant copies have the SAME stability as their originals (the flagged risk)?

If a redundant copy of inf_i is as cross-fold-FI-stable as inf_i itself, then "protect the most cross-fold-stable
member of a cluster from being dropped" cannot prefer the genuine feature over its copy -> the idea is refuted.
Measured on the manyredundant scenario (red_i_* are near-duplicates of inf_i): per-feature 5-fold RF impurity-FI
mean + std (low std = stable). Compares each informative base's stability to its redundant copies' stability.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenarios import make


def main():
    X, y, t = make("manyredundant", 0)
    cols = list(X.columns)
    # 5-fold per-feature impurity FI
    fis = []
    for tr, _ in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0).fit(X.iloc[tr], y.iloc[tr])
        fis.append(m.feature_importances_)
    F = np.vstack(fis)
    mean = F.mean(0); std = F.std(0)
    stab = pd.DataFrame({"feature": cols, "fi_mean": mean.round(5), "fi_std": std.round(5),
                         "cv_stability": (1 - std / (mean + 1e-9)).round(3)}).sort_values("fi_mean", ascending=False)
    # group: informative bases vs their redundant copies. manyredundant naming: inf_i and red_i_j (parent i).
    base = set(t["base"])
    print("=== top-20 by FI (feature, fi_mean, fi_std, cv_stability) ===")
    print(stab.head(20).to_string(index=False))
    # for each base inf_i with redundant copies red_i_*, compare stability
    print("\n=== base vs its redundant copies (R2s-6 discriminator) ===")
    refuted_cases = 0; total = 0
    for b in sorted(base):
        if not b.startswith("inf_"):
            continue
        i = b.split("_")[1]
        copies = [c for c in cols if c.startswith(f"red_{i}_") or (c.startswith("red_") and c.split("_")[1] == i)]
        if not copies:
            continue
        total += 1
        bstab = float(stab.loc[stab.feature == b, "cv_stability"].iloc[0])
        cstabs = [float(stab.loc[stab.feature == c, "cv_stability"].iloc[0]) for c in copies]
        # "copies also stable" -> the best copy's stability is within ~10% of the base's
        best_copy = max(cstabs)
        close = abs(best_copy - bstab) < 0.10 or best_copy >= bstab
        refuted_cases += int(close)
        print(f"  {b}: cv_stability={bstab:.3f} | {len(copies)} copies best_copy_stability={best_copy:.3f} "
              f"-> {'COPY ~= ORIGINAL (cannot distinguish)' if close else 'original clearly more stable'}")
    print(f"\nR2s-6 verdict: copies indistinguishable-from-original in {refuted_cases}/{total} base clusters -> "
          + ("REFUTED (stability cannot protect the original over its copy)" if refuted_cases >= total / 2 else "stability MAY help"))


if __name__ == "__main__":
    main()
