"""Why does hybrid not 'win' on scene? Discriminating test of the knn-parsimony-artifact
hypothesis: the bench's headline metric = mean(lgbm,logit,knn); knn AUC collapses with
feature count, so the mean rewards ultra-sparse selectors regardless of tree quality.

Test: take ONE fixed informative ranking on scene and sweep top-k; if knn AUC rises as k
shrinks while lgbm AUC is flat/falls, the 'hybrid loses' is a metric artifact, not a defect.
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.normpath(os.path.join(HERE, "..", "..", "src", "mlframe",
                                      "feature_selection", "_benchmarks", "fs_hybrid"))
sys.path.insert(0, BENCH)
import round4_broad_realdata_bench as B  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
import lightgbm as lgb  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402


def auc_by_model(Xtr, Xte, ytr, yte):
    o = {}
    o["lgbm"] = roc_auc_score(yte, lgb.LGBMClassifier(n_estimators=300, verbose=-1).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    o["logit"] = roc_auc_score(yte, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    o["knn"] = roc_auc_score(yte, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    return o


def main():
    X, y, note = B.load_one("scene", dict(name="scene", version=1), 3000, 1200)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    # one fixed informative ranking: LightGBM impurity on the full set
    imp = lgb.LGBMClassifier(n_estimators=300, verbose=-1).fit(Xtr, ytr).feature_importances_
    order = list(np.argsort(-imp))
    print(f"scene {X.shape}  note={note}")
    print(f"{'k':>5}{'lgbm':>9}{'logit':>9}{'knn':>9}{'mean':>9}")
    for k in [5, 9, 15, 30, 60, 90, 150, X.shape[1]]:
        cols = [X.columns[i] for i in order[:k]]
        o = auc_by_model(Xtr[cols], Xte[cols], ytr, yte)
        m = np.mean(list(o.values()))
        print(f"{k:>5}{o['lgbm']:>9.4f}{o['logit']:>9.4f}{o['knn']:>9.4f}{m:>9.4f}")


if __name__ == "__main__":
    main()
