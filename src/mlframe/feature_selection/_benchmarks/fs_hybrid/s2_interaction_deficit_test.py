"""S2 cheap falsifiable test (ShapProxiedFS interaction-deficit corrector idea).

Agent S2 hypothesis: subsets containing both operands of an interaction are UNDER-credited by the additive
SHAP proxy (base + sum main-effect |phi|), so honest_loss < proxy_loss for high within-subset interaction
mass -> corr(interaction_mass, honest-proxy) should be clearly NEGATIVE; then add interaction_mass as a
bias-corrector feature to recover synergy. Falsification gate: |corr| < 0.1 -> too weak to recover.

RESULT (this script, seed 0): corr is POSITIVE and non-trivial -- xor2 pearson +0.49 / spearman +0.51;
base +0.33 / +0.32. Wrong sign. TreeSHAP distributes an interaction's effect into the MAIN-effect values
of its operands, so the additive proxy already absorbs synergy and OVER-credits interaction-complete
subsets (honest_loss > proxy_loss). S2's "down-correct to recover synergy" premise is backwards for a
TreeSHAP proxy -> idea REJECTED (the cheap-test prevented implementing a wrong-premise corrector feature).
"""
from __future__ import annotations
import os, warnings
os.environ.setdefault("TQDM_DISABLE", "1"); warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.special import expit
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import brier_score_loss
import lightgbm as lgb, shap
from scenarios import make


def honest_brier(X, y, cols):
    if not cols:
        return 0.25
    p = cross_val_predict(lgb.LGBMClassifier(n_estimators=120, num_leaves=31, verbose=-1), X[cols], y, cv=3, method="predict_proba")[:, 1]
    return brier_score_loss(y, p)


def run(scenario, seed=0, n_anchors=45):
    X, y, t = make(scenario, seed)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    Xtr = Xtr.reset_index(drop=True); ytr = ytr.reset_index(drop=True).to_numpy()
    m = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, verbose=-1).fit(Xtr, ytr)
    expl = shap.TreeExplainer(m)
    phi = np.asarray(expl.shap_values(Xtr))
    if phi.ndim == 3:
        phi = phi[..., 1] if phi.shape[-1] == 2 else phi[..., 0]
    base = float(np.ravel(expl.expected_value)[-1])
    Phi = np.asarray(expl.shap_interaction_values(Xtr))
    if Phi.ndim == 4:
        Phi = Phi[..., 1] if Phi.shape[-1] == 2 else Phi[..., 0]
    cols = list(Xtr.columns); P = len(cols); rng = np.random.default_rng(seed)
    mass, bias = [], []
    for _ in range(n_anchors):
        k = int(rng.integers(2, 9)); idx = sorted(rng.choice(P, size=k, replace=False).tolist())
        names = [cols[i] for i in idx]
        proxy = brier_score_loss(ytr, np.clip(expit(base + phi[:, idx].sum(1)), 1e-6, 1 - 1e-6))
        honest = honest_brier(Xtr, ytr, names)
        im = float(np.mean([2 * np.abs(Phi[:, idx[a], idx[b]]).sum() for a in range(len(idx)) for b in range(a + 1, len(idx))]))
        mass.append(im)
        bias.append(honest - proxy)
    mass, bias = np.array(mass), np.array(bias)
    return pearsonr(mass, bias)[0], spearmanr(mass, bias)[0]


if __name__ == "__main__":
    for sc in ["xor2", "base"]:
        pr, sr = run(sc)
        print(f"{sc:6s} corr(interaction_mass, honest-proxy): pearson={pr:.3f} spearman={sr:.3f} "
              f"(NEGATIVE would support S2; measured POSITIVE -> rejected)")
