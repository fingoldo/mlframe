"""biz_value: usability-aware raw retention keeps linearly-usable raws whose engineered children are
LINEARLY LOSSY, so the FE selection beats raw-only downstream AND dropping those raws would crater a linear
model.

This legitimizes the behaviour surfaced by the F6_decoy create/keep/drop case (2026-07-02): a raw that
re-encodes a signal an engineered survivor captures only NONLINEARLY (ab_log=log(a*b+1) beside a fused
compound; g/k operands beside a**2/b) is NOT a droppable decoy -- the nonlinear child cannot hand a LINEAR
model that signal, so retaining the raw is a real, measurable win. The downstream no-harm Ridge guard in
_fe_raw_redundancy_drop already protects this at fit time; this test pins the OUTCOME value quantitatively.

Canonical case (from retain_usable_raw_columns' own docstring): y = a**2/b + g/k + log(c)*sin(d). The MI
greedy under-values g/k (low binned MI) but they carry genuine linear signal.
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from mlframe.feature_selection.filters.mrmr import MRMR

SEED = 42


def _make_ratio_plus_trig(n):
    rng = np.random.default_rng(SEED)
    a, b, c, d, g, k = (rng.uniform(0.2, 1.2, n) for _ in range(6))
    e = rng.uniform(0.2, 1.2, n)  # pure-noise column the selection must reject
    y = (a**2 / b) + (g / k) + np.log(c) * np.sin(d)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "g": g, "k": k, "e": e}), pd.Series(y, name="y")


def _heldout_r2(M, y):
    M = np.asarray(M, dtype=float)
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    Xtr, Xte, ytr, yte = train_test_split(M, np.asarray(y, dtype=float), test_size=0.3, random_state=SEED)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(Xtr, ytr)
    return float(model.score(Xte, yte))


def test_biz_value_usability_retention_keeps_linearly_usable_raws():
    """Measured (n=5000, 2026-07-02): full 0.842, eng-only 0.465, raw-only 0.767. Floors sit well below the
    measured values so seed noise does not trip them while a real regression (the retention dropping the
    linearly-usable raws, or FE turning harmful) does."""
    X, y = _make_ratio_plus_trig(5000)

    np.random.seed(SEED)
    fs = MRMR(verbose=0, random_seed=SEED, fe_max_steps=1)
    fs.fit(X, y)
    selected = list(fs.get_feature_names_out())
    Xt = np.asarray(fs.transform(X), dtype=float)
    Xt = pd.DataFrame(Xt, columns=selected)

    raw_cols = [c for c in selected if c in X.columns]
    eng_cols = [c for c in selected if c not in X.columns]

    r2_full = _heldout_r2(Xt.values, y)  # the actual MRMR selection (raws + engineered)
    r2_eng_only = _heldout_r2(Xt[eng_cols].values, y) if eng_cols else 0.0  # if the linearly-usable raws were dropped
    r2_raw_only = _heldout_r2(X.values, y)  # all raw features (the no-harm reference)

    # (1) the linearly-usable raws the greedy under-ranks are RETAINED (at least the g/k additive-term
    #     operands, which carry linear signal the a**2/b + trig children do not).
    assert raw_cols, "no raw features retained -- the linearly-usable raws were dropped"
    assert "g" in raw_cols or "k" in raw_cols, f"the g/k linear operands were not retained: {raw_cols}"

    # (2) FE is NOT harmful: the selection's downstream linear fit matches or beats raw-only.
    assert r2_full >= r2_raw_only - 0.01, f"FE harmful: selection R2 {r2_full:.4f} < raw-only {r2_raw_only:.4f}"

    # (3) the RETAINED raws carry large linear signal the engineered children do not: dropping them (eng-only)
    #     craters the downstream linear fit. Measured gap 0.38; floor 0.25.
    assert r2_full - r2_eng_only >= 0.25, (
        f"retained linearly-usable raws add too little (full {r2_full:.4f} vs eng-only {r2_eng_only:.4f}); the retention win is not materialising"
    )

    # (4) absolute quality floor (measured 0.842).
    assert r2_full >= 0.80, f"downstream R2 {r2_full:.4f} below the 0.80 floor"
