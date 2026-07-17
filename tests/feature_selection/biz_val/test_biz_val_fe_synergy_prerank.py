"""BIZ_VALUE: the wide-frame interaction-propensity pre-rank recovers near-zero-marginal interaction
operands on a frame WIDER than fe_synergy_screen_max_features, where the legacy skip-past-cap missed them.

Construction (the isolating regime, probed 2026-06-19): n=8000, p=300 > cap 250, almost all pure noise;
TWO planted pure sign-product pair interactions whose operands carry only a FAINT main-effect leak
(L=0.05) -- weak enough that marginal screening alone misses most operands (so the legacy skip arm engineers
nothing and recovers ~1 operand), but with enough higher-moment leakage that the second-moment pre-rank
ranks the operands into the swept top-250 and the all-pairs joint-MI sweep + raw-retention recover them.
This is the realistic L>=0.05 band, NOT the irreducible perfectly-balanced L=0 case (pinned separately in
test_fe_interaction_prerank). Multi-seed, fixed train/test split.

Measured (seed 0): pre-rank ON recovers 4/4 operands; OFF recovers 1/4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from mlframe.feature_selection.filters.mrmr._mrmr_class import MRMR

N = 8000
P = 300  # > fe_synergy_screen_max_features default 250
LEAK = 0.05
OPERANDS = (5, 60, 180, 240)  # two pairs: (5,180) and (60,240)


pytestmark = pytest.mark.timeout(60)  # untimed biz_val real-fit tier: surface a hang fast (global --timeout=600 is a coarse backstop)


def _make_wide_interaction(seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, P))
    ia, ic, ib, idd = OPERANDS
    a, b, c, d = X[:, ia], X[:, ib], X[:, ic], X[:, idd]
    logit = 2.6 * np.sign(a) * np.sign(b) + 2.6 * np.sign(c) * np.sign(d) + LEAK * 2.6 * (a + b + c + d)
    y = (rng.random(N) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    cols = [f"f{i}" for i in range(P)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y"), {f"f{i}" for i in OPERANDS}


def _fit(prerank, Xtr, ytr):
    # fe_synergy_exhaustive="never" isolates the lever under test. The default "auto" mode now escalates to the
    # FULL C(p,2) exhaustive sweep whenever it is affordable (no time budget set -> always), which runs over ALL
    # raw columns regardless of fe_synergy_prerank -- so ON and OFF become identical and the knob looks like a
    # no-op. Pinning "never" forces the pre-rank-vs-legacy-skip path this test exists to exercise (the production
    # regime where the exhaustive sweep is declined as too expensive on a very wide frame).
    m = MRMR(fe_synergy_prerank=prerank, fe_synergy_screen_max_features=250, fe_synergy_exhaustive="never")
    m.fit(Xtr, ytr)
    return m


def _auc(m, Xtr, ytr, Xte, yte):
    Ztr, Zte = m.transform(Xtr), m.transform(Xte)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Ztr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Zte)[:, 1]))


def test_prerank_recovers_more_wide_operands_than_legacy_skip():
    """Aggregate across seeds (selection recovery is the lever the knob controls; fit on the full frame --
    the regime the probe verified robust -- so the claim is not at the split-size margin)."""
    total_on = total_off = 0
    per_seed = []
    for seed in (0, 1, 2):
        X, y, operands = _make_wide_interaction(seed)
        m_on, m_off = _fit(True, X, y), _fit(False, X, y)
        rec_on = len(operands & set(m_on.get_feature_names_out()))
        rec_off = len(operands & set(m_off.get_feature_names_out()))
        total_on += rec_on
        total_off += rec_off
        per_seed.append((seed, rec_on, rec_off))
    # The pre-rank recovers strictly more near-zero-marginal interaction operands than the legacy skip-past-
    # cap (which never runs the synergy sweep on a p>cap frame), and recovers a real fraction on average.
    assert total_on > total_off, f"pre-rank did not beat legacy skip across seeds: {per_seed}"
    assert total_on / 3 >= 2.0, f"pre-rank mean recovery too low: {per_seed}"


def test_prerank_does_not_regress_heldout_auc():
    """Recovering genuine interaction signal must not hurt held-out AUC (single seeded split)."""
    seed = 1
    X, y, _ = _make_wide_interaction(seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    m_on, m_off = _fit(True, Xtr, ytr), _fit(False, Xtr, ytr)
    auc_on = _auc(m_on, Xtr, ytr, Xte, yte)
    auc_off = _auc(m_off, Xtr, ytr, Xte, yte)
    assert auc_on >= auc_off - 0.01, f"pre-rank regressed AUC: on={auc_on:.4f} off={auc_off:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--no-cov"])
