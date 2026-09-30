"""An MRMR fit's INFO stream keeps stage-level lines only: per-step / per-candidate chatter lives at DEBUG."""
import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import MRMR

_CHATTER = (
    "categorizing dataset",
    "categorized.",
    "bootstrapped eval took",
    "conditioning joint undersampled",
    "Converted targets from",
    "MRMR+ selected 2 out of",  # duplicates the final "MRMR: selected K of N" summary
)


def test_mrmr_fit_info_stream_has_no_per_step_chatter_but_keeps_final_summary(caplog):
    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = ((X.f0 + X.f1 + rng.normal(size=n) * 0.5) > 0).astype(int)
    caplog.set_level(logging.INFO)
    MRMR(verbose=1, random_seed=0, full_npermutations=3, min_features_fallback=1, fe_max_steps=0).fit(X, y)
    info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    noisy = [m for m in info if any(c in m for c in _CHATTER) or (m.startswith("MRMR+ selected") and "before the Feature" not in m)]
    assert noisy == []
    assert any(m.startswith("MRMR: selected") for m in info), info
