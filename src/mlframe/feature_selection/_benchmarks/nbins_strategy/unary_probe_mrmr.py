"""What does MRMR do with a known unary relation? Four data sets, one MRMR fit each, and a ridge model on the raw columns against the same ridge on MRMR's output.

The cases are ``y = sin(x1) + x2`` with x1 on [-6, 6] (non-monotone, two periods), the same with x1 on [-1, 1] (sin is nearly linear: nothing should be added), ``y = log(x1) + x2`` with a
lognormal x1, and ``y = x1**2 + x2``; every case has six noise columns and noise of standard deviation 0.3. The ridge score is the held-out R2 of 40 percent of the rows; the ceiling is the
R2 of the generating terms (0.943, 0.932, 0.957 and 0.971). Usage: ``python unary_probe_mrmr.py`` (CPU only; one fit takes 8 to 370 s). Result of 2026-11-24 in
``audits/2026-11-23/research_mrmr_unary_probe.md``.
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

CASES = ("sin_wide", "sin_narrow", "log_lognormal", "square")


def make(case: str, n: int = 6000, seed: int = 0) -> Tuple[pd.DataFrame, pd.Series]:
    """The frame (x1, x2, six noise columns) and the target of one case."""
    rng = np.random.default_rng(seed)
    x2 = rng.standard_normal(n)
    noise = 0.3 * rng.standard_normal(n)
    if case == "sin_wide":
        x1 = rng.uniform(-6, 6, n)
        y = np.sin(x1) + x2 + noise
    elif case == "sin_narrow":
        x1 = rng.uniform(-1, 1, n)
        y = np.sin(x1) + x2 + noise
    elif case == "log_lognormal":
        x1 = rng.lognormal(0, 1, n)
        y = np.log(x1) + x2 + noise
    elif case == "square":
        x1 = rng.standard_normal(n)
        y = x1**2 + x2 + noise
    else:
        raise ValueError(case)
    X = pd.DataFrame({"x1": x1, "x2": x2, **{f"n{k}": rng.standard_normal(n) for k in range(6)}})
    return X, pd.Series(y, name="y")


def ridge_r2(x_tr, x_te, y_tr, y_te) -> float:
    """Held-out R2 of a ridge model."""
    return float(RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(x_tr, y_tr).score(x_te, y_te))


def main() -> None:
    """Fit MRMR on every case and print the output columns and the ridge R2 before and after."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    for case in CASES:
        X, y = make(case)
        x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=1)
        base = ridge_r2(x_tr[["x1", "x2"]], x_te[["x1", "x2"]], y_tr, y_te)
        t0 = time.time()
        model = MRMR(verbose=0, random_state=0, n_jobs=1, fe_max_steps=1).fit(x_tr, y_tr)
        names = list(model.get_feature_names_out())
        z_tr = pd.DataFrame(model.transform(x_tr)).reset_index(drop=True).fillna(0)
        z_te = pd.DataFrame(model.transform(x_te)).reset_index(drop=True).fillna(0)
        r2 = ridge_r2(z_tr, z_te, y_tr.values, y_te.values)
        print(f"CASE {case}: ridge R2 raw x1,x2 = {base:.3f}; MRMR output {names}; ridge R2 on the output = {r2:.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
