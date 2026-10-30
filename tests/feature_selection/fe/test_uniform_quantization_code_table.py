"""``uniform`` quantization reserves a code for missing values, so a candidate column's codes reach ``quantization_nbins``; the table handed to ``mi_direct`` must hold them.

A table sized ``quantization_nbins`` was one entry short: the contingency kernel read past it, which crashed the process in the pair search of a 20 000 x 30 frame and, where the error was
caught (the operand marginal of the pair gate), closed the gate silently with ``+inf``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_code_bins import code_table_bins


def _mixed_frame(n: int, p: int):
    """Numeric columns, small-integer columns, a group column, a categorical and two columns with missing values; the target is an interaction plus a ratio, a sine and a modular term. These are the
    shapes whose candidate columns carry a missing-value code, in the draw order that reproduced the defect."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p), dtype=np.float32)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)], copy=False)
    for k in range(4):
        df[f"i{k}"] = rng.integers(0, 20, n).astype(np.int8)
    df["g"] = rng.integers(0, 12, n).astype(np.int8)
    df["c0"] = pd.Categorical(rng.integers(0, 15, n).astype(str))
    for k in range(2):
        col = rng.standard_normal(n).astype(np.float32)
        col[rng.random(n) < 0.1] = np.nan
        df[f"m{k}"] = col
    x0, x1, x2, x3, x4 = (df[f"x{i}"].to_numpy() for i in range(5))
    logit = 1.2 * x0 * x1 + 0.8 * x2 / (1.5 + np.abs(x3)) + 0.9 * np.sin(1.7 * x4) + 1.0 * (((df["i0"] + df["i1"]) % 3) == 0) + 0.2 * (df["g"] - 6)
    y = (logit + rng.normal(0, 0.8, n) > np.median(logit)).astype(np.int8)
    return df, y


def test_table_holds_the_reserved_missing_code():
    """Codes 0..nbins (the reserved slot) need nbins + 1 entries; codes within 0..nbins-1 keep nbins; an empty column keeps nbins."""
    assert code_table_bins(np.array([0, 3, 10]), 10) == 11
    assert code_table_bins(np.array([0, 3, 9]), 10) == 10
    assert code_table_bins(np.array([], dtype=np.int32), 10) == 10


_CHILD = """
import inspect, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, sys.argv[1])
from tests.feature_selection.fe.test_uniform_quantization_code_table import _mixed_frame
from mlframe.feature_selection.filters.mrmr import MRMR
df, y = _mixed_frame(4000, 8)
off = {f: False for f in inspect.signature(MRMR.__init__).parameters if f.startswith("fe_") and f.endswith("_enable")}
m = MRMR(verbose=0, random_seed=0, n_jobs=1, nbins_strategy=None, quantization_method="uniform", **off).fit(df, y)
print("FIT_DONE", len(m.get_feature_names_out()))
"""


@pytest.mark.timeout(1800)
def test_uniform_fit_reads_nothing_beyond_its_code_tables(tmp_path):
    """A ``uniform`` fit completes under numba's bounds checking and without the operand-marginal failure warning.

    Run in a child process with ``NUMBA_BOUNDSCHECK=1`` (and a cache of its own, since checked code is compiled separately): an out-of-bounds read raises nothing in a normal run, it returns garbage
    counts, so the defect shows only as an ``IndexError`` here. The child's log carries the warning the pair gate emits when the operand marginal raised."""
    root = Path(__file__).resolve().parents[3]
    env = {**os.environ, "NUMBA_BOUNDSCHECK": "1", "NUMBA_CACHE_DIR": str(tmp_path / "numba"), "CUDA_VISIBLE_DEVICES": "-1"}
    run = subprocess.run([sys.executable, "-c", _CHILD, str(root)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=1700)
    out = run.stdout + run.stderr
    relevant = "\n".join(ln for ln in out.splitlines() if "cupy" not in ln and "CUDA" not in ln)
    assert run.returncode == 0 and "FIT_DONE" in run.stdout, relevant[-2000:]
    assert "marginal-MI computation failed" not in out, [ln for ln in out.splitlines() if "marginal-MI" in ln][:3]
