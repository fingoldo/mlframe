"""A fit with a fixed ``random_seed`` selects the same columns whatever ``PYTHONHASHSEED`` the process started with.

Python salts the hash of a string per process, so a set of column names iterates in a different order in every process. The held-out residual prune of the selection (``redundancy_policy='drop'``, the default)
builds its design from the selected names and eliminates columns one at a time, with every raw column tied in the weakest-first order, so the order of the design decides which of several redundant copies
survives. It was built by iterating a set of names: the same fit on 50 noisy copies of each of two latents kept 20, 22, 24, 26 or 28 columns in different processes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import mlframe

_SCRIPT = """
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from mlframe.feature_selection.filters.mrmr import MRMR

rng = np.random.default_rng(101)
n, k = 1500, 50
z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
w1, w2 = rng.standard_normal((1, k)) * 0.5, rng.standard_normal((1, k)) * 0.5
X = np.hstack([z1[:, None] * w1 + 0.05 * rng.standard_normal((n, k)), z2[:, None] * w2 + 0.05 * rng.standard_normal((n, k))])
cols = [f"e1_{i}" for i in range(k)] + [f"e2_{i}" for i in range(k)]
y = 2.0 * z1 + z2 + 0.3 * rng.standard_normal(n)
sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0, random_seed=101, n_jobs=1)
sel.fit(pd.DataFrame(X, columns=cols), pd.Series(y, name="y"))
print("SUPPORT " + json.dumps(sorted(str(c) for c in sel.get_feature_names_out())))
"""


def _support_under_hash_seed(hash_seed: int) -> list:
    """Fit the fixture in a fresh interpreter started with ``PYTHONHASHSEED=hash_seed`` and return the sorted names it selected."""
    src = str(Path(mlframe.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONHASHSEED": str(hash_seed), "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    done = subprocess.run([sys.executable, "-c", _SCRIPT], env=env, capture_output=True, text=True, timeout=900, check=False)
    assert done.returncode == 0, f"the fit failed under PYTHONHASHSEED={hash_seed}:\n{done.stderr[-2000:]}"
    line = next(row for row in done.stdout.splitlines() if row.startswith("SUPPORT "))
    return json.loads(line[len("SUPPORT ") :])


def test_support_does_not_depend_on_the_string_hash_seed():
    """Two processes with different string-hash salts select exactly the same columns from 100 near-duplicate columns."""
    first = _support_under_hash_seed(1)
    second = _support_under_hash_seed(2)
    assert first, "the fit selected nothing, so the comparison says nothing"
    assert first == second, f"the support depends on PYTHONHASHSEED: {len(first)} columns against {len(second)}; only in the first: {sorted(set(first) - set(second))}, only in the second: {sorted(set(second) - set(first))}"
