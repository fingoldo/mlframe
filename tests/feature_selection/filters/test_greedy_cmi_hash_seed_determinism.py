"""The CMI-greedy constructor scans its candidates in the same order in every process: equal-CMI candidates must not be ordered by the string hash seed."""

import os
import subprocess
import sys

_SCRIPT = """
import os, warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from mlframe.feature_selection.filters import _mi_greedy_cmi_fe_steps as S
from mlframe.feature_selection.filters._mi_greedy_cmi_fe import greedy_cmi_fe_construct
scans = []
real = S._per_candidate_cmi_argmax
def recording(st, scan, *a, **k):
    scans.append(list(scan))
    return real(st, scan, *a, **k)
S._per_candidate_cmi_argmax = recording
rng = np.random.default_rng(0)
n = 4000
X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(4)})
y = ((X["x1"].to_numpy() ** 2 + 0.3 * rng.normal(size=n)) > 1).astype(int)
X_aug, scores = greedy_cmi_fe_construct(X, y, top_k=4)
picked = list(scores.index) if hasattr(scores, "index") else list(scores)
print("PICKED", picked, "FIRST_SCAN", scans[0] if scans else None)
"""


def _picked(seed: str) -> str:
    """The picked-column line printed by a fresh interpreter started with the given ``PYTHONHASHSEED``."""
    env = dict(os.environ, PYTHONHASHSEED=seed)
    out = subprocess.run([sys.executable, "-c", _SCRIPT], env=env, capture_output=True, text=True, check=True, timeout=600).stdout
    return next(line for line in out.splitlines() if line.startswith("PICKED"))


def test_same_picks_under_different_hash_seeds():
    """Six interpreters with different hash seeds scan the same candidates in the same order and pick the same columns."""
    picks = {_picked(str(seed)) for seed in (0, 1, 2, 3, 4, 5)}
    assert len(picks) == 1, picks
