"""Regression test for CON17: ``elect_all`` / ``rank_all`` iterated ``set(ELECTION_METHODS)`` / ``set(RANKING_METHODS)`` to build the result rows, making the row/column
ORDER depend on PYTHONHASHSEED. The fix iterates the declared list order (and filters order-preservingly). This test runs the pipeline under several PYTHONHASHSEED values in
subprocesses and asserts the produced order is identical.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


_SRC = str(Path(__file__).resolve().parents[2] / "src")

_SNIPPET = r"""
import sys
sys.path.insert(0, {src!r})
import pandas as pd
from mlframe.votenrank.leaderboard.leaderboard_impl import Leaderboard
t = pd.DataFrame(
    {{"taskA": [0.9, 0.5, 0.2], "taskB": [0.8, 0.4, 0.6], "taskC": [0.3, 0.7, 0.5]}},
    index=["m1", "m2", "m3"],
)
lb = Leaderboard(table=t)
elect = list(lb.elect_all()["method"])
rank = list(lb.rank_all().columns)
print("|".join(elect))
print("|".join(rank))
"""


def _run(hashseed: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    env["CUDA_VISIBLE_DEVICES"] = ""
    out = subprocess.run(
        [sys.executable, "-c", _SNIPPET.format(src=_SRC)],
        env=env,
        capture_output=True,
        text=True,
        timeout=80,
    )
    assert out.returncode == 0, out.stderr
    # Take the last two non-empty lines (avoid warning noise on earlier lines).
    lines = [l for l in out.stdout.splitlines() if "|" in l]
    return "\n".join(lines[-2:])


def test_con17_method_order_is_pythonhashseed_independent():
    outputs = {_run(hs) for hs in ("0", "1", "12345", "98765")}
    assert len(outputs) == 1, f"method/row order differs across PYTHONHASHSEED:\n{outputs}"
