"""The order-2 maxT floor must reject malformed inputs with an error, not crash the process (IMPL-1, found in mrmr_audit_2026-09-14).

``pooled_pair_permutation_null_joint_mi_floor`` hands its arrays straight to an njit kernel that indexes contingency tables by the codes.
Passing the natural-looking ``classes_y=[0, 1]`` (the distinct classes instead of one code per row), a scalar ``nbins``, or a pair index
past the last column made the kernel read out of bounds and SEGFAULT, taking the whole interpreter down with no Python traceback. Each
case below runs in a child process, so a regression shows up as a crash exit code instead of killing the test run itself.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from mlframe.feature_selection.filters._permutation_null import pooled_pair_permutation_null_joint_mi_floor

_SETUP = """
import numpy as np
from mlframe.feature_selection.filters._permutation_null import pooled_pair_permutation_null_joint_mi_floor as f
rng = np.random.default_rng(0)
n, p = 400, 5
data = rng.integers(0, 4, size=(n, p)).astype(np.int32)
nbins = np.full(p, 4, dtype=np.int32)
y = rng.integers(0, 2, size=n)
freqs = np.bincount(y).astype(np.float64) / n
pa = np.array([0, 1, 2], dtype=np.int64)
pb = np.array([1, 2, 3], dtype=np.int64)
"""

_CASES = {
    "distinct_classes_instead_of_row_codes": "f(data, nbins, pa, pb, np.array([0, 1]), freqs, n_permutations=5)",
    "scalar_nbins": "f(data, 4, pa, pb, y, freqs, n_permutations=5)",
    "pair_index_past_last_column": "f(data, nbins, np.array([0, 1, 7], dtype=np.int64), pb, y, freqs, n_permutations=5)",
    "short_nbins_vector": "f(data, nbins[:2], pa, pb, y, freqs, n_permutations=5)",
    "target_code_out_of_range": "f(data, nbins, pa, pb, np.where(np.arange(n) == 0, 9, y), freqs, n_permutations=5)",
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_malformed_input_raises_value_error_instead_of_crashing(case):
    """Each malformed call must end in a ValueError inside the child, not a native crash."""
    script = _SETUP + textwrap.dedent(
        f"""
        try:
            {_CASES[case]}
        except ValueError as exc:
            print("VALUE_ERROR", exc)
            raise SystemExit(0)
        print("NO_ERROR")
        raise SystemExit(3)
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert (
        proc.returncode == 0 and "VALUE_ERROR" in proc.stdout
    ), f"{case}: exit code {proc.returncode} (a negative or large code is a native crash); stdout={proc.stdout[-400:]!r} stderr={proc.stderr[-800:]!r}"


def test_well_formed_input_still_returns_a_floor():
    """Control: the validation must not reject the documented call shape."""
    rng = np.random.default_rng(0)
    n, p = 400, 5
    data = rng.integers(0, 4, size=(n, p)).astype(np.int32)
    nbins = np.full(p, 4, dtype=np.int32)
    y = rng.integers(0, 2, size=n)
    freqs = np.bincount(y).astype(np.float64) / n
    floor = pooled_pair_permutation_null_joint_mi_floor(data, nbins, np.array([0, 1, 2]), np.array([1, 2, 3]), y, freqs, n_permutations=5)
    assert isinstance(floor, float) and floor >= 0.0
