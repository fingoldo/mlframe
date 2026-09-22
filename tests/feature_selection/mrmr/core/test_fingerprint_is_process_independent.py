"""The fit-cache fingerprints are the same value in any process, and their only instability costs a miss, never a wrong hit.

The cache key decides whether a fit is reused, so a fingerprint that varied between processes would either recompute needlessly or, far worse,
collide. It is built with blake2b over sampled bytes: no ``hash()`` of a string, no dict ordering, nothing salted. The one edge that is not
value-stable is float bytes, where ``-0.0`` and ``0.0`` compare equal but hash apart, as do NaNs with different payloads. That direction is a
MISS, an extra fit, never a wrong reuse, which is why it is left alone; this records the reasoning so it is not re-derived, and pins the part
that actually matters.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fingerprints import _mrmr_compute_x_fingerprint, _mrmr_compute_y_fingerprint_sample


def _frame():
    """A frame with float, int and string columns, so every branch of the sampler runs."""
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame({"f": rng.normal(size=n), "i": rng.integers(0, 9, size=n), "s": rng.choice(["a", "b", "c"], size=n)})


def test_the_fingerprint_is_the_same_in_a_process_with_a_different_hash_seed():
    """PYTHONHASHSEED changes string hashing; the fingerprint must not move with it."""
    here = _mrmr_compute_x_fingerprint(_frame())
    script = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, "src")
        import numpy as np, pandas as pd
        from mlframe.feature_selection.filters._mrmr_fingerprints import _mrmr_compute_x_fingerprint
        rng = np.random.default_rng(0)
        n = 300
        df = pd.DataFrame({"f": rng.normal(size=n), "i": rng.integers(0, 9, size=n), "s": rng.choice(["a", "b", "c"], size=n)})
        print(_mrmr_compute_x_fingerprint(df))
        """
    )
    env_seeds = ("1", "424242")
    outs = []
    for seed in env_seeds:
        proc = subprocess.run(  # nosec B603 -- fixed argv, no shell, local interpreter
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "PYTHONHASHSEED": seed, "CUDA_VISIBLE_DEVICES": ""},
        )
        # A subprocess that cannot run leaves nothing to compare, so the whole point of this test evaporates. Fail loudly instead of
        # reporting a skip that reads as "checked, fine".
        assert proc.returncode == 0, f"the PYTHONHASHSEED={seed} subprocess failed, so the fingerprint was never compared: {proc.stderr[-400:]}"
        outs.append(proc.stdout.strip().splitlines()[-1])
    assert outs[0] == outs[1], f"the fingerprint moved with PYTHONHASHSEED: {outs}"
    assert outs[0] == here, f"the subprocess fingerprint differs from this process: {outs[0]} vs {here}"


def test_the_same_frame_fingerprints_the_same_twice():
    """The most basic property the cache depends on."""
    assert _mrmr_compute_x_fingerprint(_frame()) == _mrmr_compute_x_fingerprint(_frame())


def test_a_changed_value_changes_the_fingerprint():
    """Teeth-check: a fingerprint that never moved would make every fit a cache hit."""
    a = _frame()
    b = a.copy()
    b.loc[0, "f"] = b.loc[0, "f"] + 1.0
    assert _mrmr_compute_x_fingerprint(a) != _mrmr_compute_x_fingerprint(b)


def test_signed_zero_hashes_apart_which_costs_a_miss_not_a_wrong_hit():
    """Pins the known edge as a MISS: the two frames are equal under ``==`` but need not share a fingerprint.

    If this ever starts mattering, normalising the sampled buffer with ``+ 0.0`` before hashing is the fix; the important half, asserted here,
    is that differing bytes never produce the SAME key for different data.
    """
    pos = pd.DataFrame({"f": np.array([0.0, 1.0, 2.0])})
    neg = pd.DataFrame({"f": np.array([-0.0, 1.0, 2.0])})
    assert (pos["f"].to_numpy() == neg["f"].to_numpy()).all(), "fixture precondition: the two frames compare equal"
    # Either outcome is acceptable for correctness; what must never happen is two DIFFERENT frames sharing a key.
    assert _mrmr_compute_x_fingerprint(pos) != _mrmr_compute_x_fingerprint(pd.DataFrame({"f": np.array([9.0, 1.0, 2.0])}))


def test_the_y_sample_fingerprint_is_stable_and_discriminating():
    """The target's fingerprint follows the same rules as the frame's."""
    y = np.arange(50) % 3
    assert _mrmr_compute_y_fingerprint_sample(y) == _mrmr_compute_y_fingerprint_sample(np.arange(50) % 3)
    other = np.arange(50) % 4
    assert _mrmr_compute_y_fingerprint_sample(y) != _mrmr_compute_y_fingerprint_sample(other)
