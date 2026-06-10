"""Regression sensors for the 2026-06-10 composite discovery-cache audit.

- S1: data_signature hashed PyObject* pointer bytes for string/datetime/
  categorical columns, so the signature differed across processes (and even
  within one when strings re-materialised) -> the discovery cache NEVER hit on
  real frames.
- P8/S2: _row_order_fingerprint ran hash_rows() over the ENTIRE polars frame to
  keep only the first 256 hashes; slice-first is digest-identical and bounded.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.cache import data_signature, _row_order_fingerprint

pl = pytest.importorskip("polars")


def _string_frame_pd(seed=0):
    rng = np.random.default_rng(seed)
    n = 500
    cats = np.array(["alpha", "beta", "gamma", "delta"])
    return pd.DataFrame({
        "y": rng.normal(size=n),
        "city": cats[rng.integers(0, 4, n)],
        "ts": pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), unit="D"),
    })


class TestSignatureDeterministicAcrossProcesses:
    def test_string_signature_stable_in_process(self) -> None:
        """Re-materialising the string column must yield the same signature."""
        df1 = _string_frame_pd()
        df2 = _string_frame_pd()  # identical content, fresh string objects
        s1 = data_signature(df1, "y", ["city", "ts"])
        s2 = data_signature(df2, "y", ["city", "ts"])
        assert s1 == s2, "string/datetime signature not content-stable"

    def test_string_signature_stable_across_subprocess(self) -> None:
        """The signature must be identical in a FRESH interpreter (the pointer
        bug only manifests across process boundaries / hash randomisation)."""
        code = (
            "import numpy as np, pandas as pd;"
            "from mlframe.training.composite.cache import data_signature;"
            "rng=np.random.default_rng(0); n=500;"
            "cats=np.array(['alpha','beta','gamma','delta']);"
            "df=pd.DataFrame({'y':rng.normal(size=n),'city':cats[rng.integers(0,4,n)],"
            "'ts':pd.to_datetime('2020-01-01')+pd.to_timedelta(rng.integers(0,1000,n),unit='D')});"
            "print(data_signature(df,'y',['city','ts']))"
        )
        local = data_signature(_string_frame_pd(), "y", ["city", "ts"])
        # Different PYTHONHASHSEED in a full env copy: str.__hash__ randomisation
        # must NOT affect the content-based blake2b signature.
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            env={**os.environ, "PYTHONHASHSEED": "12345"}, timeout=120,
        )
        assert out.returncode == 0, out.stderr
        remote = out.stdout.strip().splitlines()[-1]
        assert local == remote, (
            f"signature differs across processes: {local} != {remote}"
        )

    def test_string_signature_changes_with_content(self) -> None:
        df1 = _string_frame_pd(seed=0)
        df2 = _string_frame_pd(seed=1)
        assert data_signature(df1, "y", ["city", "ts"]) != data_signature(
            df2, "y", ["city", "ts"]
        )

    def test_polars_string_signature_stable(self) -> None:
        rng = np.random.default_rng(2)
        n = 600
        cats = np.array(["x", "yy", "zzz", "w"])
        d = {"y": rng.normal(size=n), "g": cats[rng.integers(0, 4, n)]}
        df1 = pl.DataFrame(d)
        df2 = pl.DataFrame(d)
        assert data_signature(df1, "y", ["g"]) == data_signature(df2, "y", ["g"])


class TestRowOrderFingerprintBounded:
    def test_slice_first_digest_identical(self) -> None:
        """P8: slicing the prefix BEFORE hash_rows must give the same digest as
        hashing the whole frame then slicing (hash_rows is row-local)."""
        rng = np.random.default_rng(3)
        n = 5000
        df = pl.DataFrame({"a": rng.normal(size=n), "b": rng.integers(0, 100, n)})
        # Current (fixed) implementation.
        got = _row_order_fingerprint(df)
        # Reference: hash whole frame, slice after (the slow pre-fix shape).
        ref_hashes = df.hash_rows().slice(0, min(df.height, 256)).to_numpy()
        import hashlib
        ref = hashlib.blake2b(
            np.ascontiguousarray(ref_hashes).tobytes(), digest_size=8
        ).hexdigest()
        assert got == ref, "slice-first fingerprint diverged from whole-frame"

    def test_prefix_reorder_bursts_fingerprint(self) -> None:
        rng = np.random.default_rng(4)
        n = 2000
        df = pl.DataFrame({"a": np.arange(n).astype(float)})
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=7)
        assert _row_order_fingerprint(df) != _row_order_fingerprint(shuffled)
