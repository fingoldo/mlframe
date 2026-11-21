"""Meta-test: no builtin ``hash()`` feeds a random seed anywhere in the mlframe package.

``hash(str)`` is salted per process, so a seed built from it changes on every run and a benchmark result cannot be
reproduced in another interpreter (audit 01-F2, 40-F12). The check is the shared ``salted_hash_seed`` scanner from
py-ci-shared; an int-tuple seed that is genuinely stable is exempted on its line with a reasoned ``audit-exempt`` marker.
"""

from __future__ import annotations

from pathlib import Path

import mlframe
from py_ci_shared.code_audit import scan_salted_hash_seeds

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent


def test_no_builtin_hash_feeds_a_random_seed():
    """The scanner reports nothing across the package; a finding names the file and line to fix."""
    findings = scan_salted_hash_seeds(MLFRAME_DIR)
    assert not findings, "salted hash() feeds a seed (use zlib.crc32 or exempt an int-only use):\n" + "\n".join(
        f"  {f.file}:{f.line}  {f.snippet}" for f in findings
    )
