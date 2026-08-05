"""FE_TRANSFORMER_A-3 meta-test (2026-08-05 audit): repo-hygiene guard against the naive
``(n_query, n_anchors, d)`` broadcast-cube pairwise-squared-distance idiom recurring anywhere under
``feature_engineering/`` outside the shared GEMM helper (and benchmark scripts that legitimately need the
naive reference form to compare against).

``diffs = Xq[:, None, :] - anchors[None, :, :]`` followed by ``sq = (diffs**2).sum(axis=-1)`` materialises
a full ``(n_query, n_anchors, d)`` float32 temporary that is never even the return value -- only the final
``(n_query, n_anchors)`` result is kept. At this codebase's documented production scale (``d <= 32768``,
100+GB frames), that intermediate cube alone can reach multi-GB-to-100+GB. This exact idiom was found
independently reimplemented (instead of reusing ``_squared_dists_shared.squared_dists``, the GEMM-
decomposition drop-in already used by ``anchor_attention.py``/``class_conditional_anchor.py``) across 13
sibling files in one audit pass -- this test is the structural safety net so a future new call site (or a
regression back to the naive form) fails CI instead of silently reintroducing the OOM risk.
"""

from __future__ import annotations

import re
from pathlib import Path

import mlframe

_PATTERN = re.compile(r"^[ \t]*\w+[ \t]*=[ \t]*\w+\[:,\s*None,\s*:\][ \t]*-[ \t]*\w+\[None,\s*:,\s*:\]", re.MULTILINE)

# The canonical GEMM-decomposition helper itself legitimately documents/implements the naive form it replaces.
_ALLOWED_FILES = {"_squared_dists_shared.py"}


def test_no_naive_broadcast_cube_distance_idiom_outside_shared_helper():
    """No file under feature_engineering/ (outside the shared helper and _benchmarks/) may reimplement
    the naive broadcast-cube pairwise-distance formula -- route through
    feature_engineering.transformer._squared_dists_shared.squared_dists instead."""
    fe_root = Path(mlframe.__file__).parent / "feature_engineering"
    offenders = []
    for path in fe_root.rglob("*.py"):
        if path.name in _ALLOWED_FILES:
            continue
        if "_benchmarks" in path.parts:
            continue  # benchmark scripts legitimately need the naive reference form to compare against
        text = path.read_text(encoding="utf-8")
        if _PATTERN.search(text):
            offenders.append(str(path.relative_to(fe_root)))
    assert not offenders, (
        f"found the naive (n_query, n_anchors, d) broadcast-cube distance idiom (real OOM risk at this "
        f"codebase's documented production d/n scale) in: {offenders} -- use "
        f"feature_engineering.transformer._squared_dists_shared.squared_dists instead"
    )
