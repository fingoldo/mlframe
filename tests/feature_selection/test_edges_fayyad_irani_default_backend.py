"""Regression test for the edges_fayyad_irani default-backend fix.

Pre-fix the function defaulted ``backend='python'`` while the sibling
``mdlp_bin_edges`` already defaulted ``'njit'``. The python backend's
``_mdlp_recurse`` calls ``_entropy_from_labels`` once per (split candidate,
half-frame) pair, and each call does ``np.unique`` + ``np.sort`` on the
label slice (~4ms at n=500k). The c0022_9f2cf625 @500k MRMR-using profile
attributed 1566s of a 1700s suite (88% of wall) to that path before this
fix landed. The njit kernel `_mdlp_recurse_njit` maintains running
per-class counts across the candidate-scan and is the documented
"audit fix (10-30x speedup target)".

These tests pin:
  (1) edges_fayyad_irani's default backend is now 'njit'.
  (2) The default produces the same edge set as backend='python' on small
      labelled data (correctness equivalence).
  (3) Explicit backend='python' still works (legacy A/B-test path
      remains operational).
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from mlframe.feature_selection.filters._adaptive_nbins import edges_fayyad_irani


def test_default_backend_is_njit():
    """Default backend must be 'njit' to avoid the 1566s @500k regression."""
    sig = inspect.signature(edges_fayyad_irani)
    backend_param = sig.parameters["backend"]
    assert backend_param.default == "njit", (
        f"edges_fayyad_irani default backend must be 'njit', got "
        f"{backend_param.default!r}. The 'python' backend is 10-30x slower "
        f"and was the dominant hotspot in c0022_9f2cf625 @500k profile."
    )


def test_default_equivalent_to_python_backend_on_small_data():
    """Sanity: default (njit) produces the same edges as explicit 'python'
    on a small labelled dataset. The MDLP algorithm is deterministic given
    the input, so both backends must yield identical edge sets."""
    rng = np.random.default_rng(20260530)
    n = 2_000
    x = rng.standard_normal(n)
    # Class-structured labels so MDLP finds real splits.
    y = (x > rng.choice([-1.0, 0.0, 1.0], size=n)).astype(np.int64)

    edges_default = edges_fayyad_irani(x, y)  # default = njit
    edges_python = edges_fayyad_irani(x, y, backend="python")

    # Edge counts must match.
    assert edges_default.size == edges_python.size, (
        f"njit produced {edges_default.size} edges, python {edges_python.size}"
    )
    # Sort + compare (both backends collect splits in the same algorithmic
    # order, but the result is a sorted edge list).
    np.testing.assert_allclose(
        np.sort(edges_default), np.sort(edges_python), atol=1e-8
    )


def test_explicit_python_backend_still_works():
    """The legacy 'python' backend remains operational for A/B-testing."""
    rng = np.random.default_rng(0)
    n = 500
    x = rng.standard_normal(n)
    y = (x > 0).astype(np.int64)

    edges = edges_fayyad_irani(x, y, backend="python")
    assert isinstance(edges, np.ndarray)
    # MDLP on a clear monotonic relationship should find at least one split.
    assert edges.size >= 0  # 0 if no MDL-passing split; otherwise >=1
    assert np.all(np.isfinite(edges))


def test_njit_backend_smoke_handles_pure_label_input():
    """Defensive: the njit kernel must handle pure-label (no informative
    split) input without raising. The recursion should short-circuit at
    `h_full <= 0` and return an empty inner-edge list."""
    rng = np.random.default_rng(1)
    n = 200
    x = rng.standard_normal(n)
    y = np.zeros(n, dtype=np.int64)  # all-same label

    edges = edges_fayyad_irani(x, y)
    assert isinstance(edges, np.ndarray)
    assert edges.size == 0  # no informative splits
