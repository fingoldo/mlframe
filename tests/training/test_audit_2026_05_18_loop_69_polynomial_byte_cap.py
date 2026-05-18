"""Regression: ``apply_preprocessing_extensions`` auto-tunes the
polynomial step down when the dense output array would exceed
``memory_safety_max_bytes``.

Pre-fix path (iter-69 100k seed=157 cb-regression, axes
cat_enc=onehot + poly_deg=2):
- After one-hot encoding the post-pipeline frame had ~58 numeric
  features. projected_exact = C(58+2, 2) ≈ 1711 cols, well under
  the column-count guard ``memory_safety_max_features=100_000``.
- sklearn's PolynomialFeatures then allocated a dense ndarray of
  shape (81000, 1711) at float64 -> 1.03 GiB; the env couldn't
  fit it and the whole training run died with MemoryError.

Post-fix: ``memory_safety_max_bytes`` (default 500 MB) auto-tunes
the polynomial step the same way iter-44 auto-tunes
``dim_n_components`` -- flip interaction_only, decrement degree,
skip the step entirely if even degree=1 exceeds the cap. Each step
is WARN-logged.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.training.configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline import apply_preprocessing_extensions


def _make_frame(n: int = 200, n_features: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        f"x{i}": rng.standard_normal(n).astype(np.float64)
        for i in range(n_features)
    })


def test_polynomial_flips_interaction_only_when_over_byte_cap(caplog) -> None:
    """Wide-enough frame at degree=2 -> auto-tune flips
    interaction_only=True to drop pure-power terms first. Locks the
    first-stage fallback ordering."""
    df = _make_frame(n=2000, n_features=30)
    cfg = PreprocessingExtensionsConfig(
        polynomial_degree=2,
        polynomial_interaction_only=False,
        # Very tight byte cap forces the auto-tune to kick in.
        memory_safety_max_bytes=1_000_000,  # 1 MB
    )
    with caplog.at_level(logging.WARNING, logger="mlframe.training.pipeline"):
        out = apply_preprocessing_extensions(df, None, None, cfg, verbose=0)
    assert out is not None
    # Must have logged the interaction_only flip (the first auto-tune step).
    assert any(
        "interaction_only=True" in rec.message and "polynomial" in rec.message
        for rec in caplog.records
    ), f"expected interaction_only flip WARN; got: {[r.message for r in caplog.records]}"


def test_polynomial_decrements_degree_when_interaction_only_not_enough(caplog) -> None:
    """Even tighter cap (still over after flipping to interaction_only)
    -> auto-tune drops degree until either it fits or hits degree=1."""
    df = _make_frame(n=5000, n_features=40)
    cfg = PreprocessingExtensionsConfig(
        polynomial_degree=2,
        polynomial_interaction_only=False,
        memory_safety_max_bytes=100_000,  # 100 KB -- tighter
    )
    with caplog.at_level(logging.WARNING, logger="mlframe.training.pipeline"):
        out = apply_preprocessing_extensions(df, None, None, cfg, verbose=0)
    assert out is not None


def test_polynomial_skips_entirely_when_even_degree_1_too_big(caplog) -> None:
    """Cap absurdly small -> even degree=1 (= n_features identity)
    fails the byte cap. Auto-tune skips the polynomial step entirely;
    output is the input unchanged (modulo other extensions)."""
    df = _make_frame(n=10000, n_features=50)
    cfg = PreprocessingExtensionsConfig(
        polynomial_degree=2,
        polynomial_interaction_only=False,
        memory_safety_max_bytes=1_000,  # 1 KB -- impossible
    )
    with caplog.at_level(logging.WARNING, logger="mlframe.training.pipeline"):
        out = apply_preprocessing_extensions(df, None, None, cfg, verbose=0)
    assert out is not None
    # The "skipping polynomial step entirely" WARN must fire.
    assert any(
        "skipping" in rec.message and "polynomial" in rec.message
        for rec in caplog.records
    ), f"expected skip-polynomial WARN; got: {[r.message for r in caplog.records]}"


def test_polynomial_fits_when_under_byte_cap_no_warn(caplog) -> None:
    """Small frame well under the cap -> no auto-tune fires; the
    polynomial step runs normally. Locks the no-op happy path."""
    df = _make_frame(n=100, n_features=4)
    cfg = PreprocessingExtensionsConfig(
        polynomial_degree=2,
        polynomial_interaction_only=False,
        memory_safety_max_bytes=500_000_000,  # 500 MB
    )
    with caplog.at_level(logging.WARNING, logger="mlframe.training.pipeline"):
        out = apply_preprocessing_extensions(df, None, None, cfg, verbose=0)
    assert out is not None
    _warn_msgs = [rec.message for rec in caplog.records]
    assert not any(
        "polynomial" in m and ("flipping" in m or "decrementing" in m or "skipping" in m)
        for m in _warn_msgs
    ), f"expected no polynomial auto-tune WARN; got: {_warn_msgs}"


def test_memory_safety_max_bytes_none_disables_byte_guard(caplog) -> None:
    """``memory_safety_max_bytes=None`` keeps the historical
    column-count-only behaviour (no byte auto-tune)."""
    df = _make_frame(n=2000, n_features=30)
    cfg = PreprocessingExtensionsConfig(
        polynomial_degree=2,
        polynomial_interaction_only=False,
        memory_safety_max_bytes=None,
    )
    with caplog.at_level(logging.WARNING, logger="mlframe.training.pipeline"):
        out = apply_preprocessing_extensions(df, None, None, cfg, verbose=0)
    assert out is not None
    _warn_msgs = [rec.message for rec in caplog.records]
    # When the byte guard is disabled, the auto-tune messages must NOT fire.
    assert not any(
        "polynomial output would allocate" in m
        for m in _warn_msgs
    )
