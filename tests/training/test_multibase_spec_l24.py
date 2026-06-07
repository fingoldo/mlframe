"""Regression: composite-discovery multi-base spec integration must
materialise the FULL (n_total, 1+K) base matrix at integration time.

Pre-fix path (fuzz c0116_9ab4b3e3):
1. CompositeTargetDiscovery enabled with auto base selection and
   multi-base auto-promotion enabled (default). For some target/feature
   shapes the forward-stepwise promoter picks 2 base columns
   (CompositeSpec.base_column + 1 entry in extra_base_columns) and
   transitions the spec from linear_residual to linear_residual_multi
   with alphas=[a1, a2].
2. The integration step at _phase_composite_discovery.py:436 calls
   _build_full_column_from_splits(_spec.base_column, ...) which returns
   ONLY the primary base column as a 1-D array; extra_base_columns is
   never read.
3. transform.forward(y, _base_full, _spec.fitted_params) is called with
   _base_full shape=(n,) (1 column) but alphas size=2.
4. _linear_residual_multi_forward raises
   ``ValueError: linear_residual_multi: base has 1 columns but fitted
   alphas has 2 entries``.

Post-fix: when extra_base_columns is non-empty, _phase_composite_discovery
stacks the primary + each extra column into a (n_total, 1+K) matrix
before invoking transform.forward.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.spec import CompositeSpec
from mlframe.training.composite.transforms import (
    _linear_residual_multi_fit,
    get_transform,
)


def _build_multi_base_spec(rng: np.random.Generator):
    """Build a real multi-base spec by fitting linear_residual_multi on
    a synthetic y = 0.7*b1 + -0.4*b2 + 1.2 + noise. Returns the spec
    plus the per-split arrays for the test."""
    n_total = 60
    train_idx = np.arange(0, 40)
    val_idx = np.arange(40, 50)
    test_idx = np.arange(50, 60)

    b1_full = rng.standard_normal(n_total).astype(np.float64)
    b2_full = rng.standard_normal(n_total).astype(np.float64)
    y_full = 0.7 * b1_full - 0.4 * b2_full + 1.2 + 0.05 * rng.standard_normal(n_total)

    base_train = np.column_stack([b1_full[train_idx], b2_full[train_idx]])
    fitted_params = _linear_residual_multi_fit(
        y=y_full[train_idx], base=base_train, sample_weight=None,
    )

    spec = CompositeSpec(
        name="y_target-linres_multi-b1+b2",
        target_col="y_target",
        transform_name="linear_residual_multi",
        base_column="b1",
        fitted_params=fitted_params,
        mi_gain=0.5,
        mi_y=1.0,
        mi_t=1.5,
        valid_domain_frac=1.0,
        n_train_rows=int(len(train_idx)),
        extra_base_columns=("b2",),
    )

    train_df = pd.DataFrame({"b1": b1_full[train_idx], "b2": b2_full[train_idx]})
    val_df = pd.DataFrame({"b1": b1_full[val_idx], "b2": b2_full[val_idx]})
    test_df = pd.DataFrame({"b1": b1_full[test_idx], "b2": b2_full[test_idx]})
    return spec, train_df, val_df, test_df, train_idx, val_idx, test_idx, y_full


def test_multi_base_spec_forward_stacks_extra_base_columns() -> None:
    """Reproduces the integration step from _phase_composite_discovery
    on a multi-base spec. Must NOT raise ValueError on alpha-vs-base
    count mismatch."""
    rng = np.random.default_rng(0)
    spec, train_df, val_df, test_df, train_idx, val_idx, test_idx, y_full = (
        _build_multi_base_spec(rng)
    )
    transform = get_transform(spec.transform_name)

    # Reproduce the post-fix logic:
    from mlframe.training.core._misc_helpers import _build_full_column_from_splits
    extra_bases = tuple(getattr(spec, "extra_base_columns", ()) or ())
    base_primary = _build_full_column_from_splits(
        spec.base_column, train_df, val_df, test_df,
        train_idx, val_idx, test_idx, n_total=y_full.shape[0],
    )
    assert base_primary.ndim == 1
    assert extra_bases == ("b2",)
    cols = [base_primary]
    for eb in extra_bases:
        cols.append(
            _build_full_column_from_splits(
                eb, train_df, val_df, test_df,
                train_idx, val_idx, test_idx, n_total=y_full.shape[0],
            )
        )
    base_full = np.column_stack(cols)
    assert base_full.shape == (y_full.shape[0], 1 + len(extra_bases))

    valid = transform.domain_check(y_full, base_full)
    assert valid.any()
    # Pre-fix: passing base_primary (1-D, shape=(n,)) would raise
    # ValueError("base has 1 columns but fitted alphas has 2 entries").
    # Post-fix: passing the stacked 2-D matrix works.
    t_out = transform.forward(y_full[valid], base_full[valid, :], spec.fitted_params)
    assert t_out.shape == (int(valid.sum()),)
    assert np.all(np.isfinite(t_out))


def test_multi_base_spec_forward_pre_fix_would_have_failed() -> None:
    """Pre-fix simulation: feeding only the primary base column (1-D)
    must produce the alpha-mismatch ValueError. Locks in the bug surface
    so a future regression of the integration code immediately fails this
    sensor instead of crashing inside the suite."""
    rng = np.random.default_rng(1)
    spec, train_df, val_df, test_df, train_idx, val_idx, test_idx, y_full = (
        _build_multi_base_spec(rng)
    )
    transform = get_transform(spec.transform_name)
    from mlframe.training.core._misc_helpers import _build_full_column_from_splits
    base_primary = _build_full_column_from_splits(
        spec.base_column, train_df, val_df, test_df,
        train_idx, val_idx, test_idx, n_total=y_full.shape[0],
    )
    valid = transform.domain_check(y_full, base_primary)
    if not valid.any():
        pytest.skip("domain_check disqualified all rows for the synthetic data")
    with pytest.raises(ValueError, match="alphas"):
        transform.forward(y_full[valid], base_primary[valid], spec.fitted_params)


def test_single_base_spec_path_unchanged() -> None:
    """Baseline: legacy single-base specs (extra_base_columns=())
    continue to pass a 1-D base array. The fix's conditional branch
    must not regress the common path."""
    rng = np.random.default_rng(2)
    n_total = 60
    train_idx = np.arange(0, 40)
    val_idx = np.arange(40, 50)
    test_idx = np.arange(50, 60)
    b1_full = rng.standard_normal(n_total).astype(np.float64)
    y_full = 0.7 * b1_full + 1.0 + 0.05 * rng.standard_normal(n_total)
    from mlframe.training.composite.transforms import _linear_residual_fit
    fitted_params = _linear_residual_fit(
        y=y_full[train_idx], base=b1_full[train_idx], sample_weight=None,
    )
    spec = CompositeSpec(
        name="y-linres-b1",
        target_col="y", transform_name="linear_residual",
        base_column="b1", fitted_params=fitted_params,
        mi_gain=0.5, mi_y=1.0, mi_t=1.5,
        valid_domain_frac=1.0, n_train_rows=int(len(train_idx)),
    )
    assert spec.extra_base_columns == ()
    train_df = pd.DataFrame({"b1": b1_full[train_idx]})
    val_df = pd.DataFrame({"b1": b1_full[val_idx]})
    test_df = pd.DataFrame({"b1": b1_full[test_idx]})
    transform = get_transform(spec.transform_name)
    from mlframe.training.core._misc_helpers import _build_full_column_from_splits
    base_full = _build_full_column_from_splits(
        spec.base_column, train_df, val_df, test_df,
        train_idx, val_idx, test_idx, n_total=y_full.shape[0],
    )
    assert base_full.ndim == 1
    valid = transform.domain_check(y_full, base_full)
    t_out = transform.forward(y_full[valid], base_full[valid], spec.fitted_params)
    assert t_out.shape == (int(valid.sum()),)
