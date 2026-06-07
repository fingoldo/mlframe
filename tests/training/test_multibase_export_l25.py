"""Regression: CompositeTargetDiscovery.export_specs must preserve
``extra_base_columns`` so downstream multi-base consumers don't see
a stub spec.

Pre-fix path (fuzz c0047_701a2067):
1. CompositeTargetDiscovery's forward-stepwise auto-promoter produces
   a CompositeSpec with extra_base_columns=("num_dep",) and
   base_column="num_1".
2. ``export_specs()`` snapshots specs to plain dicts for
   ``metadata["composite_target_specs"]`` storage. The original
   implementation copied 10 fields but DROPPED extra_base_columns.
3. Every downstream consumer that reads specs from metadata —
   ``_phase_dummy_baselines`` (y-scale inverse step) and
   ``_phase_composite_post`` (OOF holdout per-spec base matrix) —
   sees a dict where ``extra_base_columns`` is implicitly empty,
   so ``transform.forward`` / ``transform.inverse`` get a 1-D base
   while fitted alphas has K>=2 entries.
4. Result: spurious WARNINGS in dummy_baselines and OOF holdout
   crashes ("base has 1 columns but fitted alphas has 2 entries").

Post-fix: ``export_specs`` includes the field; downstream consumers
that read ``spec_dict.get("extra_base_columns")`` see the full tuple
and can build the (n, 1+K) matrix correctly.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.training.composite.discovery import CompositeTargetDiscovery
from mlframe.training.composite_spec import CompositeSpec


def _make_discovery_with_multibase_spec():
    """Build a discovery shell with a multi-base CompositeSpec pinned
    into specs_. We don't run the actual MI / tiny-model loop --
    the test only exercises export_specs's dict-builder layer."""
    from mlframe.training.composite.transforms import _linear_residual_multi_fit

    rng = np.random.default_rng(0)
    n = 50
    b1 = rng.standard_normal(n).astype(np.float64)
    b2 = rng.standard_normal(n).astype(np.float64)
    y = 0.7 * b1 - 0.4 * b2 + 1.0 + 0.05 * rng.standard_normal(n)

    fitted_params = _linear_residual_multi_fit(
        y=y, base=np.column_stack([b1, b2]), sample_weight=None,
    )

    spec = CompositeSpec(
        name="y_target-linresM-b1+b2",
        target_col="y_target",
        transform_name="linear_residual_multi",
        base_column="b1",
        fitted_params=fitted_params,
        mi_gain=0.5,
        mi_y=1.0,
        mi_t=1.5,
        valid_domain_frac=1.0,
        n_train_rows=n,
        extra_base_columns=("b2",),
    )
    # CompositeTargetDiscovery's __init__ requires configuration; we
    # bypass it by instantiating with the bare specs_ attribute.
    disc = CompositeTargetDiscovery.__new__(CompositeTargetDiscovery)
    disc.specs_ = [spec]
    return disc, spec


def test_export_specs_preserves_extra_base_columns() -> None:
    """Multi-base spec round-trips through export_specs with
    extra_base_columns intact."""
    disc, spec = _make_discovery_with_multibase_spec()
    exported = disc.export_specs()
    assert len(exported) == 1
    d = exported[0]
    assert d["base_column"] == "b1"
    assert "extra_base_columns" in d, "export_specs dropped extra_base_columns"
    assert tuple(d["extra_base_columns"]) == ("b2",)
    # Sanity: primary fields still present.
    assert d["name"] == spec.name
    assert d["transform_name"] == "linear_residual_multi"


def test_export_specs_legacy_single_base_returns_empty_tuple() -> None:
    """Legacy single-base specs export an empty tuple under the new
    key so downstream `dict.get("extra_base_columns") or ()` returns
    `()` and the 1-D fast path stays active."""
    from mlframe.training.composite.transforms import _linear_residual_fit

    rng = np.random.default_rng(1)
    n = 50
    b1 = rng.standard_normal(n).astype(np.float64)
    y = 0.7 * b1 + 1.0 + 0.05 * rng.standard_normal(n)
    fitted_params = _linear_residual_fit(y=y, base=b1, sample_weight=None)

    spec = CompositeSpec(
        name="y-linres-b1",
        target_col="y",
        transform_name="linear_residual",
        base_column="b1",
        fitted_params=fitted_params,
        mi_gain=0.5, mi_y=1.0, mi_t=1.5,
        valid_domain_frac=1.0, n_train_rows=n,
        # extra_base_columns defaults to ()
    )
    disc = CompositeTargetDiscovery.__new__(CompositeTargetDiscovery)
    disc.specs_ = [spec]
    d = disc.export_specs()[0]
    assert d["extra_base_columns"] == ()


def test_exported_dict_contract_for_downstream_consumers() -> None:
    """The downstream pattern is::

        extras = tuple(spec_dict.get("extra_base_columns") or ())

    This must yield the expected tuple shape for both legacy and
    multi-base specs without raising or returning None."""
    disc, _ = _make_discovery_with_multibase_spec()
    multibase_d = disc.export_specs()[0]
    assert tuple(multibase_d.get("extra_base_columns") or ()) == ("b2",)

    # An old-format dict (pre-fix payload from cache) should still
    # round-trip through the same .get(..., None) idiom.
    legacy_format_d = {
        "name": "x", "target_col": "y", "transform_name": "linear_residual",
        "base_column": "b1", "fitted_params": {},
        "mi_gain": 0.0, "mi_y": 0.0, "mi_t": 0.0,
        "valid_domain_frac": 1.0, "n_train_rows": 1,
    }
    assert tuple(legacy_format_d.get("extra_base_columns") or ()) == ()
