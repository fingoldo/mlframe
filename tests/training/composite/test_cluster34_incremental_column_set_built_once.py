"""Audit 2026-11-20 finding 34-F12: the warm-start rescore builds the frame's column set once per spec, not once per candidate column.

The check uses a counting spy on ``_frame_columns`` and a real polars frame; the call count must not grow with the number of
feature columns.
"""

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from mlframe.training.composite.discovery import _incremental
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def test_column_set_is_built_once_per_spec(monkeypatch):
    """With 40 feature columns the column-set helper runs at most twice (base check plus the one set), not 40 times."""
    rng = np.random.default_rng(0)
    n = 200
    data = {f"f{i}": rng.normal(size=n) for i in range(40)}
    data["y"] = rng.normal(size=n)
    df = pl.DataFrame(data)
    calls = []
    real = _incremental._frame_columns

    def counting(frame):
        """Count calls and delegate."""
        calls.append(1)
        return real(frame)

    monkeypatch.setattr(_incremental, "_frame_columns", counting)

    class _Spec:
        """A unary spec: no base column, so only the feature-column filter touches the frame."""

        name = "spec_identity"
        fitted_params = {}
        transform_name = "identity"
        base_column = None
        extra_base_columns = ()

    class _Identity:
        """Unary identity transform: every row is in its domain and forward returns y unchanged."""

        requires_base = False

        def domain_check(self, y, base=None):
            """Accept every row."""
            return np.ones(len(y), dtype=bool)

        def forward(self, y, base=None, params=None):
            """Return y unchanged."""
            return y

    monkeypatch.setattr(_incremental, "get_transform", lambda name: _Identity())
    _incremental._rescore_spec_gain(_Spec(), df, "y", [f"f{i}" for i in range(40)] + ["y"], np.arange(n), CompositeTargetDiscoveryConfig())
    assert len(calls) <= 2, len(calls)
