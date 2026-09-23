"""Naming a base in ``base_candidates`` overrules the corr filter, which is what that filter's log promises (DSC-26).

The corr-threshold filter's INFO line told operators that a legitimate lag dropped by ``forbidden_base_corr_threshold``
could be recovered by passing it via ``base_candidates=[...]``. The explicit path then kept only the features that had
survived that same filter, so the advice was a dead end and the only working knob was the threshold. The explicit list
now readmits a base the corr filter took - and only that filter: a non-numeric or constant base still cannot be fitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.discovery import CompositeTargetDiscovery
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _frame(n: int = 400) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """A frame whose ``lag`` column is almost y (so the corr filter takes it) next to an ordinary feature."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=n) * 5.0 + 20.0
    df = pd.DataFrame({"lag": y + rng.normal(scale=1e-3, size=n), "x": rng.normal(size=n), "const": np.ones(n)})
    return df, y, np.arange(n)


def _discovery(**cfg) -> CompositeTargetDiscovery:
    """A discovery instance with ``target`` as its target column."""
    disc = CompositeTargetDiscovery(config=CompositeTargetDiscoveryConfig(enabled=True, **cfg))
    disc._target_col = "target"
    return disc


def _usable(disc, df, y, rows) -> list[str]:
    """The feature list the filter leaves, recording the corr-filtered names on ``disc``."""
    from mlframe.training.composite.discovery._filter import _filter_features

    return list(_filter_features(disc, df, list(df.columns), y, rows))


def test_the_corr_filter_takes_the_lag_and_records_it():
    """Baseline: the near-copy lag is filtered out and its name is kept for the explicit path."""
    df, y, rows = _frame()
    disc = _discovery(base_candidates=["lag"], forbidden_base_corr_threshold=0.99)
    usable = _usable(disc, df, y, rows)
    assert "lag" not in usable and "x" in usable
    assert "lag" in disc._corr_filtered_bases_


def test_an_explicitly_named_base_is_readmitted(caplog):
    """The advice works: the explicit list gets the lag back, and the readmission is logged with its correlation."""
    df, y, rows = _frame()
    disc = _discovery(base_candidates=["lag"], forbidden_base_corr_threshold=0.99)
    usable = _usable(disc, df, y, rows)
    with caplog.at_level("INFO", logger="mlframe.training.composite.discovery"):
        bases = disc._resolve_base_candidates(df, "target", usable, y, rows)
    assert bases == ["lag"]
    assert any("readmitted past the corr filter" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("name", ["const", "missing"])
def test_only_the_corr_filter_is_overruled(name, caplog):
    """A constant base and an unknown name are still dropped, and the drop is reported without blaming the corr filter."""
    df, y, rows = _frame()
    disc = _discovery(base_candidates=[name], forbidden_base_corr_threshold=0.99)
    usable = _usable(disc, df, y, rows)
    with caplog.at_level("WARNING", logger="mlframe.training.composite.discovery"):
        bases = disc._resolve_base_candidates(df, "target", usable, y, rows)
    assert bases == []
    dropped = [r.getMessage() for r in caplog.records if "explicit base_candidates dropped" in r.getMessage()]
    assert dropped and "leak-corr" not in dropped[0]
