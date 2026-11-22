"""Audit 2026-11-20 finding 34-F13: routing a spec column over many groups extracts each base column once, not once per group.

The routed values must still match a brute-force per-group computation.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.training.composite.discovery._per_group_discovery import route_spec_column_by_group
from mlframe.training.composite.transforms.shared import call_transform
from mlframe.training.composite.transforms import get_transform


def _discovery(groups):
    """A stub with one per-group spec ('lr') per group, all using the linear_residual transform on 'base'."""
    specs = {}
    for g, (alpha, beta) in groups.items():
        specs[g] = [SimpleNamespace(name="lr", transform_name="linear_residual", base_column="base", extra_base_columns=(), fitted_params={"alpha": alpha, "beta": beta})]
    return SimpleNamespace(
        config=SimpleNamespace(per_group_column="g"),
        specs_by_group_=specs,
        specs_=[],
        _target_col="y",
    )


def test_base_column_is_extracted_once_and_values_match_brute_force(monkeypatch):
    """Five groups: the base column is pulled from the frame once, and each row's value equals the per-group transform output."""
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({"g": rng.integers(0, 5, size=n), "base": rng.normal(size=n), "y": rng.normal(size=n)})
    groups = {g: (0.5 * g, 1.0 + 0.1 * g) for g in range(5)}
    disc = _discovery(groups)

    from mlframe.training.composite.discovery import screening

    calls = []
    real_extract = screening._extract_column_array

    def counting(frame, col, *a, **k):
        """Count base-column extractions and delegate."""
        calls.append(col)
        return real_extract(frame, col, *a, **k)

    monkeypatch.setattr(screening, "_extract_column_array", counting)
    out = route_spec_column_by_group(disc, df, "lr")
    assert calls.count("base") <= 1, calls

    tr = get_transform("linear_residual")
    expected = np.empty(n)
    for g, (alpha, beta) in groups.items():
        m = df["g"].to_numpy() == g
        expected[m] = call_transform(tr, "forward", df["y"].to_numpy()[m], df["base"].to_numpy()[m], {"alpha": alpha, "beta": beta})
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)  # the extractor returns float32
