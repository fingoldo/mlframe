"""Wave 9.1 loop-iter-22 regression: ``_apply_target_encoding`` MUST
raise when ``unknown_strategy='raise'`` and an input row hits an unseen
``(a, b)`` cell.

Pre-fix at ``engineered_recipes.py:405``: the function silently
substituted ``global_mean`` for unseen cells regardless of the
``unknown_strategy`` setting. Diverged from:
- ``_apply_factorize:512`` which raises on -1 sentinels.
- ``_apply_factorize_kway:556`` (iter-13 fix) which raises at each
  chain step.

Effect: a model whose target-encoded features were silently filled
with the train ``global_mean`` degraded predictions on drifted data,
but passed any user try/except guard that expected a raise. Production
monitoring relying on "raise on schema drift" failed open.

Severity: P1 (silent data corruption when user explicitly opted into
strict mode; asymmetric contract vs factorize family).

Fix at engineered_recipes.py:404: add the same guard as factorize -
when ``unknown_strategy == 'raise'`` and any ``cell_idx < 0``, raise
``ValueError`` with the unseen-row count.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def te_recipe():
    """target_encoding recipe with -1 sentinels at unseen slots."""
    from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe
    return EngineeredRecipe(
        name="te_test", kind="target_encoding",
        src_names=("a", "b"), factorize_nbins=(2, 2),
        unknown_strategy="raise",
        extra={
            # 4-cell factorize lookup; (0,0) and (1,1) seen, (0,1) and (1,0) unseen.
            "factorize_lookup": np.array([0, -1, -1, 1], dtype=np.int64),
            "cell_means": np.array([0.7, 0.3], dtype=np.float64),
            "global_mean": 0.5,
        },
    )


def test_raise_on_unseen_combo(te_recipe):
    """Unseen cell (0,1) with raise strategy must raise ValueError."""
    from mlframe.feature_selection.filters.engineered_recipes import _apply_target_encoding
    X = pd.DataFrame({"a": [0], "b": [1]})
    with pytest.raises(ValueError, match="combinations not seen"):
        _apply_target_encoding(te_recipe, X)


def test_clip_strategy_substitutes_global_mean(te_recipe):
    """Negative control: clip strategy returns global_mean for unseen."""
    from mlframe.feature_selection.filters.engineered_recipes import _apply_target_encoding
    X = pd.DataFrame({"a": [0], "b": [1]})
    out = _apply_target_encoding(replace(te_recipe, unknown_strategy="clip"), X)
    assert out.shape == (1,)
    assert out[0] == 0.5


def test_sentinel_strategy_substitutes_global_mean(te_recipe):
    """Negative control: sentinel strategy returns global_mean for unseen."""
    from mlframe.feature_selection.filters.engineered_recipes import _apply_target_encoding
    X = pd.DataFrame({"a": [0], "b": [1]})
    out = _apply_target_encoding(replace(te_recipe, unknown_strategy="sentinel"), X)
    assert out.shape == (1,)
    assert out[0] == 0.5


def test_raise_only_seen_combos_succeeds(te_recipe):
    """A row of only-seen combinations must NOT raise under raise strategy."""
    from mlframe.feature_selection.filters.engineered_recipes import _apply_target_encoding
    X = pd.DataFrame({"a": [0, 1], "b": [0, 1]})  # (0,0) and (1,1) - both seen
    out = _apply_target_encoding(te_recipe, X)
    assert out.shape == (2,)
    assert abs(out[0] - 0.7) < 1e-12
    assert abs(out[1] - 0.3) < 1e-12


def test_raise_message_includes_unseen_count_and_column_names(te_recipe):
    """Error message must surface unseen row count + column names for
    actionable diagnostics.
    """
    from mlframe.feature_selection.filters.engineered_recipes import _apply_target_encoding
    # 3 unseen rows
    X = pd.DataFrame({"a": [0, 1, 0], "b": [1, 0, 1]})
    with pytest.raises(ValueError) as exc_info:
        _apply_target_encoding(te_recipe, X)
    msg = str(exc_info.value)
    assert "3 row(s)" in msg
    assert "X[a]" in msg
    assert "X[b]" in msg
