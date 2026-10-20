"""An unknown panel token names the allowed tokens with their descriptions, not only the bare names.

Regression: the validators raised ``Allowed: [...]`` (names only), so a typo in a ``*_panels`` template left the user
to find the description elsewhere. The message now comes from the catalogue, so the description text is in it.
"""

import numpy as np
import pytest

from mlframe.reporting.catalog import unknown_panel_tokens_message
from mlframe.reporting.charts.binary import compose_binary_figure


def test_unknown_binary_token_error_carries_catalogue_descriptions():
    """A typo in a binary panel template names the allowed tokens with their catalogue descriptions."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.random(200)
    with pytest.raises(ValueError) as exc:
        compose_binary_figure(y, s, panels_template="ROC NOT_A_PANEL")
    msg = str(exc.value)
    assert "NOT_A_PANEL" in msg
    # A description that exists only in the catalogue, not in the token name list.
    assert "TPR vs FPR with the chance diagonal" in msg
    assert "[binary_classification]" in msg
    # Other task types' blocks are not dragged in.
    assert "[regression]" not in msg


def test_unknown_panel_tokens_message_is_limited_to_the_task_block():
    """The error text carries only the offending task type's catalogue block."""
    text = unknown_panel_tokens_message("regression", "regression", ["BOGUS"])
    assert text.startswith("Unknown regression panel tokens ['BOGUS']")
    assert "SCATTER" in text
    assert "[binary_classification]" not in text
