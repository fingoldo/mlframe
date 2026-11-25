"""Cluster 26 regression: the SHAP confidence plot skips visibly when matplotlib is absent (audit 2026-11-20, finding 26-F36).

With matplotlib missing the import guard leaves ``plt`` as None, and the function then raised on ``plt.get_fignums()``. It now logs a WARNING and
returns; no figure work is attempted.
"""

import logging
from unittest.mock import patch

from mlframe.training import _confidence_analysis as ca


def test_skips_with_warning_when_plt_is_none(caplog):
    """With plt None the function returns None and logs a WARNING instead of raising (26-F36)."""
    with patch.object(ca, "plt", None), caplog.at_level(logging.WARNING):
        out = ca._shap_confidence_plot(True, None, None, 5, "viridis", 0.5, "y", "t", "f.png", (4, 3))
    assert out is None
    assert any("matplotlib is not installed" in r.getMessage() for r in caplog.records)
