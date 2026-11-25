"""Cluster 09 regression: an empty quintile stratum is announced before it falls back to the whole fold (audit 2026-11-20, finding 09-F45).

Source-level pin: the fallback path logs a WARNING naming the stratum, and the log comes before the assignment that substitutes the fold
pool. Triggering an empty stratum needs a specific edge layout, so the behaviour is pinned through the code path.
"""

from pathlib import Path

from mlframe.feature_engineering.transformer import y_quintile_baseline_knn as yq

_SRC = Path(yq.__file__).resolve()


def test_empty_stratum_warns_before_falling_back():
    """The WARNING precedes the fold-pool fallback assignment (09-F45)."""
    text = _SRC.read_text(encoding="utf-8")
    warn = text.index("stratum %d is empty; its features fall back to the whole train fold")
    fallback = text.index("                X_q = Xt_s\n                pred_q = pred_train\n")
    assert warn < fallback
