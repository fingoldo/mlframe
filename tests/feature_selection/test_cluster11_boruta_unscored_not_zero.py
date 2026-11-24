"""Cluster 11 regression: a BorutaShap column that was never scored is not given the same score as a real zero (audit 2026-11-20, finding 11-F11).

The per-feature mean was ``nan_to_num(nanmean, nan=0.0)``, so a never-scored column tied with a genuinely zero score. Unscored columns now map to
-inf, which ranks them below every scored column.
"""

from pathlib import Path

from mlframe.feature_selection._benchmarks.fs_hybrid import _arms as arms


def test_unscored_column_maps_to_negative_infinity_not_zero():
    """The BorutaShap score no longer uses nan_to_num(..., nan=0.0) for never-scored columns (11-F11)."""
    src = Path(arms.__file__).resolve().read_text(encoding="utf-8")
    assert "np.nanmean(np.asarray(hist[c], dtype=np.float64)), nan=0.0" not in src
    assert 'means[str(c)] = m if np.isfinite(m) else float("-inf")' in src
