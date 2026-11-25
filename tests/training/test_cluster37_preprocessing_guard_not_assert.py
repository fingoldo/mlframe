"""Cluster 37 regression: the preprocessing-config guard is an explicit raise, not an assert (audit 2026-11-20, finding 37-F43).

``assert`` is stripped under ``python -O``, so a None preprocessing config would reach the pipeline and fail somewhere unrelated. The guard is an
``if ...: raise RuntimeError`` that survives optimisation.
"""

from pathlib import Path

from mlframe.training.core import _main_train_suite as suite


def test_guard_is_an_explicit_raise_not_an_assert():
    """The preprocessing_config check raises RuntimeError and is not an assert (37-F43)."""
    src = Path(suite.__file__).resolve().read_text(encoding="utf-8")
    assert "assert st.preprocessing_config is not None" not in src
    assert "if st.preprocessing_config is None:" in src
    assert 'raise RuntimeError("_main_train_suite: preprocessing_config must be resolved' in src
