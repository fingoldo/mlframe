"""Cluster 11 regression: importing the stat-study core does not change the process-wide warning filters (audit 2026-11-20, finding 11-F6).

The module used to call ``warnings.simplefilter("ignore")`` at import, silencing warnings for every importer. The call is gone.
"""

from pathlib import Path

from mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study import core


def test_import_does_not_install_an_ignore_all_filter():
    """The module source has no global simplefilter call (11-F6)."""
    src = Path(core.__file__).resolve().read_text(encoding="utf-8")
    assert 'warnings.simplefilter("ignore")' not in src
