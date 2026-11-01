"""The import probe's child interpreter does not import a module planted in the current directory (SUB2-01).

``python -c`` and ``python -m`` put the working directory first on ``sys.path``, so a ``polars.py`` in a directory the process runs from
used to execute in the probe. The children now start with ``-P``.
"""

import pytest

from mlframe.utils import native_import_probe as probe


def test_planted_module_in_the_working_directory_is_not_executed_by_the_probe(tmp_path, monkeypatch):
    """A probe run from a directory that holds a ``planted`` module must not import it."""
    marker = tmp_path / "planted_ran.txt"
    (tmp_path / "planted.py").write_text(f"open({str(marker)!r}, 'w').write('ran')\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    probe._probe_child("planted")
    assert not marker.exists(), "the child interpreter imported a module from the working directory"


@pytest.mark.parametrize("name", ["json"])
def test_probe_still_reports_a_real_stdlib_module_as_importable(name):
    """The flag only changes the search path, so an ordinary module still imports in the child."""
    completed = probe._probe_child(name)
    assert completed is not None and completed.returncode == 0
