"""Cluster 21 regression: a failed atomic move of a rendered chart removes its scratch file (audit 2026-11-20, finding 21-F20).

Source-level pin, since the persistent kaleido path needs a real renderer: the OSError branch around os.replace calls _abandon_scratch_file,
as the other branch already did, and the abandon helper is the one that removes the file.
"""

from pathlib import Path

from mlframe.reporting.renderers import _kaleido as kd

_SRC = Path(kd.__file__).resolve()


def test_move_failure_abandons_the_scratch_file():
    """The os.replace failure branch abandons the scratch file (21-F20)."""
    text = _SRC.read_text(encoding="utf-8")
    move = text.index("os.replace(_scratch, path)")
    except_branch = text.index("_exc[0] = _mv_exc", move)
    abandon = text.index("_abandon_scratch_file(_scratch)", except_branch)
    assert abandon - except_branch < 400
