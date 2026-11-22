"""Audit 2026-11-20 finding 29-F18: a blank line in a GloVe file is skipped, not a crash, and skipped rows are counted in a WARNING.

The reader used to index values[0] on a blank line and abort the file. The test writes a file with a blank line and a multi-word key, and
checks the good rows load and the two skipped lines are reported.
"""

import logging

from mlframe.utils.text import read_glove_embeddings


def test_blank_and_multiword_lines_are_skipped_and_counted(tmp_path, caplog):
    """The two valid vectors load; the blank line and the multi-word key are skipped with a WARNING naming the count."""
    path = tmp_path / "glove.txt"
    path.write_text("cat 0.1 0.2\n\nnew york 0.3 0.4\ndog 0.5 0.6\n", encoding="utf8")
    with caplog.at_level(logging.WARNING):
        vectors = read_glove_embeddings(fpath=str(path))
    assert set(vectors) == {"cat", "dog"}
    assert any("skipped 2" in r.getMessage() for r in caplog.records)
