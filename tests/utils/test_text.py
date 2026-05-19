"""Smoke tests for mlframe.utils.text (E-P1.4)."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


@pytest.mark.fast
def test_import_text_module():
    """Module imports cleanly with public callable present."""
    from mlframe.utils import text as tmod

    assert callable(tmod.read_glove_embeddings)


@pytest.mark.fast
def test_read_glove_embeddings_happy_path(tmp_path):
    """Reads a minimal GloVe-format file into a dict[word]->vector."""
    from mlframe.utils.text import read_glove_embeddings

    f = tmp_path / "tiny_glove.txt"
    f.write_text("cat 0.1 0.2 0.3\ndog 0.4 0.5 0.6\n", encoding="utf8")

    out = read_glove_embeddings(fpath=str(f))
    assert isinstance(out, dict)
    assert set(out.keys()) == {"cat", "dog"}
    assert out["cat"].shape == (3,)
    np.testing.assert_allclose(out["dog"], np.array([0.4, 0.5, 0.6], dtype="float32"))


@pytest.mark.fast
def test_read_glove_embeddings_merges_into_existing(tmp_path):
    """When embeddings_dict supplied, function merges into it."""
    from mlframe.utils.text import read_glove_embeddings

    f = tmp_path / "more.txt"
    f.write_text("fish 1.0 2.0\n", encoding="utf8")

    seed = {"bird": np.array([9.0, 9.0], dtype="float32")}
    out = read_glove_embeddings(embeddings_dict=seed, fpath=str(f))
    assert out is seed  # mutated in place AND returned
    assert "bird" in out and "fish" in out
