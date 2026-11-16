"""Regression: ranker run metadata.json must be written atomically (a failed serialisation must not truncate the previous file)."""
import orjson
import pytest

from mlframe.training.ranking import _ranker_suite_train_helpers as h


def test_failed_metadata_serialisation_keeps_previous_file(tmp_path):
    """A failed metadata serialisation leaves the previous file untouched."""
    save_dir = str(tmp_path)
    h._train_mlframe_rank_save_dir(save_dir, "m", [], {}, False, {"a": 1})
    path = tmp_path / "m_metadata.json"
    before = path.read_bytes()
    assert orjson.loads(before) == {"a": 1}
    with pytest.raises(TypeError):
        h._train_mlframe_rank_save_dir(save_dir, "m", [], {}, False, {"a": object()})
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["m_metadata.json"]
