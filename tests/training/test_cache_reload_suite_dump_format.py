"""Regression: the use-cache reload reads the suite's own ``.dump`` (zstd + dill model entry) through ``load_mlframe_model``.

The reload used to call ``safe_joblib_load`` on that file. Suite dumps are zstd-compressed dill entries, not joblib tuples, so the
joblib unpickler failed on the zstd magic and the failure was not in the reload's except list.
"""

from types import SimpleNamespace

import pytest

from mlframe.training import _trainer_train_and_evaluate_parts as parts

_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def _write_suite_dump(path):
    """Write a file that starts with the zstd magic, as the suite does for its .dump files."""
    path.write_bytes(_ZSTD_MAGIC + b"\x00" * 32)


def test_cache_reload_returns_model_and_pre_pipeline_from_suite_entry(tmp_path, monkeypatch):
    """The reload returns the entry model and pre-pipeline read from the suite dump."""
    dump = tmp_path / "lgb.dump"
    _write_suite_dump(dump)
    monkeypatch.setattr(parts, "load_mlframe_model", lambda _path: SimpleNamespace(model="fitted-model", pre_pipeline="fitted-pp"))

    model, pre_pipeline = parts._train_and_evaluate_use_cache_exists_model(True, str(dump), None, None, None)

    assert model == "fitted-model"
    assert pre_pipeline == "fitted-pp"


def test_cache_reload_keeps_inputs_when_dump_does_not_load(tmp_path, monkeypatch):
    """An unreadable dump leaves the caller model and pre-pipeline untouched so the target retrains."""
    dump = tmp_path / "lgb.dump"
    _write_suite_dump(dump)
    monkeypatch.setattr(parts, "load_mlframe_model", lambda _path: None)

    model, pre_pipeline = parts._train_and_evaluate_use_cache_exists_model(True, str(dump), None, "live-model", "live-pp")

    assert model == "live-model"
    assert pre_pipeline == "live-pp"


@pytest.mark.parametrize("use_cache", [False])
def test_cache_reload_is_skipped_when_cache_disabled(tmp_path, monkeypatch, use_cache):
    """With use_cache off the dump is never read."""
    dump = tmp_path / "lgb.dump"
    _write_suite_dump(dump)

    def _must_not_load(_path):
        """Fail the test if the loader is called."""
        raise AssertionError("cache disabled: the dump must not be read")

    monkeypatch.setattr(parts, "load_mlframe_model", _must_not_load)

    model, pre_pipeline = parts._train_and_evaluate_use_cache_exists_model(use_cache, str(dump), None, "live-model", "live-pp")

    assert (model, pre_pipeline) == ("live-model", "live-pp")
