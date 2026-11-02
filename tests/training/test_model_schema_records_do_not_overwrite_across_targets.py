"""Schema records of the same model name trained for different targets or pre-pipelines keep separate keys."""

from __future__ import annotations

from mlframe.training.core._phase_train_one_target_schema import _schema_store_key


def _store(schemas, name, target, pipeline):
    """Store a record the way the writer does and return its key."""
    rec = {"target_name": target, "pre_pipeline_name": pipeline}
    key = _schema_store_key(schemas, name, rec)
    schemas[key] = rec
    return key


def test_second_target_does_not_overwrite_the_first():
    """cb trained for y1 and y2 leaves two records."""
    schemas: dict = {}
    k1 = _store(schemas, "cb", "y1", "ord")
    k2 = _store(schemas, "cb", "y2", "ord")
    assert k1 == "cb" and k2 != k1 and len(schemas) == 2


def test_second_pre_pipeline_does_not_overwrite_the_first():
    """cb after the ordinary and after the MRMR pre-pipeline leaves two records."""
    schemas: dict = {}
    _store(schemas, "cb", "y1", "ord")
    _store(schemas, "cb", "y1", "mrmr")
    assert len(schemas) == 2


def test_retraining_the_same_target_and_pipeline_replaces_its_own_record():
    """Same identity keeps the bare key and overwrites (negative control)."""
    schemas: dict = {}
    _store(schemas, "cb", "y1", "ord")
    _store(schemas, "cb", "y1", "ord")
    assert list(schemas) == ["cb"]
