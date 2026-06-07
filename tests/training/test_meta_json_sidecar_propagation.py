"""Wave-19 follow-up sensor: .meta.json sidecar pattern reused at the
three remaining persistence boundaries flagged by the audit.

Original wave-19 P0 #1 (49d68ab) added the sidecar to save_mlframe_model
in training/io.py. This sensor pins three additional integrations:

1. P0 #3 -- training/ranker_suite.py write loop: per-flavor booster
   joblib.dump now triggers _write_save_meta_sidecar so CB/LGB/XGB
   minor-upgrade skew gets a WARN log at load instead of a cryptic
   AttributeError deep in predict().

2. P1 -- calibration/post.py per-calibrator joblib.dump now writes a
   sidecar. _PerClassIsotonicCalibrator / _PostHocMultiCalibratedModel
   carry attributes (n_classes, is_exclusive, _target_type) whose
   semantics could shift across mlframe versions.

3. P1 -- inference/predict.py read_trained_models calls
   validate_load_meta_sidecar before joblib.load so the operator sees
   library-version drift instead of chasing a downstream crash.

All three sites are source-level guards: a behavioural fixture would
require full ranker / calibrator / inference fixtures which already
exist in their own dedicated test files; what we pin HERE is the
contract that the sidecar helper is wired in at each call site.
Failure mode: future refactor removes the wire-up, this sensor names
the boundary that lost its version check.
"""
from __future__ import annotations

import pathlib

import pytest


def _read_src(rel_path: str) -> str:
    """Read a source file under src/mlframe. A flat module that became a
    subpackage (``X.py`` -> ``X/__init__.py`` + submodules) is read as the
    package __init__ plus every submodule so source-pattern sensors match."""
    import mlframe as _mlframe
    _pkg = pathlib.Path(_mlframe.__file__).resolve().parent
    _path = _pkg / rel_path
    if not _path.exists() and _path.suffix == ".py":
        _sub_pkg = _path.with_suffix("")
        _init = _sub_pkg / "__init__.py"
        if _init.exists():
            parts = [_init.read_text(encoding="utf-8")]
            for _sub in sorted(_sub_pkg.glob("*.py")):
                if _sub.name != "__init__.py":
                    parts.append(_sub.read_text(encoding="utf-8"))
            return "\n".join(parts)
    return _path.read_text(encoding="utf-8")


def test_ranker_suite_per_flavor_dump_writes_sidecar():
    """training/ranking write loop calls _write_save_meta_sidecar
    immediately after each joblib.dump."""
    src = _read_src("training/ranking.py")
    # Both the helper-import line and the call must be present in the loop body.
    assert "_write_save_meta_sidecar as _wsms" in src, (
        "Wave 19 P0 #3 regression: ranker_suite no longer imports the "
        "_write_save_meta_sidecar helper; booster artefacts will be saved "
        "without library-version envelope."
    )
    assert "_wsms(artefact_path, durable=False)" in src, (
        "Wave 19 P0 #3 regression: the sidecar call site is gone from the "
        "ranker_suite per-flavor dump loop."
    )


def test_calibrator_post_dump_writes_sidecar():
    """calibration/post.py joblib.dump for each calibrator now triggers
    the sidecar write."""
    src = _read_src("calibration/post.py")
    assert "_write_save_meta_sidecar as _wsms" in src, (
        "Wave 19 P1 regression: calibration/post no longer imports the "
        "_write_save_meta_sidecar helper; per-calibrator dumps have no "
        "version envelope."
    )
    assert "_wsms(calib_fpath, durable=False)" in src, (
        "Wave 19 P1 regression: sidecar call site missing in calibrator "
        "post-hoc save loop."
    )


def test_inference_read_trained_models_validates_sidecar():
    """inference/predict.py read_trained_models loop calls
    validate_load_meta_sidecar before joblib.load."""
    src = _read_src("inference/predict.py")
    assert "validate_load_meta_sidecar as _vlms" in src, (
        "Wave 19 P1 regression: inference/predict.read_trained_models "
        "no longer imports validate_load_meta_sidecar; library-version "
        "drift will not surface to operators."
    )
    assert "_vlms(model_file, strict=False)" in src, (
        "Wave 19 P1 regression: sidecar validation call site missing in "
        "the inference read_trained_models loop."
    )


def test_sidecar_helpers_remain_importable_from_io():
    """The reusable sidecar helpers MUST stay public in training/io for
    callers (ranker_suite / calibrator / inference predict) to use them."""
    from mlframe.training.io import (
        _write_save_meta_sidecar,
        validate_load_meta_sidecar,
        load_save_meta_sidecar,
        _meta_sidecar_path,
        _collect_lib_versions,
    )
    # These are private-but-shared (single-underscore prefix) helpers; we
    # rely on the contract that the public + adjacent callers use them.
    assert callable(_write_save_meta_sidecar)
    assert callable(validate_load_meta_sidecar)
    assert callable(load_save_meta_sidecar)
    assert callable(_meta_sidecar_path)
    assert callable(_collect_lib_versions)
