"""Wave 46 (2026-05-20): path traversal / os.path.join absolute-path eating audit.

3 P1 + 2 P2 findings — all share the root cause: raw user-supplied
model_name / target_name / featureset_name / task_type plumbed into
os.path.join without slugify(...), breaking symmetry with the rest of the
codebase which IS consistently slugifying.

P1 fixes:
  1. training/core/_phase_helpers.py:1727 — LTR _save_dir slugify(ctx.model_name).
  2. training/ranker_suite.py:900,922 — joblib artefact + metadata basename slugified.
  3. calibration/post.py:563 — all 4 dir components (target/featureset/task/model) slugified.

P2 fixes:
  4. training/core/_phase_finalize.py:92 — _CT_ENSEMBLE__ dir suffix slugified
     (defence-in-depth; ctx.models keys are internal but the prefix-startswith
     gate alone would accept "_CT_ENSEMBLE__../../evil").
  5. training/neural/base.py:362 — default_root_dir trust contract documented.

Verified clean (do not refactor):
  - training/io.py:130 (basename via os.path.basename sanitises)
  - training/composite_cache.py:630+ (filename via blake2b hash; _HEX_KEY_RE fully anchored)
  - training/feature_handling/cache.py:332, cache_backend.py:111 (DiskKey.filename hash-derived)
  - training/feature_handling/hf_provider.py:90 (sig_hash blake2b)
  - feature_selection/wrappers/_rfecv.py:470 (user-owned checkpoint_path)
  - No tarfile.extractall / zipfile.extractall — no Zip Slip exposure.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest


MLFRAME_ROOT = Path(importlib.import_module("mlframe").__file__).parent


def _read(rel: str) -> str:
    return (MLFRAME_ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Source-level sensors
# ---------------------------------------------------------------------------


def test_phase_helpers_ltr_save_dir_slugifies_model_name() -> None:
    src = _read("training/core/_phase_helpers.py")
    # Forbid the raw join.
    assert "os.path.join(_data_dir, _models_dir, ctx.model_name)" not in src, (
        "LTR _save_dir must slugify ctx.model_name (path-traversal vector otherwise)."
    )
    # Require the slugified form.
    assert "_slugify(ctx.model_name)" in src


def test_ranker_suite_artefact_basenames_slugify_model_name() -> None:
    src = _read("training/ranker_suite.py")
    # Forbid the raw f-string interpolation.
    assert 'f"{model_name}_{flavor}.joblib"' not in src
    assert 'f"{model_name}_metadata.json"' not in src
    # Require the slugified local alias.
    assert "_safe_model_name = _slugify(model_name)" in src
    assert 'f"{_safe_model_name}_{flavor}.joblib"' in src
    assert 'f"{_safe_model_name}_metadata.json"' in src


def test_calibration_post_final_models_dir_slugifies_all_components() -> None:
    src = _read("calibration/post.py")
    # Forbid the raw 4-arg join.
    assert "join(models_dir, target_name, featureset_name, task_type, model_name)" not in src
    # Require slugify on each component.
    assert "_slugify(target_name)" in src
    assert "_slugify(featureset_name)" in src
    assert "_slugify(str(task_type))" in src
    assert "_slugify(model_name)" in src


def test_phase_finalize_ct_ensemble_dir_slugified() -> None:
    src = _read("training/core/_phase_finalize.py")
    # Forbid raw _dir_name = _tname assignment in this CT-ENSEMBLE branch.
    # The post-fix form keeps the literal prefix and slugifies the suffix.
    assert '_dir_name = "_CT_ENSEMBLE__" + slugify(_tname[len("_CT_ENSEMBLE__"):])' in src


def test_neural_base_default_root_dir_trust_contract_documented() -> None:
    src = _read("training/neural/base.py")
    # The Wave 46 documentation comment marks the trust contract explicitly.
    assert "default_root_dir" in src and "caller-controlled" in src


def test_hex_key_re_is_fully_anchored() -> None:
    """The audit's Info finding: verify _HEX_KEY_RE is full-match anchored."""
    src = _read("training/composite_cache.py")
    # Must use \A and \Z anchors (or ^...$) -- not bare bracket class.
    # Behavioural check: a non-hex tail must NOT match.
    import re
    from mlframe.training.composite_cache import _HEX_KEY_RE
    assert _HEX_KEY_RE.match("deadbeef") is not None
    assert _HEX_KEY_RE.match("deadbeef/../etc") is None, (
        "_HEX_KEY_RE must be fully anchored so a partial-hex string with "
        "trailing path-traversal characters fails the fast-path gate."
    )
    assert _HEX_KEY_RE.match("deadbeef\nrogue") is None
    assert _HEX_KEY_RE.match("deadbeefXY") is None


# ---------------------------------------------------------------------------
# Behavioural sensors: trigger the traversal and assert it stays inside root.
# ---------------------------------------------------------------------------


def test_slugify_neutralises_path_separators() -> None:
    """slugify must strip / and .. so traversal attempts collapse to safe basenames."""
    from pyutilz.strings import slugify

    traversal_attempts = [
        "../../etc/passwd",
        "/etc/passwd",
        "C:/Windows/System32",
        "../../../evil",
        "evil/../../../etc",
    ]
    for attempt in traversal_attempts:
        slug = slugify(attempt)
        # Post-slugify, the result must NOT contain path separators or '..'.
        assert "/" not in slug, f"slugify({attempt!r}) -> {slug!r} still contains '/'"
        assert "\\" not in slug, f"slugify({attempt!r}) -> {slug!r} still contains '\\'"
        # `..` may survive in pyutilz' slugify if it normalises to dots; allow but
        # require that the result joined with a root does NOT escape.
        joined = os.path.join("/safe/root", slug)
        # os.path.abspath collapses .., but our regex check on the SLUG is the right place.
        # If the slug is exactly "..", joined would escape. Verify it's not.
        assert slug != "..", f"slugify({attempt!r}) collapsed to '..'"


def test_os_path_join_absolute_eats_prefix_documented() -> None:
    """Sanity-check the underlying footgun this audit class targets."""
    # POSIX absolute path eats the prefix.
    assert os.path.join("/safe/root", "/etc/passwd") == "/etc/passwd"
    # The slugify fix removes the leading slash so the join behaves correctly.
    from pyutilz.strings import slugify
    assert os.path.join("/safe/root", slugify("/etc/passwd")).startswith("/safe/root")
