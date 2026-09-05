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
import ast
import os
from pathlib import Path

MLFRAME_ROOT = Path(importlib.import_module("mlframe").__file__).parent


def _read(rel: str) -> str:
    """Read a source file under src/mlframe. A flat module that became a
    subpackage (``X.py`` -> ``X/__init__.py`` + submodules) is read as the
    package __init__ plus every submodule so source-pattern sensors match."""
    _path = MLFRAME_ROOT / rel
    if not _path.exists() and _path.suffix == ".py":
        # Monolith-split compat: the flat module became a subpackage
        # (``X.py`` -> ``X/__init__.py`` + submodules). Read __init__ + every submodule.
        _pkg = _path.with_suffix("")
        _init = _pkg / "__init__.py"
        if _init.exists():
            _parts = [_init.read_text(encoding="utf-8")]
            for _sub in sorted(_pkg.glob("*.py")):
                if _sub.name != "__init__.py":
                    _parts.append(_sub.read_text(encoding="utf-8"))
            primary = "\n".join(_parts)
        else:
            primary = _path.read_text(encoding="utf-8")
    else:
        primary = _path.read_text(encoding="utf-8")
    return primary


# ---------------------------------------------------------------------------
# Source-level sensors
# ---------------------------------------------------------------------------


# Names that carry caller-controlled text into a path. A traversal in any of them escapes the
# artefact root ("../../evil"), or eats the prefix entirely when absolute ("/foo", "C:/x"), because
# os.path.join drops everything before an absolute component.
_UNTRUSTED_PATH_COMPONENTS = frozenset({"model_name", "target_name", "featureset_name", "_tname", "_target_name", "_model_name"})


def _leaf_name(node):
    """The trailing identifier of a Name or Attribute expression, or None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_slugified(node):
    """Is this expression wrapped in a slugify() call (however the module aliased it)?"""
    return isinstance(node, ast.Call) and (_leaf_name(node.func) or "").endswith("slugify")


def _is_os_path_join(func):
    """os.path.join only -- NOT str.join, which shares the attribute name and joins no paths."""
    if not (isinstance(func, ast.Attribute) and func.attr == "join"):
        return False
    return _leaf_name(func.value) == "path"


def _unslugified_path_uses(tree):
    """Yield (lineno, name) for every caller-controlled name reaching a path unslugified."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_os_path_join(node.func):
            # os.path.join's FIRST argument is the root; a traversal there is the caller's own root.
            for arg in node.args[1:]:
                if _is_slugified(arg):
                    continue
                name = _leaf_name(arg)
                if name in _UNTRUSTED_PATH_COMPONENTS:
                    yield node.lineno, name
        elif isinstance(node, ast.JoinedStr):
            # An artefact BASENAME built by interpolation -- "{model_name}_{flavor}.joblib".
            # Must END in an artefact extension. Merely CONTAINING a separator matched dict keys
            # ("{_ttype}/{_tname}/{_mn}") and error messages ("Drop NaN/inf rows"), and a security
            # check that cries wolf gets muted.
            literals = "".join(v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
            if not literals.endswith((".joblib", ".json", ".pkl", ".bin", ".parquet")):
                continue
            for value in node.values:
                if not isinstance(value, ast.FormattedValue) or _is_slugified(value.value):
                    continue
                name = _leaf_name(value.value)
                if name in _UNTRUSTED_PATH_COMPONENTS:
                    yield node.lineno, name


def test_no_caller_controlled_name_reaches_a_path_unslugified() -> None:
    """The rule the four removed spelling-pins each covered one site of.

    Behavioural since 2026-09-03. Those asserted that `os.path.join(_data_dir, _models_dir,
    ctx.model_name)` is absent and `_slugify(ctx.model_name)` present, that
    `f"{model_name}_{flavor}.joblib"` is absent and `f"{_safe_model_name}_{flavor}.joblib"`
    present, and so on -- four hand-maintained pairs, each naming one call site by its exact
    current spelling, each already rewritten once when its module was split, and none of them
    saying anything about a FIFTH site added tomorrow.

    This reads the parse tree instead: every caller-controlled name reaching os.path.join past the
    root argument, or interpolated into an artefact basename, must be wrapped in slugify. The
    behavioural sensors below already pin what slugify itself guarantees; this pins that the call
    sites use it.
    """
    offenders = []
    unparseable = []
    scanned = 0
    for path in sorted(MLFRAME_ROOT.rglob("*.py")):
        rel = path.relative_to(MLFRAME_ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            # Reported, never skipped in silence. For a security scanner, "scanned nothing" and
            # "found nothing" produce the same green tick, and a file that stops parsing is exactly
            # the one somebody just edited.
            unparseable.append(f"{rel}: {exc}")
            continue
        scanned += 1
        for lineno, name in _unslugified_path_uses(tree):
            offenders.append(f"{rel}:{lineno} ({name})")

    assert not unparseable, "modules this check could not parse, so it did not scan them: " + ", ".join(unparseable)
    assert scanned > 100, f"only {scanned} modules scanned -- the tree walk is not reaching mlframe/"
    assert not offenders, "caller-controlled names reach a path unslugified at: " + ", ".join(offenders)


def test_neural_base_default_root_dir_trust_contract_documented() -> None:
    """Neural base default root dir trust contract documented."""
    src = _read("training/neural/base.py")
    # The Wave 46 documentation comment marks the trust contract explicitly.
    assert "default_root_dir" in src and "caller-controlled" in src


def test_hex_key_re_is_fully_anchored() -> None:
    """The audit's Info finding: verify _HEX_KEY_RE is full-match anchored."""
    _read("training/composite/cache_store.py")
    # Must use \A and \Z anchors (or ^...$) -- not bare bracket class.
    # Behavioural check: a non-hex tail must NOT match.
    from mlframe.training.composite.cache_store import _HEX_KEY_RE

    assert _HEX_KEY_RE.match("deadbeef") is not None
    assert (
        _HEX_KEY_RE.match("deadbeef/../etc") is None
    ), "_HEX_KEY_RE must be fully anchored so a partial-hex string with trailing path-traversal characters fails the fast-path gate."
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
        os.path.join("/safe/root", slug)
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
