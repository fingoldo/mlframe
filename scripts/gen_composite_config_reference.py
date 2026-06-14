#!/usr/bin/env python
"""Generate the markdown field reference for ``CompositeTargetDiscoveryConfig``.

Introspects the pydantic ``CompositeTargetDiscoveryConfig`` and emits a stable
markdown table of every field -- name, type, default, and the field's own source
comment -- to ``docs/composite_config_reference.md``.

The field docstrings live as ``#`` comment blocks immediately above each field
assignment (not in pydantic ``Field(description=...)``), so the comment text is
harvested from the class source via ``ast`` + a line-prefix scan rather than from
``FieldInfo.description``. Types and defaults come from the resolved pydantic
``model_fields`` so they always reflect the actual runtime model.

Importable + idempotent: ``render_markdown()`` returns the exact bytes-content
that would be written; ``main()`` writes it and the doc-drift test
(``tests/test_composite_config_reference_drift.py``) re-runs ``render_markdown()``
in memory and asserts the committed file matches. A new config field without a
doc entry therefore fails CI.

Run to regenerate after editing the config:
  python scripts/gen_composite_config_reference.py
Check-only (non-zero exit on drift):
  python scripts/gen_composite_config_reference.py --check
"""
from __future__ import annotations

import argparse
import ast
import inspect
import sys
import typing
from pathlib import Path

# Newline used for the committed doc + the generated source it is compared to.
# The repo is CRLF; pin it so the drift test is byte-stable on every platform.
_DOC_NEWLINE = "\r\n"

_DOC_RELPATH = Path("docs") / "composite_config_reference.md"
_CONFIG_CLASS_NAME = "CompositeTargetDiscoveryConfig"


def _load_config_class() -> type:
    """Import and return the pydantic config class (lazy, no side effects)."""
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    return CompositeTargetDiscoveryConfig


def _type_repr(annotation: object) -> str:
    """Human-readable type string for a field annotation.

    ``typing`` constructs render via ``str`` (``Optional[int]`` not the raw
    ``typing.Optional[int]`` repr noise); plain classes render by ``__name__``.
    """
    if annotation is None:
        return "None"
    if isinstance(annotation, type):
        return annotation.__name__
    text = str(annotation)
    # Strip the leading ``typing.`` qualifier that ``str(...)`` keeps so the
    # table reads ``Optional[str]`` rather than ``typing.Optional[str]``.
    return text.replace("typing.", "")


def _default_repr(field) -> str:
    """Render a field default, materialising ``default_factory`` values once."""
    factory = getattr(field, "default_factory", None)
    if factory is not None:
        try:
            value = factory()
        except Exception:  # pragma: no cover - defensive: never crash the doc gen
            return "<factory>"
        return repr(value)
    default = getattr(field, "default", None)
    # pydantic uses a sentinel (``PydanticUndefined``) for required fields; the
    # config has none, but render it as ``(required)`` if one ever appears.
    if default is None and getattr(field, "is_required", lambda: False)():
        return "(required)"
    return repr(default)


def _harvest_field_comments(config_class: type) -> dict[str, str]:
    """Map field name -> its leading ``#`` comment block from the class source.

    Walks every class in the MRO that declares fields (the config plus any carved
    base classes, whose source lives in a separate module), so a base-class field's
    source comment is still harvested after a monolith split. A subclass comment
    wins over a base comment for the same field name. See ``_harvest_one`` for the
    per-class AST walk.
    """
    merged: dict[str, str] = {}
    # Walk base -> derived so a derived-class override wins on name collision.
    for klass in reversed(config_class.__mro__):
        if klass is object:
            continue
        try:
            merged.update(_harvest_one(klass))
        except (OSError, TypeError, SyntaxError, IndexError, AssertionError):
            # No retrievable / parseable source for this base (builtins, C-exts,
            # pydantic internals): skip; its fields simply carry no comment.
            continue
    return merged


def _harvest_one(config_class: type) -> dict[str, str]:
    """Harvest field source-comments from a single class body.

    Finds each ``AnnAssign`` (``name: T = ...``) field via ``ast``, then scans the
    physical source lines immediately above that statement, collecting the
    contiguous run of ``#`` comment lines as the field's doc. A blank line breaks
    the run. Comments belonging to the field above (separated by its own
    assignment) are naturally excluded by the ``AnnAssign`` line bounds.
    """
    source = inspect.getsource(config_class)
    lines = source.splitlines()
    tree = ast.parse(source)
    class_def = tree.body[0]
    assert isinstance(class_def, ast.ClassDef)

    # Line number (1-based, into ``lines``) where each field assignment starts.
    field_start_line: dict[str, int] = {}
    # Track every line that is part of a field statement so the comment scan does
    # not walk up into a preceding multi-line ``Field(default_factory=...)`` body.
    statement_lines: set[int] = set()
    for node in class_def.body:
        for ln in range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1):
            statement_lines.add(ln)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            field_start_line[node.target.id] = node.lineno

    comments: dict[str, str] = {}
    for name, start in field_start_line.items():
        block: list[str] = []
        idx = start - 1  # 1-based -> walk upward from the line above the field.
        while idx >= 1:
            text = lines[idx - 1]
            stripped = text.strip()
            if stripped.startswith("#"):
                block.append(stripped.lstrip("#").strip())
                idx -= 1
                continue
            # Stop at a blank line or any non-comment code (incl. the prior
            # field's own statement lines).
            break
        block.reverse()
        comments[name] = " ".join(p for p in block if p)
    return comments


def _md_escape(text: str) -> str:
    """Escape pipe + collapse newlines so a comment is one safe markdown cell."""
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def render_markdown() -> str:
    """Return the full markdown reference content (idempotent, deterministic).

    Field order follows ``model_fields`` insertion order, which mirrors the
    declaration order in the source, so re-running on an unchanged config yields
    byte-identical output.
    """
    config_class = _load_config_class()
    fields = config_class.model_fields
    comments = _harvest_field_comments(config_class)

    parts: list[str] = []
    parts.append(f"# `{_CONFIG_CLASS_NAME}` field reference")
    parts.append("")
    parts.append(
        "Auto-generated by `scripts/gen_composite_config_reference.py` from "
        f"`mlframe.training.configs.{_CONFIG_CLASS_NAME}`. Do not edit by hand -- "
        "regenerate after changing the config (a doc-drift test fails CI otherwise)."
    )
    parts.append("")
    parts.append(
        f"The config exposes **{len(fields)} fields**. Each row's description is the "
        "field's own source comment."
    )
    parts.append("")
    parts.append("## Passing the config")
    parts.append("")
    parts.append(
        "`train_mlframe_models_suite(..., composite_target_discovery_config=...)` "
        "accepts EITHER a `CompositeTargetDiscoveryConfig` instance OR a plain "
        "`dict` of field overrides. The dict is converted via `_ensure_config`, "
        "which is STRICT: an unknown key (e.g. a typo `enabledd=True`) raises "
        "`ValueError` instead of being silently dropped, so misspelled knobs fail "
        "loud. `None` uses all defaults."
    )
    parts.append("")
    parts.append("```python")
    parts.append("# instance form")
    parts.append("from mlframe.training.configs import CompositeTargetDiscoveryConfig")
    parts.append("cfg = CompositeTargetDiscoveryConfig(enabled=True, auto_base_top_k=5)")
    parts.append("train_mlframe_models_suite(..., composite_target_discovery_config=cfg)")
    parts.append("")
    parts.append("# equivalent dict form (validated + type-checked the same way)")
    parts.append(
        "train_mlframe_models_suite("
        '..., composite_target_discovery_config={"enabled": True, "auto_base_top_k": 5})'
    )
    parts.append("```")
    parts.append("")
    parts.append(
        "`MLFRAME_DISABLE_COMPOSITE=1` forces composite discovery off regardless of "
        "the config."
    )
    parts.append("")
    parts.append("## Fields")
    parts.append("")
    parts.append("| Field | Type | Default | Description |")
    parts.append("| --- | --- | --- | --- |")
    for name, field in fields.items():
        type_str = _type_repr(field.annotation)
        default_str = _default_repr(field)
        comment = comments.get(name, "")
        parts.append(
            f"| `{name}` | `{_md_escape(type_str)}` | `{_md_escape(default_str)}` "
            f"| {_md_escape(comment)} |"
        )
    parts.append("")
    return _DOC_NEWLINE.join(parts)


def _doc_path() -> Path:
    """Absolute path to the committed doc (repo-root relative to this script)."""
    return (Path(__file__).resolve().parent.parent / _DOC_RELPATH).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the committed doc differs from the generated content.",
    )
    args = parser.parse_args(argv)

    content = render_markdown()
    path = _doc_path()

    if args.check:
        if not path.exists():
            print(f"MISSING: {path} does not exist; run the generator.", file=sys.stderr)
            return 1
        existing = path.read_text(encoding="utf-8", newline="")
        if existing != content:
            print(
                f"DRIFT: {path} is out of date; run "
                "`python scripts/gen_composite_config_reference.py`.",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {path} is up to date.")
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    # ``newline=""`` keeps the explicit CRLF in ``content`` exactly (no translation).
    path.write_text(content, encoding="utf-8", newline="")
    print(f"WROTE: {path} ({len(content)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
