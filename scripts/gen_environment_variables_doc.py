#!/usr/bin/env python
"""Generate ``docs/ENVIRONMENT_VARIABLES.md``: every environment variable read anywhere in ``src/mlframe/``.

The scan is ``py_ci_shared.readme_env_var_parity.find_env_var_reads``, the same one the parity meta-test uses, so the
inventory and the check cannot disagree. It sees literal names, names bound to a module-level string constant
(``_ENV_VAR = "MLFRAME_..."``) and reads through the package's own helpers (``READER_FUNCS``); the constant-bound
names were missing from the hand-run inventory this replaces, among them the switches that turn safety mechanisms off.

Run to regenerate:
  python scripts/gen_environment_variables_doc.py
Check-only (non-zero exit on drift):
  python scripts/gen_environment_variables_doc.py --check
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

from py_ci_shared.readme_env_var_parity import find_env_var_reads

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "mlframe"
DOC = ROOT / "docs" / "ENVIRONMENT_VARIABLES.md"

#: Keyword arguments that hand an env-var name to a library which reads it (``safe_load(..., env_var=_ENV_VAR)``).
DELEGATED_KEYWORDS = frozenset({"env_var"})

#: The package's own env readers, each taking the variable name as its first argument.
READER_FUNCS = frozenset({"env_flag", "env_int", "env_float", "_read_int_env", "_float_env", "_env_gpu_default_on"})

HEADER = """## Environment variables

Every environment variable read anywhere in `src/mlframe/`, generated from the source by `scripts/gen_environment_variables_doc.py` (name, first read site, and its default when one is passed at that site). A read counts whether the name is written inline, bound to a module-level string constant, or passed to one of the package's env helpers (`env_flag` / `env_int` / `env_float` and a few module-local ones). This is a mechanically-generated inventory, not a hand-written guide: it documents *that* a var is read and its default, not *why* it exists or what it tunes; see the linked file for that.

| Variable | Default | First read at |
|---|---|---|
"""


def source_files() -> list[Path]:
    """Every ``.py`` under ``src/mlframe`` in a stable order, so "first read" does not depend on the filesystem."""
    return sorted((p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts), key=lambda p: p.relative_to(SRC).as_posix())


def _module_constants(tree: ast.AST) -> dict[str, ast.expr]:
    """Module-level ``NAME = <literal>`` bindings, used to show a default passed by name as its value."""
    out: dict[str, ast.expr] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            out[node.target.id] = node.value
    return out


def _literal_source(expr: ast.expr) -> str | None:
    try:
        ast.literal_eval(expr)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    return ast.unparse(expr)


def _delegated_reads(files: list[Path]) -> dict[str, tuple[Path, int, str | None]]:
    """Env vars whose name is handed to a library via a ``DELEGATED_KEYWORDS`` keyword, resolved through constants."""
    out: dict[str, tuple[Path, int, str | None]] = {}
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        consts = _module_constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg not in DELEGATED_KEYWORDS:
                    continue
                value = consts.get(kw.value.id, kw.value) if isinstance(kw.value, ast.Name) else kw.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    out.setdefault(value.value, (path, node.lineno, None))
    return out


def render_markdown() -> str:
    files = source_files()
    reads = find_env_var_reads(files, reader_funcs=READER_FUNCS, allow_unparsed=True)
    for name, site in _delegated_reads(files).items():
        reads.setdefault(name, site)
    trees: dict[Path, dict[str, ast.expr]] = {}
    rows = []
    for name in sorted(reads):
        path, line, default = reads[name]
        if default is not None and default.isidentifier():
            if path not in trees:
                trees[path] = _module_constants(ast.parse(path.read_text(encoding="utf-8")))
            bound = trees[path].get(default)
            resolved = _literal_source(bound) if bound is not None else None
            default = resolved if resolved is not None else default
        rel = path.resolve().relative_to(ROOT).as_posix()
        shown = f"`{default}`" if default is not None else "—"
        rows.append(f"| `{name}` | {shown.replace('|', '&#124;')} | [{rel}](../{rel}#L{line}) |")
    return HEADER + "\n".join(rows) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed doc differs from the generated one")
    args = parser.parse_args(argv)
    text = render_markdown()
    current = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    if args.check:
        if current != text:
            sys.stderr.write(f"{DOC.relative_to(ROOT)} is stale; run python scripts/gen_environment_variables_doc.py\n")
            return 1
        return 0
    DOC.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
