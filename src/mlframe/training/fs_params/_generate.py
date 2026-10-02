"""Regenerate (or verify) the strict-parameter model modules in this package from the selector constructors.

    python -m mlframe.training.fs_params._generate            # rewrite every generated module
    python -m mlframe.training.fs_params._generate --check    # exit 1 if any module on disk differs from what would be generated
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from pyutilz.dev.signature_models import render_model_source

from ._spec import SPECS, SelectorSpec

PACKAGE_DIR = Path(__file__).parent

# Annotations naming a class from any other module are written as ``Any``: the generated modules must not import the selectors' packages
# (MRMR alone takes seconds), which is the whole point of generating them instead of validating against the live signature.
LIGHT_MODULES = ("collections", "numpy", "pandas", "sklearn")


def render(spec: SelectorSpec) -> str:
    """Source of the module for one selector."""
    return render_model_source(
        spec.target(),
        spec.class_name,
        exclude=spec.exclude,
        overrides=spec.all_overrides(),
        regenerate_hint="python -m mlframe.training.fs_params._generate",
        allowed_modules=LIGHT_MODULES,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write or check every generated module; ``--check`` returns 1 and names the stale files."""
    parser = argparse.ArgumentParser(prog="python -m mlframe.training.fs_params._generate")
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 if a module is out of date")
    args = parser.parse_args(argv)
    stale = []
    for spec in SPECS:
        path = PACKAGE_DIR / f"{spec.key}.py"
        source = render(spec)
        if args.check:
            current = path.read_text(encoding="utf-8").replace("\r\n", "\n") if path.exists() else ""
            if current != source:
                stale.append(path.name)
        else:
            path.write_text(source, encoding="utf-8", newline="\n")
    if stale:
        sys.stdout.write(f"out of date: {', '.join(stale)}; run python -m mlframe.training.fs_params._generate\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
