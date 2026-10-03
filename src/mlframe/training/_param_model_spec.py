"""Shared machinery of the generated strict-parameter packages (``fs_params``, ``calibration_params``).

A ``SelectorSpec`` names a callable (a class or a function); a package's ``_generate`` renders one module per spec from the callable's signature
with ``pyutilz.dev.signature_models``, so the config models cannot drift from the signatures (a sync test per package fails when they do).
Enum-like parameters are constrained with the callee's OWN accepted-value tuples, so the allowed values have one source.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from pyutilz.dev.signature_models import render_model_source

# Annotations naming a class from any other module are written as ``Any``: the generated modules must not import the callee's package
# (MRMR alone takes seconds), which is the whole point of generating them instead of validating against the live signature.
LIGHT_MODULES = ("collections", "numpy", "pandas", "sklearn")

#: ``[tool.black] line-length`` of pyproject.toml.
BLACK_LINE_LENGTH = 160


def literal_source(values: Tuple[Any, ...]) -> str:
    """``Literal[...]`` source for ``values``; a ``None`` member makes the whole annotation ``Optional``."""
    present = tuple(v for v in values if v is not None)
    text = f"Literal[{', '.join(repr(v) for v in present)}]"
    return f"Optional[{text}]" if None in values else text


@dataclass(frozen=True)
class SelectorSpec:
    """One selector: where its constructor lives, which parameters the suite owns (excluded), and the enum / constraint overrides."""

    key: str
    class_name: str
    target_module: str
    target_attr: str
    exclude: Tuple[str, ...] = ()
    enums: Callable[[], Dict[str, Tuple[Any, ...]]] = field(default=lambda: {})
    overrides: Dict[str, str] = field(default_factory=dict)

    def target(self) -> Any:
        """The selector class (imported lazily; MRMR alone costs seconds)."""
        return getattr(importlib.import_module(self.target_module), self.target_attr)

    def all_overrides(self) -> Dict[str, str]:
        """Enum-derived ``Literal`` annotations plus the hand-written ``overrides`` (the latter win)."""
        out = {name: literal_source(values) for name, values in self.enums().items()}
        out.update(self.overrides)
        return out


def render_spec(spec: SelectorSpec, regenerate_hint: str) -> str:
    """Source of the module for one spec."""
    source = str(
        render_model_source(
            spec.target(),
            spec.class_name,
            exclude=spec.exclude,
            overrides=spec.all_overrides(),
            regenerate_hint=regenerate_hint,
            allowed_modules=LIGHT_MODULES,
        )
    )
    import black

    # Formatted the way the repository formats its code, so the committed module equals the render byte for byte and ``--check`` stays exact.
    return black.format_str(source, mode=black.Mode(line_length=BLACK_LINE_LENGTH))


def run_generator(specs: Sequence[SelectorSpec], package_dir: Path, module_name: str, argv: Optional[Sequence[str]] = None) -> int:
    """Write (or with ``--check`` verify) one module per spec in ``package_dir``; ``--check`` returns 1 and names the stale files."""
    prog = f"python -m {module_name}"
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 if a module is out of date")
    args = parser.parse_args(argv)
    stale = []
    for spec in specs:
        path = package_dir / f"{spec.key}.py"
        source = render_spec(spec, prog)
        if args.check:
            current = path.read_text(encoding="utf-8").replace("\r\n", "\n") if path.exists() else ""
            if current != source:
                stale.append(path.name)
        else:
            path.write_text(source, encoding="utf-8", newline="\n")
    if stale:
        sys.stdout.write(f"out of date: {', '.join(stale)}; run {prog}\n")
        return 1
    return 0
