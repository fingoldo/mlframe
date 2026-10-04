"""CUDA kernel sources must not use the platform-width ``long``.

NVRTC on Windows (LLP64) compiles ``long`` as 4 bytes, so ``(long)cand * n`` and ``long off = ...`` silently wrap past 2**31 elements while behaving
correctly on Linux, and no test on a small GPU ever sees it. ``long long`` is 8 bytes everywhere.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "mlframe"
_PLATFORM_LONG = re.compile(r"(?<!long )(?<!unsigned )\blong\b(?! long)(?!\s*(?:long|double))")
_LINE_COMMENT = re.compile(r"//[^\n]*")


def platform_long_uses(kernel_source: str) -> list:
    """Offending fragments of a CUDA source string (``long`` used without ``long long``), comments ignored."""
    code = _LINE_COMMENT.sub("", kernel_source)
    return [m.group(0) for m in _PLATFORM_LONG.finditer(code)]


def _kernel_strings():
    """Yield ``(path, line, text)`` for every string constant in the package that is CUDA kernel source."""
    for path in SRC.rglob("*.py"):
        if "_benchmarks" in path.parts or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "__global__" not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__global__" in node.value:
                yield path, node.lineno, node.value


def test_detector_flags_platform_long_and_accepts_long_long():
    """The detector catches ``(long)`` casts and ``long`` locals, and accepts ``long long`` / ``unsigned long long``."""
    assert platform_long_uses("const float* xi = X + (long)i * f;") == ["long"]
    assert platform_long_uses("long off = (long)level * width;") == ["long", "long"]
    assert platform_long_uses("long long off = (long long)level * width; unsigned long long k = 0;") == []
    assert platform_long_uses("int x = 0; // a long comment") == []


def test_no_cuda_kernel_source_uses_platform_width_long():
    """Every CUDA kernel source in src/mlframe spells 64-bit indices as ``long long``."""
    offenders = [f"{path.relative_to(SRC)}:{line}" for path, line, text in _kernel_strings() if platform_long_uses(text)]
    assert offenders == []
