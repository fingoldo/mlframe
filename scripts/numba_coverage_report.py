"""Parse a coverage.xml produced by the nightly NUMBA_DISABLE_JIT=1 workflow and emit a structured JSON report.

Walks every ``@njit`` / ``@cuda.jit`` -decorated module under ``src/mlframe/`` and produces a per-module diff of:

* lines covered when numba JIT was disabled (kernel bodies visible)
* lines covered in the regular daily run if a baseline coverage XML is provided
* the delta (extra lines newly covered)

The output JSON is a list of records sorted by absolute delta descending so the worst-blinded kernel surfaces first.

Invocation::

    python scripts/numba_coverage_report.py \
        --numba-disabled coverage.xml \
        --baseline coverage-baseline.xml \
        --out audit/critique_2026_05_24/reports/numba_coverage_<date>.json

Both ``--baseline`` and ``--out`` are optional; without ``--baseline`` only the numba-disabled snapshot is emitted, without
``--out`` the JSON is printed to stdout.

The script intentionally has zero hard dependencies beyond the stdlib so it can be invoked from the
``numba-coverage-nightly`` workflow without an extra ``pip install`` step. The XML it reads is the standard cobertura-style
schema produced by ``pytest --cov-report=xml``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_NUMBA_DECORATOR_RE = re.compile(r"^\s*@(?:numba\.)?(?:njit|cuda\.jit|jit)\b")


@dataclass
class ModuleCoverage:
    """One record per source module that hosts at least one @njit / @cuda.jit decorator."""

    module: str  # dotted path, e.g. ``mlframe.feature_selection.filters.permutation``
    file_path: str  # source-relative path, e.g. ``src/mlframe/feature_selection/filters/permutation.py``
    statements: int
    statements_covered_numba_disabled: int
    statements_covered_baseline: Optional[int]
    line_rate_numba_disabled: float
    line_rate_baseline: Optional[float]
    delta_lines: Optional[int]
    delta_rate: Optional[float]
    numba_decorator_count: int


def _normalise_path(raw: str) -> str:
    """Lower-noise file-path key: forward slashes, drop leading ``./``, drop a single leading ``src/`` segment.

    Both cobertura XMLs and the on-disk scan are normalised through this helper so a coverage entry stored as
    ``src/mlframe/feature_selection/filters/permutation.py`` matches a scan entry stored as
    ``mlframe/feature_selection/filters/permutation.py``."""
    norm = raw.replace("\\", "/").lstrip("./")
    if norm.startswith("src/"):
        norm = norm[len("src/") :]
    return norm


def _parse_cobertura(xml_path: Path) -> dict[str, tuple[int, int]]:
    """Return {normalised_file_path: (statements, covered)} from a cobertura ``coverage.xml``.

    File paths are normalised via ``_normalise_path`` so cobertura ``src/mlframe/...`` entries collide with on-disk
    scan paths of the form ``mlframe/...``. The cobertura format stores per-file ``<class>`` nodes with ``filename``
    + per-line ``<line>`` entries carrying a ``hits`` attribute; we treat any line with hits >= 1 as covered.
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"coverage XML not found: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result: dict[str, tuple[int, int]] = {}
    for cls in root.iter("class"):
        filename = cls.attrib.get("filename", "")
        if not filename:
            continue
        norm = _normalise_path(filename)
        lines_node = cls.find("lines")
        if lines_node is None:
            continue
        total = 0
        covered = 0
        for line in lines_node.findall("line"):
            total += 1
            hits = int(line.attrib.get("hits", "0"))
            if hits >= 1:
                covered += 1
        # Cobertura emits one <class> per python class; multiple classes per file collapse into one row.
        prev_total, prev_covered = result.get(norm, (0, 0))
        result[norm] = (prev_total + total, prev_covered + covered)
    return result


def _scan_numba_modules(src_root: Path) -> dict[str, int]:
    """Walk ``src_root`` and return ``{normalised_path: njit_decorator_count}`` for every .py with @njit / @cuda.jit.

    Paths are reported relative to ``src_root.parent`` and then run through ``_normalise_path`` so they match the
    cobertura-side keys regardless of whether the XML stored ``src/mlframe/...`` or ``mlframe/...``."""
    out: dict[str, int] = {}
    for path in src_root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        count = sum(1 for line in text.splitlines() if _NUMBA_DECORATOR_RE.match(line))
        if count:
            rel = path.relative_to(src_root.parent).as_posix()
            out[_normalise_path(rel)] = count
    return out


def _file_path_to_dotted(file_path: str) -> str:
    p = _normalise_path(file_path)
    if p.endswith(".py"):
        p = p[: -len(".py")]
    return p.replace("/", ".")


def build_report(
    numba_disabled_xml: Path,
    baseline_xml: Optional[Path],
    src_root: Path,
) -> list[ModuleCoverage]:
    """Combine the numba-disabled XML, optional baseline XML, and on-disk module scan into one report."""
    numba_cov = _parse_cobertura(numba_disabled_xml)
    baseline_cov: Optional[dict[str, tuple[int, int]]] = None
    if baseline_xml is not None:
        baseline_cov = _parse_cobertura(baseline_xml)

    numba_modules = _scan_numba_modules(src_root)

    rows: list[ModuleCoverage] = []
    for file_path, decorator_count in sorted(numba_modules.items()):
        nd_stmts, nd_covered = numba_cov.get(file_path, (0, 0))
        if nd_stmts == 0:
            # File not present in coverage report (probably never imported by the test selection); record zeros so the
            # nightly run surfaces the gap rather than silently dropping the module.
            line_rate_nd = 0.0
        else:
            line_rate_nd = nd_covered / nd_stmts

        bl_covered: Optional[int] = None
        line_rate_bl: Optional[float] = None
        delta_lines: Optional[int] = None
        delta_rate: Optional[float] = None
        if baseline_cov is not None:
            bl_stmts, bl_covered_val = baseline_cov.get(file_path, (0, 0))
            bl_covered = bl_covered_val
            line_rate_bl = (bl_covered_val / bl_stmts) if bl_stmts > 0 else 0.0
            delta_lines = nd_covered - bl_covered_val
            delta_rate = line_rate_nd - line_rate_bl

        rows.append(
            ModuleCoverage(
                module=_file_path_to_dotted(file_path),
                file_path=file_path,
                statements=nd_stmts,
                statements_covered_numba_disabled=nd_covered,
                statements_covered_baseline=bl_covered,
                line_rate_numba_disabled=round(line_rate_nd, 4),
                line_rate_baseline=round(line_rate_bl, 4) if line_rate_bl is not None else None,
                delta_lines=delta_lines,
                delta_rate=round(delta_rate, 4) if delta_rate is not None else None,
                numba_decorator_count=decorator_count,
            )
        )

    # Sort by absolute delta_lines descending; ties broken by numba_decorator_count desc (more kernels = higher risk).
    def _sort_key(row: ModuleCoverage) -> tuple[float, int]:
        delta = abs(row.delta_lines) if row.delta_lines is not None else row.statements_covered_numba_disabled
        return (-float(delta), -row.numba_decorator_count)

    rows.sort(key=_sort_key)
    return rows


def _emit(rows: list[ModuleCoverage], out_path: Optional[Path]) -> None:
    payload = {
        "schema_version": 1,
        "rows": [asdict(r) for r in rows],
        "summary": {
            "modules": len(rows),
            "total_decorators": sum(r.numba_decorator_count for r in rows),
            "total_statements_covered_numba_disabled": sum(r.statements_covered_numba_disabled for r in rows),
            "total_delta_lines": sum(r.delta_lines or 0 for r in rows),
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if out_path is None:
        sys.stdout.write(text + "\n")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    logger.info("Wrote numba-coverage report to %s (%d modules)", out_path, len(rows))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numba-disabled", required=True, type=Path, help="coverage.xml from a NUMBA_DISABLE_JIT=1 run")
    parser.add_argument("--baseline", type=Path, default=None, help="optional baseline coverage.xml (regular daily run)")
    parser.add_argument("--src-root", type=Path, default=Path("src/mlframe"), help="source root to scan for @njit / @cuda.jit decorators")
    parser.add_argument("--out", type=Path, default=None, help="output JSON path; defaults to stdout")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    rows = build_report(
        numba_disabled_xml=args.numba_disabled,
        baseline_xml=args.baseline,
        src_root=args.src_root,
    )
    _emit(rows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
