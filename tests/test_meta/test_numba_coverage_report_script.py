"""Sensor for ``scripts/numba_coverage_report.py``: end-to-end parse + diff a tiny synthetic cobertura XML pair and
assert the resulting JSON has the expected fields, per-module decorator counts, and delta-lines ranking.

Closes the W10 task that asks for a structured JSON bench-report alongside the AP5 nightly workflow so the
NUMBA_DISABLE_JIT=1 coverage XML is consumable by reviewers without manual XML parsing."""

from __future__ import annotations

import orjson
import subprocess  # nosec B404 -- test-only local trusted subprocess invocation (fixed argv, no shell, no untrusted input)
import sys
import textwrap
from pathlib import Path

import pytest


_FAKE_COBERTURA_NUMBA_DISABLED = textwrap.dedent(
    """\
    <?xml version="1.0" ?>
    <coverage>
      <packages>
        <package>
          <classes>
            <class filename="src/mlframe/feature_selection/filters/permutation.py">
              <lines>
                <line number="1" hits="1"/>
                <line number="2" hits="1"/>
                <line number="3" hits="1"/>
                <line number="4" hits="1"/>
                <line number="5" hits="0"/>
              </lines>
            </class>
          </classes>
        </package>
      </packages>
    </coverage>
    """
)

_FAKE_COBERTURA_BASELINE = textwrap.dedent(
    """\
    <?xml version="1.0" ?>
    <coverage>
      <packages>
        <package>
          <classes>
            <class filename="src/mlframe/feature_selection/filters/permutation.py">
              <lines>
                <line number="1" hits="1"/>
                <line number="2" hits="0"/>
                <line number="3" hits="0"/>
                <line number="4" hits="0"/>
                <line number="5" hits="0"/>
              </lines>
            </class>
          </classes>
        </package>
      </packages>
    </coverage>
    """
)


@pytest.fixture
def tmp_src_root(tmp_path: Path) -> Path:
    """Create a tiny synthetic src tree with one @njit decorator so the scan picks up exactly one module."""
    src = tmp_path / "src" / "mlframe" / "feature_selection" / "filters"
    src.mkdir(parents=True)
    target = src / "permutation.py"
    target.write_text(
        "import numba\n\n@numba.njit\ndef kernel():\n    return 1\n",
        encoding="utf-8",
    )
    return tmp_path / "src" / "mlframe"


def test_numba_coverage_report_emits_per_module_delta(tmp_path: Path, tmp_src_root: Path) -> None:
    nd_xml = tmp_path / "nd.xml"
    bl_xml = tmp_path / "bl.xml"
    out_json = tmp_path / "report.json"
    nd_xml.write_text(_FAKE_COBERTURA_NUMBA_DISABLED, encoding="utf-8")
    bl_xml.write_text(_FAKE_COBERTURA_BASELINE, encoding="utf-8")

    script = Path(__file__).resolve().parents[2] / "scripts" / "numba_coverage_report.py"
    proc = subprocess.run(  # nosec B603 -- fixed local argv (sys.executable/git + literal args), no shell, no untrusted input
        [
            sys.executable,
            str(script),
            "--numba-disabled",
            str(nd_xml),
            "--baseline",
            str(bl_xml),
            "--src-root",
            str(tmp_src_root),
            "--out",
            str(out_json),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"script failed: stdout={proc.stdout!r}, stderr={proc.stderr!r}"
    payload = orjson.loads(out_json.read_bytes())

    assert payload["schema_version"] == 1
    assert payload["summary"]["modules"] == 1
    assert payload["summary"]["total_decorators"] == 1
    assert len(payload["rows"]) == 1

    row = payload["rows"][0]
    assert row["module"] == "mlframe.feature_selection.filters.permutation"
    assert row["statements"] == 5
    assert row["statements_covered_numba_disabled"] == 4
    assert row["statements_covered_baseline"] == 1
    assert row["delta_lines"] == 3
    assert row["delta_rate"] == pytest.approx(0.6, abs=1e-3)
    assert row["numba_decorator_count"] == 1
    assert row["file_path"] == "mlframe/feature_selection/filters/permutation.py"


def test_numba_coverage_report_stdout_when_no_out_arg(tmp_path: Path, tmp_src_root: Path) -> None:
    nd_xml = tmp_path / "nd.xml"
    nd_xml.write_text(_FAKE_COBERTURA_NUMBA_DISABLED, encoding="utf-8")

    script = Path(__file__).resolve().parents[2] / "scripts" / "numba_coverage_report.py"
    proc = subprocess.run(  # nosec B603 -- fixed local argv (sys.executable/git + literal args), no shell, no untrusted input
        [
            sys.executable,
            str(script),
            "--numba-disabled",
            str(nd_xml),
            "--src-root",
            str(tmp_src_root),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    payload = orjson.loads(proc.stdout)
    assert payload["summary"]["modules"] == 1
    assert payload["rows"][0]["statements_covered_baseline"] is None
    assert payload["rows"][0]["delta_lines"] is None
