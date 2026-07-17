"""Tests for build_combined_report: one navigable, dependency-free HTML per (model, split).

Covers:
- unit: ChartEntry / dict / tuple entry coercion, PNG reference vs base64-inline, plotly fragment
  embedding (verbatim, no re-render), section grouping + nav, anchor uniqueness, missing-artifact note,
  empty input, valid standalone HTML.
- biz_value: N entries across sections -> N nav links + N panels + N img/fragments, opens as valid HTML,
  and in reference mode the file size stays ~scaffold (no chart-byte duplication).
- cProfile: assembly is O(entries) and cheap (200 charts well under 1s).
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import struct
import zlib


from mlframe.reporting.report_html import (
    DEFAULT_INLINE_PNG_MAX_BYTES,
    ChartEntry,
    build_combined_report,
)


# -----------------------------------------------------------------------------
# Helpers: write a minimal real PNG of a chosen byte size.
# -----------------------------------------------------------------------------


def _png_chunk(tag: bytes, body: bytes) -> bytes:
    """Helper: Png chunk."""
    return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)


def _write_png(path: str, pad_bytes: int = 0) -> str:
    """Write a valid 1x1 PNG, optionally padding with a tEXt chunk to reach a target size."""
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00\x00")
    chunks = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat)
    if pad_bytes > 0:
        chunks += _png_chunk(b"tEXt", b"pad\x00" + b"x" * pad_bytes)
    chunks += _png_chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(chunks)
    return path


# -----------------------------------------------------------------------------
# Unit tests
# -----------------------------------------------------------------------------


def test_chart_entry_coercion_forms(tmp_path):
    """Chart entry coercion forms."""
    png = _write_png(str(tmp_path / "a.png"))
    entries = [
        ChartEntry("Sec", "from-dataclass", png_path=png),
        {"section": "Sec", "label": "from-dict", "png_path": png},
        ("Sec", "from-tuple", png),
        ("Sec2", "tuple-with-fragment", None, "<div>frag</div>"),
    ]
    out = build_combined_report(entries, title="t", out_path=str(tmp_path / "r.html"))
    text = open(out, encoding="utf-8").read()
    for lbl in ("from-dataclass", "from-dict", "from-tuple", "tuple-with-fragment"):
        assert lbl in text
    assert "<div>frag</div>" in text


def test_bad_entry_skipped_with_note_not_raised(tmp_path):
    """A malformed entry is skipped with a visible note; it must not abort the whole report."""
    png = _write_png(str(tmp_path / "a.png"))
    out = build_combined_report(
        [42, ChartEntry("Sec", "good", png_path=png)],
        title="t",
        out_path=str(tmp_path / "r.html"),
    )
    text = open(out, encoding="utf-8").read()
    assert "skipped malformed entry" in text
    assert "good" in text  # the valid entry still renders
    assert text.rstrip().endswith("</html>")


def test_small_png_inlined_as_base64(tmp_path):
    """Small png inlined as base64."""
    png = _write_png(str(tmp_path / "small.png"))
    assert os.path.getsize(png) <= DEFAULT_INLINE_PNG_MAX_BYTES
    out = build_combined_report(
        [ChartEntry("S", "c", png_path=png)],
        title="t",
        out_path=str(tmp_path / "r.html"),
    )
    text = open(out, encoding="utf-8").read()
    assert "data:image/png;base64," in text
    assert "small.png" not in text  # inlined, not referenced


def test_large_png_referenced_by_relative_path(tmp_path):
    """Large png referenced by relative path."""
    os.makedirs(str(tmp_path / "sub"), exist_ok=True)
    png = _write_png(str(tmp_path / "sub" / "big.png"), pad_bytes=2000)
    out = build_combined_report(
        [ChartEntry("S", "c", png_path=png)],
        title="t",
        out_path=str(tmp_path / "r.html"),
        inline_png_max_bytes=10,  # force reference mode
    )
    text = open(out, encoding="utf-8").read()
    assert "data:image/png;base64," not in text
    assert "sub/big.png" in text  # relative path, forward-slashed


def test_missing_png_emits_note(tmp_path):
    """Missing png emits note."""
    out = build_combined_report(
        [ChartEntry("S", "ghost", png_path=str(tmp_path / "nope.png"))],
        title="t",
        out_path=str(tmp_path / "r.html"),
    )
    text = open(out, encoding="utf-8").read()
    assert "missing image" in text


def test_entry_with_no_artifact_emits_note(tmp_path):
    """Entry with no artifact emits note."""
    out = build_combined_report(
        [ChartEntry("S", "empty")],
        title="t",
        out_path=str(tmp_path / "r.html"),
    )
    text = open(out, encoding="utf-8").read()
    assert "no png_path or plotly_html_fragment" in text


def test_plotly_fragment_embedded_verbatim(tmp_path):
    """Plotly fragment embedded verbatim."""
    frag = '<div id="plotly-xyz"><script>/* fake plotly */</script></div>'
    out = build_combined_report(
        [ChartEntry("Interactive", "roc", plotly_html_fragment=frag)],
        title="t",
        out_path=str(tmp_path / "r.html"),
    )
    text = open(out, encoding="utf-8").read()
    assert frag in text  # not re-rendered or sanitised


def test_sections_grouped_and_ordered(tmp_path):
    """Sections grouped and ordered."""
    png = _write_png(str(tmp_path / "a.png"))
    entries = [
        ChartEntry("Calibration", "c1", png_path=png),
        ChartEntry("ROC", "r1", png_path=png),
        ChartEntry("Calibration", "c2", png_path=png),
    ]
    out = build_combined_report(entries, title="t", out_path=str(tmp_path / "r.html"))
    text = open(out, encoding="utf-8").read()
    # First-appearance section order preserved: Calibration before ROC.
    assert text.index("Calibration") < text.index("ROC")
    assert text.count('<details class="section"') == 2  # two distinct collapsible sections


def test_duplicate_labels_get_unique_anchors(tmp_path):
    """Duplicate labels get unique anchors."""
    png = _write_png(str(tmp_path / "a.png"))
    entries = [ChartEntry("S", "same", png_path=png), ChartEntry("S", "same", png_path=png)]
    out = build_combined_report(entries, title="t", out_path=str(tmp_path / "r.html"))
    text = open(out, encoding="utf-8").read()
    anchors = [seg.split('"')[0] for seg in text.split('id="')[1:]]
    panel_anchors = [a for a in anchors if a.startswith("s-same")]
    assert len(panel_anchors) == len(set(panel_anchors)) == 2  # unique


def test_label_is_html_escaped(tmp_path):
    """Label is html escaped."""
    png = _write_png(str(tmp_path / "a.png"))
    out = build_combined_report(
        [ChartEntry("S", "<script>alert(1)</script>", png_path=png)],
        title="t",
        out_path=str(tmp_path / "r.html"),
    )
    text = open(out, encoding="utf-8").read()
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;" in text


def test_empty_entries_valid_html(tmp_path):
    """Empty entries valid html."""
    out = build_combined_report([], title="nothing", out_path=str(tmp_path / "r.html"))
    text = open(out, encoding="utf-8").read()
    assert text.startswith("<!DOCTYPE html>")
    assert text.rstrip().endswith("</html>")
    assert "No charts to display" in text


def test_creates_parent_dir(tmp_path):
    """Creates parent dir."""
    out_path = str(tmp_path / "deep" / "nested" / "r.html")
    build_combined_report([], title="t", out_path=out_path)
    assert os.path.exists(out_path)


# -----------------------------------------------------------------------------
# biz_value
# -----------------------------------------------------------------------------


def test_biz_val_report_html_n_entries_yield_n_nav_and_n_panels(tmp_path):
    """N entries across S sections -> S section TOC links + N child TOC links, S collapsible sections, N panels."""
    png = _write_png(str(tmp_path / "chart.png"))
    n, n_sections = 12, 3
    entries = []
    for i in range(n):
        sec = f"section{i % n_sections}"
        if i % 2 == 0:
            entries.append(ChartEntry(sec, f"chart {i}", png_path=png))
        else:
            entries.append(ChartEntry(sec, f"chart {i}", plotly_html_fragment=f"<div>frag{i}</div>"))
    out = build_combined_report(entries, title="model / test", out_path=str(tmp_path / "r.html"))
    text = open(out, encoding="utf-8").read()

    assert text.count('<a class="child" href="#') == n, "one child nav link per entry"
    # Section-level TOC links: total anchors minus child anchors.
    assert text.count('<a href="#') == n_sections, "one section TOC link per distinct section"
    assert text.count('<details class="section"') == n_sections, "one collapsible section per distinct section"
    assert text.count('class="panel"') == n, "one panel per entry"
    rendered = text.count("data:image/png;base64,") + sum(f"<div>frag{i}</div>" in text for i in range(1, n, 2))
    assert rendered == n, "every entry produced an inline image or an embedded fragment"
    assert text.startswith("<!DOCTYPE html>") and text.rstrip().endswith("</html>")


def test_biz_val_report_html_reference_mode_no_byte_duplication(tmp_path):
    """In reference (non-inline) PNG mode, the HTML must NOT duplicate chart bytes.

    Measured: 4 references to a ~55KB chart -> HTML ~13KB (scaffold only), vs 4*55KB if bytes were
    duplicated. Bound: HTML < the summed referenced PNG bytes (proves references, not copies).
    """
    big = _write_png(str(tmp_path / "big.png"), pad_bytes=50_000)
    png_bytes = os.path.getsize(big)
    entries = [ChartEntry("S", f"chart {i}", png_path=big) for i in range(4)]
    out = build_combined_report(
        entries,
        title="t",
        out_path=str(tmp_path / "r.html"),
        inline_png_max_bytes=0,
    )
    html_bytes = os.path.getsize(out)
    assert html_bytes < png_bytes, (
        f"reference-mode HTML ({html_bytes}B) should stay below one referenced PNG ({png_bytes}B); the chart bytes must not be duplicated into the page"
    )


def test_sections_are_collapsible_details_first_open(tmp_path):
    """Each section is a <details>; the first is open by default, the rest collapsed."""
    png = _write_png(str(tmp_path / "a.png"))
    entries = [
        ChartEntry("First", "c1", png_path=png),
        ChartEntry("Second", "c2", png_path=png),
        ChartEntry("Third", "c3", png_path=png),
    ]
    out = build_combined_report(entries, title="t", out_path=str(tmp_path / "r.html"))
    text = open(out, encoding="utf-8").read()
    assert text.count("<details ") == 3
    assert text.count("</details>") == 3
    assert "<summary>First</summary>" in text
    # Exactly one open section (the first); collapsed sections carry no `open` attribute.
    assert text.count('class="section" id="section-first" open') == 1
    assert text.count(" open>") == 1


def test_nav_anchors_resolve_to_section_and_panel_ids(tmp_path):
    """Every TOC href (#anchor) must resolve to a matching id= in the document body."""
    png = _write_png(str(tmp_path / "a.png"))
    entries = [
        ChartEntry("Alpha", "one", png_path=png),
        ChartEntry("Alpha", "two", png_path=png),
        ChartEntry("Beta", "three", png_path=png),
    ]
    out = build_combined_report(entries, title="t", out_path=str(tmp_path / "r.html"))
    text = open(out, encoding="utf-8").read()
    href_anchors = {seg.split('"')[0] for seg in text.split('href="#')[1:]}
    id_anchors = {seg.split('"')[0] for seg in text.split('id="')[1:]}
    assert href_anchors, "report must contain navigation anchors"
    assert href_anchors <= id_anchors, "every nav anchor resolves to a section/panel id"


def test_topbar_shows_title_and_subtitle(tmp_path):
    """Topbar shows title and subtitle."""
    png = _write_png(str(tmp_path / "a.png"))
    out = build_combined_report(
        [ChartEntry("S", "c", png_path=png)],
        title="My Model Report",
        subtitle="LightGBM / holdout - 2026-06-11",
        out_path=str(tmp_path / "r.html"),
    )
    text = open(out, encoding="utf-8").read()
    assert 'class="topbar"' in text
    assert "My Model Report" in text
    assert "LightGBM / holdout - 2026-06-11" in text


def test_section_descriptions_render_under_header(tmp_path):
    """Section descriptions render under header."""
    png = _write_png(str(tmp_path / "a.png"))
    out = build_combined_report(
        [ChartEntry("Calibration", "c", png_path=png)],
        title="t",
        out_path=str(tmp_path / "r.html"),
        section_descriptions={"Calibration": "Reliability + Brier decomposition."},
    )
    text = open(out, encoding="utf-8").read()
    assert "Reliability + Brier decomposition." in text
    assert 'class="secdesc"' in text


def test_no_cdn_refs_in_png_mode(tmp_path):
    """PNG-only reports must be fully self-contained: no http(s):// CDN references."""
    png = _write_png(str(tmp_path / "a.png"))
    entries = [ChartEntry(f"sec{i}", f"c{i}", png_path=png) for i in range(5)]
    out = build_combined_report(entries, title="t", out_path=str(tmp_path / "r.html"))
    text = open(out, encoding="utf-8").read()
    assert "http://" not in text and "https://" not in text


def test_empty_entries_no_layout_just_notice(tmp_path):
    """Empty entries no layout just notice."""
    out = build_combined_report([], title="nothing", out_path=str(tmp_path / "r.html"))
    text = open(out, encoding="utf-8").read()
    assert text.startswith("<!DOCTYPE html>") and text.rstrip().endswith("</html>")
    assert "No charts to display" in text
    assert "<details" not in text  # no empty section scaffolding


# -----------------------------------------------------------------------------
# cProfile
# -----------------------------------------------------------------------------


def test_cprofile_report_assembly_is_cheap_at_200_entries(tmp_path):
    """200 charts assemble in well under 1s; assembly is O(entries) (no quadratic string concat)."""
    png = _write_png(str(tmp_path / "tiny.png"))
    entries = [ChartEntry(f"sec{i % 5}", f"chart {i}", png_path=png) for i in range(200)]
    out_path = str(tmp_path / "big.html")

    pr = cProfile.Profile()
    pr.enable()
    build_combined_report(entries, title="200-chart report", out_path=out_path)
    pr.disable()

    total = pstats.Stats(pr, stream=io.StringIO()).total_tt
    assert total < 1.0, f"200-entry assembly took {total:.3f}s; expected < 1s (O(entries) assembly)"
    text = open(out_path, encoding="utf-8").read()
    assert text.count('class="panel"') == 200
