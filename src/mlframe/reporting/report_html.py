"""Single combined HTML report per (model, split) so a DS does not hunt scattered chart files.

A training suite drops many per-(model, split) chart artifacts (calibration PNG, ROC PNG, drift
heatmap, SHAP summary, ...) across a directory. This module assembles them into ONE navigable,
dependency-free HTML page: a sticky sidebar grouped by section, and each chart inline as an ``<img>``
(PNG, by relative path or small-PNG base64) or an embedded plotly HTML fragment (reused verbatim).

This is ASSEMBLY ONLY -- it never re-renders, re-bins, or re-plots. The PNGs and plotly fragments were
already produced by the renderers; ``build_combined_report`` only stitches references / fragments into a
template. PNG mode needs NO CDN (the page is just text + ``<img>`` tags resolving relative paths); plotly
fragments embed whatever plotly.js loader the renderer baked in. File size therefore stays ~the size of
the HTML scaffold plus any base64-inlined small PNGs -- the referenced PNGs are NOT duplicated into the page.
"""

from __future__ import annotations

import base64
import html
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

# Small PNGs are inlined as base64 (one self-contained file, no broken relative links when the page is
# moved); larger ones are referenced by relative path so the HTML stays light. 256 KB keeps a typical
# 80-100 KB chart inline while a heavy hexbin/heatmap stays external.
DEFAULT_INLINE_PNG_MAX_BYTES: int = 256 * 1024


@dataclass(frozen=True)
class ChartEntry:
    """One chart artifact to place in the combined report.

    Exactly one of ``png_path`` / ``plotly_html_fragment`` should be set (png takes precedence when both
    are given). ``section`` groups entries in the sidebar; ``label`` is the per-chart heading + nav text.
    """

    section: str
    label: str
    png_path: Optional[str] = None
    plotly_html_fragment: Optional[str] = None
    caption: str = ""


def _coerce_entry(entry) -> ChartEntry:
    """Accept a ChartEntry, a dict, or a ``(section, label, png_path)`` / ``(section, label, png, fragment)`` tuple."""
    if isinstance(entry, ChartEntry):
        return entry
    if isinstance(entry, dict):
        return ChartEntry(
            section=str(entry.get("section", "")),
            label=str(entry.get("label", "")),
            png_path=entry.get("png_path"),
            plotly_html_fragment=entry.get("plotly_html_fragment"),
            caption=str(entry.get("caption", "")),
        )
    if isinstance(entry, (tuple, list)):
        section = str(entry[0]) if len(entry) > 0 else ""
        label = str(entry[1]) if len(entry) > 1 else ""
        png = entry[2] if len(entry) > 2 else None
        frag = entry[3] if len(entry) > 3 else None
        return ChartEntry(section=section, label=label, png_path=png, plotly_html_fragment=frag)
    raise TypeError(
        f"chart entry must be a ChartEntry, dict, or (section, label, png[, fragment]) tuple; got {type(entry).__name__}"
    )


def _slug(text: str, used: Dict[str, int]) -> str:
    """URL-safe, collision-free anchor id from arbitrary label text."""
    base = "".join(c if c.isalnum() else "-" for c in text.strip().lower()).strip("-") or "panel"
    n = used.get(base, 0)
    used[base] = n + 1
    return base if n == 0 else f"{base}-{n}"


def _img_tag(png_path: str, out_dir: str, inline_png_max_bytes: int) -> str:
    """``<img>`` tag for a PNG: base64-inline when small enough, else a relative-path reference.

    Reading a small PNG to inline it is the only IO this module does; it never opens a large PNG (size is
    checked via ``os.path.getsize`` first), so a heavy chart costs one ``stat`` and a relative-path string.
    """
    try:
        size = os.path.getsize(png_path)
    except OSError:
        return f'<div class="missing">missing image: {html.escape(png_path)}</div>'
    if size <= inline_png_max_bytes:
        with open(png_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        return f'<img loading="lazy" alt="" src="data:image/png;base64,{b64}"/>'
    rel = os.path.relpath(png_path, out_dir) if out_dir else png_path
    rel = rel.replace(os.sep, "/")
    return f'<img loading="lazy" alt="" src="{html.escape(rel)}"/>'


_PAGE_CSS = """\
:root{--bg:#fff;--fg:#1a1a1a;--muted:#6b7280;--line:#e5e7eb;--accent:#b91c1c;--bar:#111827;--sidebg:#fafafa}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg)}
.topbar{position:sticky;top:0;z-index:10;background:var(--bar);color:#fff;padding:10px 20px}
.topbar h1{font-size:16px;margin:0;font-weight:600}
.topbar .sub{font-size:12px;color:#cbd5e1;margin-top:2px}
.layout{display:flex;align-items:flex-start}
nav{position:sticky;top:46px;align-self:flex-start;width:240px;height:calc(100vh - 46px);overflow:auto;padding:16px;border-right:1px solid var(--line);background:var(--sidebg)}
nav .toctitle{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin:0 0 8px}
nav a{display:block;padding:3px 0;color:var(--fg);text-decoration:none;font-size:13px}
nav a:hover{color:var(--accent)}
nav a.child{padding-left:12px;font-size:12px;color:var(--muted)}
nav a.child:hover{color:var(--accent)}
main{flex:1;padding:24px 32px;max-width:1100px}
details.section{margin:0 0 20px;border:1px solid var(--line);border-radius:6px;scroll-margin-top:54px}
details.section>summary{cursor:pointer;list-style:none;padding:10px 14px;font-size:15px;font-weight:600;background:var(--sidebg);border-radius:6px 6px 0 0;border-bottom:1px solid var(--line)}
details.section>summary::-webkit-details-marker{display:none}
details.section>summary::before{content:"\\25B8 ";color:var(--muted)}
details.section[open]>summary::before{content:"\\25BE ";color:var(--muted)}
details.section>summary:hover{color:var(--accent)}
.secdesc{color:var(--muted);font-size:12px;padding:8px 14px 0}
.secbody{padding:8px 14px 14px}
.panel{margin:0 0 28px;scroll-margin-top:54px}
.panel:last-child{margin-bottom:0}
.panel h3{font-size:14px;margin:0 0 6px;border-bottom:1px solid var(--line);padding-bottom:4px}
.panel .cap{color:var(--muted);font-size:12px;margin:0 0 8px}
.panel img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:4px}
.missing{color:var(--accent);font-style:italic}
.empty{color:var(--muted);font-style:italic;padding:24px 32px}
"""


def build_combined_report(
    chart_entries: Sequence,
    *,
    title: str,
    out_path: str,
    subtitle: str = "",
    section_descriptions: Optional[Mapping[str, str]] = None,
    inline_png_max_bytes: int = DEFAULT_INLINE_PNG_MAX_BYTES,
) -> str:
    """Assemble ``chart_entries`` into one navigable, dependency-free HTML file at ``out_path``.

    The page has a sticky top bar (title + ``subtitle`` context line), a sticky side table-of-contents
    that jumps to every section (and, indented, to each chart), and one collapsible ``<details>`` block
    per section -- the first section is open by default so the report greets the reader with content.

    Parameters
    ----------
    chart_entries : sequence of ChartEntry / dict / ``(section, label, png[, fragment])`` tuples.
        The already-produced artifacts for one (model, split). PNGs are referenced (relative path) or
        inlined (base64, when under ``inline_png_max_bytes``); plotly HTML fragments are embedded verbatim.
        A malformed entry is skipped with an inline note rather than aborting the whole report.
    title : page title + headline shown in the top bar.
    out_path : destination ``.html`` file. Its parent directory anchors the relative ``<img>`` paths.
    subtitle : optional context line under the title (e.g. ``"LightGBM / holdout - generated 2026-06-11"``).
    section_descriptions : optional one-line description per section name, shown under the section header.
    inline_png_max_bytes : PNGs at or below this size are base64-inlined; larger ones referenced by path.

    Returns
    -------
    str
        ``out_path`` (the written file). The page contains one TOC entry + one collapsible section per
        distinct section and one panel per entry, opens as valid standalone HTML, and -- in reference
        (non-inline) PNG mode -- does NOT duplicate the underlying chart bytes.
    """
    descriptions = dict(section_descriptions or {})
    out_dir = os.path.dirname(os.path.abspath(out_path))

    used_ids: Dict[str, int] = {}
    # Preserve section order of first appearance; group panels under their section. Malformed entries
    # never crash the report -- they land in a synthetic section as a visible skipped-entry note.
    section_order: List[str] = []
    section_ids: Dict[str, str] = {}
    by_section: Dict[str, List[Tuple[str, ChartEntry, Optional[str]]]] = {}
    for raw in chart_entries:
        try:
            e: Optional[ChartEntry] = _coerce_entry(raw)
            err: Optional[str] = None
        except (TypeError, ValueError, AttributeError) as exc:
            e, err = None, f"skipped malformed entry: {exc}"
        sec = (e.section if e else "") or "Other"
        if sec not in by_section:
            by_section[sec] = []
            section_order.append(sec)
            section_ids[sec] = _slug(f"section-{sec}", used_ids)
        label = e.label if e else "(malformed)"
        anchor = _slug(f"{sec}-{label}", used_ids)
        by_section[sec].append((anchor, e, err))

    nav_parts: List[str] = ['<div class="toctitle">Contents</div>']
    main_parts: List[str] = []
    for idx, sec in enumerate(section_order):
        sec_id = section_ids[sec]
        nav_parts.append(f'<a href="#{sec_id}">{html.escape(sec)}</a>')

        panel_parts: List[str] = []
        for anchor, e, err in by_section[sec]:
            label = e.label if e else "(malformed)"
            nav_parts.append(f'<a class="child" href="#{anchor}">{html.escape(label)}</a>')
            body = (
                f'<div class="missing">{html.escape(err)}</div>'
                if e is None
                else _render_entry_body(e, out_dir, inline_png_max_bytes)
            )
            cap = f'<p class="cap">{html.escape(e.caption)}</p>' if (e and e.caption) else ""
            panel_parts.append(
                f'<section class="panel" id="{anchor}">'
                f"<h3>{html.escape(label)}</h3>{cap}{body}</section>"
            )

        desc = descriptions.get(sec, "")
        desc_html = f'<p class="secdesc">{html.escape(desc)}</p>' if desc else ""
        is_open = " open" if idx == 0 else ""
        main_parts.append(
            f'<details class="section" id="{sec_id}"{is_open}>'
            f"<summary>{html.escape(sec)}</summary>{desc_html}"
            f'<div class="secbody">{"".join(panel_parts)}</div></details>'
        )

    sub_html = f'<div class="sub">{html.escape(subtitle)}</div>' if subtitle else ""
    if section_order:
        body_html = (
            f'<div class="layout"><nav>{"".join(nav_parts)}</nav>'
            f'<main>{"".join(main_parts)}</main></div>'
        )
    else:
        body_html = '<p class="empty">No charts to display.</p>'

    page = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>'
        '<meta name="viewport" content="width=device-width,initial-scale=1"/>'
        f"<title>{html.escape(title)}</title><style>{_PAGE_CSS}</style></head><body>"
        f'<div class="topbar"><h1>{html.escape(title)}</h1>{sub_html}</div>'
        f"{body_html}</body></html>"
    )

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(page)
    return out_path


def _render_entry_body(entry: ChartEntry, out_dir: str, inline_png_max_bytes: int) -> str:
    """Inner HTML for one chart: PNG ``<img>`` (preferred) or embedded plotly fragment, else a missing note."""
    if entry.png_path:
        return _img_tag(entry.png_path, out_dir, inline_png_max_bytes)
    if entry.plotly_html_fragment:
        # Fragment is reused verbatim -- the renderer already produced it; we do not re-render or sanitise the chart.
        return entry.plotly_html_fragment
    return '<div class="missing">no png_path or plotly_html_fragment provided</div>'


__all__ = [
    "build_combined_report",
    "ChartEntry",
    "DEFAULT_INLINE_PNG_MAX_BYTES",
]
