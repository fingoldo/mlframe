"""Per-token caption fragments for the composers that render a caller-chosen mix of panels.

A composer like ``compose_binary_figure`` takes a ``panels_template`` and renders whatever tokens it names. The
caption, however, was a single figure-level string written for the DEFAULT template -- so a caller asking for
``"KS,GAIN"`` got a paragraph explaining the ROC and PR curves that are not on their figure, and a caller asking
for a token the paragraph never mentioned got no explanation of it at all. A caption that describes panels the
reader cannot see is worse than a short one.

Each composer keeps a ``{token: fragment}`` catalogue beside its ``_TOKEN_BUILDERS``, and builds the caption from
the fragments of the tokens it ACTUALLY rendered, in render order, behind a lead sentence that stays true for any
mix. ``test_caption_fragments_cover_every_token`` pins that the two catalogues have identical keys, so a new panel
cannot ship without its sentence.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional


def caption_for_tokens(lead: str, tokens: Iterable[str], fragments: Dict[str, str], *, tail: Optional[str] = None) -> str:
    """``lead`` plus one fragment per rendered token (deduplicated, render order), plus an optional closing note.

    A token with no fragment contributes nothing rather than raising: a caption is a reading aid, and a composer
    that refused to build a figure over a missing sentence would be trading a real panel for a cosmetic one.
    """
    seen: Dict[str, None] = {}
    for tok in tokens:
        frag = fragments.get(tok)
        if frag and frag not in seen:
            seen[frag] = None
    parts = [lead.strip()] if lead else []
    parts.extend(seen)
    if tail:
        parts.append(tail.strip())
    return " ".join(p for p in parts if p)


__all__ = ["caption_for_tokens"]
