"""SEC6 (DOC): the safe_pickle module docstring must state the sidecar is an integrity check, not an authenticity control.

The sha256 sidecar defends against accidental corruption / un-sidecar'd swaps, NOT against an attacker with directory write access who
rewrites both payload and sidecar. This sensor pins that caveat so a future edit cannot silently drop the threat-model disclosure.

The full caveat now lives in pyutilz.core.safe_pickle (the canonical shared implementation,
2026-07-06); mlframe.utils.safe_pickle is a thin re-export shim whose own docstring just
points there, so this sensor checks pyutilz's docstring (the actual source of truth) and
mlframe's shim for the pointer.
"""

import pyutilz.core.safe_pickle as pyutilz_sp

import mlframe.utils.safe_pickle as sp


def test_module_docstring_states_integrity_not_authenticity():
    # Collapse whitespace so the caveat matches regardless of line wrapping.
    doc = " ".join((pyutilz_sp.__doc__ or "").lower().split())
    assert "not an authenticity" in doc
    assert "integrity" in doc
    assert "write access" in doc
    assert "hmac" in doc or "signature" in doc


def test_mlframe_shim_docstring_points_to_canonical_module():
    doc = " ".join((sp.__doc__ or "").lower().split())
    assert "pyutilz.core.safe_pickle" in doc
