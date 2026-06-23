"""SEC6 (DOC): the safe_pickle module docstring must state the sidecar is an integrity check, not an authenticity control.

The sha256 sidecar defends against accidental corruption / un-sidecar'd swaps, NOT against an attacker with directory write access who
rewrites both payload and sidecar. This sensor pins that caveat so a future edit cannot silently drop the threat-model disclosure.
"""
import mlframe.utils.safe_pickle as sp


def test_module_docstring_states_integrity_not_authenticity():
    # Collapse whitespace so the caveat matches regardless of line wrapping.
    doc = " ".join((sp.__doc__ or "").lower().split())
    assert "not an authenticity" in doc
    assert "integrity" in doc
    assert "write access" in doc
    assert "hmac" in doc or "signature" in doc
