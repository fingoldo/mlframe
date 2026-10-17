"""No test may assert on SOURCE TEXT: call the code and assert on what it does.

Memory rule ``feedback_behavioral_tests``: a test that reads prod source and checks a substring passes against dead
code, an inverted condition and a comment. The detector is ``py_ci_shared.source_text_claims``, an AST walk shared
by every fingoldo repo. It replaces three local gates here, each of which saw one spelling: ``inspect.getsource``
(``test_no_inspect_getsource``), ``"literal" in <read_text var>`` with one level of ``_read`` helper
(``test_no_source_text_proxy`` / ``_source_proxy_scan``), and byte positions via ``.find`` / ``.index``. The shared
detector sees all three plus aliases, slices and ``.sql`` files. Parsing a file with ``ast.parse`` and walking it is
not a claim: that is how every meta-linter here works.

Whole files whose static checks ARE the intended gate stay in ``_ALLOWLIST`` with the reason each was accepted; an
entry for a file that no longer reads source fails. Everything else that exists is recorded in
``_source_text_baseline.json`` keyed ``file::function::kind``, so a line shift does not re-report it, and a stale key
fails too. Refresh after an intentional change::

    pytest tests/test_meta/test_no_source_text_claims.py --refresh-source-text-baseline
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.source_text_claims import assert_no_new_source_text_claims

_TESTS_DIR = Path(__file__).resolve().parents[1]
_BASELINE_PATH = Path(__file__).resolve().parent / "_source_text_baseline.json"

# File -> why its source-text checks are accepted as a whole. Carried over from test_no_inspect_getsource's WHITELIST
# with each group's reason; files that no longer read source were dropped.
_ALLOWLIST: dict[str, str] = {
    "feature_selection/mrmr/test_audit_verification_gaps.py": "Documented known-debt (148 call sites across 28 files, 2026-07): each asserts on prod source text -- comment-marker presence/count, exact log-message wording, or the literal shape of a closure body -- rather than runtime behaviour. 34 sibling sites in this same audit-file family were already converted to behavioural tests with empirical bug-reproduction verification; these remaining sites need either heavy fixtures (Lightning networks, thread-pool Futures in a specific pending state) or have no behavioural equivalent at all (a pure comment-marker-count check). Listed explicitly, mirroring SOURCE_POSITION_PROXY_WHITELIST below, so the gate stays live for NEW test files while tracking this remainder rather than silently passing.",
    "training/test_audit_numpy_view_copy_contracts.py": "Documented known-debt (148 call sites across 28 files, 2026-07): each asserts on prod source text -- comment-marker presence/count, exact log-message wording, or the literal shape of a closure body -- rather than runtime behaviour. 34 sibling sites in this same audit-file family were already converted to behavioural tests with empirical bug-reproduction verification; these remaining sites need either heavy fixtures (Lightning networks, thread-pool Futures in a specific pending state) or have no behavioural equivalent at all (a pure comment-marker-count check). Listed explicitly, mirroring SOURCE_POSITION_PROXY_WHITELIST below, so the gate stays live for NEW test files while tracking this remainder rather than silently passing.",
}


def test_no_new_source_text_claims():
    """No test asserts on source text beyond the allowlist and the recorded debt, and neither holds a stale entry."""
    files = sorted(p for p in _TESTS_DIR.rglob("*.py") if "__pycache__" not in p.parts)
    # ~3600 test modules today; a walk that found fewer than 1000 has stopped reaching the tree.
    assert len(files) >= 1000, f"only {len(files)} test files found under {_TESTS_DIR}; Check _TESTS_DIR"
    assert_no_new_source_text_claims(files, _TESTS_DIR, allowlist=_ALLOWLIST, baseline_path=_BASELINE_PATH, min_files=1000)
