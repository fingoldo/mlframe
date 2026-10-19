# tiny_cupy_detector_2026-10-19 — audit brief

Scope: the static meta-test `tests/test_meta/test_no_tiny_cupy_ops_in_python_loops.py` (launch-bound Python loops of small cupy calls) and its baseline `tests/test_meta/_tiny_cupy_loops_baseline.json`, plus the sibling `test_perm_loop_invariants_hoisted.py` where the same call-following logic applies.

Commit under audit: `3e00704d8` (call-following and chunk policy), on top of `2c76e8748`.

Owner requests, verbatim in substance: critique the detector, extend it, improve it; every proposal from the critique agent is a finding and gets a disposition. Nothing is collapsed or dropped.

## Method

1. Read the detector and its self-tests. Each finding names the file and line of the detector (`test_no_tiny_cupy_ops_in_python_loops.py:<line>`) or the source site it concerns.
2. Verdict per finding: CONFIRMED (reproduced with a snippet), PLAUSIBLE (reasoned from code, not reproduced), or REJECTED (with reason).
3. Disposition per finding: FIX (implemented in this wave), DEFER (kept open with a reason and a target), DISMISS (with reason).
4. A finding is closed only when its disposition is executed and its regression test passes.

## Ground rules (repo standard)

- Every behaviour fix ships with a regression test that fails before the fix and passes after.
- The baseline is a worklist: entries are removed only when the detector stops flagging a site for a real code reason. Entries are never removed to make a test pass.
- No test may pin a known false negative as correct behaviour.
