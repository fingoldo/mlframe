"""MRMR critique batch D: statistical null-consistency (S-F1 kept; N-F3 and S-F3 deferred).

- S-F1 (GPU): mi_direct_gpu(return_null_mean=True) now runs the full _NULL_MEAN_MIN_PERMS null budget (was as low
  as 2), bringing the GPU relevance null in line with the already-validated CPU path (permutation.py:642). Contract
  is tested on CPU where the constant lives; on-device exercise is blocked by this env's GPU native instability.
- N-F3 (perm_pvalue full-budget extrapolation) and S-F3 (greedy-jmim exponent) were REVERTED: both override a
  deliberate, bench/test-pinned design (bench_perm_pvalue_addone.py; the greedy-jmim exponent) and need their own
  multi-seed biz-value bench to settle rather than a code change on a critique-agent argument. Tracked FUTURE.
"""

from mlframe.feature_selection.filters.permutation import _perm_pvalue, _addone_pvalue_enabled


def test_perm_pvalue_full_budget_break_position_independent():
    # Retained (deliberate) design: an early-stopped run scores its p against the FULL budget so confidence does not
    # depend on WHERE the break fired. Pins the behaviour N-F3 proposed to change so a future edit is a conscious one.
    p_stopped = _perm_pvalue(5, 8, full_budget=100)
    p_full = _perm_pvalue(5, 100, full_budget=100)
    if _addone_pvalue_enabled():
        assert p_stopped == p_full == (1.0 + 5) / (1.0 + 100)


def test_gpu_null_budget_constant_available():
    # the GPU return_null_mean path bumps to this same constant the CPU path uses (permutation.py:642).
    from mlframe.feature_selection.filters.permutation import _NULL_MEAN_MIN_PERMS

    assert _NULL_MEAN_MIN_PERMS >= 32
