"""Guard (MRMR critique S-F3): the JMIM ``**(nexisting+1)`` discount exponent stays the DEFAULT (bench-rejected correction).

S-F3 flagged that the exponent amplifies a JMIM joint MI > 1. A multi-seed selection bench
(bench_sf3_jmim_exponent_selection.py) found the discount-only correction never improved selection (0/8 seed wins),
was identical on 7/8 and regressed 1 seed -> the exponent is load-bearing. It is kept as the default; the correction is
an off-by-default option (``MLFRAME_JMIM_EXPONENT_DISCOUNT_ONLY``) for re-testing. This guard pins the default OFF so a
future change cannot silently flip to the regressing behaviour, and documents that the flag exists.
"""



def test_jmim_exponent_discount_only_is_off_by_default(monkeypatch):
    # Read per call and passed to the njit kernel as an argument; with no env override it must be OFF (exponent applied).
    """Jmim exponent discount only is off by default."""
    from mlframe.feature_selection.filters import evaluation

    monkeypatch.delenv("MLFRAME_JMIM_EXPONENT_DISCOUNT_ONLY", raising=False)
    assert evaluation.jmim_exponent_discount_only() is False
