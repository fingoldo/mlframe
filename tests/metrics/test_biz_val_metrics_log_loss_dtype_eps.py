"""Regression sensor: fast_log_loss's dtype-DEPENDENT clip eps is correct -- do not flip it to a fixed float64 eps.

qual-11 investigated the "log-loss clip eps inflates the metric" hypothesis (probability post-processing default mining). Verdict: REJECT -- the legacy
``eps = np.finfo(y_pred.dtype).eps`` default is the correct precision-matched floor, NOT a bug.

Why a fixed float64 eps (2.22e-16) on float32 inputs is WRONG: float32 cannot represent probabilities in (1 - 1.19e-7, 1). A confident ``1 - 1e-8``
collapses to exactly ``1.0`` on the cast to float32. Clipping that ``1.0`` at the tiny float64 eps yields ``-log(2.22e-16) ~= 36`` for a confident-wrong
row, OVERSHOOTING the honest ``-log(1e-8) ~= 18``. The float32 machine eps is the right floor for float32 inputs precisely because it matches the
representable precision near the 0/1 boundary. The bias direction flips with eps choice, so there is no unambiguously-better dtype-independent default --
hence REJECT, keep the dtype-dependent legacy default. See bench: ``mlframe/metrics/_benchmarks/bench_log_loss_dtype_eps_qual11.py``.

These tests pin BOTH sides so a future "just always use float64 eps" flip trips immediately.
"""
import numpy as np
from sklearn.metrics import log_loss as sklearn_log_loss

from mlframe.metrics.core import fast_log_loss


def _overconfident_wrong(seed: int, n: int = 100_000, conf: float = 1e-8, wrong_frac: float = 0.05):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(np.int64)
    wrong = rng.random(n) < wrong_frac
    p1 = np.where(y == 1, 1 - conf, conf)
    p1 = np.where(wrong, 1 - p1, p1)
    return y, np.clip(p1, 1e-12, 1 - 1e-12)


def test_float64_log_loss_matches_sklearn_exactly():
    """The float64 path uses eps=2.22e-16 and reproduces sklearn's effectively-no-clip value (the honest target) within 1e-3 on all 5 seeds."""
    for seed in range(5):
        y, p = _overconfident_wrong(seed)
        ml64 = fast_log_loss(y, p.astype(np.float64))
        honest = sklearn_log_loss(y, p)
        assert abs(ml64 - honest) < 1e-3, f"seed{seed}: ml64={ml64} vs honest={honest}"


def test_float32_uses_dtype_eps_not_float64_eps():
    """float32 default must clip at float32 eps (1.19e-7), NOT float64 eps. A flip to a fixed float64 eps would push the value far ABOVE honest
    (-log(2.22e-16)~=36 per unrepresentable-near-1 confident-wrong row); pin that the float32 default does NOT overshoot honest by >5%."""
    overshoots = 0
    for seed in range(5):
        y, p = _overconfident_wrong(seed)
        ml32 = fast_log_loss(y, p.astype(np.float32))  # default eps == finfo(float32).eps
        honest = sklearn_log_loss(y, p)
        if ml32 > honest * 1.05:
            overshoots += 1
    assert overshoots == 0, f"float32 default eps overshot honest on {overshoots}/5 seeds -- dtype-eps default was flipped to float64 eps"


def test_explicit_float64_eps_on_float32_overshoots_documents_reject():
    """Documents WHY the reject holds: forcing the float64 eps onto float32 input overshoots the honest value (>20% high) -- the failure the
    legacy dtype-dependent default avoids. If this ever stops overshooting, the float32 representation behaviour changed and the verdict needs review."""
    y, p = _overconfident_wrong(0)
    honest = sklearn_log_loss(y, p)
    forced64 = fast_log_loss(y, p.astype(np.float32), eps=float(np.finfo(np.float64).eps))
    assert forced64 > honest * 1.2, f"forcing float64 eps on float32 no longer overshoots (forced64={forced64}, honest={honest}); revisit qual-11 reject"
