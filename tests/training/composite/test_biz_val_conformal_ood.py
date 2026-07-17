"""Unit + biz_value + cProfile tests for OOD-adaptive Mondrian conformal width and
the do-not-deploy verdict (feature T2-K).

Mechanism under test (``composite/conformal.py``): a predict-time group that was
UNSEEN at calibration -- or seen but too small to certify the level -- no longer gets
the raw pooled ``global`` radius (which under-covers exactly the OOD groups whose
residual spread exceeds the pooled bulk). Instead it gets an OOD-adaptive radius: a
conservative group-level conformal upper quantile of the calibration groups' OWN radii
(``ceil((G+1)(1-alpha))`` order statistic, floored at the pooled radius), MEASURED from
the between-group dispersion -- no magic constant. Every such row is flagged
low-confidence (``return_ood=True`` mask); the fraction is surfaced in ``runtime_stats_``
and via ``mondrian_ood_summary``. The per-SEEN-CERTIFIED-group path is bit-identical.

cProfile (calibrate + predict, 200k rows / 500 groups, 2 alphas; see
``_profile_mondrian_ood``): calibrate wall ~106ms, predict wall ~31ms. Top hotspots by
tottime are all PRE-EXISTING grouping cost, not the OOD additions:
  1. ``numpy.ndarray.argsort`` (~0.11s) -- the CPX29 factorize+stable-argsort grouping.
  2. ``conformal_quantile`` sort (3012 calls) -- the per-group radius quantiles.
  3. ``pandas.factorize_array`` / ``_isna_string_dtype`` -- object-label factorize.
Verdict: NO actionable speedup. The OOD-adaptive additions cost 2 extra small
``conformal_quantile`` calls per alpha (over G group radii) plus a boolean flag gather,
< 1% of the wall the G per-group quantiles + factorize+argsort already dominate.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite import conformal as conf
from mlframe.training.composite.conformal import conformal_quantile


class _Stub:
    """Minimal CompositeTargetEstimator surface the conformal functions read: an
    ``estimator_`` flag, a ``predict`` returning zeros (so residual == y_cal and the
    band is symmetric ``+/- radius``), and a mutable ``runtime_stats_`` dict."""

    def __init__(self) -> None:
        self.estimator_ = object()
        self.runtime_stats_: dict = {}

    def predict(self, X):
        """Predict."""
        return np.zeros(len(X), dtype=np.float64)


def _cal_stub(y_cal, groups_cal, alpha):
    """Cal stub."""
    st = _Stub()
    conf.calibrate_conformal_mondrian(st, np.arange(len(y_cal)), y_cal, groups_cal, alpha)
    return st


def _grouped(rng, spec, prefix):
    """Build (y, groups) where ``spec`` is a list of (sigma, n_rows); residuals are
    N(0, sigma) per group and the point prediction is 0, so the group's own radius
    is directly its |residual| quantile."""
    ys, gs = [], []
    for i, (sigma, n) in enumerate(spec):
        ys.append(rng.normal(0.0, sigma, n))
        gs.append(np.full(n, f"{prefix}{i}", dtype=object))
    return np.concatenate(ys), np.concatenate(gs)


# --------------------------------------------------------------------------- #
# Seen-group bit-identity: the OOD change must not touch the certified path.
# --------------------------------------------------------------------------- #
class TestSeenGroupBitIdentity:
    """Groups tests covering seen group bit identity."""
    def test_stored_seen_radius_equals_reference_quantile(self) -> None:
        # The stored per-group radius for a CERTIFIED group must be exactly the
        # independent split-conformal reference. This assertion FAILS if the seen
        # path is ever changed to inflate / route through the OOD fallback.
        """Stored seen radius equals reference quantile."""
        rng = np.random.default_rng(0)
        y, g = _grouped(rng, [(0.7, 500), (2.5, 500), (1.2, 500)], "cal")
        alpha = 0.1
        st = _cal_stub(y, g, alpha)
        table = st._mondrian_q_[round(alpha, 6)]
        for lab in ("cal0", "cal1", "cal2"):
            ref = conformal_quantile(y[g == lab], alpha)  # residual == y (pred is 0)
            assert table[lab] == ref, f"{lab}: seen radius drifted from reference"
            # And a certified seen radius must NOT have been inflated to the OOD radius.
            assert table[lab] != st._mondrian_ood_[round(alpha, 6)] or ref == st._mondrian_ood_[round(alpha, 6)]

    def test_predict_seen_radius_bit_identical_across_ood_toggle(self) -> None:
        # Flipping the OOD gate must leave the applied radius on a SEEN certified
        # group bit-identical (only unseen / uncertifiable rows may change).
        """Predict seen radius bit identical across ood toggle."""
        rng = np.random.default_rng(1)
        y, g = _grouped(rng, [(0.6, 600), (3.0, 600)], "cal")
        alpha = 0.1
        st = _cal_stub(y, g, alpha)
        # test set is all seen certified rows (labels cal0/cal1)
        yt, gt = _grouped(np.random.default_rng(9), [(0.6, 200), (3.0, 200)], "cal")
        st.conformal_ood_adaptive = True
        lo_on, hi_on = conf.predict_interval_mondrian(st, np.arange(len(yt)), gt, alpha)
        st.conformal_ood_adaptive = False
        lo_off, hi_off = conf.predict_interval_mondrian(st, np.arange(len(yt)), gt, alpha)
        assert np.array_equal(hi_on - lo_on, hi_off - lo_off), "seen radius moved with the OOD gate"
        # And each equals the stored certified radius exactly.
        applied = (hi_on - lo_on) / 2.0
        table = st._mondrian_q_[round(alpha, 6)]
        assert applied[0] == table["cal0"] and applied[-1] == table["cal1"]


# --------------------------------------------------------------------------- #
# OOD unseen + too-small routing and the low-confidence flag.
# --------------------------------------------------------------------------- #
class TestOODRouting:
    """Groups tests covering o o d routing."""
    def _dispersed(self, seed=0):
        """Dispersed."""
        rng = np.random.default_rng(seed)
        spec = [(0.6, 180)] * 20 + [(2.8, 180)] * 6  # modest bulk + wide tail
        y, g = _grouped(rng, spec, "cal")
        return _cal_stub(y, g, 0.1), y, g

    def test_unseen_group_gets_inflated_radius_not_raw_global(self) -> None:
        """Unseen group gets inflated radius not raw global."""
        st, _, _ = self._dispersed()
        key = round(0.1, 6)
        ood_r = st._mondrian_ood_[key]
        global_r = st._mondrian_q_[key][None]
        assert ood_r > global_r, "OOD radius must inflate above the pooled global"
        unseen = np.full(5, "brand_new", dtype=object)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lo, hi, flag = conf.predict_interval_mondrian(st, np.arange(5), unseen, 0.1, return_ood=True)
        applied = (hi - lo) / 2.0
        assert np.allclose(applied, ood_r) and not np.allclose(applied, global_r)
        assert flag.all(), "every unseen row must be flagged OOD"

    def test_too_small_group_stored_global_but_predicts_inflated(self) -> None:
        # A 2-row group cannot certify alpha=0.1 -> its STORED radius stays the pooled
        # global (bit-identical), but at PREDICT time it routes to the OOD inflation
        # and is flagged. Regression: too-small groups must fall back WITH inflation.
        """Too small group stored global but predicts inflated."""
        rng = np.random.default_rng(2)
        spec = [(0.6, 180)] * 15 + [(2.8, 180)] * 5 + [(1.0, 2)]  # last = tiny
        y, g = _grouped(rng, spec, "cal")
        # rename the tiny group so we can address it
        tiny_lab = "cal20"
        st = _cal_stub(y, g, 0.1)
        key = round(0.1, 6)
        assert tiny_lab in st._mondrian_uncertified_[key]
        assert st._mondrian_q_[key][tiny_lab] == st._mondrian_q_[key][None]  # stored == global
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lo, hi, flag = conf.predict_interval_mondrian(
                st,
                np.arange(3),
                np.full(3, tiny_lab, dtype=object),
                0.1,
                return_ood=True,
            )
        applied = (hi - lo) / 2.0
        assert np.allclose(applied, st._mondrian_ood_[key]), "too-small group must predict inflated"
        assert flag.all(), "too-small (uncertifiable) rows must be flagged OOD"

    def test_flag_exact_on_ood_rows_clear_on_seen(self) -> None:
        """Flag exact on ood rows clear on seen."""
        st, _, _ = self._dispersed()
        # mix: seen certified (cal0), unseen (zzz), and stays certified
        labels = np.array(["cal0", "zzz", "cal1", "zzz", "cal2"], dtype=object)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, flag = conf.predict_interval_mondrian(st, np.arange(5), labels, 0.1, return_ood=True)
        assert np.array_equal(flag, np.array([False, True, False, True, False]))

    def test_adaptive_off_uses_raw_global_still_flagged(self) -> None:
        """Adaptive off uses raw global still flagged."""
        st, _, _ = self._dispersed()
        st.conformal_ood_adaptive = False
        key = round(0.1, 6)
        unseen = np.full(4, "nope", dtype=object)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lo, hi, flag = conf.predict_interval_mondrian(st, np.arange(4), unseen, 0.1, return_ood=True)
        applied = (hi - lo) / 2.0
        assert np.allclose(applied, st._mondrian_q_[key][None]), "gate off -> raw global"
        assert flag.all(), "provenance flag is independent of the inflation gate"

    def test_unseen_group_warns_with_global_radius_message(self) -> None:
        """Unseen group warns with global radius message."""
        st, _, _ = self._dispersed()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            conf.predict_interval_mondrian(st, np.arange(2), np.full(2, "ghost", dtype=object), 0.1)
        assert any("global radius" in str(w.message) for w in rec)


# --------------------------------------------------------------------------- #
# Degenerate cases.
# --------------------------------------------------------------------------- #
class TestDegenerate:
    """Groups tests covering degenerate."""
    def test_all_groups_seen_no_ood_flag(self) -> None:
        """All groups seen no ood flag."""
        rng = np.random.default_rng(0)
        y, g = _grouped(rng, [(0.6, 400), (1.5, 400)], "cal")
        st = _cal_stub(y, g, 0.1)
        yt, gt = _grouped(np.random.default_rng(5), [(0.6, 150), (1.5, 150)], "cal")
        _, _, flag = conf.predict_interval_mondrian(st, np.arange(len(yt)), gt, 0.1, return_ood=True)
        assert not flag.any()
        assert conf.mondrian_ood_summary(st, 0.1)["fraction_ood"] == 0.0

    def test_single_calibration_group(self) -> None:
        """Single calibration group."""
        rng = np.random.default_rng(0)
        y, g = _grouped(rng, [(1.0, 500)], "cal")
        st = _cal_stub(y, g, 0.1)
        key = round(0.1, 6)
        # With one certified group the OOD radius = that group's radius, floored at global.
        assert np.isfinite(st._mondrian_ood_[key])
        assert st._mondrian_ood_[key] >= st._mondrian_q_[key][None]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lo, _hi, flag = conf.predict_interval_mondrian(st, np.arange(3), np.full(3, "new", dtype=object), 0.1, return_ood=True)
        assert lo.shape == (3,) and flag.all()

    def test_empty_unseen_set_fraction_zero(self) -> None:
        """Empty unseen set fraction zero."""
        rng = np.random.default_rng(0)
        y, g = _grouped(rng, [(0.6, 300), (2.0, 300)], "cal")
        st = _cal_stub(y, g, 0.1)
        conf.predict_interval_mondrian(st, np.arange(4), np.array(["cal0"] * 4, dtype=object), 0.1)
        assert st.runtime_stats_["mondrian_ood_fraction"] == 0.0

    def test_summary_zeros_before_any_predict(self) -> None:
        """Summary zeros before any predict."""
        rng = np.random.default_rng(0)
        y, g = _grouped(rng, [(1.0, 400)], "cal")
        st = _cal_stub(y, g, 0.1)
        s = conf.mondrian_ood_summary(st, 0.1)
        assert s == {"n_rows": 0, "n_ood": 0, "fraction_ood": 0.0}


# --------------------------------------------------------------------------- #
# Verdict accessor + runtime_stats_ aggregate.
# --------------------------------------------------------------------------- #
class TestVerdictAccessor:
    """Groups tests covering verdict accessor."""
    def test_runtime_stats_and_summary_record_fraction(self) -> None:
        """Runtime stats and summary record fraction."""
        rng = np.random.default_rng(0)
        y, g = _grouped(rng, [(0.6, 300)] * 10 + [(2.8, 300)] * 3, "cal")
        st = _cal_stub(y, g, 0.1)
        # 6 rows: 3 seen (cal0), 3 unseen -> 50% OOD
        labels = np.array(["cal0", "cal0", "cal0", "u", "u", "u"], dtype=object)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, flag = conf.predict_interval_mondrian(st, np.arange(6), labels, 0.1, return_ood=True)
        assert flag.sum() == 3
        s = conf.mondrian_ood_summary(st, 0.1)
        assert s == {"n_rows": 6, "n_ood": 3, "fraction_ood": 0.5}
        assert st.runtime_stats_["mondrian_ood_fraction"] == pytest.approx(0.5)

    def test_return_ood_shape_and_dtype(self) -> None:
        """Return ood shape and dtype."""
        rng = np.random.default_rng(0)
        y, g = _grouped(rng, [(1.0, 400), (2.0, 400)], "cal")
        st = _cal_stub(y, g, 0.1)
        out = conf.predict_interval_mondrian(st, np.arange(5), np.array(["cal0"] * 5, dtype=object), 0.1, return_ood=True)
        assert len(out) == 3
        _lo, _hi, flag = out
        assert flag.dtype == bool and flag.shape == (5,)
        # default (return_ood=False) stays a 2-tuple.
        out2 = conf.predict_interval_mondrian(st, np.arange(5), np.array(["cal0"] * 5, dtype=object), 0.1)
        assert len(out2) == 2

    def test_predict_without_calibration_raises(self) -> None:
        """Predict without calibration raises."""
        st = _Stub()
        with pytest.raises(RuntimeError, match="no Mondrian radius"):
            conf.predict_interval_mondrian(st, np.arange(1), np.array(["a"], dtype=object), 0.1)


# --------------------------------------------------------------------------- #
# biz_value: OOD-adaptive covers unseen groups; legacy global under-covers.
# --------------------------------------------------------------------------- #
class TestBizValueOODCoverage:
    """Groups tests covering biz value o o d coverage."""
    def _run(self, seed):
        """Calibrates on a bimodal-spread grouped set, evaluates coverage on an unseen wider-spread group, and returns the comparison."""
        alpha = 0.1
        rng = np.random.default_rng(seed)
        # calibration: modest bulk (sigma 0.6) + a real wide tail (sigma 2.8)
        cal_spec = [(0.6, 180)] * 30 + [(2.8, 180)] * 10
        yc, gc = _grouped(rng, cal_spec, "cal")
        st = _cal_stub(yc, gc, alpha)
        # unseen test groups: spread clearly LARGER than the modest bulk (sigma 2.2)
        yt, gt = _grouped(rng, [(2.2, 180)] * 20, "unseen")
        # a block of SEEN certified test rows (reuse a modest calibration label)
        ys = rng.normal(0.0, 0.6, 2000)
        gs = np.full(2000, "cal0", dtype=object)

        st.conformal_ood_adaptive = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lo, hi = conf.predict_interval_mondrian(st, np.arange(len(yt)), gt, alpha)
            lo_s_on, hi_s_on = conf.predict_interval_mondrian(st, np.arange(len(ys)), gs, alpha)
        cov_ood = float(np.mean((yt >= lo) & (yt <= hi)))
        width_seen_on = float(np.mean(hi_s_on - lo_s_on))

        st.conformal_ood_adaptive = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lo2, hi2 = conf.predict_interval_mondrian(st, np.arange(len(yt)), gt, alpha)
            lo_s_off, hi_s_off = conf.predict_interval_mondrian(st, np.arange(len(ys)), gs, alpha)
        cov_legacy = float(np.mean((yt >= lo2) & (yt <= hi2)))
        width_seen_off = float(np.mean(hi_s_off - lo_s_off))
        return cov_ood, cov_legacy, width_seen_on, width_seen_off

    def test_biz_val_mondrian_ood_covers_unseen_while_global_undercovers(self) -> None:
        """OOD-adaptive width reaches >= 1-alpha on UNSEEN (larger-spread) groups
        while the legacy raw-global fallback UNDER-covers the very same rows.

        Measured (4 seeds): cov_ood 0.96-0.97, cov_legacy 0.68-0.74, delta ~0.23-0.28.
        Floors set with margin. Seen-group width must be IDENTICAL across the gate
        (the OOD change must not blow up seen intervals)."""
        oods, legs = [], []
        for seed in range(4):
            cov_ood, cov_legacy, w_on, w_off = self._run(seed)
            # Contract: OOD-adaptive achieves marginal coverage >= 1-alpha on unseen.
            assert cov_ood >= 0.90, f"seed {seed}: OOD-adaptive under-covered unseen: {cov_ood:.3f}"
            # Legacy global fallback demonstrably under-covers the same rows.
            assert cov_legacy < 0.82, f"seed {seed}: legacy did not under-cover: {cov_legacy:.3f}"
            # Seen-group width unchanged by the OOD mechanism (no needless blow-up).
            assert w_on == w_off, f"seed {seed}: seen width moved with the OOD gate ({w_on} vs {w_off})"
            oods.append(cov_ood)
            legs.append(cov_legacy)
        assert float(np.mean(oods)) - float(np.mean(legs)) >= 0.15, "OOD gain over legacy too small"


# --------------------------------------------------------------------------- #
# Integration: the real estimator binds return_ood + the verdict accessors.
# --------------------------------------------------------------------------- #
class TestRealEstimatorIntegration:
    """Groups tests covering real estimator integration."""
    def test_real_estimator_return_ood_and_runtime_stats(self) -> None:
        """Real estimator return ood and runtime stats."""
        rng = np.random.default_rng(0)
        n = 1500
        b = rng.normal(0.0, 1.0, n)
        f = rng.normal(0.0, 1.0, n)
        y = b + 0.5 * f + rng.normal(0.0, 1.0, n)
        X = pd.DataFrame({"b": b, "feat": f})
        groups = np.array(["a", "b"], dtype=object)[rng.integers(0, 2, n)]
        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="linear_residual",
            base_column="b",
        ).fit(X, y)
        est.calibrate_conformal_mondrian(X, y, groups, 0.1)
        Xt = X.iloc[:6]
        gt = np.array(["a", "a", "a", "ood1", "ood2", "b"], dtype=object)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _lo, _hi, flag = est.predict_interval_mondrian(Xt, gt, 0.1, return_ood=True)
        assert flag.tolist() == [False, False, False, True, True, False]
        assert est.runtime_stats_["mondrian_ood_fraction"] == pytest.approx(2 / 6)
        assert conf.mondrian_ood_summary(est, 0.1)["n_ood"] == 2


def _profile_mondrian_ood(n: int = 200_000, n_groups: int = 500) -> None:
    """cProfile harness for calibrate + predict at a representative shape. Run manually:
    ``python -c "from tests.training.composite.test_biz_val_conformal_ood import _profile_mondrian_ood as p; p()"``.
    See the module docstring for the captured top-3 hotspots + the no-actionable-speedup verdict.
    """
    import cProfile
    import io
    import pstats

    rng = np.random.default_rng(0)
    groups = np.array([f"g{i}" for i in rng.integers(0, n_groups, size=n)], dtype=object)
    sig = 0.5 + 2.5 * rng.random(n_groups)
    y = rng.normal(0.0, sig[rng.integers(0, n_groups, size=n)])
    X = np.arange(n)
    tg = np.array([f"g{i}" for i in rng.integers(0, n_groups, size=n)], dtype=object)
    alphas = [0.05, 0.1]
    st = _Stub()
    conf.calibrate_conformal_mondrian(st, X, y, groups, alphas)  # warm
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(3):
        conf.calibrate_conformal_mondrian(st, X, y, groups, alphas)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            conf.predict_interval_mondrian(st, X, tg, 0.1, return_ood=True)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(14)
    print(s.getvalue())
