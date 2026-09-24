"""The selection gates that decide which composite targets exist, tested one function at a time (TST-14).

These functions ran only inside end-to-end ``fit`` calls whose assertions were about something else, so a scale mismatch or
a NaN pass-through inside them had no test that could catch it. Writing these found one: the MI-gain gate compared
``mi_gain <= eps`` and ``NaN <= eps`` is False, so an unmeasured spec was admitted and then sorted arbitrarily into the
top-k.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from mlframe.training.composite.discovery import _tiny_rerank_waic
from mlframe.training.composite.discovery._filter_and_gate import filter_sort_and_gate_candidates
from mlframe.training.composite.discovery._fit_helpers import maybe_boost_mi_strata_for_heavy_tail, no_base_candidates_report_entry
from mlframe.training.composite.discovery._rejection_ledger import ledger_init
from mlframe.training.composite.discovery._tiny_rerank_waic import apply_honest_oof_floor
from mlframe.training.composite.spec import CompositeSpec
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _spec(name: str, mi_gain: float = 0.5, transform: str = "diff", base: str = "b") -> CompositeSpec:
    """A spec with just the fields the gates read."""
    return CompositeSpec(name=name, target_col="y", transform_name=transform, base_column=base, fitted_params={},
                         mi_gain=mi_gain, mi_y=0.0, mi_t=0.0, valid_domain_frac=1.0, n_train_rows=100)


def _gate_self(**cfg) -> types.SimpleNamespace:
    """A discovery stand-in whose later gates are switched off, so only the MI-gain gate and the top-k decide."""
    base = dict(enabled=True, eps_mi_gain=0.01, mi_gain_fdr_control=False, structural_fragility_gate_enabled=False,
                detect_linear_residual_alpha_drift=False, top_k_after_mi=10)
    base.update(cfg)
    return types.SimpleNamespace(config=CompositeTargetDiscoveryConfig(**base))


def _run_gate(self, candidates) -> list[str]:
    """Names of the specs the filter keeps."""
    kept = filter_sort_and_gate_candidates(self, candidates, df=None, train_idx=np.arange(10), y_full=np.zeros(10),
                                           y_train=np.zeros(10), extract_column_array=None)
    return [s.name for s in kept]


# ---------------------------------------------------------------------------
# filter_sort_and_gate_candidates
# ---------------------------------------------------------------------------


def test_a_spec_whose_gain_could_not_be_measured_is_rejected_with_a_reason():
    """A NaN MI gain is an unmeasured spec: it must not clear the gate that exists to require a positive gain."""
    candidates = [{"spec": _spec("good", 0.5)}, {"spec": _spec("unmeasured", float("nan"))}]
    assert _run_gate(_gate_self(), candidates) == ["good"]
    assert "not finite" in candidates[1]["reason"]
    assert not candidates[1].get("kept")


def test_a_nan_lower_confidence_bound_rejects_even_with_a_finite_point_gain():
    """Under bootstrap the gate reads the LCB; a NaN LCB is as unmeasured as a NaN point estimate."""
    candidates = [{"spec": _spec("lcb_nan", 0.5), "mi_gain_lcb": float("nan")}]
    assert _run_gate(_gate_self(), candidates) == []


def test_the_eps_gate_and_the_top_k_rank_by_gain_then_by_name():
    """At or below eps is rejected; ties in gain are broken by name, so the cut is the same on every run."""
    candidates = [{"spec": _spec(n, g)} for n, g in [("c", 0.3), ("a", 0.3), ("b", 0.3), ("low", 0.01), ("top", 0.9)]]
    assert _run_gate(_gate_self(top_k_after_mi=3), candidates) == ["top", "a", "b"]


def test_already_rejected_and_fdr_dropped_entries_stay_out():
    """An entry with no spec, or one the FDR control dropped, is not re-admitted by the gate."""
    candidates = [{"spec": None, "reason": "earlier"}, {"spec": _spec("fdr", 0.9), "fdr_dropped": True}, {"spec": _spec("ok", 0.2)}]
    assert _run_gate(_gate_self(), candidates) == ["ok"]


# ---------------------------------------------------------------------------
# apply_honest_oof_floor
# ---------------------------------------------------------------------------


def _floor_self(**cfg) -> types.SimpleNamespace:
    """A discovery stand-in with the floor's two settings and a fresh rejection ledger."""
    self = types.SimpleNamespace(config=types.SimpleNamespace(honest_oof_floor_reject_enabled=True,
                                                              honest_oof_selection_tolerance=1.05, **cfg))
    ledger_init(self)
    return self


def test_the_floor_drops_a_spec_at_or_above_it_and_keeps_the_scores_aligned():
    """10 * 1.05 = 10.5 is the threshold: 10.5 and 12 go, 9 stays, and each survivor keeps its own score."""
    specs = [_spec("a"), _spec("at_floor"), _spec("b"), _spec("worse")]
    agg = [1.0, 2.0, 3.0, 4.0]
    honest = {"a": 9.0, "at_floor": 10.5, "b": 9.5, "worse": 12.0}
    self = _floor_self()
    kept, kept_agg = apply_honest_oof_floor(self, specs, agg, honest, 10.0)
    assert [s.name for s in kept] == ["a", "b"]
    assert kept_agg == [1.0, 3.0], "a dropped spec shifted the scores of the ones after it"
    assert self._tiny_rerank_scores == {"a": 1.0, "b": 3.0}
    assert {r["spec_name"] for r in self.rejection_ledger_} == {"at_floor", "worse"}


def test_an_unmeasured_spec_keeps_its_cv_rank():
    """A spec with no honest measurement, or a NaN one, is not judged by a floor it was never compared with."""
    specs = [_spec("missing"), _spec("nan"), _spec("worse")]
    kept, kept_agg = apply_honest_oof_floor(_floor_self(), specs, [1.0, 2.0, 3.0], {"nan": float("nan"), "worse": 20.0}, 10.0)
    assert [s.name for s in kept] == ["missing", "nan"]
    assert kept_agg == [1.0, 2.0]


@pytest.mark.parametrize("baseline", [float("nan"), float("inf")])
def test_no_usable_baseline_leaves_every_spec_in_place(baseline):
    """Without a finite floor there is nothing to compare against, so the gate is a no-op rather than a mass rejection."""
    specs = [_spec("a"), _spec("b")]
    kept, kept_agg = apply_honest_oof_floor(_floor_self(), specs, [1.0, 2.0], {"a": 1e9, "b": 1e9}, baseline)
    assert kept is specs and kept_agg == [1.0, 2.0]


def test_the_switch_turns_the_floor_off():
    """``honest_oof_floor_reject_enabled=False`` keeps every spec, however far above the floor."""
    self = _floor_self()
    self.config.honest_oof_floor_reject_enabled = False
    specs = [_spec("worse")]
    kept, _ = apply_honest_oof_floor(self, specs, [1.0], {"worse": 1e9}, 1.0)
    assert [s.name for s in kept] == ["worse"]


# ---------------------------------------------------------------------------
# _apply_waic_tiebreak
# ---------------------------------------------------------------------------


def _tiebreak(monkeypatch, waics: list[float | None], agg: list[float]) -> list[int]:
    """Run the tie-break on additive specs whose WAICs are ``waics`` (None = not scorable); return the new order."""
    from mlframe.training.composite.transforms import get_transform

    n = 200
    specs = [_spec(f"s{i}", transform="linear_residual", base=f"b{i}") for i in range(len(waics))]
    for sp in specs:
        object.__setattr__(sp, "fitted_params", get_transform("linear_residual").fit(np.linspace(2.0, 3.0, n), np.linspace(1.0, 2.0, n)))
    cache = {f"b{i}": (np.linspace(1.0, 2.0, n), np.random.default_rng(i).normal(size=(n, 2))) for i in range(len(specs))}
    it = iter(waics)

    def _fake(*_a, **_k):
        """The next scripted WAIC, invalid when scripted as None."""
        w = next(it)
        return types.SimpleNamespace(valid=w is not None and math.isfinite(w), waic=float("nan") if w is None else float(w))

    monkeypatch.setattr("mlframe.training.composite.discovery._eval_waic.compute_transform_waic", _fake)
    self = types.SimpleNamespace(config=types.SimpleNamespace(transform_waic_n_folds=2, random_state=0, top_m_after_tiny=10))
    out = _tiny_rerank_waic._apply_waic_tiebreak(self, np.arange(len(specs)), specs, agg, [s.name for s in specs],
                                                 y_screen=np.linspace(2.0, 3.0, n), per_base_cache=cache)
    return [int(i) for i in out]


def test_a_band_with_an_unscorable_member_keeps_its_rmse_order(monkeypatch):
    """Only a band whose every member has a valid WAIC may be re-ordered; one failure leaves the whole band alone."""
    assert _tiebreak(monkeypatch, [0.0, None, 5.0], [1.0, 1.001, 1.002]) == [0, 1, 2]


def test_a_nan_waic_counts_as_unscorable(monkeypatch):
    """A NaN WAIC must not sort anywhere: the band keeps its RMSE order."""
    assert _tiebreak(monkeypatch, [0.0, float("nan")], [1.0, 1.001]) == [0, 1]


def test_equal_waics_fall_back_to_the_name_order(monkeypatch):
    """A tie in WAIC is broken by name, so the order does not depend on thread completion order."""
    assert _tiebreak(monkeypatch, [3.0, 3.0, 3.0], [1.002, 1.001, 1.0]) == [0, 1, 2]


def test_specs_outside_the_noise_band_are_not_reordered(monkeypatch):
    """Two specs 10% apart in RMSE are not a tie, whatever their WAICs say."""
    assert _tiebreak(monkeypatch, [0.0, 100.0], [1.0, 1.1]) == [0, 1]


# ---------------------------------------------------------------------------
# _fit_helpers
# ---------------------------------------------------------------------------


def test_the_empty_discovery_report_says_why():
    """When every base was filtered out the report has one entry naming the cause, not an ambiguous empty list."""
    [entry] = no_base_candidates_report_entry()
    assert entry["name"] == "__no_base_candidates__" and entry["rejected"] is True and entry["kept"] is False
    assert "no usable base candidates" in entry["reason"]
    assert math.isnan(entry["mi_gain"])


def test_a_heavy_tailed_target_raises_mi_strata_on_a_copy_of_the_config():
    """The boost applies to this discovery's config only; the caller's shared config object keeps its own value."""
    shared = CompositeTargetDiscoveryConfig(enabled=True, mi_n_strata=10, mi_n_strata_heavy_tail=30)
    self = types.SimpleNamespace(config=shared)
    y = np.random.default_rng(0).standard_t(df=2, size=5000)  # heavy tails: kurtosis well above 5
    maybe_boost_mi_strata_for_heavy_tail(self, y)
    assert self.config.mi_n_strata == 30
    assert shared.mi_n_strata == 10, "the caller's config was mutated"


@pytest.mark.parametrize("y", [np.random.default_rng(1).normal(size=5000), np.random.default_rng(2).standard_t(df=2, size=50)])
def test_a_light_tailed_or_tiny_target_leaves_the_strata_alone(y):
    """A Gaussian target has nothing to boost, and fewer than 100 rows cannot support a kurtosis estimate."""
    self = types.SimpleNamespace(config=CompositeTargetDiscoveryConfig(enabled=True, mi_n_strata=10, mi_n_strata_heavy_tail=30))
    maybe_boost_mi_strata_for_heavy_tail(self, y)
    assert self.config.mi_n_strata == 10


def test_the_boost_never_lowers_a_user_floor():
    """A user who already asked for more strata than the heavy-tail value keeps their own number."""
    self = types.SimpleNamespace(config=CompositeTargetDiscoveryConfig(enabled=True, mi_n_strata=50, mi_n_strata_heavy_tail=30))
    maybe_boost_mi_strata_for_heavy_tail(self, np.random.default_rng(0).standard_t(df=2, size=5000))
    assert self.config.mi_n_strata == 50
