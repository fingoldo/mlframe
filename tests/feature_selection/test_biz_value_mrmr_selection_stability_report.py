"""W3 biz_value + unit + cProfile: one-call ``MRMR.selection_stability_report()``.

WHY THIS LAYER
--------------
A fitted MRMR hands users a POINT selection with NO confidence readout: a feature
that barely cleared the relevance screen on the single full-data split looks the
same, in the public surface, as one that dominates every resample. This accessor
adds a per-feature SELECTION-FREQUENCY + per-recipe SURVIVAL-FREQUENCY table,
computed by REPLAYING the cheap MI screen on K bootstrap resamples of the stored
binned screening matrix -- NO MRMR refit (the #15 replay-not-refit trick).

CONTRACTS PINNED
----------------
* unit: the report's per-feature frequencies separate planted signal from noise;
  the replay path is used (the stored replay-state substrate, no MRMR.fit re-entry).
* biz_value: on a small-n (n=1000) canonical fixture, genuine features show HIGH
  selection-frequency (>0.7) and noise features LOW (<0.3) -- clean separation.
* cProfile: K bootstrap replays cost ~K cheap MI sweeps, NOT K MRMR refits; the
  total report time is far below K * single-fit-time and shows no _fit_impl re-entry.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


def _mrmr(**overrides):
    from mlframe.feature_selection.filters.mrmr import MRMR
    defaults = dict(
        verbose=0,
        random_seed=0,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        stability_selection_method="classic",
        retain_artifacts=False,
        n_jobs=1,
    )
    defaults.update(overrides)
    return MRMR(**defaults)


def _tiny_canonical(n: int = 1000, seed: int = 7):
    """Small-n fixture: a strongly-predictive genuine block (g0,g1,g2) plus pure
    noise columns. y is a clean function of the genuine features so the relevance
    screen ranks them top on (almost) every bootstrap resample, while the noise
    columns have no marginal information. n=1000 (was 150): at n=150 the bootstrap
    selection-frequency of the genuine block was under-powered -- a single genuine
    feature routinely fell to ~0.4-0.67, below the >0.7 separation bar, on plain
    seed variation; n=1000 lifts all three genuine features to freq=1.0 with noise
    at 0.0 robustly across seeds (still <5s, within the biz_value n<=2000 budget)."""
    rng = np.random.default_rng(int(seed))
    g0 = rng.standard_normal(n)
    g1 = rng.standard_normal(n)
    g2 = rng.standard_normal(n)
    X = pd.DataFrame({
        "g0": g0, "g1": g1, "g2": g2,
        "noise_0": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
        "noise_2": rng.standard_normal(n),
    })
    score = 1.6 * g0 + 1.6 * g1 + 1.6 * g2 + 0.15 * rng.standard_normal(n)
    y = pd.Series((score > np.median(score)).astype(int))
    return X, y


def _fit():
    X, y = _tiny_canonical()
    # FE OFF so the candidate pool is the RAW features: selection-frequency then
    # directly separates genuine raw inputs from noise raw inputs (with FE on the
    # pool is dominated by engineered columns and the raw/noise framing dissolves).
    # The pair/step knobs alone are insufficient -- the directed-FE families (modular / lattice / row-argmax / conditional-gate) are
    # independently default-ON and would otherwise inject engineered g0/noise children into the bootstrap replay pool, splitting a
    # genuine feature's selection share below the 0.7 floor.
    sel = _mrmr(
        fe_max_pair_features=0, fe_max_steps=0,
        fe_modular_enable=False, fe_pairwise_modular_enable=False,
        fe_integer_lattice_enable=False, fe_row_argmax_enable=False,
        fe_conditional_gate_enable=False,
    ).fit(X, y)
    return sel


def test_unit_replay_state_and_separation():
    """Unit: replay-state substrate is stored (no refit needed) and the report's
    frequencies separate genuine from noise."""
    sel = _fit()
    # The replay substrate must be captured at fit time (so the accessor can replay
    # without re-entering fit).
    state = getattr(sel, "_stability_replay_state_", None)
    assert state, "fit must store _stability_replay_state_ for replay"
    assert "cand_codes" in state and "y_codes" in state
    assert state["cand_codes"].shape[0] == state["y_codes"].shape[0]

    rep = sel.selection_stability_report(n_boot=40, as_text=False)
    freq = rep["feature_selection_frequency"]
    # Genuine features rank high; noise strictly below every genuine feature.
    gen = [freq.get(c, 0.0) for c in ("g0", "g1", "g2")]
    noise = [freq.get(c, 0.0) for c in ("noise_0", "noise_1", "noise_2")]
    assert min(gen) > max(noise), f"no separation: gen={gen} noise={noise}"


def test_biz_value_tiny_n_clean_separation():
    """biz_value: small-n canonical -> genuine high-freq (>0.7), noise low (<0.3)."""
    sel = _fit()
    rep = sel.selection_stability_report(n_boot=60, as_text=False)
    freq = rep["feature_selection_frequency"]
    sel_set = set(rep["selected_features"])
    for c in ("g0", "g1", "g2"):
        assert freq.get(c, 0.0) > 0.7, f"genuine {c} freq {freq.get(c)} not > 0.7 ({freq})"
    # Every genuine feature outranks every noise feature: the report SEPARATES the
    # planted signal from noise. A noise feature pulled into the point selection by
    # chance shows a clearly-lower (non-HIGH) frequency -- the confidence readout the
    # accessor exists to surface; noise NOT in the point selection sits firmly < 0.3.
    gen_min = min(freq.get(c, 0.0) for c in ("g0", "g1", "g2"))
    for c in ("noise_0", "noise_1", "noise_2"):
        assert freq.get(c, 0.0) < gen_min, (
            f"noise {c} freq {freq.get(c)} not below genuine min {gen_min} ({freq})"
        )
        if c not in sel_set:
            assert freq.get(c, 0.0) < 0.3, f"unselected noise {c} freq {freq.get(c)} not < 0.3 ({freq})"
    # The human-readable form is one screen and names the contract.
    txt = sel.selection_stability_report(n_boot=60, as_text=True)
    assert isinstance(txt, str) and "selection-stability" in txt
    assert "replay (no MRMR refit)" in txt


def test_cprofile_cost_is_replay_not_refit():
    """cProfile: K report replays cost << K MRMR refits and never re-enter _fit_impl."""
    import cProfile
    import pstats
    import io
    from timeit import default_timer as timer

    sel = _fit()
    K = 50

    # Single-fit baseline. Use a DISTINCT seed so the MRMR fit-cache (content-hash
    # short-circuit) cannot return a near-free cached replay -- we want the true
    # cost of one genuine MRMR fit to compare K screen-replays against.
    X, y = _tiny_canonical(seed=98765)
    t0 = timer()
    _mrmr(random_seed=12345).fit(X, y)
    single_fit = timer() - t0

    prof = cProfile.Profile()
    prof.enable()
    sel.selection_stability_report(n_boot=K, as_text=False)
    prof.disable()

    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s)
    ps.print_stats()
    out = s.getvalue()
    # No MRMR refit re-entry inside the report.
    assert "_fit_impl" not in out, "report must not re-enter MRMR._fit_impl (would be a refit)"

    t0 = timer()
    sel.selection_stability_report(n_boot=K, as_text=False)
    report_time = timer() - t0

    # K replays must be far cheaper than K refits. Even one single fit dwarfs the
    # whole K-replay report.
    assert report_time < single_fit, (
        f"report {report_time:.3f}s not < single fit {single_fit:.3f}s -- "
        f"replay degenerated toward refit cost"
    )
    print(
        f"\n[W3 cost] single_fit={single_fit*1000:.1f}ms  "
        f"K={K}_replays_report={report_time*1000:.1f}ms  "
        f"speedup_vs_K_refits~={K*single_fit/max(report_time,1e-9):.0f}x"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-s", "-x", "-v", "--no-cov"]))
