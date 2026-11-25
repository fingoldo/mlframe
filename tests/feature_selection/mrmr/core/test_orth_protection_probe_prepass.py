"""The significance probes of the orth-basis protection pass run on a pool before the pass reads them; the pass keeps the same columns."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl._friend_graph_and_redundancy import _group2 as G


def _recipe(kind, *sources):
    """A recipe stub: its kind and the names of its source columns."""
    return SimpleNamespace(kind=kind, src_names=sources)


def test_the_sources_a_recipe_offers_to_the_protection_pass():
    """One source, or the two of a binned aggregate; a hinge leg, a pair and a recipe without sources offer none."""
    assert G._orth_protection_sources(_recipe("orth_basis", "a")) == ("a",)
    assert G._orth_protection_sources(_recipe("binned_numeric_agg", "g", "v")) == ("g", "v")
    assert G._orth_protection_sources(_recipe("hinge_basis", "a")) is None
    assert G._orth_protection_sources(_recipe("orth_basis", "a", "b")) is None
    assert G._orth_protection_sources(_recipe("orth_basis")) is None
    assert G._orth_protection_sources(None) is None


def _gate_inputs(n_sources=8):
    """Columns s0.., one basis column per source (plus a hinge leg and a pair basis that the pass skips), the first two sources already selected."""
    sources = [f"s{i}" for i in range(n_sources)]
    feats = [f"{s}__b" for s in sources] + ["leg__hinge", "pair__b"]
    cols = sources + feats
    recipes = {f"{s}__b": _recipe("orth_basis", s) for s in sources}
    recipes["leg__hinge"] = _recipe("hinge_basis", "s3")
    recipes["pair__b"] = _recipe("orth_basis", "s4", "s5")
    return sources, feats, cols, recipes


def _run_pass(monkeypatch, probe_min_sources):
    """Run the pass with a probe that records its calls and calls every odd source significant; the held-out check accepts every column."""
    calls = []

    def probe(src_name, _index, _data, _target, _nbins, memo):
        """The significance of a source, memoised by name as the real probe does: the odd ones."""
        if src_name not in memo:
            calls.append(src_name)
            memo[src_name] = int(src_name[1:]) % 2 == 1
        return memo[src_name]

    monkeypatch.setattr(G, "_orth_source_is_signal_impl", probe)
    monkeypatch.setattr(G, "_PARALLEL_PROBE_MIN_SOURCES", probe_min_sources)
    _sources, feats, cols, recipes = _gate_inputs()
    n = 50
    X = pd.DataFrame({c: np.arange(n, dtype=np.float32) for c in cols})
    out = G._orth_feature_heldout_gate_pass(
        feats, [0, 1], True, cols, np.zeros((n, len(cols)), dtype=np.int8), (0,), np.full(len(cols), 4), recipes, {c: np.arange(n, dtype=np.float32) for c in feats}, X, lambda values, _extra: 1.0, False
    )
    return calls, list(out)


def test_the_prepass_probes_each_needed_source_once_and_the_pass_keeps_the_same_columns(monkeypatch):
    """With the pool and without it the pass re-adds the same columns; each source that needed a probe is probed once, none of the selected ones, and the pool probes them all before the pass."""
    serial_calls, serial_out = _run_pass(monkeypatch, 10**9)
    pooled_calls, pooled_out = _run_pass(monkeypatch, 4)
    assert pooled_out == serial_out and len(serial_out) > 2
    assert "s0" not in pooled_calls and "s1" not in pooled_calls  # the selected sources are never probed
    assert sorted(set(serial_calls)) == sorted(set(pooled_calls)) or set(serial_calls) <= set(pooled_calls)
    assert len(pooled_calls) == len(set(pooled_calls))  # no source is probed twice
    assert {f"s{i}" for i in range(2, 8)} <= set(pooled_calls)
