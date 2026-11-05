"""Hybrid_orth variant groups 1, 3 and 4 run together: the fit equals the sequential one, and the fold keeps the first owner of every name."""

import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl._hybrid_orth_family_variants._concurrent import _drop_repeated_features, _fold_recipes


class _Est:
    """A bare estimator holding ``hybrid_orth_features_``."""


def test_fold_keeps_the_first_owner_of_a_recipe_name():
    """A later group's recipe for a name an earlier group already owns, or for a skipped name, is not folded in."""
    target = {"base": 0, "a": "first"}
    _fold_recipes(target, {"base": 0}, {"base": 0, "a": "second", "b": 2, "c": 3}, skip={"c"})
    assert target == {"base": 0, "a": "first", "b": 2}


def test_repeated_hybrid_feature_names_are_listed_once():
    """A name two variants emit stays once, in its first position."""
    est = _Est()
    est.hybrid_orth_features_ = ["x4__fourier3", "x4__fourier3", "x4__He3"]
    _drop_repeated_features(est)
    assert est.hybrid_orth_features_ == ["x4__fourier3", "x4__He3"]


def test_concurrent_groups_give_the_sequential_fit(monkeypatch):
    """Same selected features, hybrid features and transformed values with the groups run together and one by one."""
    from mlframe.feature_selection._benchmarks.profiling import family_scale as Fm
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = Fm._make(4000, 12, 3)
    kw = Fm.all_families_kwargs()
    fits = {}
    for mode in ("0", "1"):
        monkeypatch.setenv("MLFRAME_FE_HYBRID_VARIANT_CONCURRENCY", mode)
        MRMR._FIT_CACHE.clear()
        m = MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(X, y)
        fits[mode] = (list(map(str, m.get_feature_names_out())), list(map(str, m.hybrid_orth_features_ or [])), m.transform(X.iloc[:500]))
    assert fits["0"][0] == fits["1"][0]
    assert fits["0"][1] == fits["1"][1]
    pd.testing.assert_frame_equal(fits["0"][2], fits["1"][2])


def test_both_switches_must_allow_the_concurrent_groups(monkeypatch):
    """The groups run together only when the cascade-level switch and the group-level switch are both on."""
    from mlframe.feature_selection.filters._mrmr_fit_impl._hybrid_orth_family_variants._concurrent import hybrid_variant_concurrency_enabled

    monkeypatch.delenv("MLFRAME_FE_STAGE_CONCURRENCY", raising=False)
    monkeypatch.delenv("MLFRAME_FE_HYBRID_VARIANT_CONCURRENCY", raising=False)
    assert hybrid_variant_concurrency_enabled() is True
    monkeypatch.setenv("MLFRAME_FE_HYBRID_VARIANT_CONCURRENCY", "0")
    assert hybrid_variant_concurrency_enabled() is False
    monkeypatch.delenv("MLFRAME_FE_HYBRID_VARIANT_CONCURRENCY")
    monkeypatch.setenv("MLFRAME_FE_STAGE_CONCURRENCY", "0")
    assert hybrid_variant_concurrency_enabled() is False


def test_groups_run_one_after_another_when_their_peaks_do_not_fit_the_device(monkeypatch):
    """The group peaks, scaled to the rows, are checked against the device as the FE cascades' are: when they do not fit, ``None`` goes back and the caller runs the groups in order."""
    import pandas as pd

    from mlframe.feature_selection.filters._mrmr_fit_impl import _fe_stage_concurrency as C
    from mlframe.feature_selection.filters._mrmr_fit_impl._hybrid_orth_family_variants._concurrent import _GROUP_VRAM_WEIGHTS, run_groups_concurrently

    seen = []

    def _no_fit(vram_weights, n_rows):
        """Record the check's inputs; the device is full."""
        seen.append((tuple(vram_weights), int(n_rows)))
        return False

    monkeypatch.setattr(C, "_stages_fit_device", _no_fit)
    X = pd.DataFrame({"a": range(10)})
    ran = []
    groups = tuple((lambda X, **k: ran.append(1) or X) for _ in range(4))
    assert run_groups_concurrently(_Est(), X, {"_hybrid_orth_pre_recipes": {}}, groups) is None
    assert seen == [(_GROUP_VRAM_WEIGHTS, 10)]
    assert ran == []
