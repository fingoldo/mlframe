"""Weight-aware feature selection through ``train_mlframe_models_suite``.

``FeatureSelectionConfig.use_sample_weights_in_fs`` is the only knob that makes FS weight-aware: when True the suite stamps
``_mlframe_use_sample_weights_in_fs_=True`` on every MRMR / RFECV instance, and ``_passthrough_cols_fit_transform`` forwards the
active suite-level ``sample_weight`` into ``MRMR.fit``. MRMR consumes it by resampling rows with replacement proportional to the
weights (``_maybe_resample_for_sample_weight``) before MI screening, so a heavily-weighted minority slice can flip which feature
the selector keeps. Prior coverage only stamped/checked the marker at the unit level; nothing ran the suite end-to-end with a
``sample_weight``-producing extractor and asserted (a) the weights actually reach ``MRMR.fit`` only under the flag, (b) they
change the selection, and (c) the FS-cache reuse invariant (flag=False -> exactly one MRMR fit per target across weight schemas).
The flag was inert because the per-strategy ``clone()`` stripped the setattr-applied ``_mlframe_use_sample_weights_in_fs_``
marker; it is now re-asserted on the clone so ``_passthrough_cols_fit_transform`` forwards the weights.

Design of the discriminating synthetic
---------------------------------------
Two disjoint subpopulations in one frame:
  * 80% of rows ("A-driven"): y is a clean linear function of feature ``A``; ``B`` is pure noise there.
  * 20% of rows ("B-driven"): y is a clean linear function of feature ``B``; ``A`` is pure noise there.
Under UNIFORM weighting the A-slice (4x larger) dominates the MI screen, so MRMR keeps ``A``. The extractor emits a ``weighted``
schema that puts a large weight on the 20% B-slice and a tiny weight on the 80% A-slice; when those weights reach MRMR's
resample step the effective sample is dominated by B-rows, so the weighted MI screen keeps ``B`` instead. flag=False -> A wins
(weights ignored); flag=True -> B wins (weights honoured).

The selection-content assertions (A vs B) are the business signal but are selection-noise-sensitive, so they are paired with a
robust mechanism pin: a monkeypatch spy on ``MRMR.fit`` that records, per call, whether ``sample_weight is not None``. The spy is
the load-bearing sensor -- weights-not-None must appear iff the flag is True.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.configs import TargetTypes
from mlframe.training import FeatureSelectionConfig, OutputConfig
from mlframe.feature_selection.filters import MRMR

from .shared import SimpleFeaturesAndTargetsExtractor

_SEED = 20260610
_N_A = 400  # 80% A-driven rows
_N_B = 100  # 20% B-driven rows (heavily up-weighted under the 'weighted' schema)
_W_BIG = 50.0  # weight on the B-slice
_W_SMALL = 0.02  # weight on the A-slice


def _make_two_subpop_frame(seed: int = _SEED) -> pd.DataFrame:
    """A frame where A explains y on the 80% majority and B explains y on the 20% minority.

    Crucially A and B are independent noise on the OTHER slice, so the MI screen's verdict depends entirely on which slice
    dominates the (possibly weighted) sample -- exactly the lever ``use_sample_weights_in_fs`` controls.

    ``seed`` is per-test-distinct so the process-global ``_PRE_PIPELINE_CACHE`` (keyed by train_df content signature) never
    collides across tests in the same session; without this a later test's identical-content frame would hit the cache and
    skip the MRMR.fit the spy must observe.
    """
    rng = np.random.default_rng(seed)
    n = _N_A + _N_B

    a = rng.standard_normal(n).astype(np.float64)
    b = rng.standard_normal(n).astype(np.float64)
    # Two extra pure-noise distractors so MRMR has a non-trivial field to choose from.
    c = rng.standard_normal(n).astype(np.float64)
    d = rng.standard_normal(n).astype(np.float64)

    is_b_slice = np.zeros(n, dtype=bool)
    is_b_slice[_N_A:] = True  # last 100 rows are the B-driven minority

    y = np.empty(n, dtype=np.float64)
    noise = 0.05 * rng.standard_normal(n)
    y[~is_b_slice] = 3.0 * a[~is_b_slice] + noise[~is_b_slice]  # majority: y = f(A)
    y[is_b_slice] = 3.0 * b[is_b_slice] + noise[is_b_slice]  # minority: y = f(B)

    return pd.DataFrame(
        {
            "A": a,
            "B": b,
            "C": c,
            "D": d,
            "_is_b_slice": is_b_slice.astype(np.int8),  # carried only to build weights; dropped as a feature below
            "target": y,
        }
    )


class _WeightedSubpopExtractor(SimpleFeaturesAndTargetsExtractor):
    """Regression extractor that emits the requested weight schemas, up-weighting the B-slice for the 'weighted' schema.

    The base ``SimpleFeaturesAndTargetsExtractor`` only knows 'uniform' / 'recency'; here we override ``transform`` to inject a
    'weighted' array tied to the ``_is_b_slice`` marker column. ``_is_b_slice`` is added to ``columns_to_drop`` so it never
    reaches the models or the selector as a feature.
    """

    def __init__(self, weight_schemas):
        super().__init__(target_column="target", regression=True, weight_schemas=weight_schemas)

    def transform(self, df):
        out = list(super().transform(df))
        # out = (df, target_by_type, group_ids_raw, group_ids, timestamps, artifacts, cols_to_drop, sample_weights)
        n = df.shape[0]
        is_b = np.asarray(df["_is_b_slice"].values, dtype=bool)
        sample_weights = dict(out[7])
        if "weighted" in (self.weight_schemas or ()):
            w = np.where(is_b, _W_BIG, _W_SMALL).astype(np.float64)
            sample_weights["weighted"] = w
        out[7] = sample_weights
        cols_to_drop = list(out[6])
        if "_is_b_slice" not in cols_to_drop:
            cols_to_drop.append("_is_b_slice")
        out[6] = cols_to_drop
        return tuple(out)


def _mrmr_fs_config(use_sample_weights_in_fs: bool) -> FeatureSelectionConfig:
    """Tiny simple-mode MRMR config; deterministic seed; CPU-only."""
    return FeatureSelectionConfig(
        use_mrmr_fs=True,
        use_sample_weights_in_fs=use_sample_weights_in_fs,
        mrmr_kwargs={
            "verbose": 0,
            "max_runtime_mins": 1,
            "n_workers": 1,
            "quantization_nbins": 5,
            "use_simple_mode": True,
            "random_seed": _SEED,
        },
    )


def _install_mrmr_fit_spy(monkeypatch):
    """Monkeypatch ``MRMR.fit`` to record per-call (sample_weight_is_not_none, selected_feature_names).

    Returns the records list. The wrapper delegates to the real ``MRMR.fit`` so selection behaviour is unchanged; it only
    observes whether ``sample_weight`` arrived and what got selected. ``selected_feature_names`` is read post-fit from
    ``get_feature_names_out`` (falls back to ``support_`` over the input columns) -- best-effort, never raises.
    """
    records: list[dict] = []
    real_fit = MRMR.fit

    def spy_fit(self, X, y, groups=None, sample_weight=None, **fit_params):
        result = real_fit(self, X, y, groups=groups, sample_weight=sample_weight, **fit_params)
        selected = None
        try:
            selected = list(self.get_feature_names_out())
        except Exception:
            try:
                cols = list(X.columns) if hasattr(X, "columns") else None
                if cols is not None and getattr(self, "support_", None) is not None:
                    sup = np.asarray(self.support_, dtype=bool)
                    selected = [c for c, k in zip(cols, sup) if k]
            except Exception:
                selected = None
        records.append({"sw_not_none": sample_weight is not None, "selected": selected})
        return result

    monkeypatch.setattr(MRMR, "fit", spy_fit, raising=True)
    return records


def _run_suite(df, fs_config, temp_data_dir, common_init_params, fast_iterations, weight_schemas):
    fte = _WeightedSubpopExtractor(weight_schemas=weight_schemas)
    models, metadata = train_mlframe_models_suite(
        df=df,
        target_name="weight_aware_fs",
        model_name="waf",
        features_and_targets_extractor=fte,
        mlframe_models=["lgb"],
        hyperparams_config={"iterations": fast_iterations},
        reporting_config=common_init_params,
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
        verbose=0,
        feature_selection_config=fs_config,
    )
    return models, metadata


def _union_selected(records):
    """All feature names selected across every recorded MRMR fit (defensive against per-fit None)."""
    out: set = set()
    for r in records:
        if r["selected"]:
            out.update(r["selected"])
    return out


def _top_ranked(records):
    """First (highest-relevance) feature of the last MRMR fit that selected anything, or None.

    ``get_feature_names_out`` returns features in MRMR's relevance ranking; at small n both informative features
    (A and B) survive selection so the weight-aware signal is the RANK flip (A-first uniform vs B-first weighted),
    not the selection-set difference."""
    for r in reversed(records):
        if r["selected"]:
            return r["selected"][0]
    return None


@pytest.mark.slow
class TestWeightAwareFeatureSelectionSuite:
    """End-to-end weight-aware FS through ``train_mlframe_models_suite``."""

    def test_flag_false_ignores_weights_and_selects_majority_feature_A(self, temp_data_dir, common_init_params, fast_iterations, monkeypatch):
        """flag=False: ``sample_weight`` must NOT reach MRMR.fit (spy records all-None) and the uniform MI screen, dominated
        by the 80% A-slice, must keep ``A``. The spy's sw-None pin is the robust sensor; the A-in-selection check is the
        business signal."""
        df = _make_two_subpop_frame(seed=_SEED + 1)
        records = _install_mrmr_fit_spy(monkeypatch)

        models, _ = _run_suite(
            df,
            _mrmr_fs_config(use_sample_weights_in_fs=False),
            temp_data_dir,
            common_init_params,
            fast_iterations,
            weight_schemas=("uniform", "weighted"),
        )

        assert TargetTypes.REGRESSION in models
        assert len(records) >= 1, "MRMR.fit was never called -- FS did not run"
        # Mechanism pin: with the flag OFF, weights never reach the selector.
        assert all(not r["sw_not_none"] for r in records), "use_sample_weights_in_fs=False but sample_weight reached MRMR.fit"
        # Business signal (selection-noise-tolerant via union across fits): majority feature A is kept.
        selected = _union_selected(records)
        if selected:
            assert "A" in selected, f"uniform-weighted FS should keep majority feature A, got {sorted(selected)}"

    def test_flag_true_honours_weights_and_selects_minority_feature_B(self, temp_data_dir, common_init_params, fast_iterations, monkeypatch):
        """flag=True with the 'weighted' schema up-weighting the 20% B-slice: ``sample_weight`` MUST reach MRMR.fit on the
        weighted schema (spy records sw-not-None at least once) and the weight-aware MI screen, now dominated by B-rows,
        must keep ``B``."""
        df = _make_two_subpop_frame(seed=_SEED + 2)
        records = _install_mrmr_fit_spy(monkeypatch)

        models, _ = _run_suite(
            df,
            _mrmr_fs_config(use_sample_weights_in_fs=True),
            temp_data_dir,
            common_init_params,
            fast_iterations,
            weight_schemas=("weighted",),
        )

        assert TargetTypes.REGRESSION in models
        assert len(records) >= 1, "MRMR.fit was never called -- FS did not run"
        # Mechanism pin (load-bearing, correct behaviour): under the flag, the weighted schema's weights reach MRMR.fit.
        # The per-strategy ``clone()`` strips the setattr-applied marker; it is re-asserted on the clone so _wants_sw
        # stays True and ``sample_weight`` is forwarded into MRMR.fit.
        assert any(r["sw_not_none"] for r in records), "use_sample_weights_in_fs=True but sample_weight never reached MRMR.fit"
        # Business signal: the up-weighted B-slice flips the selection to B.
        selected = _union_selected(records)
        if selected:
            assert "B" in selected, f"weight-aware FS should flip selection to up-weighted minority feature B, got {sorted(selected)}"

    def test_weights_change_which_feature_is_selected(self, temp_data_dir, common_init_params, fast_iterations, monkeypatch):
        """Two statistically-equivalent frames (same generative process, distinct seeds), two suite runs differing ONLY in
        the flag: the top-ranked feature must flip (A-first under uniform vs B-first under weighted). A stable-across-flag
        ranking means weights are inert. Distinct seeds also keep the process-global pre-pipeline cache from letting the
        second run skip MRMR.fit on identical content."""
        df_off = _make_two_subpop_frame(seed=_SEED + 3)
        df_on = _make_two_subpop_frame(seed=_SEED + 30)

        records_off = _install_mrmr_fit_spy(monkeypatch)
        _run_suite(
            df_off,
            _mrmr_fs_config(use_sample_weights_in_fs=False),
            temp_data_dir,
            common_init_params,
            fast_iterations,
            weight_schemas=("uniform", "weighted"),
        )
        # Snapshot the off-run records BEFORE installing the second spy: monkeypatch stacks, so the on-run's spy
        # delegates through the first spy, which would re-append on-run fits to ``records_off`` and pollute the pin.
        off_sw_flags = [r["sw_not_none"] for r in records_off]
        top_off = _top_ranked(records_off)

        records_on = _install_mrmr_fit_spy(monkeypatch)
        _run_suite(
            df_on,
            _mrmr_fs_config(use_sample_weights_in_fs=True),
            temp_data_dir,
            common_init_params,
            fast_iterations,
            weight_schemas=("weighted",),
        )
        top_on = _top_ranked(records_on)

        # Mechanism contrast: weights reach the selector only under the flag. flag=False side is a hard pin (weights
        # must NOT leak); flag=True side is the correct behaviour the marker-stripping clone broke.
        assert all(not f for f in off_sw_flags)
        assert any(r["sw_not_none"] for r in records_on), "use_sample_weights_in_fs=True but sample_weight never reached MRMR.fit"

        # Business signal: the weight-aware MI screen flips the top-ranked feature from the majority A (uniform) to the
        # up-weighted minority B (weighted). Both A and B survive selection at this n, so the rank flip is the signal.
        if top_off is not None and top_on is not None:
            assert top_off != top_on, f"weight-aware FS produced identical top-ranked feature to uniform FS (both {top_off!r}); weights inert"
            assert top_on == "B", f"weighted FS should rank up-weighted minority B first, got {top_on!r}"

    def test_flag_false_reuses_single_fs_fit_across_weight_schemas(self, temp_data_dir, common_init_params, fast_iterations, monkeypatch):
        """FS-cache reuse invariant. flag=False is the default-OFF contract: FS is computed ONCE per target and reused across
        weight schemas. With two schemas in one run ('uniform','weighted') exactly ONE real MRMR fit must occur for the single
        target -- the second schema hits the FS cache. A second fit would mean the cache key spuriously folds the weight schema
        even though weights are weight-blind here."""
        df = _make_two_subpop_frame(seed=_SEED + 4)
        records = _install_mrmr_fit_spy(monkeypatch)

        _run_suite(
            df,
            _mrmr_fs_config(use_sample_weights_in_fs=False),
            temp_data_dir,
            common_init_params,
            fast_iterations,
            weight_schemas=("uniform", "weighted"),
        )

        # Weight-blind FS across both schemas: the cache collapses to one fit per target. A second fit would mean
        # the FS-cache key spuriously folds the weight schema even though weights are FS-blind here.
        assert all(not r["sw_not_none"] for r in records)
        n_fits = len(records)
        assert n_fits == 1, (
            f"use_sample_weights_in_fs=False ran MRMR.fit {n_fits} times for one target across 2 weight schemas; "
            "the FS-cache reuse invariant expects exactly 1 (the second schema must hit the cache)"
        )


@pytest.mark.parametrize("use_flag", [False, True])
def test_weight_aware_fs_marker_reaches_selector_via_suite(use_flag, temp_data_dir, common_init_params, fast_iterations, monkeypatch):
    """Fast representative covering both flag values: the spy confirms ``sample_weight`` is forwarded to MRMR.fit iff the flag
    is True. Tiny single-schema run so it stays well under the per-test budget; the heavy selection-content assertions live in
    the slow class above."""
    # Per-param distinct seed so the two parametrisations never share a process-global pre-pipeline cache entry.
    df = _make_two_subpop_frame(seed=_SEED + (50 if use_flag else 5))
    records = _install_mrmr_fit_spy(monkeypatch)

    schemas = ("weighted",) if use_flag else ("uniform",)
    models, _ = _run_suite(
        df,
        _mrmr_fs_config(use_sample_weights_in_fs=use_flag),
        temp_data_dir,
        common_init_params,
        fast_iterations,
        weight_schemas=schemas,
    )

    assert TargetTypes.REGRESSION in models
    assert len(records) >= 1
    if use_flag:
        # Correct behaviour: weights reach MRMR.fit under the flag (marker re-asserted on the per-strategy clone).
        assert any(r["sw_not_none"] for r in records), "use_sample_weights_in_fs=True but sample_weight never reached MRMR.fit"
    else:
        assert all(not r["sw_not_none"] for r in records), "flag=False: sample_weight must NOT reach MRMR.fit"


def test_per_strategy_clone_preserves_weight_aware_marker():
    """Regression: ``sklearn.clone`` strips the setattr-applied ``_mlframe_use_sample_weights_in_fs_`` marker, which
    made weight-aware FS inert at the per-strategy clone in ``_train_one_target``. ``_forward_selector_sticky_attrs``
    re-asserts it on the clone, so the marker (and the selector-kind tag) survive."""
    from sklearn.base import clone
    from mlframe.training.core._phase_train_one_target_body import _forward_selector_sticky_attrs

    mrmr = MRMR(verbose=0, use_simple_mode=True, n_workers=1)
    mrmr._mlframe_use_sample_weights_in_fs_ = True
    mrmr._mlframe_selector_kind_ = "MRMR"

    cloned = clone(mrmr)
    assert not getattr(cloned, "_mlframe_use_sample_weights_in_fs_", False), "clone unexpectedly carried the marker"

    _forward_selector_sticky_attrs(mrmr, cloned)
    assert cloned._mlframe_use_sample_weights_in_fs_ is True
    assert cloned._mlframe_selector_kind_ == "MRMR"
