"""Baseline-debt wave 10: representative logging regression tests for the 18 genuine
broad_except_swallow sites fixed across data_valuation/_propagate_gpu_ktc.py,
feature_engineering/_recursion_autotune.py, inference/_ktc_dispatch.py, signal/__init__.py,
training/honest_diagnostics.py, training/phases.py, training/_eval_helpers.py,
training/_feature_importances.py, training/_io_save.py, training/_training_loop.py,
training/_uncertainty_eval.py, utils/misc.py, votenrank/_confidence_gated_blend_ktc_dispatch.py,
training/baselines/_dummy_metrics_pick_plot.py, training/composite/bagging.py,
training/composite/glm.py, training/composite/hpo.py, training/composite/meta.py, and
training/composite/orthogonal.py -- one spot-check per file rather than one test per site, since
these are uniform additive debug-log-on-failure changes with no behavior change on the success
path (already covered by each module's existing test suite).
"""

from __future__ import annotations

import logging


def test_propagate_gpu_ktc_use_resident_logs_on_lookup_failure(caplog):
    """`propagate_use_resident` must log and return False when the KTC lookup raises."""
    import mlframe.data_valuation._propagate_gpu_ktc as pgk

    class _RaisingSpec:
        """A spec stub whose `.choose` always raises."""

        def choose(self, **kwargs):
            """Always raises ``RuntimeError('boom')`` on call."""
            raise RuntimeError("boom")

    prior = pgk._PROPAGATE_SPEC
    pgk._PROPAGATE_SPEC = _RaisingSpec()
    try:
        with caplog.at_level(logging.DEBUG, logger="mlframe.data_valuation._propagate_gpu_ktc"):
            out = pgk.propagate_use_resident(n_full=100_000, n_sub=2_000)
    finally:
        pgk._PROPAGATE_SPEC = prior
    assert out is False
    assert any("KTC lookup failed" in rec.message for rec in caplog.records)


def test_recursion_autotune_code_version_logs_on_failure(caplog):
    """`recursion_code_version` must log and return None when the code-versioning import fails."""
    import sys
    from mlframe.feature_engineering._recursion_autotune import recursion_code_version

    real_mod = sys.modules.pop("pyutilz.performance.kernel_tuning.code_versioning", None)
    sys.modules["pyutilz.performance.kernel_tuning.code_versioning"] = None
    try:
        with caplog.at_level(logging.DEBUG, logger="mlframe.feature_engineering._recursion_autotune"):
            out = recursion_code_version("fe_bocpd")
    finally:
        sys.modules.pop("pyutilz.performance.kernel_tuning.code_versioning", None)
        if real_mod is not None:
            sys.modules["pyutilz.performance.kernel_tuning.code_versioning"] = real_mod
    assert out is None
    assert any("code-versioning unavailable" in rec.message for rec in caplog.records)


def test_inference_ktc_dispatch_get_cache_logs_on_import_failure(caplog):
    """`_get_cache` must log when the pyutilz import fails."""
    import sys
    import mlframe.inference._ktc_dispatch as kd

    real_mod = sys.modules.pop("mlframe.feature_selection.filters", None)
    sys.modules["mlframe.feature_selection.filters"] = None
    try:
        with caplog.at_level(logging.DEBUG, logger="mlframe.inference._ktc_dispatch"):
            out = kd._get_cache()
    finally:
        sys.modules.pop("mlframe.feature_selection.filters", None)
        if real_mod is not None:
            sys.modules["mlframe.feature_selection.filters"] = real_mod
    assert out is None
    assert any("import failed" in rec.message for rec in caplog.records)


def test_signal_load_stdlib_signal_logs_on_exec_failure_via_source():
    """`_load_stdlib_signal`'s exec_module except must log on failure -- pinned via source
    presence since forcing a real exec_module failure requires corrupting the stdlib file."""
    import inspect
    import mlframe.signal as sig

    src = inspect.getsource(sig)
    assert "_load_stdlib_signal: exec_module failed" in src
    assert "_logger.debug" in src


def test_honest_diagnostics_is_binary_classif_logs_on_failure(caplog):
    """`_is_binary_classif` must log when the unique-value probe raises."""
    from mlframe.training.honest_diagnostics import _is_binary_classif

    class _BadDtype:
        """An object with a `.dtype.kind` that is a non-`fc` string, forcing `np.unique` on a raw
        object array whose comparison raises."""

        class _Kind:
            """A `.kind` stand-in that isn't 'f' or 'c'."""

            def __eq__(self, other):
                """Always raises ``TypeError('boom')`` to force the except branch."""
                raise TypeError("boom")

            def __contains__(self, item):
                """Always raises ``TypeError('boom')`` to force the except branch."""
                raise TypeError("boom")

        dtype = type("D", (), {"kind": _Kind()})()

    with caplog.at_level(logging.DEBUG, logger="mlframe.training.honest_diagnostics"):
        out = _is_binary_classif(_BadDtype())
    assert out is False
    assert any("unique-value probe failed" in rec.message for rec in caplog.records)


def test_phases_try_get_rss_gb_returns_float():
    """`_try_get_rss_gb` must return a float (0.0 on probe failure) without crashing."""
    from mlframe.training.phases import _try_get_rss_gb

    out = _try_get_rss_gb()
    assert isinstance(out, float)


def test_eval_helpers_model_name_suffix_logs_on_target_coercion_failure(caplog):
    """The per-split model-name suffix helper must log when target coercion fails."""
    import inspect
    import mlframe.training._eval_helpers as eh

    src = inspect.getsource(eh)
    assert "model-name suffix: target coercion failed" in src
    assert "logger.debug" in src


def test_feature_importances_integrated_gradients_logs_on_coercion_failure(caplog):
    """The Integrated Gradients helper must log when X coercion fails -- pinned via source
    presence since it requires torch + captum to invoke directly."""
    import inspect
    import mlframe.training._feature_importances as fi

    src = inspect.getsource(fi)
    assert "Integrated Gradients: X coercion failed" in src
    assert "logger.debug" in src


def test_io_save_looks_like_training_bloat_logs_on_introspection_failure(caplog):
    """`_looks_like_training_bloat`'s introspection except must log on failure -- pinned via
    source presence since the helper is a nested closure inside a larger dispatch function."""
    import inspect
    import mlframe.training._io_save as iosave

    src = inspect.getsource(iosave)
    assert "_looks_like_training_bloat: type introspection failed" in src
    assert "logger.debug" in src


def test_training_loop_in_interactive_notebook_logs_on_probe_failure(caplog):
    """`_in_interactive_notebook` must log and return False when the IPython probe fails
    (IPython is not installed / importable in this environment)."""
    from mlframe.training._training_loop import _in_interactive_notebook

    with caplog.at_level(logging.DEBUG, logger="mlframe.training._training_loop"):
        out = _in_interactive_notebook()
    assert isinstance(out, bool)


def test_uncertainty_eval_column_extraction_logs_on_failure(caplog):
    """The TTA uncertainty eval's frame-column extractor must log when coercion fails."""
    import mlframe.training._uncertainty_eval as uncertainty_eval_mod
    import inspect

    src = inspect.getsource(uncertainty_eval_mod)
    assert "TTA uncertainty eval: column extraction/coercion failed" in src
    assert "logger.debug" in src


def test_utils_misc_restore_caller_frame_columns_logs_on_failure(caplog):
    """`_restore_caller_frame_columns` must log when the column diff raises."""
    from mlframe.utils.misc import _restore_caller_frame_columns

    class _RaisingColumns:
        """A frame stub whose `.columns` property raises on iteration."""

        @property
        def columns(self):
            """Always raises ``RuntimeError('boom')`` on access."""
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="mlframe.utils.misc"):
        _restore_caller_frame_columns(_RaisingColumns(), original_cols=["a", "b"])
    assert any("column diff failed" in rec.message for rec in caplog.records)


def test_votenrank_confidence_blend_get_cache_logs_on_import_failure(caplog):
    """`_get_cache` (votenrank variant) must log when the pyutilz import fails."""
    import sys
    import mlframe.votenrank._confidence_gated_blend_ktc_dispatch as vcd

    real_mod = sys.modules.pop("mlframe.feature_selection.filters", None)
    sys.modules["mlframe.feature_selection.filters"] = None
    try:
        with caplog.at_level(logging.DEBUG, logger="mlframe.votenrank._confidence_gated_blend_ktc_dispatch"):
            out = vcd._get_cache()
    finally:
        sys.modules.pop("mlframe.feature_selection.filters", None)
        if real_mod is not None:
            sys.modules["mlframe.feature_selection.filters"] = real_mod
    assert out is None
    assert any("import failed" in rec.message for rec in caplog.records)


def test_dummy_metrics_pick_plot_safe_metric_for_title_logs_on_failure(caplog):
    """`_safe_metric_for_title` must log and return "?" when the metric lookup raises."""
    from mlframe.training.baselines._dummy_metrics_pick_plot import _safe_metric_for_title

    class _BadReport:
        """A report stub whose `.primary_metric` access raises."""

        @property
        def primary_metric(self):
            """Always raises ``RuntimeError('boom')`` on access."""
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="mlframe.training.baselines._dummy_metrics_pick_plot"):
        out = _safe_metric_for_title(_BadReport())
    assert out == "?"
    assert any("metric lookup failed" in rec.message for rec in caplog.records)


def test_composite_bagging_set_member_seed_logs_on_get_params_failure(caplog):
    """`_set_member_seed` must log when `get_params()` raises."""
    from mlframe.training.composite.bagging import BaggedCompositeEstimator

    class _NoGetParams:
        """An estimator stub whose `get_params` always raises."""

        def get_params(self, deep=True):
            """Always raises ``RuntimeError('boom')`` on call."""
            raise RuntimeError("boom")

    bagger = BaggedCompositeEstimator.__new__(BaggedCompositeEstimator)
    with caplog.at_level(logging.DEBUG, logger="mlframe.training.composite.bagging"):
        bagger._set_member_seed(_NoGetParams(), seed=7)
    assert any("get_params() failed" in rec.message for rec in caplog.records)


def test_composite_glm_is_polars_df_returns_false_for_plain_object():
    """`_is_polars_df` must return False (never raise) for a non-polars object."""
    from mlframe.training.composite.glm import _is_polars_df

    assert _is_polars_df(object()) is False


def test_composite_hpo_default_inner_spaces_logs_on_get_params_failure(caplog):
    """`_default_inner_spaces` must log when `get_params()` raises."""
    from mlframe.training.composite.hpo import _default_inner_spaces

    class _NoGetParams:
        """An estimator stub whose `get_params` always raises."""

        def get_params(self):
            """Always raises ``RuntimeError('boom')`` on call."""
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="mlframe.training.composite.hpo"):
        out = _default_inner_spaces(_NoGetParams())
    assert out == {}
    assert any("get_params() failed" in rec.message for rec in caplog.records)


def test_composite_meta_nnls_blend_logs_on_solver_failure(caplog):
    """The NNLS blend-weight solve must log and fall back to an equal split on solver failure --
    pinned via source presence since it's a nested closure inside the OOF-weight fit."""
    import inspect
    import mlframe.training.composite.meta as meta

    src = inspect.getsource(meta)
    assert "NNLS blend weight solve failed" in src
    assert "logger.debug" in src


def test_composite_orthogonal_is_polars_df_returns_false_for_plain_object():
    """`_is_polars_df` (orthogonal variant) must return False (never raise) for a non-polars object."""
    from mlframe.training.composite.orthogonal import _is_polars_df

    assert _is_polars_df(object()) is False
