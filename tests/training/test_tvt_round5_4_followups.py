"""Round-5.4 follow-up tests covering 3 surgical fixes from the TVT
2026-05-24 production run analysis:

* Fix #1: MLP-skip on extreme-AR + group-aware regressions. The MLP
  cannot learn a transferable residual when lag1_corr >= 0.99 and the
  splitter is group-aware (R2=-286 on test, ensemble quality-gate
  always excludes MLP). Skip saves 2.7 min training + 126 MB save
  dump per run. Mirror of composite-discovery extreme_ar_group_aware_skip
  from round 5.3.

* Fix #2: CT_ENSEMBLE for raw-only targets. When 0 composite specs
  exist (extreme-AR skip fired), the legacy entry guard bypasses
  dummy-floor gate + lag_predict injection, leaving the suite shipping
  an arithmetic ensemble that's WORSE than the best single component.
  Synthesise raw-only entries so the gates run.

* Fix #3: verdict table best_model column falls back to TEST when VAL
  is unpopulated. Prevents the suite-end summary from showing "-" when
  the trained models did get evaluated (just not on val).
"""

from __future__ import annotations

from pathlib import Path


def _module_source(mod) -> str:
    """Read a module's source via ``Path.read_text``. ``inspect.getsource``
    is forbidden in tests per ``feedback_behavioral_tests`` (the meta-test
    ``tests/test_meta/test_no_inspect_getsource.py`` enforces this);
    file-based read achieves the same source-grep contract."""
    return Path(mod.__file__).read_text(encoding="utf-8")


class TestMlpExtremeArGroupAwareSkip:
    def test_behavior_config_has_mlp_knobs_default_off(self) -> None:
        """Skip is a TrainingBehaviorConfig field (2026-05-25 rework:
        previously env-var-gated to avoid the kwarg-splat rejection,
        but configure_training_params now has **_unused_behavior_kwargs
        catch-all so adding behavior knobs is safe). Default off so
        MLP trains by default; opt-in for extreme-AR group-aware regimes."""
        from mlframe.training._model_configs import TrainingBehaviorConfig

        fields = getattr(TrainingBehaviorConfig, "model_fields", {})
        assert "mlp_extreme_ar_group_aware_skip" in fields
        assert "mlp_extreme_ar_threshold" in fields
        cfg = TrainingBehaviorConfig()
        assert cfg.mlp_extreme_ar_group_aware_skip is False
        assert cfg.mlp_extreme_ar_threshold == 0.99

    def test_skip_logic_reads_from_behavior_config(self) -> None:
        """Lock in that the per-model loop reads the skip flag from
        behavior_config (not env). Catches a future refactor that
        re-introduces env-var gating without updating the config."""
        # Monolith-split compat: the extreme-AR gate body was carved out of
        # ``_phase_train_one_target_body`` into the ``_phase_train_one_target_post``
        # sibling; concatenate both so the marker sensor still matches the
        # relocated skip logic.
        from mlframe.training.core import (
            _phase_train_one_target_body,
            _phase_train_one_target_post,
        )

        src = _module_source(_phase_train_one_target_body) + "\n" + _module_source(_phase_train_one_target_post)
        assert "mlp_extreme_ar_group_aware_skip" in src
        assert "mlp_extreme_ar_threshold" in src
        assert "lag1_autocorr_per_group" in src
        assert "prefer_group_aware" in src
        assert "extreme-AR + group-aware skip fired" in src
        assert 'if mlframe_model_name == "mlp":' in src
        # No leftover env-var references for the MLP skip.
        assert "MLFRAME_MLP_EXTREME_AR_GROUP_AWARE_SKIP" not in src
        assert "MLFRAME_MLP_EXTREME_AR_THRESHOLD" not in src


class TestAlwaysBuildCtEnsembleForRaw:
    def test_config_knob_default_on(self) -> None:
        from mlframe.training._composite_target_discovery_config import (
            CompositeTargetDiscoveryConfig,
        )

        fields = getattr(CompositeTargetDiscoveryConfig, "model_fields", None)
        assert fields is not None
        assert "always_build_ct_ensemble_for_raw" in fields
        cfg = CompositeTargetDiscoveryConfig()
        assert cfg.always_build_ct_ensemble_for_raw is True

    def test_phase_composite_post_synthesises_raw_specs(self) -> None:
        """When composite_specs_by_target_type is empty AND the knob
        is on AND a regression target has trained models, synthesise
        an empty-spec entry so the existing loop covers raw-only."""
        from mlframe.training.core import _phase_composite_post

        src = _module_source(_phase_composite_post)
        assert "always_build_ct_ensemble_for_raw" in src
        assert "synthesised raw-only entries" in src
        # The synthesis must skip composite-named targets (they would
        # double-count once their specs land).
        assert "is_composite_target_name" in src

    def test_target_types_import_path_is_correct(self) -> None:
        """Regression guard: the raw-only synthesis block does ``from
        ..target_types import TargetTypes`` (BROKEN -- no such module;
        TargetTypes lives in ``mlframe.training.configs``). TVT prod
        2026-05-25 surfaced this as ModuleNotFoundError at the end of
        Phase 4, killing the suite right before the ensemble verdict.
        The import must resolve from ``mlframe.training.configs``.
        """
        from mlframe.training.core import _phase_composite_post

        src = _module_source(_phase_composite_post)
        assert "from ..target_types import" not in src
        assert "from ..configs import TargetTypes" in src
        # Round-trip the import itself to be sure.
        from mlframe.training.configs import TargetTypes  # noqa: F401


class TestVerdictTableTestFallback:
    def _summary_src(self) -> str:
        """Concatenated source of both candidate modules: the legacy
        ``_phase_composite_post`` (kept for back-compat) and the
        carved-out ``_phase_composite_post_summary`` introduced when
        the file crossed the monolith threshold. Either one may host
        the fallback logic depending on the current split."""
        from mlframe.training.core import (
            _phase_composite_post,
            _phase_composite_post_summary,
        )

        return _module_source(_phase_composite_post) + "\n" + _module_source(_phase_composite_post_summary)

    def test_fallback_logic_present(self) -> None:
        """When val metrics are missing for all trained models, the
        suite-end summary must fall back to test metrics (tagged so
        the operator sees the cross-split comparison)."""
        src = self._summary_src()
        # Both passes (val first, then test fallback) must be present.
        assert '_entry_metric(_m, "val"' in src
        assert '_entry_metric(_m, "test"' in src
        assert "test fallback" in src
        # The "_best_split" tracker prevents tagging val-found models
        # as test fallback by accident.
        assert "_best_split" in src

    def test_fallback_tag_in_display_name(self) -> None:
        """The fallback tag '(test fallback)' is what operators see in
        the verdict table; lock in the exact string so log-grep
        playbooks remain stable."""
        src = self._summary_src()
        assert '"(test fallback)"' in src or "'(test fallback)'" in src


class TestSlidingWindowGrMatchBatchedAgreesWithLoop:
    """Lock in numerical equivalence of the batched matmul variant of
    sliding-window GR match with the original per-row loop. The
    batched variant lives in wellbore.py (contests_wellbore repo) but
    has no upstream mlframe dependency; this test mirrors the same
    logic so a regression here is caught even if wellbore.py drifts.
    """

    def test_batched_matches_loop_within_fp32_tolerance(self) -> None:
        import numpy as np

        def _loop(h_gr, tw_gr, tw_tvt, W=15):
            n = len(h_gr)
            out_tvt = np.full(n, np.nan)
            out_score = np.full(n, np.nan)
            out_idx = np.full(n, np.nan)
            L = 2 * W + 1
            tw_win = np.lib.stride_tricks.sliding_window_view(tw_gr, L)
            tw_win_has_nan = ~np.all(np.isfinite(tw_win), axis=1)
            tw_centers_tvt = tw_tvt[W : len(tw_gr) - W]
            tw_mean = tw_win.mean(axis=1, keepdims=True)
            tw_std = tw_win.std(axis=1, keepdims=True) + 1e-6
            tw_win_n = (tw_win - tw_mean) / tw_std
            h_pad = np.concatenate([np.full(W, np.nan), h_gr, np.full(W, np.nan)]).astype(np.float64)
            h_win = np.lib.stride_tricks.sliding_window_view(h_pad, L)
            h_finite = np.all(np.isfinite(h_win), axis=1)
            for i in range(n):
                if not h_finite[i]:
                    continue
                w = h_win[i]
                wn = (w - w.mean()) / (w.std() + 1e-6)
                scores = tw_win_n @ wn / L
                scores[tw_win_has_nan] = -np.inf
                j = int(np.argmax(scores))
                out_tvt[i] = tw_centers_tvt[j]
                out_score[i] = float(scores[j])
                out_idx[i] = float(j)
            return out_tvt, out_score, out_idx

        def _batched(h_gr, tw_gr, tw_tvt, W=15, chunk=8192):
            n = len(h_gr)
            out_tvt = np.full(n, np.nan)
            out_score = np.full(n, np.nan)
            out_idx = np.full(n, np.nan)
            L = 2 * W + 1
            inv_L = np.float32(1.0 / L)
            tw_win = np.lib.stride_tricks.sliding_window_view(tw_gr, L)
            tw_win_has_nan = ~np.all(np.isfinite(tw_win), axis=1)
            tw_centers_tvt = tw_tvt[W : len(tw_gr) - W]
            tw_mean = tw_win.mean(axis=1, keepdims=True)
            tw_std = tw_win.std(axis=1, keepdims=True) + 1e-6
            tw_win_n = np.ascontiguousarray(((tw_win - tw_mean) / tw_std).astype(np.float32, copy=False))
            h_pad = np.concatenate([np.full(W, np.nan), h_gr, np.full(W, np.nan)]).astype(np.float64)
            h_win = np.lib.stride_tricks.sliding_window_view(h_pad, L)
            h_finite = np.all(np.isfinite(h_win), axis=1)
            for start in range(0, n, chunk):
                stop = min(start + chunk, n)
                m = h_finite[start:stop]
                if not m.any():
                    continue
                rows = np.ascontiguousarray(h_win[start:stop][m]).astype(np.float32, copy=False)
                mu = rows.mean(axis=1, keepdims=True)
                s = rows.std(axis=1, keepdims=True) + np.float32(1e-6)
                rows_n = (rows - mu) / s
                scores = (rows_n @ tw_win_n.T) * inv_L
                if tw_win_has_nan.any():
                    scores[:, tw_win_has_nan] = -np.inf
                j = scores.argmax(axis=1)
                best = scores[np.arange(scores.shape[0]), j]
                gi = np.flatnonzero(m) + start
                out_tvt[gi] = tw_centers_tvt[j]
                out_score[gi] = best
                out_idx[gi] = j.astype(np.float64)
            return out_tvt, out_score, out_idx

        rng = np.random.default_rng(0)
        h = rng.standard_normal(500)
        tw = rng.standard_normal(800)
        tvt = rng.standard_normal(800)
        h[7] = np.nan
        tw[40:45] = np.nan
        a = _loop(h, tw, tvt, W=15)
        b = _batched(h, tw, tvt, W=15)
        for name, x, y in zip(("tvt", "score", "idx"), a, b):
            # NaN mask must match exactly.
            import numpy as np

            assert (np.isnan(x) == np.isnan(y)).all(), f"NaN-mask mismatch on {name}"
            m = np.isfinite(x) & np.isfinite(y)
            if name == "idx":
                # Argmax may tie-break differently between sequential
                # and matmul paths when two windows score identically;
                # allow off-by-one ties only if scores at both indices
                # are within fp32 tolerance. On random data ties are
                # vanishingly rare so we require exact match here.
                import numpy as np

                assert (x[m] == y[m]).all(), f"argmax differs on {name}"
            elif name == "tvt":
                import numpy as np

                assert (x[m] == y[m]).all(), f"selected tvt differs"
            else:
                import numpy as np

                assert np.allclose(x[m], y[m], atol=1e-4), f"score mismatch on {name}"


class TestDtwBincountAggregation:
    """Lock in equivalence of the bincount-vectorised DTW per-row
    aggregation with the original Python loop. Same locking-test
    pattern as the sliding-window batched fixture above.
    """

    def test_bincount_matches_python_loop(self) -> None:
        import numpy as np

        def _loop(path, n_q, tw_tvt):
            sum_tvt = np.zeros(n_q, dtype=np.float64)
            count = np.zeros(n_q, dtype=np.float64)
            for i, j in path:
                if 0 <= i < n_q and 0 <= j < len(tw_tvt):
                    sum_tvt[i] += tw_tvt[j]
                    count[i] += 1.0
            valid = count > 0
            pred = np.full(n_q, np.nan, dtype=np.float64)
            pred[valid] = sum_tvt[valid] / count[valid]
            return pred, count

        def _batched(path, n_q, tw_tvt):
            path_arr = np.asarray(path, dtype=np.int64)
            i_idx = path_arr[:, 0]
            j_idx = path_arr[:, 1]
            n_tw = len(tw_tvt)
            mask = (i_idx >= 0) & (i_idx < n_q) & (j_idx >= 0) & (j_idx < n_tw)
            i_idx = i_idx[mask]
            j_idx = j_idx[mask]
            sum_tvt = np.bincount(
                i_idx,
                weights=tw_tvt[j_idx],
                minlength=n_q,
            ).astype(np.float64, copy=False)
            count = np.bincount(i_idx, minlength=n_q).astype(np.float64)
            valid = count > 0
            pred = np.full(n_q, np.nan, dtype=np.float64)
            pred[valid] = sum_tvt[valid] / count[valid]
            return pred, count

        rng = np.random.default_rng(0)
        n_q = 200
        tw_tvt = rng.standard_normal(300)
        # Synthesise a plausible DTW path: monotone non-decreasing in
        # both indices, with some many-to-one and one-to-many cells.
        path = [(0, 0)]
        i, j = 0, 0
        while i < n_q - 1 or j < len(tw_tvt) - 1:
            step = int(rng.integers(0, 3))
            if step == 0 and i < n_q - 1:
                i += 1
            elif step == 1 and j < len(tw_tvt) - 1:
                j += 1
            else:
                if i < n_q - 1:
                    i += 1
                if j < len(tw_tvt) - 1:
                    j += 1
            path.append((i, j))
        # Add a few out-of-bounds entries to verify the mask works.
        path = [(-1, -1), *path, (n_q + 5, len(tw_tvt) + 10)]
        pred_a, count_a = _loop(path, n_q, tw_tvt)
        pred_b, count_b = _batched(path, n_q, tw_tvt)
        import numpy as np

        np.testing.assert_array_equal(count_a, count_b)
        # NaN locations must match.
        assert (np.isnan(pred_a) == np.isnan(pred_b)).all()
        m = np.isfinite(pred_a) & np.isfinite(pred_b)
        np.testing.assert_allclose(pred_a[m], pred_b[m], atol=1e-12)
