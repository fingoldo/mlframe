"""End-to-end test that ``group_field`` in SimpleFeaturesAndTargetsExtractor
propagates to ``make_train_test_split`` and groups stay within one split.

User-reported confusion 2026-05-18: passed ``group_field="well_id"`` plus
``TrainingSplitConfig(val_placement="backward")`` (no timestamps), saw the
warning "val/test rows are randomly mixed across time" and assumed groups
were being ignored. They WEREN'T - groups were respected; only TEMPORAL
ordering was downgraded. This test pins the behaviour: every group ends
up entirely in train OR val OR test, never split across.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestGroupFieldEndToEnd:
    """``group_field`` from extractor must reach the splitter and prevent
    per-row group leakage."""

    def _build_problem(self, n_groups: int = 20, rows_per_group: int = 100, seed: int = 42):
        """Build problem."""
        rng = np.random.default_rng(seed)
        rows = []
        for g in range(n_groups):
            x_a = rng.normal(size=rows_per_group)
            x_b = rng.normal(size=rows_per_group)
            y = 1.5 * x_a + 0.5 * x_b + rng.normal(0, 0.3, rows_per_group)
            for i in range(rows_per_group):
                rows.append(
                    {
                        "x_a": float(x_a[i]),
                        "x_b": float(x_b[i]),
                        "y": float(y[i]),
                        "well_id": f"well_{g:03d}",
                    }
                )
        return pd.DataFrame(rows)

    @pytest.mark.timeout(60)
    def test_groups_never_leak_across_splits(self) -> None:
        """Groups never leak across splits."""
        from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor
        from mlframe.training.configs import TrainingSplitConfig
        from mlframe.training.splitting import make_train_test_split

        df = self._build_problem(n_groups=30, rows_per_group=50)

        # Reproduce the user's exact extractor + split config (minus timestamps).
        extractor = SimpleFeaturesAndTargetsExtractor(
            regression_targets=["y"],
            group_field="well_id",
            verbose=0,
        )
        df_out, _target_by_type, _group_ids_raw, group_ids, _timestamps, *_ = extractor.transform(df)
        assert group_ids is not None, "extractor with group_field set should produce group_ids"
        assert len(group_ids) == len(df), "group_ids shape mismatch"

        cfg = TrainingSplitConfig(
            shuffle_val=True,
            shuffle_test=True,
            test_size=0.1,
            val_size=0.1,
            val_placement="backward",
        )
        assert cfg.use_groups is True, "use_groups default must be True"

        train_idx, val_idx, test_idx, *_ = make_train_test_split(
            df=df_out,
            timestamps=None,
            stratify_y=None,
            groups=group_ids,
            **cfg.model_dump(
                exclude={
                    "use_groups",
                    "calib_size",
                    "conformal_size",
                    # config-only fields consumed in the phase-helper before the splitter
                    "composite_cardinality_cap",
                    "bucket_stratify",
                    "time_column",
                    "cv_strategy",
                    "cv_purge",
                }
            ),
        )

        # Confirm: no well_id appears in more than one split.
        wells = df["well_id"].values
        wells_train = set(wells[train_idx])
        wells_val = set(wells[val_idx])
        wells_test = set(wells[test_idx])
        cross_train_val = wells_train & wells_val
        cross_train_test = wells_train & wells_test
        cross_val_test = wells_val & wells_test
        assert not cross_train_val, f"well_id leakage train<->val: {cross_train_val}"
        assert not cross_train_test, f"well_id leakage train<->test: {cross_train_test}"
        assert not cross_val_test, f"well_id leakage val<->test: {cross_val_test}"

    @pytest.mark.timeout(60)
    def test_use_groups_false_explicitly_disables(self) -> None:
        """When ``use_groups=False`` the splitter ignores group_ids and
        runs IID row-shuffle (per-well leakage allowed). Documents the
        explicit-opt-out path."""
        from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor
        from mlframe.training.configs import TrainingSplitConfig
        from mlframe.training.splitting import make_train_test_split

        df = self._build_problem(n_groups=20, rows_per_group=50)
        extractor = SimpleFeaturesAndTargetsExtractor(
            regression_targets=["y"],
            group_field="well_id",
            verbose=0,
        )
        df_out, *_, group_ids, _, _, _, _ = extractor.transform(df)

        cfg = TrainingSplitConfig(
            shuffle_val=True,
            shuffle_test=True,
            test_size=0.1,
            val_size=0.1,
            val_placement="backward",
            use_groups=False,  # explicit opt-out
        )
        # When the suite's phase code derives _groups, use_groups=False
        # makes _groups=None. We replicate that here.
        _groups = group_ids if cfg.use_groups else None
        train_idx, val_idx, test_idx, *_ = make_train_test_split(
            df=df_out,
            timestamps=None,
            stratify_y=None,
            groups=_groups,
            **cfg.model_dump(
                exclude={
                    "use_groups",
                    "calib_size",
                    "conformal_size",
                    # config-only fields consumed in the phase-helper before the splitter
                    "composite_cardinality_cap",
                    "bucket_stratify",
                    "time_column",
                    "cv_strategy",
                    "cv_purge",
                }
            ),
        )
        # With use_groups=False the IID path is used; with 1000 rows
        # across 20 wells, wells very likely leak across splits.
        wells = df["well_id"].values
        wells_train = set(wells[train_idx])
        wells_val = set(wells[val_idx])
        wells_test = set(wells[test_idx])
        any_overlap = bool((wells_train & wells_val) or (wells_train & wells_test) or (wells_val & wells_test))
        # Probabilistically guaranteed with these sizes; if this ever
        # flakes, increase n_groups / rows_per_group.
        assert any_overlap, "use_groups=False should produce well leakage on this size; got isolated wells across splits which is suspicious"


class TestSplitterEmitsGroupAwareInfoLine:
    """The splitter must log a clear "Group-aware splitting: ENABLED/disabled"
    info line so the operator sees the active mode without grepping for
    GroupShuffleSplit internals."""

    def test_enabled_log_when_groups_supplied(self, caplog) -> None:
        """Enabled log when groups supplied."""
        import logging
        from mlframe.training.splitting import make_train_test_split

        rng = np.random.default_rng(0)
        n = 500
        df = pd.DataFrame(
            {
                "x_a": rng.normal(size=n).astype(np.float32),
                "y": rng.normal(size=n).astype(np.float32),
            }
        )
        groups = np.repeat(np.arange(25), n // 25)
        with caplog.at_level(logging.INFO, logger="mlframe.training.splitting"):
            make_train_test_split(
                df=df,
                timestamps=None,
                stratify_y=None,
                groups=groups,
                test_size=0.1,
                val_size=0.1,
                shuffle_val=True,
                shuffle_test=True,
                val_placement="forward",
            )
        assert any(
            "Group-aware splitting: ENABLED" in r.message for r in caplog.records
        ), "splitter must log 'Group-aware splitting: ENABLED' when groups given"

    def test_disabled_log_when_groups_none(self, caplog) -> None:
        """Disabled log when groups none."""
        import logging
        from mlframe.training.splitting import make_train_test_split

        rng = np.random.default_rng(0)
        n = 500
        df = pd.DataFrame(
            {
                "x_a": rng.normal(size=n).astype(np.float32),
                "y": rng.normal(size=n).astype(np.float32),
            }
        )
        with caplog.at_level(logging.INFO, logger="mlframe.training.splitting"):
            make_train_test_split(
                df=df,
                timestamps=None,
                stratify_y=None,
                groups=None,
                test_size=0.1,
                val_size=0.1,
                shuffle_val=True,
                shuffle_test=True,
                val_placement="forward",
            )
        assert any("Group-aware splitting: disabled" in r.message for r in caplog.records)
