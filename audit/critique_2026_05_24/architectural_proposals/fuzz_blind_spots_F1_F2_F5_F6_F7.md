# Fuzz blind spots requiring AXES expansion (F1, F2, F5, F6, F7)

Defer rationale: adding a new key to `AXES` in `_fuzz_combo.py` requires
matching:

1. New field on `FuzzCombo` dataclass (~10 LOC).
2. `enumerate_combos` pairwise-coverage logic (no change — picks up new axis automatically once dataclass and AXES are aligned).
3. `canonical_key()` rule(s) (~5-15 LOC) to collapse no-op axis values for combos where the axis is irrelevant, otherwise pairwise sampler wastes budget on duplicate combos.
4. `_configs_for_combo` / `_config_for_models` wiring in `test_fuzz_suite.py` (passing the axis value to the actual suite call).
5. `build_frame_for_combo` if the axis affects frame shape.
6. `_required_combos` entries forcing key combo cells.

The blast radius is ~30-80 LOC per axis. Doing all five in this wave
risks regressing the established fuzz pool (currently ~150 combos pinned
with hashed seeds in `_fuzz_results.jsonl`). User approval needed before
landing.

## Findings to expand

| # | Axis name | Values | Touch points | Canon rule | Required combo pin |
|---|---|---|---|---|---|
| F1 | `enable_crash_reporting_cfg` | (False, True) | suite call kwarg `enable_crash_reporting`; gate to Windows only since non-Windows is no-op | canonicalise to False when `platform.system() != "Windows"` | one Windows-only combo with `True` |
| F2 | (existing axes) | `prefer_polarsds=True` x `target_type="learning_to_rank"` | already wired; need `_required_combos` entry | none | force pair into pinned combo set |
| F5 | (canon-rule only) | `prep_ext_pysr_enabled_cfg=True` x `inject_inf_nan=True` | canon rule | if `inject_inf_nan`: force `prep_ext_pysr_enabled_cfg=False` (PySR cannot consume inf/nan) | none |
| F6 | (existing) | `composite_discovery_enabled_cfg=True` x `outlier_detection in {lof,ocsvm}` | already wired; need regression sensor for known 0-row-val cluster | none | one sensor in `test_fuzz_regression_sensors.py` |
| F7 | (existing) | `dummy_baselines_enabled_cfg=False` x `baseline_diagnostics_enabled_cfg=True` | already wired; need assertion that diagnostics auto-enable baselines OR raise | none | regression sensor asserting non-silent behaviour |

## Path forward

Bundle into a follow-up wave once Wave 3 lands:

1. Single PR adds all 5 axes / canon rules.
2. Refresh `_fuzz_results.jsonl` baselines once new combos enumerate.
3. Verify pairwise coverage report still meets target (150 unique combos
   covering all 2-way interactions of the touched axes).

## What this wave DOES land (in scope)

- F3 sensor (`weight_schemas=("recency",)` x `recurrent_model_cfg`)
  added directly to `test_fuzz_regression_sensors.py` -- focused
  deterministic combo, no axis expansion needed.
- F4 sensor (`mrmr_nan_strategy_cfg="fillna_zero"` x `inject_all_nan_col`
  x `use_mrmr_fs=True`) added similarly.
- F8 metamorphic check (`multilabel_strategy_cfg="chain"` x
  `multilabel_chain_order_cfg="random"` reproducibility) added to
  `test_fuzz_metamorphic.py`.

These are 3 of 8 blind spots closed in-band; the rest deferred per the
arch-defer protocol.
