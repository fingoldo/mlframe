# Audit 06: MRMR feature-engineering stages (wave 2)

Scope: `_mrmr_fe_step/`, `_feature_engineering_pairs/`, cascades, pure-form retention and sides, accuracy gates, dedup, group counts, column stats. Read-only, Haiku model, nothing executed.
Scope notes from the auditor: `_engineered_dedup.py` is not under `filters/` (it is `_mrmr_fit_impl/_fit_impl_stages/_engineered_dedup.py`); the cascades are in `_mrmr_fit_impl/`.

Tracker rows: AUD23-38, 39, 40, 41, 42, 43, 46 (FE part).

1. Prewarp replay can use a different warp than the one scored (train/apply skew). Severity high.
   - `_mrmr_fe_step/_step_score_parts2.py:222-223`: pair-scoped key tried first, then var-keyed fallback `st._prewarp_specs.get(var_idx)`.
   - `_feature_engineering_pairs/_pairs_gates.py:80-83` documents the var-keyed entry is whichever chunk merged last, so not the spec that built a given pair's column.
   - Trigger: operand var warped in two chunks or steps with different fits, and the pair key absent. The recipe stores the wrong warp.
   - Fallback confirmed in code. Whether the pair key is ever absent is hypothesis: `_pairs_core_helpers.py:406-411` writes the pair key only when `_prewarp_spec_by_var.get(_v)` is non-None.

2. Recipe emitted with `prewarp` but no spec; fit succeeds, transform raises. Severity medium.
   - `_step_score_parts2.py:222-223` yields None when both keys miss; the recipe still gets `unary_a_name="prewarp"`.
   - `engineered_recipes/_recipe_unary_binary.py:133-136` raises KeyError at replay.
   - `_recipe_unary_binary.py:338-341` disables quantization by `prewarp_a/b` being non-None, not by the unary name. Confirmed code path; trigger hypothesis.

3. gate_med median keyed by var only. Severity low.
   - `_step_score_parts2.py:228-229` reads `st._gate_med_specs.get(var_idx)`, merged with `.update` across chunks (`_step_core_parts.py:289`). Replay raises KeyError if `gate_med_{side}_median` is missing (`_recipe_unary_binary.py:93-98`). The median comes from the shared subsample (`_pairs_setup.py:166-192`), so probably identical across chunks. Hypothesis.

4. Name-keyed counting can undercount recipes. Severity medium (ledger only).
   - `_fe_group_counts.py:32-36`: `group_counts` dedups by name with `setdefault`; two distinct recipes with the same name count once, so emitted, kept_for_screen and selected can disagree with the recipes held. `record_cascade_emitted` (`:56`) uses this too. Dedup by name confirmed; duplicate names occurring is hypothesis.

5. Name-based skips of distinct recipes. Severity medium.
   - `_fe_pure_form_retention.py:802-806`: `if name in existing_names: continue` silently drops a candidate whose recipe differs but whose name collides.
   - `_mrmr_fit_impl/_fe_stage_cascade_mid_a.py:278`: names colliding with existing columns are dropped from the support, while the recipes may still be in the registry counted by `record_cascade_emitted`. Filtering confirmed; ledger mismatch is hypothesis.

6. column_stats_ fate can come from the wrong row. Severity low to medium.
   - `_column_stats.py:116-119`: `drop_duplicates("feature_name")` keeps the first provenance row; if one name has both a dropped and a kept row, the fate reflects the first.
   - `_column_stats.py:74`: `is_engineered` is name membership in `feature_names_in_`, so an engineered column whose name equals a raw name is labelled raw. Confirmed code; occurrence hypothesis.

7. Bare or silent failures.
   - `_mrmr_fe_step/_helpers.py:39-43`: on dtype-introspection failure returns `set()`, marking every column numeric. The comment admits this can crash downstream; logged only at debug. Confirmed.
   - `_fe_accuracy_gate.py:235-237` and `:386`: a probe exception returns None, which `keep_engineered_over_source` treats as keep. Documented fail-open (`:176-178`, `:384-385`): a broken candidate passes the gate. Confirmed, by design.
   - `_fe_accuracy_gate.py:173-177`: `_cut` memoised by name across source and engineered columns; needs a name clash. Hypothesis.
   - `_pairs_*` chunk and recompute `except Exception` blocks (e.g. `_pairs_chunks.py:303`, `_pairs_score_recompute.py:207`) null the slot or skip and log. No silent pass found.
