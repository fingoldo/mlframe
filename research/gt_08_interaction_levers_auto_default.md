# gt_08: Auto-activated interaction levers — `proxy_mode="auto"` as the new ShapProxiedFS default

Read `research/README.md` (shared conventions) first. This document is self-contained otherwise.

## 1. Problem & motivation

ShapProxiedFS's additive coalition proxy `base + Σ_{j∈S} φ_j` is structurally blind to pure
interactions: for `y = sign(a*b)` each operand's marginal φ ≈ 0, so XOR-style pairs never earn
credit and are silently dropped. Two existing opt-in levers fix this on interaction-heavy data:

- `proxy_mode="interaction"` — re-scores candidates under
  `base + Σφ_j + 2·Σ_{i<j∈S} Φ_ij` using the TreeSHAP interaction tensor. Bench
  (`src/mlframe/feature_selection/_benchmarks/bench_shap_interaction_proxy.py`): wins the
  competing-XOR bed by ~+0.24 honest-holdout AUC replicated 3/3 seeds, but only 1/6 beds and
  slightly regresses one additive-redundant seed. Rejected as unconditional default per the
  majority-win rule.
- `su_seeded_interactions=True` — a cheap pairwise-SU synergy screen
  (`su_synergy_screen`) rescues interaction operand pairs at O(P)+O(K) cost with a built-in
  permutation-null SNR gate. Measured wins: +0.388 AUC on a pure-interaction bed, +0.072 on synth,
  correct NO-OP on hard_synth (noise-buried interactions correctly not seeded).

Both stay opt-in, so the default user gets zero interaction handling. The fix requested here:
make interaction handling the DEFAULT — but as a **data-driven auto-gate**, not an unconditional
flip (the unconditional flip is bench-rejected; the repo rule is "gate a big win on its safe
condition"). The SNR-gated synergy screen already IS that safe condition detector: it fires only
when statistically significant synergy exists, and no-ops cleanly (empty `kept` list) otherwise.

## 2. Existing machinery (all paths verified against current source)

File: `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_interactions.py`

```python
def su_synergy_screen(
    X, y, *,
    n_bins: int = 8, top_k: int = 8, max_screen_cols: int = 120,
    snr_z: float = 3.0, snr_null_quantile: float = 0.99, snr_abs_floor: float = 1e-3,
    n_permutations: int = 3,
    importance: np.ndarray | None = None, rng=None,
) -> tuple[list, dict]:   # at :342
```
- `X`: pandas DataFrame of proxy/unit columns; `y`: array-like (continuous y is quantile-binned).
- Returns `(kept, info)`. `kept`: list of `(synergy, joint_su, col_a, col_b)` tuples,
  best-synergy-first, `len <= top_k`; **empty list == the SNR gate cleared nothing (no-op signal)**.
  `info` keys: `gate`, `null_quantile`, `null_std`, `n_screened_cols`, `n_pairs`, `best_synergy`,
  `n_kept`. Synergy score: `SU(joint_bin(a,b); y) - max(SU(a;y), SU(b;y))`, thresholded against a
  target-shuffle permutation null. Cost: O(P) binning + O(min(P, max_screen_cols)²) discrete SU.

```python
def sparse_interaction_candidates(
    model_template, X_proxy, y, kept_pairs, *,
    classification, metric=None, min_card=1, max_card=None, top_n=30, rng=None,
) -> tuple[list, dict]:   # at :505
```
- Consumes `su_synergy_screen` output (reads `pair[-2], pair[-1]`), engineers one `a*b` product
  column per pair, fits ONE in-sample model on `[X_proxy | products]`, returns
  `(candidates, product_to_operands)`; `([], {})` when `kept_pairs` is empty.

Wiring seams (file `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxied_fit_interactions.py`):
- `resolve_su_seeded_pairs(self, phi, X_proxy, y_phi, unit_to_members, working_cols, X_cols,
  report, _stage)` at `:47` — currently gated on `self.su_seeded_interactions` being truthy
  (`if self.su_seeded_interactions and phi.shape[1] >= 2:`). Runs the screen (or reuses the
  prefilter-stage screen result `self._su_seeded_pairs_orig`), returns
  `(_su_kept_pairs, _su_screen_info, _su_rescue_proxy_idx)`.
- `augment_candidates_with_interactions(...)` at `:121` — merges su_seeded sparse candidates into
  the search's candidate list; block gated on `if self.su_seeded_interactions:` at `:211`.
- The prescreen keep-set union of rescued operand indices lives in
  `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxied_fit.py` (search for
  `_su_rescue_proxy_idx` in the prescreen block, near the `noise_floor_rescue_keep_set` call).
- Constructor: `src/mlframe/feature_selection/shap_proxied_fs/__init__.py` — `proxy_mode`
  validated at `:305` (`("additive", "interaction")`), `su_seeded_interactions` + its 7 tuning
  params at `:125-132`. NOTE the sklearn-clone rule: store raw, validate without mutating.

There is ALSO a prefilter-stage screen invocation (`_shap_proxied_fit.py`, the
`su_seeded interaction rescue (pre-prefilter)` block, ~`:422-458`) which rescues operands past the
PREFILTER cut — gated on the same flag. The auto mode must activate this one too, since on wide
frames interaction operands die at the prefilter before the proxy-space screen ever runs.

## 3. Design

### 3.1 New mode value and default flip
- Extend the `proxy_mode` validator to `("additive", "interaction", "auto")` and **change the
  default to `"auto"`** in the same change (repo rule: corrective mechanisms flip ON by default
  when the fix closes a real bug class; the gate makes this safe).
- Semantics of `"auto"`:
  1. ALWAYS run the su_seeded synergy screen (both the pre-prefilter invocation and the
     proxy-space `resolve_su_seeded_pairs`) regardless of `su_seeded_interactions` flag.
  2. If the screen's `kept` is non-empty → enable the full su_seeded path: operand rescue past the
     prefilter, operand rescue past the prescreen, `sparse_interaction_candidates` augmentation,
     bare-operand-pair injection. Exactly the behaviour of `su_seeded_interactions=True` today.
  3. If `kept` is empty → the additive path runs **byte-identically** to `proxy_mode="additive"`
     (the screen result is discarded; no extra candidates, no rescued indices; the only cost is the
     screen itself).
  4. `"auto"` does NOT enable the O(P²) TreeSHAP interaction tensor path
     (`proxy_mode="interaction"` / `interaction_aware`) — that stays opt-in. Rationale: the sparse
     su_seeded path achieves the interaction wins at O(K) cost and no-ops safely; the tensor path
     is gated to P≤16 and bench-rejected as a default. Document this in the `proxy_mode` docstring.
- Backward compatibility:
  - `proxy_mode="additive"` → legacy exact behaviour (no screen at all) — the escape hatch.
  - `proxy_mode="interaction"` → unchanged.
  - `su_seeded_interactions=True` with `proxy_mode="auto"` → forces the su_seeded path even if the
    screen keeps nothing? NO — keep it simple: `su_seeded_interactions=True` behaves exactly as
    today (runs the screen, gate decides), and under `"auto"` the flag is redundant. Emit no
    warning; just document that `"auto"` subsumes it.

### 3.2 Implementation steps
1. `__init__.py`: extend validator tuple; change default `proxy_mode: str = "auto"`; update the
   long `proxy_mode` comment block (it currently documents the additive-default rationale — extend
   it with the auto-gate rationale and the bench table from §5).
2. Introduce ONE resolved boolean helper (new small module or a function in
   `_shap_proxied_methods.py`): `_su_screen_enabled(self) -> bool` returning
   `self.su_seeded_interactions or str(self.proxy_mode).lower() == "auto"`. Replace the two
   `if self.su_seeded_interactions` gates (`_shap_proxied_fit.py` pre-prefilter block;
   `_shap_proxied_fit_interactions.py` `:69` and `:211`) with calls to it. Grep for ALL flag
   read sites first (repo rule: class-level change → one pass over every site): also check
   `_shap_proxy_prefilter.py` and report-population sites.
3. Telemetry: under `"auto"`, always write `report["su_seeded_interactions"]` (screen info incl.
   `best_synergy`, `gate`, `n_kept`) plus a new key `report["proxy_mode_resolved"]` ∈
   `{"additive", "additive(auto:gate-silent)", "interaction(auto:gate-fired)"}` so users see which
   branch ran.
4. cProfile harness: extend `shap_proxied_fs/_benchmarks/profile_shap_proxied_fit.py` (or add a
   sibling) measuring screen overhead at widths {2000, 10000}; acceptance: screen ≤3% of e2e wall.

### 3.3 Mandatory pre-flip benchmark
Re-run ALL 6 beds of `bench_shap_interaction_proxy.py` × 3 seeds, three arms:
`additive` / `interaction` / `auto`. Flip criteria (all must hold):
- On XOR/pure-interaction beds: `auto` matches or beats `interaction`'s honest-holdout AUC win.
- On additive/additive-redundant beds: `auto`'s `selected_features_` is IDENTICAL to `additive`'s
  (the gate must not fire; assert `report["su_seeded_interactions"]["n_kept"] == 0`).
- Screen overhead ≤3% e2e wall on the widest bed.
If the gate fires spuriously on any additive bed, tighten `snr_z`/`snr_abs_floor` defaults for the
auto path only (do not change the opt-in flag's defaults) and re-run; if it cannot be made silent,
ship `"auto"` as opt-in instead and document the failed flip (REJECTED ≠ DELETED).

## 4. biz_val tests

File: `tests/feature_selection/shap_proxied/test_biz_val_shap_proxied_proxy_mode_auto.py`.
Synthetic beds via `make_regime_dataset` (`src/mlframe/feature_selection/_benchmarks/_shap_proxy_regime_data.py`)
or inline numpy (pattern: `tests/feature_selection/shap_proxied/test_biz_val_shap_proxied_parsimony_tol_recall.py`).

1. `test_biz_val_proxy_mode_auto_gate_fires_on_xor_bed` — n=2000, p=200: 4 additive informative
   (weight 1.0) + 1 XOR pair (weight 1.5, `sign(a*b)`) + noise. Assert:
   `report["proxy_mode_resolved"]` shows gate-fired; XOR operand recall = 2/2 with `auto` vs 0/2
   with `additive`; downstream-AUC(auto) ≥ downstream-AUC(additive) + 0.03 (bench measured +0.24
   on the harder bed; floor generously).
2. `test_biz_val_proxy_mode_auto_gate_silent_on_additive_bed` — pure additive bed (6 informative +
   noise, no interactions). Assert: `n_kept == 0`; `selected_features_` EXACTLY equals the
   `proxy_mode="additive"` run's (same random_state); wall overhead of the auto run ≤ 1.10× the
   additive run (generous; target ≤1.03).
3. `test_biz_val_proxy_mode_auto_noop_on_noise_buried_interactions` — hard_synth-style bed
   (interaction pair buried in 200 noise cols at low SNR): gate must stay silent (this pins the
   permutation-null behaviour that made su_seeded safe).

Unit tests (same file or `test_shap_proxied_knobs.py` extension): validator accepts `"auto"`,
rejects garbage; default is `"auto"`; `proxy_mode="additive"` skips the screen entirely (assert
`"su_seeded_interactions" not in report`); sklearn `clone()` round-trips.

## 5. Acceptance criteria
- Default `proxy_mode="auto"`; all three biz_val tests green locally
  (`CUDA_VISIBLE_DEVICES="" pytest ... --no-cov -p no:anyio`).
- 6-bed × 3-seed bench table committed under `_benchmarks/` with the three-arm comparison; flip
  criteria of §3.3 met (or the documented fallback to opt-in taken).
- Byte-identity proof for the silent-gate case included in the bench output.
- Existing suite `tests/feature_selection/shap_proxied/` green (batch pattern if the monolithic
  run times out; see PLAN_wide_dataframe_improvements.md housekeeping section for the 3-batch recipe).

## 6. Known risks / rejected alternatives
- Unconditional `proxy_mode="interaction"` default: bench-rejected (1/6 beds, regresses an
  additive seed, O(k²) tensor cost) — do not resurrect.
- Gate flakiness near the SNR threshold: mitigate by pinning `rng` derivation from `random_state`
  (the screen already accepts an isolated rng — `resolve_su_seeded_pairs` uses
  `np.random.default_rng(int(self.random_state) + 7919)`; keep that).
- Screen cost on very wide proxies: bounded by `max_screen_cols=120` — verify it consumes the
  post-prescreen/importance-ranked head, not raw width, at the pre-prefilter call site.
