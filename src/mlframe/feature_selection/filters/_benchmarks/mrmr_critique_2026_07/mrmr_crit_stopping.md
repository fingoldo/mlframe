# MRMR stopping / fallback / empty-support critique

- ST-1 [P2] `_fit_impl_core.py:9854-9857` — UAED elbow trim applies a combined raw+engineered gain-trace elbow to a RAW-only support_, then sets n_features_ = support_.size, desyncing transform width (still emits engineered) vs n_features_/mrmr_gains_. Only with uaed_auto_size=True (default off) AND engineered present. Fix: trim in a consistent index space; n_features_ = len(support_)+n_engineered_out.
- ST-2 [P2] `_finalise.py:254-259` — count-floor top-up gates on `_mi > _abs_floor` where _abs_floor=min_relevance_gain default 0.0, so any non-constant column qualifies. Documented ≥K count-floor contract; honest improvement = prefer significance-passing candidates + set fallback_metadata_["uninformative"] when a topped-up column sits within its null.
- ST-3 [Low] `_finalise.py:281` — `self.support_ = np.array(_topk)` omits dtype=int64 → int32 on Windows, inconsistent with every other support_ assignment (9125/9516/9656/9752). Fix: dtype=np.int64.
- ST-4 [Low] `_fit_impl_core.py:6711-6717/9791` — ran_out_of_time_ set only on FE-loop deadline, not a screen_predictors internal timeout. Fix: OR-in a screen-level timeout signal.
Verified-correct: significance-gate determinism, p-value index `_sig[3]`, `>= alpha ⇒ drop` sense, min_features_fallback empty-screen-only contract, empty-path int64 support, mutually-redundant-cluster never-empty rescue.
Improvements: honor self.random_seed in rescue null draws; principled scan bound (until rescue_cap accepted OR MI<floor); clarify min_features_fallback docstring.
