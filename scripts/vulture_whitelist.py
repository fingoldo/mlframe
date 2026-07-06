"""Vulture whitelist: confirmed false positives, not to be re-flagged.

Each entry below is a name vulture's static analysis cannot see is actually used --
dynamic ``getattr`` dispatch, a framework-mandated callback/protocol signature, or an
explicitly documented back-compat no-op. Verified by hand (2026-07-07 vulture sweep);
adding a name here means "confirmed reviewed", not "silenced without looking".

Format follows vulture's own ``--make-whitelist`` convention: a bare name as an
expression statement counts as a "use" in vulture's model (an assignment does NOT --
it creates a new binding instead of referencing an existing one).

This file is never imported by production code. Pass it alongside the real target on
the vulture command line: ``vulture src/mlframe scripts/vulture_whitelist.py``.
"""

# --- votenrank/leaderboard/Leaderboard.py: class-body imports invoked via
# getattr(self, f"{method}_election") dynamic dispatch in elect_all(). ---
from mlframe.votenrank.leaderboard._rules import (
    baldwin_election,
    borda_election,
    condorcet_election,
    copeland_election,
    dowdall_election,
    mean_election,
    minimax_election,
    optimality_gap_election,
    plurality_election,
    threshold_election,
)

baldwin_election
borda_election
condorcet_election
copeland_election
dowdall_election
mean_election
minimax_election
optimality_gap_election
plurality_election
threshold_election

# --- Framework-mandated / documented-no-op parameter names (unused-variable findings). ---

# PyTorch Lightning Callback hook signature (on_validation_batch_end) --
# dataloader_idx is part of the Lightning contract, not consumed by these callbacks.
dataloader_idx

# pydantic BaseModel.model_post_init(self, __context) -- framework-mandated signature.
__context

# Python context-manager protocol: __exit__(self, exc_type, exc_val, exc_tb).
exc_val

# Protocol-class interface stubs (feature_handling/protocols.py) -- abstract method
# signatures that concrete providers implement; params unused in the Protocol body itself.
train_texts
finetune_lr_mult

# numba @intrinsic calling convention: first positional arg is always `typingctx`.
typingctx

# Explicitly documented "retained for back-compat, no effect" parameters (each has an
# in-file docstring/comment saying so -- see the referenced module for the full note).
# NOTE: `array_size` (metrics/calibration/_calibration_metrics.py) and `default_level`
# (training/baselines/_dummy_report_type.py) are deliberately NOT whitelisted -- both
# take real distinct data / promise DEBUG-level behavior that the body never delivers;
# left visible in vulture's output as an open follow-up rather than silenced.
clf_thresh  # feature_selection/filters/_fe_linear_explainability.py
marginal_mi_a  # feature_selection/filters/_interaction_information.py
marginal_mi_b
astropy_sample_size  # feature_selection/filters/discretization/__init__.py
ensure_target_influence  # feature_selection/filters/screen.py
interactions_max_order
pre_pipeline_cache_max  # training/_trainer_configure.py -- consumed via behavior_config, not this signature
external_holdout_base_per_spec  # training/composite/ensemble/__init__.py
force_windows  # training/mlp_runtime_defaults.py
train_batch_size
_disp  # feature_selection/_benchmarks/kernel_tuning_cache/_auto_tune_sweeps_a.py -- import for module init side effect
double_scarcenes_after  # preprocessing/cleaning.py -- superseded by log_scarceness_divisor; old logic kept as an inert triple-quoted comment
n_bits  # feature_engineering/transformer/ib_baseline_codes.py -- design moved to a fixed binary-median split, comment explains
calib_set_size  # training/evaluation.py -- historical calib-on-test-slice approach removed (leaked); param vestigial
stydy_type  # models/tuning.py ParamsOptimizer.create_study -- explicitly documented STUB/no-op
need_training_continuation  # models/tuning.py CatboostParamsOptimizer -- part of the same documented-stub class
product_id  # utils/experiments.py create_experiment -- explicitly documented STUB/no-op
eval_class_weight  # training/lgb_shim.py -- sklearn/LightGBM fit() API-parity shim signature
decoder_query_dim  # feature_selection/filters/_vendored/infonet -- vendored third-party code, left at upstream shape
neighbours_xy  # feature_selection/filters/_ksg.py -- deprecated kernel retained for A/B parity (see docstring)
concentration  # feature_selection/filters/_neural_mi.py dpmine_mi -- explicitly marked EXPERIMENTAL/SKELETON
composite_config  # training/_precompute.py -- the function's own docstring documents this precompute path as deferred/NotImplementedError
