"""cProfile the mixed-dtypes training to find hotspots in train_mlframe_models_suite.

Mirrors the user's real scenario (9M rows, 587 cols, CatBoost only, iterations=200,
early_stopping_rounds=1000, show_perf_chart=True, show_fi=True, verbose=True).

Runs at 1/10th scale (n_rows configurable) so a full train+val+test cycle completes
in minutes and exposes where time is actually spent — particularly after training,
during report_model_perf where the user's run hung for hours.
"""
import cProfile, pstats, io, logging, os, sys, gc, tempfile
from pathlib import Path
from time import perf_counter as timer

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))
gc.collect()

N_ROWS = int(os.environ.get("PROFILE_N_ROWS", "500000"))
ITERATIONS = int(os.environ.get("PROFILE_ITERATIONS", "200"))
EARLY_STOPPING_ROUNDS = int(os.environ.get("PROFILE_EARLY_STOP", "1000"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
log = logging.getLogger("profile")

from tests.training.test_mixed_dtypes_training import _make_synthetic_mixed_df
from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.configs import (
    ModelHyperparamsConfig,
    PolarsPipelineConfig,
    TrainingBehaviorConfig,
    TrainingSplitConfig,
)
from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor


log.info(f"Generating synthetic DF: n_rows={N_ROWS:_}")
t0 = timer()
df = _make_synthetic_mixed_df(n_rows=N_ROWS)
log.info(f"DF shape: {df.shape}, built in {timer()-t0:.1f}s")


def run_suite(tmp_path: Path) -> None:
    ft_extractor = SimpleFeaturesAndTargetsExtractor(
        classification_targets=["cl_act_total_hired"],
        classification_lower_thresholds=dict(cl_act_total_hired=1),
        ts_field="job_posted_at",
        columns_to_drop={"uid", "job_posted_at", "job_status", "cl_id"},
        verbose=1,
    )

    train_mlframe_models_suite(
        df=df,
        target_name="H2",
        model_name="prod_jobsdetails",
        features_and_targets_extractor=ft_extractor,
        mlframe_models=["cb"],
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        pipeline_config=PolarsPipelineConfig(
            use_polarsds_pipeline=False,
            categorical_encoding=None,
            scaler_name=None,
            imputer_strategy=None,
        ),
        split_config=TrainingSplitConfig(
            shuffle_val=False,
            shuffle_test=False,
            test_size=0.1,
            val_size=0.1,
            wholeday_splitting=False,
        ),
        hyperparams_config=ModelHyperparamsConfig(
            iterations=ITERATIONS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        ),
        behavior_config=TrainingBehaviorConfig(
            prefer_calibrated_classifiers=False,
        ),
        init_common_params={"show_perf_chart": True, "show_fi": True},
        data_dir=str(tmp_path / "data"),
        verbose=True,
    )


profiler = cProfile.Profile()

with tempfile.TemporaryDirectory() as tmp:
    log.info("=" * 80)
    log.info(f"Starting train_mlframe_models_suite (iterations={ITERATIONS}, early_stop={EARLY_STOPPING_ROUNDS})")
    log.info("=" * 80)
    t_start = timer()
    profiler.enable()
    run_suite(Path(tmp))
    profiler.disable()
    log.info(f"Suite finished in {timer()-t_start:.1f}s")

profiler.dump_stats("profile_mixed_dtypes.prof")

for sort_key, label in [("cumulative", "CUMULATIVE"), ("tottime", "TOTAL TIME")]:
    print(f"\n{'='*80}\nTOP 50 by {label}\n{'='*80}")
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(sort_key)
    ps.print_stats(50)
    print(s.getvalue())

print(f"\n{'='*80}\nTOP 40 in mlframe code only (by cumulative)\n{'='*80}")
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
ps.print_stats("mlframe", 40)
print(s.getvalue())

print(f"\n{'='*80}\nTOP 30 in evaluation.py / report_model_perf (by cumulative)\n{'='*80}")
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
ps.print_stats("evaluation", 30)
print(s.getvalue())
