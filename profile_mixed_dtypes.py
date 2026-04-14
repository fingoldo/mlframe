"""cProfile the mixed-dtypes test to find hotspots in train_mlframe_models_suite."""
import cProfile, pstats, io, tempfile, sys, gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
gc.collect()

from tests.training.test_mixed_dtypes_training import (
    _make_synthetic_mixed_df, TestMixedDtypesTraining,
)

df = _make_synthetic_mixed_df(n_rows=50_000)
print(f"DF shape: {df.shape}")

t = TestMixedDtypesTraining()
profiler = cProfile.Profile()

with tempfile.TemporaryDirectory() as tmp:
    profiler.enable()
    t.test_catboost_trains_on_mixed_dtypes(df, Path(tmp))
    profiler.disable()

profiler.dump_stats("profile_mixed_dtypes.prof")

for sort_key, label in [("cumulative", "CUMULATIVE"), ("tottime", "TOTAL TIME")]:
    print(f"\n{'='*80}\nTOP 50 by {label}\n{'='*80}")
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(sort_key)
    ps.print_stats(50)
    print(s.getvalue())

print(f"\n{'='*80}\nTOP 30 in mlframe code only\n{'='*80}")
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
ps.print_stats("mlframe", 30)
print(s.getvalue())
