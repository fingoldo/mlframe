"""Run one registered kernel tuner and report how long its sweep takes, dumping every thread's stack if it exceeds a limit. Usage: ``python time_tuner.py <kernel_name> <seconds>``."""
import sys, time, faulthandler, warnings
warnings.simplefilter("ignore")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # this checkout, not an installed copy
from mlframe.feature_selection._benchmarks.profiling._paths import add_src_to_path  # noqa: E402
from mlframe.system.kernel_tuning_cache import discover_specs
s = discover_specs("mlframe")[sys.argv[1]]
faulthandler.dump_traceback_later(float(sys.argv[2]), exit=True)
t = time.perf_counter()
r = s.tuner()
print("done", sys.argv[1], None if r is None else len(r), round(time.perf_counter() - t, 1), "s", flush=True)
