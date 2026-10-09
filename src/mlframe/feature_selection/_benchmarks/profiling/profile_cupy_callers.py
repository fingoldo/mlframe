"""mlframe functions ranked by how many cupy calls they make directly: finds the launch-heavy call sites. Reads ``PROF_FILE`` written by ``profile_fit_gpu.py``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # this checkout, not an installed copy
from mlframe.feature_selection._benchmarks.profiling._paths import PROF_FILE  # noqa: E402

import pstats, collections
p = pstats.Stats(str(PROF_FILE)); st = p.stats
cnt = collections.Counter(); tim = collections.Counter()
for callee, v in st.items():
    fn = callee[0]
    is_cupy = ('cupy' in fn) or ('_ndarray_base' in callee[2] and fn == '~')
    if not is_cupy:
        continue
    for caller, cv in v[4].items():
        if 'mlframe' in caller[0]:
            key = (caller[0].replace(chr(92), '/').split('/')[-1], caller[1], caller[2])
            cnt[key] += cv[0]; tim[key] += cv[2]
print("mlframe functions by number of direct cupy calls")
for k, c in cnt.most_common(30):
    print(f"{c:6d} {tim[k]:6.2f}s  {k[0]}:{k[1]}({k[2]})")
print("total direct cupy calls from mlframe:", sum(cnt.values()))
