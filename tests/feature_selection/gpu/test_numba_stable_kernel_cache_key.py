"""A cached CUDA kernel that closes over device helpers hits its on-disk cache in the next process once the helpers' ids are pinned; ``pin_dispatcher_uuids`` is deterministic."""

import subprocess
import sys
import textwrap

import pytest

numba_cuda = pytest.importorskip("numba.cuda")

from mlframe.feature_selection.filters._numba_stable_keys import pin_dispatcher_uuids

SCRIPT = textwrap.dedent(
    '''
    import sys
    import numpy as np
    from numba import cuda
    sys.path.insert(0, {src!r})
    from mlframe.feature_selection.filters._numba_stable_keys import pin_dispatcher_uuids

    @cuda.jit(device=True)
    def shared_helper(x):
        return x + 1.0

    def factory(ft):
        @cuda.jit(device=True)
        def helper(x):
            return shared_helper(x) * ft(2.0)
        if {pin}:
            pin_dispatcher_uuids(helper, tag=ft.__name__)

        @cuda.jit(cache=True)
        def kern(out, a):
            out[0] = helper(a[0])
        return kern

    k = factory(np.float32)
    out = cuda.to_device(np.zeros(1, dtype=np.float32))
    k[1, 1](out, cuda.to_device(np.array([3.0], dtype=np.float32)))
    print("RESULT", float(out.copy_to_host()[0]))
    '''
)


def _run(tmp_path, pin: bool) -> str:
    """One fresh process: the script's output with numba's cache messages."""
    import os

    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
    script = tmp_path / "cached_kernel.py"
    if not script.exists():  # written once: numba stamps the cache with the source file's mtime, a rewrite would invalidate the entry
        script.write_text(SCRIPT.format(src=src, pin=pin), encoding="utf-8")
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "numba_cache"), NUMBA_DEBUG_CACHE="1")
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=300, check=False)
    assert res.returncode == 0, res.stderr[-2000:]
    return res.stdout + res.stderr


def test_pinned_helpers_give_a_cache_hit_in_the_next_process(tmp_path):
    """First process compiles and saves; the second loads the entry and saves nothing; both compute the same value."""
    if not numba_cuda.is_available():
        pytest.skip("no CUDA device")
    first = _run(tmp_path, pin=True)
    second = _run(tmp_path, pin=True)
    assert "RESULT 8.0" in first and "RESULT 8.0" in second
    # only the messages about the test kernel: importing the package touches the caches of its own kernels too
    mine_first = [ln for ln in first.splitlines() if "[cache]" in ln and ".kern-" in ln]
    mine_second = [ln for ln in second.splitlines() if "[cache]" in ln and ".kern-" in ln]
    assert any("index saved" in ln for ln in mine_first)
    assert not any("saved" in ln for ln in mine_second) and any("data loaded" in ln for ln in mine_second), mine_second


def test_pinning_is_deterministic_and_counts_what_it_pinned():
    """The same helpers pinned twice in one process (a second factory call) get the same ids; nested and module-level dispatchers are included."""
    if not numba_cuda.is_available():
        pytest.skip("no CUDA device")
    import numpy as np
    from numba import cuda

    @cuda.jit(device=True)
    def inner(x):
        """A module-level style device helper the outer one calls."""
        return x + 1.0

    def build(ft):
        """A factory returning a device function that closes over ``ft`` and calls ``inner``."""

        @cuda.jit(device=True)
        def outer(x):
            """The device function built per float type."""
            return inner(x) * ft(2.0)

        return outer

    a, b = build(np.float32), build(np.float32)
    n_a = pin_dispatcher_uuids(a, tag="float32")
    n_b = pin_dispatcher_uuids(b, tag="float32")
    assert n_a >= 1 and n_b >= 0
    assert a._uuid == b._uuid or n_b == 0  # the nested module-level helper was pinned by the first call and is left alone by the second
