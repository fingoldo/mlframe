"""FS_BENCHMARKS_A-1 (2026-08-05 audit): every consumable binding in
``feature_selection._benchmarks._datasets`` (``SCENARIOS``, ``CPU_SCENARIOS``, ``GPU_SCENARIOS``,
``make_scenario_data``, ``list_scenarios``) was indented one level inside an
``if __name__ == "__main__":`` guard -- so none of them existed on the module object after a normal
``import``/``from . import _datasets``, and ``bench_mrmr.py`` (the package's own documented entry point)
crashed with ``AttributeError`` the moment it referenced any of them, before a single benchmark scenario
ran. Behavioral checks (not source-inspection): exercise the module exactly the way ``bench_mrmr.py`` does.
"""

from __future__ import annotations

import mlframe.feature_selection._benchmarks._datasets as ds


def test_scenarios_dict_is_module_level_and_complete():
    """SCENARIOS must be a real module attribute (not trapped inside a __main__ guard) with all 8 scenarios."""
    assert hasattr(ds, "SCENARIOS"), "SCENARIOS is not importable -- likely trapped inside an if __name__ guard"
    assert len(ds.SCENARIOS) == 8


def test_cpu_gpu_scenario_partition():
    """CPU_SCENARIOS/GPU_SCENARIOS must be module-level and correctly partitioned by use_gpu."""
    assert hasattr(ds, "CPU_SCENARIOS") and hasattr(ds, "GPU_SCENARIOS")
    assert all(not s.use_gpu for s in ds.CPU_SCENARIOS)
    assert all(s.use_gpu for s in ds.GPU_SCENARIOS)
    assert len(ds.CPU_SCENARIOS) + len(ds.GPU_SCENARIOS) == len(ds.SCENARIOS)


def test_list_scenarios_include_gpu_ordering():
    """list_scenarios must be callable at module scope and append GPU scenarios last when included."""
    assert ds.list_scenarios(include_gpu=False) == list(ds.CPU_SCENARIOS)
    assert ds.list_scenarios(include_gpu=True) == ds.CPU_SCENARIOS + ds.GPU_SCENARIOS


def test_make_scenario_data_generates_expected_shape():
    """make_scenario_data must be callable at module scope and actually generate data -- exercises the real
    data-generation code path bench_mrmr.py depends on, on the smallest scenario for test speed."""
    scenario = ds.SCENARIOS["n10k_p100_clf"]
    X, y = ds.make_scenario_data(scenario, random_state=1)
    assert X.shape == (10_000, 100)
    assert len(y) == 10_000


def test_gpu_scenario_flag_sanity():
    """Sanity: the GPU-tagged scenario is actually tagged use_gpu=True."""
    assert ds.SCENARIOS["n100k_p100_clf_gpu"].use_gpu is True
