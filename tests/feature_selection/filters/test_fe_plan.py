"""The planner of the fit as a scheduling problem: exact on small instances (against enumeration), valid and bounded on simulated machines, and never trusting its solver."""

import itertools
import random
import time

import pytest

from mlframe.feature_selection.filters._fe_plan import Instance, Mode, Placement, Plan, Task, cpsat_available, critical_path, energy_bound, list_schedule, lower_bound, plan, violations

needs_cpsat = pytest.mark.skipif(not cpsat_available(), reason="ortools is not installed")


def _task(name, *modes, after=()):
    """A task from ``(machine, duration, demand)`` triples."""
    return Task(name, tuple(Mode(m, float(d), dict(demand)) for m, d, demand in modes), tuple(after))


def _random_instance(seed, n=5):
    """A small instance with integer data: two machines, two resources, a random DAG, one transfer delay."""
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        after = tuple(f"t{j}" for j in range(i) if rng.random() < 0.35)
        modes = [("cpu", rng.randint(1, 5), {"cores": rng.randint(1, 3)})]
        if rng.random() < 0.7:
            modes.append(("gpu", rng.randint(1, 4), {"gpu": rng.randint(1, 2), "cores": 1}))
        tasks.append(_task(f"t{i}", *modes, after=after))
    return Instance(tuple(tasks), {"cores": 3, "gpu": 2}, {("cpu", "gpu"): 1.0, ("gpu", "cpu"): 1.0})


def _enumerated_optimum(instance):
    """The optimal makespan by brute force: every assignment of modes and every precedence-feasible order, each placed at the earliest integer start where it fits (the active schedules contain an optimum)."""
    tasks = instance.by_name()
    names = list(tasks)
    best = float("inf")
    horizon = int(sum(max(m.duration for m in t.modes) for t in tasks.values()) + 10 * (len(names) + 2))
    for choice in itertools.product(*[instance.feasible_modes(tasks[n]) for n in names]):
        picked = dict(zip(names, choice))
        for order in itertools.permutations(names):
            if any(a not in order[:k] for k, n in enumerate(order) for a in tasks[n].after):
                continue
            usage = {r: [0.0] * horizon for r in instance.capacity}
            end, machine, finish = {}, {}, 0
            for n in order:
                mode = tasks[n].modes[picked[n]]
                d = int(mode.duration)
                t = int(max((end[a] + instance.delay(machine[a], mode.machine) for a in tasks[n].after), default=0))
                while any(usage[r][t + k] + a > instance.capacity[r] + 1e-9 for r, a in mode.demand.items() for k in range(d)):
                    t += 1
                for r, a in mode.demand.items():
                    for k in range(d):
                        usage[r][t + k] += a
                end[n], machine[n] = t + d, mode.machine
                finish = max(finish, t + d)
            best = min(best, finish)
    return float(best)


@needs_cpsat
@pytest.mark.parametrize("seed", range(12))
def test_cpsat_equals_the_enumerated_optimum_on_small_instances(seed):
    """The plan of ``plan`` is valid, proved optimal and as short as the best of all schedules; the list schedule is valid and never shorter; the bound never exceeds the optimum."""
    instance = _random_instance(seed)
    optimum = _enumerated_optimum(instance)
    exact = plan(instance, time_limit_s=5.0)
    assert violations(instance, exact) == []
    assert exact.makespan == pytest.approx(optimum, abs=1e-6) and exact.status == "optimal"
    greedy = list_schedule(instance)
    assert violations(instance, greedy) == [] and greedy.makespan >= optimum - 1e-9
    assert lower_bound(instance) <= optimum + 1e-9 and exact.lower_bound <= optimum + 1e-9


def test_one_machine_gives_the_sequential_order():
    """With one device holding everything, tasks cannot overlap: the makespan is the sum of the durations and a chain keeps its order."""
    tasks = (_task("a", ("m", 3, {"dev": 1})), _task("b", ("m", 4, {"dev": 1}), after=("a",)), _task("c", ("m", 2, {"dev": 1})))
    instance = Instance(tasks, {"dev": 1})
    result = plan(instance)
    assert violations(instance, result) == [] and result.makespan == pytest.approx(9.0) and result.status == "optimal"
    placed = result.placements
    assert placed["a"].end <= placed["b"].start + 1e-9


def test_a_machine_with_128_cores_and_no_gpu_runs_the_tasks_in_waves():
    """Forty independent tasks of 8 cores for 10 s on 128 cores: sixteen at a time, three waves, 30 s; the bound is the energy 25 s."""
    tasks = tuple(_task(f"s{i}", ("cpu", 10, {"cores": 8})) for i in range(40))
    instance = Instance(tasks, {"cores": 128})
    assert energy_bound(instance) == pytest.approx(25.0)
    result = plan(instance, time_limit_s=1.0)
    assert violations(instance, result) == [] and result.makespan == pytest.approx(30.0)


def test_eight_cores_and_one_gpu_split_the_work_between_them():
    """Six tasks that run 10 s on 4 cores or 3 s on the whole GPU (2 GB of the card's 4): the GPU takes four (12 s) while two overlap on the cores (10 s): 12 s, against 18 on the GPU alone and 30 on the cores."""
    tasks = tuple(_task(f"s{i}", ("cpu", 10, {"cores": 4}), ("gpu0", 3, {"gpu0": 1.0, "vram0": 2.0})) for i in range(6))
    instance = Instance(tasks, {"cores": 8, "gpu0": 1.0, "vram0": 4.0})
    result = plan(instance, time_limit_s=2.0)
    assert violations(instance, result) == [] and result.makespan <= 12.0 + 1e-6
    assert result.makespan < 18.0


def test_two_gpus_of_different_size_keep_the_big_stages_on_the_big_card():
    """Stages that need 3 GB cannot be candidates of the 2 GB card; the small ones use both cards; every placement fits its device."""
    big = [_task(f"big{i}", ("gpu0", 2, {"g0": 1.0, "v0": 3.0})) for i in range(4)]
    small = [_task(f"small{i}", ("gpu0", 2, {"g0": 1.0, "v0": 1.0}), ("gpu1", 5, {"g1": 1.0, "v1": 1.0})) for i in range(4)]
    wrong = [Task(f"big{i}", (*big[i].modes, Mode("gpu1", 1.0, {"g1": 1.0, "v1": 3.0}))) for i in range(4)]  # a fast mode on the small card that cannot fit
    instance = Instance((*wrong, *small), {"g0": 1.0, "v0": 4.0, "g1": 1.0, "v1": 2.0})
    result = plan(instance, time_limit_s=2.0)
    assert violations(instance, result) == []
    assert all(result.placements[f"big{i}"].machine == "gpu0" for i in range(4))
    assert result.makespan == pytest.approx(12.0) and result.status == "optimal"  # gpu0 takes the four big and two small ones (12 s), gpu1 the other two small (10 s); gpu0 alone needs 16 s


def test_a_stage_that_does_not_fit_the_card_runs_on_the_cpu():
    """A GPU mode that needs more memory than the card has is not a candidate: the slower CPU mode is used."""
    task = _task("wide", ("gpu0", 1, {"gpu0": 1.0, "vram0": 8.0}), ("cpu", 7, {"cores": 2}))
    instance = Instance((task,), {"gpu0": 1.0, "vram0": 4.0, "cores": 8})
    result = plan(instance)
    assert result.placements["wide"].machine == "cpu" and result.makespan == pytest.approx(7.0)


def test_a_transfer_delay_can_make_the_slower_machine_better():
    """A runs 1 s on the GPU or 3 s on the CPU, B only on the CPU; the data crossing costs 5 s: A on the CPU (4 s in all) beats the GPU (7 s)."""
    tasks = (_task("a", ("gpu0", 1, {"gpu0": 1.0}), ("cpu", 3, {"cores": 1})), _task("b", ("cpu", 1, {"cores": 1}), after=("a",)))
    instance = Instance(tasks, {"gpu0": 1.0, "cores": 4}, {("gpu0", "cpu"): 5.0, ("cpu", "gpu0"): 5.0})
    result = plan(instance)
    assert violations(instance, result) == [] and result.placements["a"].machine == "cpu" and result.makespan == pytest.approx(4.0)


def test_the_bounds_are_the_critical_path_and_the_energy():
    """A chain of 2 + 3 with a side task: the path is 5; resources of capacity 1 held by 4 s of work give 4."""
    tasks = (_task("a", ("m", 2, {"r": 1})), _task("b", ("m", 3, {"r": 0}), after=("a",)), _task("c", ("m", 2, {"r": 1})))
    instance = Instance(tasks, {"r": 1})
    assert critical_path(instance) == pytest.approx(5.0) and energy_bound(instance) == pytest.approx(4.0) and lower_bound(instance) == pytest.approx(5.0)


def test_a_bad_instance_is_refused():
    """A duplicate name, an unknown predecessor, a cycle and a task that fits no mode raise ``ValueError`` before any solver runs."""
    ok = _task("a", ("m", 1, {"r": 1}))
    with pytest.raises(ValueError, match="duplicate"):
        plan(Instance((ok, ok), {"r": 1}))
    with pytest.raises(ValueError, match="unknown"):
        plan(Instance((_task("b", ("m", 1, {}), after=("zzz",)),), {"r": 1}))
    with pytest.raises(ValueError, match="cycle"):
        plan(Instance((_task("x", ("m", 1, {}), after=("y",)), _task("y", ("m", 1, {}), after=("x",))), {"r": 1}))
    with pytest.raises(ValueError, match="no mode"):
        plan(Instance((_task("big", ("m", 1, {"r": 5})),), {"r": 1}))


def test_the_validator_catches_a_tampered_plan():
    """An early start, an overloaded resource and a wrong makespan are reported."""
    tasks = (_task("a", ("m", 2, {"r": 1})), _task("b", ("m", 2, {"r": 1}), after=("a",)))
    instance = Instance(tasks, {"r": 1})
    good = plan(instance)
    assert violations(instance, good) == []
    early = Plan({"a": good.placements["a"], "b": Placement("b", 0, "m", 1.0, 3.0)}, 3.0, 4.0, "feasible", "test")
    assert any("before a ends" in p for p in violations(instance, early))
    loose = Instance((*tasks[:1], _task("b", ("m", 2, {"r": 1}))), {"r": 1})
    overlap = Plan({"a": Placement("a", 0, "m", 0.0, 2.0), "b": Placement("b", 0, "m", 1.0, 3.0)}, 3.0, 2.0, "feasible", "test")
    assert any("r:" in p for p in violations(loose, overlap))
    wrong_span = Plan(dict(good.placements), 99.0, 4.0, "feasible", "test")
    assert any("makespan" in p for p in violations(instance, wrong_span))


def test_a_larger_instance_is_planned_within_the_time_limit_with_a_gap_reported():
    """Sixty random tasks: a valid plan comes back in about the time limit, with a lower bound below the makespan."""
    rng = random.Random(7)
    tasks = []
    for i in range(60):
        after = tuple(f"t{j}" for j in rng.sample(range(i), min(i, rng.randint(0, 2)))) if i else ()
        tasks.append(_task(f"t{i}", ("cpu", rng.randint(1, 9), {"cores": rng.randint(1, 4)}), ("gpu", rng.randint(1, 5), {"gpu": rng.randint(1, 3), "cores": 1}), after=after))
    instance = Instance(tuple(tasks), {"cores": 6, "gpu": 3}, {("cpu", "gpu"): 0.5, ("gpu", "cpu"): 0.5})
    elapsed_runs = []
    for _ in range(3):
        started = time.perf_counter()
        result = plan(instance, time_limit_s=0.5)
        elapsed_runs.append(time.perf_counter() - started)
        assert violations(instance, result) == [] and result.lower_bound <= result.makespan + 1e-9 and 0.0 <= result.gap < 1.0
    assert min(elapsed_runs) < 6.0
    assert result.makespan <= list_schedule(instance).makespan + 1e-9


def test_without_the_exact_solver_the_list_schedule_is_returned(monkeypatch):
    """With ortools unavailable the plan is the list schedule, valid, with a bound."""
    from mlframe.feature_selection.filters._fe_plan import _solve

    monkeypatch.setattr(_solve, "available", lambda: False)
    instance = _random_instance(3)
    result = plan(instance)
    assert result.solver == "list-scheduling" and violations(instance, result) == [] and result.lower_bound <= result.makespan + 1e-9


@needs_cpsat
def test_a_solver_plan_that_fails_the_check_is_replaced_by_the_list_schedule(monkeypatch):
    """A defective exact plan (a task started at time 0 regardless of its predecessor) never reaches the caller."""
    from mlframe.feature_selection.filters._fe_plan import _solve

    instance = _random_instance(1, n=6)
    broken = list_schedule(instance)
    broken = Plan({n: Placement(n, p.mode, p.machine, 0.0, p.end - p.start) for n, p in broken.placements.items()}, 0.5, 0.1, "optimal", "cp-sat")
    monkeypatch.setattr(_solve, "solve_cpsat", lambda *a, **k: broken)
    result = plan(instance)
    assert violations(instance, result) == [] and result.solver == "list-scheduling" and any("rejected" in n for n in result.notes)
