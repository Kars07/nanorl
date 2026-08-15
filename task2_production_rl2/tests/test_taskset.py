import tomllib
from pathlib import Path

import pytest
from tiny_repo_repair_v1.servers.tool import WORKSPACE, E2BRepoToolset
from tiny_repo_repair_v1.taskset import (
    CASES,
    TinyRepoRepairConfig,
    TinyRepoRepairTaskset,
)


def test_loads_production_sized_typed_tasks() -> None:
    tasks = list(TinyRepoRepairTaskset(TinyRepoRepairConfig()))
    assert len(tasks) == 150
    assert len(CASES) == 150
    assert len({task.data.category for task in tasks}) == 150


def test_taskset_override_is_narrow() -> None:
    tasks = list(TinyRepoRepairTaskset(TinyRepoRepairConfig(num_tasks=3)))
    assert [task.data.idx for task in tasks] == [0, 1, 2]


def test_train_eval_slices_are_disjoint() -> None:
    train = list(TinyRepoRepairTaskset(TinyRepoRepairConfig(num_tasks=100, start=0)))
    validation = list(
        TinyRepoRepairTaskset(TinyRepoRepairConfig(num_tasks=25, start=100))
    )
    test = list(TinyRepoRepairTaskset(TinyRepoRepairConfig(num_tasks=25, start=125)))
    splits = [
        {task.data.idx for task in train},
        {task.data.idx for task in validation},
        {task.data.idx for task in test},
    ]
    assert [len(split) for split in splits] == [100, 25, 25]
    assert all(a.isdisjoint(b) for idx, a in enumerate(splits) for b in splits[idx + 1 :])


def test_production_configs_wire_train_validation_and_sealed_test_splits() -> None:
    rl = tomllib.loads(Path("configs/rl/repo_repair_production.toml").read_text())
    test = tomllib.loads(Path("configs/eval/repo_repair_test.toml").read_text())
    train_taskset = rl["orchestrator"]["train"]["source"][0]["env"]["taskset"]
    validation_taskset = rl["orchestrator"]["eval"]["source"][0]["env"]["taskset"]
    test_taskset = test["env"]["taskset"]
    assert (train_taskset["start"], train_taskset["num_tasks"]) == (0, 100)
    assert (validation_taskset["start"], validation_taskset["num_tasks"]) == (
        100,
        25,
    )
    assert (test_taskset["start"], test_taskset["num_tasks"]) == (125, 25)


def test_seed_repositories_do_not_contain_hidden_checker() -> None:
    for task in TinyRepoRepairTaskset(TinyRepoRepairConfig()):
        assert task.data.check_command not in "\n".join(task.data.files.values())


def test_e2b_path_accepts_relative_and_workspace_absolute() -> None:
    expected = f"{WORKSPACE}/solution.py"
    assert E2BRepoToolset._path("solution.py") == expected
    assert E2BRepoToolset._path(expected) == expected


@pytest.mark.parametrize("path", ["/etc/passwd", "../secret", "a/../../secret"])
def test_e2b_path_rejects_escape(path: str) -> None:
    with pytest.raises(ValueError):
        E2BRepoToolset._path(path)
