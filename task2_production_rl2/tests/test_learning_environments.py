import pytest

import verifiers.v1 as vf
from proposer_solver_v1.servers.tool import ProposerSolverToolset
from proposer_solver_v1.taskset import (
    ProposerSolverConfig,
    ProposerSolverEnv,
    ProposerSolverEnvConfig,
    ProposerSolverTaskset,
)
from tiny_browser_v1.servers.tool import TinyBrowserToolset
from tiny_browser_v1.taskset import TinyBrowserConfig, TinyBrowserTaskset
from tiny_long_horizon_v1.servers.tool import TinyLongHorizonToolset
from tiny_long_horizon_v1.taskset import TinyLongHorizonConfig, TinyLongHorizonTaskset
from tiny_terminal_v1.servers.tool import TinyTerminalToolset
from tiny_terminal_v1.taskset import TinyTerminalConfig, TinyTerminalTaskset
from tiny_tooluse_v1.servers.tool import TinyTooluseToolset
from tiny_tooluse_v1.taskset import TinyTooluseConfig, TinyTooluseTaskset


@pytest.mark.asyncio
async def test_terminal_objective_final_state() -> None:
    task = next(iter(TinyTerminalTaskset(TinyTerminalConfig(num_tasks=1))))
    tools = TinyTerminalToolset(vf.ToolsetConfig())
    await tools.setup_task(task.data)
    tools.write("status.txt", "healthy")
    assert tools.submit() == {"passed": True}
    assert tools.state.score == 1.0


@pytest.mark.asyncio
async def test_browser_requires_evidence_navigation() -> None:
    task = next(iter(TinyBrowserTaskset(TinyBrowserConfig(num_tasks=1))))
    tools = TinyBrowserToolset(vf.ToolsetConfig())
    await tools.setup_task(task.data)
    assert tools.submit("2")["passed"] is False
    assert "evidence required" in tools.submit("2")["error"]
    assert tools.search("Mars") == ["planets"]
    assert "two moons" in tools.open_page("planets")
    assert tools.submit("2") == {"passed": True}
    assert tools.state.visited == ["planets"]


@pytest.mark.asyncio
async def test_structured_tool_use() -> None:
    task = next(iter(TinyTooluseTaskset(TinyTooluseConfig(num_tasks=1))))
    tools = TinyTooluseToolset(vf.ToolsetConfig())
    await tools.setup_task(task.data)
    assert tools.query("ISSUE-2")["owner"] == "Lin"
    assert tools.submit("Lin") == {"passed": True}
    assert tools.state.queries == 1


@pytest.mark.asyncio
async def test_long_horizon_rejects_out_of_order_step() -> None:
    task = next(iter(TinyLongHorizonTaskset(TinyLongHorizonConfig(num_tasks=1))))
    tools = TinyLongHorizonToolset(vf.ToolsetConfig())
    await tools.setup_task(task.data)
    with pytest.raises(ValueError, match="expected 'inspect'"):
        tools.complete("edit")
    for step in task.data.steps:
        assert tools.next_step() == step
        tools.complete(step)
    assert tools.submit() == {"passed": True}


@pytest.mark.asyncio
async def test_proposer_solver_objective_pair() -> None:
    task = next(iter(ProposerSolverTaskset(ProposerSolverConfig(num_tasks=1))))
    tools = ProposerSolverToolset(vf.ToolsetConfig())
    await tools.setup_task(task.data)
    tools.propose("reverse abc")
    tools.solve("cba")
    assert tools.submit() == {"passed": True}


def test_proposer_solver_is_real_three_seat_env() -> None:
    assert issubclass(ProposerSolverEnv, vf.Env)
    assert {"proposer", "solver", "n", "train_proposer", "train_solver"} <= ProposerSolverEnvConfig.model_fields.keys()
    config = ProposerSolverEnvConfig()
    assert config.n == 2
    assert config.max_concurrent_agents == 2


@pytest.mark.parametrize(
    ("taskset", "config", "expected"),
    [
        (TinyTerminalTaskset, TinyTerminalConfig, 2),
        (TinyBrowserTaskset, TinyBrowserConfig, 2),
        (TinyTooluseTaskset, TinyTooluseConfig, 2),
        (TinyLongHorizonTaskset, TinyLongHorizonConfig, 2),
        (ProposerSolverTaskset, ProposerSolverConfig, 2),
    ],
)
def test_taskset_count_overrides(taskset, config, expected: int) -> None:
    assert len(list(taskset(config(num_tasks=expected)))) == expected
