from learning_harness.harness import (
    LearningHarnessHarness,
    LearningHarnessHarnessConfig,
)
from learning_harness.taskset import LearningHarnessConfig, LearningHarnessTaskset


def test_custom_harness_capabilities_are_explicit() -> None:
    harness = LearningHarnessHarness(LearningHarnessHarnessConfig(id="learning-harness"))
    assert harness.APPENDS_SYSTEM_PROMPT
    assert not harness.SUPPORTS_MCP
    assert not harness.SUPPORTS_RESUME
    assert not harness.EXECUTES_CODE
    assert not harness.NEEDS_CONTAINER


def test_custom_harness_taskset_is_bounded() -> None:
    tasks = list(LearningHarnessTaskset(LearningHarnessConfig(num_tasks=2)))
    assert [task.data.expected for task in tasks] == ["amber", "cobalt"]
