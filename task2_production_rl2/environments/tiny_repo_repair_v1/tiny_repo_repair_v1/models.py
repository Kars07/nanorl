import verifiers.v1 as vf


class TinyRepoRepairData(vf.TaskData):
    files: dict[str, str]
    check_command: str
    category: str


class RepairState(vf.State):
    sandbox_id: str | None = None
    submitted: bool = False
    score: float = 0.0
    checker_stdout: str = ""
    checker_stderr: str = ""
    command_count: int = 0


class E2BToolConfig(vf.ToolsetConfig):
    timeout_seconds: int = 900
    command_timeout_seconds: int = 120
    max_output_chars: int = 20_000
    allow_internet_access: bool = False

