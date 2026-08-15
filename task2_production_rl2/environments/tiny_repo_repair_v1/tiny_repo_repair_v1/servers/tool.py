from __future__ import annotations

import os
import posixpath
import inspect
import signal
from typing import Any

import verifiers.v1 as vf
from e2b import Sandbox

from tiny_repo_repair_v1.models import E2BToolConfig, RepairState, TinyRepoRepairData

WORKSPACE = "/home/user/workspace"


def _ensure_e2b_transport_compatibility() -> None:
    """Adapt the E2B transport only in a Verifiers child with old pyqwest.

    Verifiers' subprocess runtime can place its own dependency overlay ahead of
    the parent venv. pyqwest 0.6 lacks the keyword-only system-certificate and
    proxy parameters used by E2B 2.35. Direct E2B connections do not need an
    HTTP proxy, so the compatibility path fails closed if one is configured.
    """
    import e2b.envd.client_sync as client_sync

    transport = client_sync.SyncHTTPTransport
    parameters = inspect.signature(transport).parameters
    if "tls_include_system_certs" in parameters:
        return

    def compatible_transport(**kwargs: Any) -> Any:
        kwargs.pop("tls_include_system_certs", None)
        proxy = kwargs.pop("proxy", None)
        if proxy is not None:
            raise RuntimeError("old pyqwest transport cannot safely use an HTTP proxy")
        return transport(**kwargs)

    client_sync.SyncHTTPTransport = compatible_transport


class E2BRepoToolset(vf.Toolset[E2BToolConfig, RepairState]):
    """Rollout-scoped MCP tools backed by one isolated E2B microVM."""

    TOOL_PREFIX = "e2b"

    def __init__(self, config: E2BToolConfig) -> None:
        super().__init__(config)
        self._sandbox: Sandbox | None = None
        self._task: TinyRepoRepairData | None = None

    async def setup_task(self, task: TinyRepoRepairData) -> None:
        _ensure_e2b_transport_compatibility()
        api_key = os.environ.get("E2B_API_KEY")
        key_file = os.environ.get("E2B_KEY_FILE")
        if not api_key and key_file:
            with open(key_file, encoding="utf-8") as handle:
                api_key = handle.read().strip()
        if not api_key:
            raise RuntimeError("E2B_API_KEY or E2B_KEY_FILE is required by E2BRepoToolset")
        self._task = TinyRepoRepairData.model_validate(task.model_dump())
        self._sandbox = Sandbox.create(
            api_key=api_key,
            timeout=self.config.timeout_seconds,
            secure=True,
            allow_internet_access=self.config.allow_internet_access,
            metadata={"owner": "task2-production-rl2", "task": str(task.idx)},
        )
        self._exit_stack.callback(self._sandbox.kill)

        # Verifiers normally tears the MCP server subprocess down with SIGTERM.
        # A process signal bypasses AsyncExitStack, so explicitly release the
        # remote VM before exiting; otherwise repeated RL rollouts accumulate
        # until the E2B account concurrency quota is exhausted.
        sandbox = self._sandbox

        def terminate(signum: int, _frame: Any) -> None:
            try:
                sandbox.kill()
            finally:
                raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        self.state.sandbox_id = self._sandbox.sandbox_id
        self._sandbox.commands.run(f"mkdir -p {WORKSPACE}")
        for relative_path, content in self._task.files.items():
            target = self._path(relative_path)
            self._sandbox.commands.run(f"mkdir -p {posixpath.dirname(target)}")
            self._sandbox.files.write(target, content)

    def _require_sandbox(self) -> Sandbox:
        if self._sandbox is None:
            raise RuntimeError("E2B sandbox has not been provisioned")
        return self._sandbox

    @staticmethod
    def _path(path: str) -> str:
        if path == WORKSPACE:
            return WORKSPACE
        if path.startswith(f"{WORKSPACE}/"):
            path = path[len(WORKSPACE) + 1 :]
        elif path.startswith("/"):
            raise ValueError(f"absolute path must be under {WORKSPACE}")
        normalized = posixpath.normpath(path)
        if normalized == ".." or normalized.startswith("../"):
            raise ValueError(f"path escapes {WORKSPACE}")
        return f"{WORKSPACE}/{normalized}"

    def _bounded(self, value: str) -> str:
        limit = self.config.max_output_chars
        if len(value) <= limit:
            return value
        return value[:limit] + f"\n...[truncated at {limit} characters]"

    @vf.tool
    def execute(self, command: str) -> dict[str, Any]:
        """Run a shell command inside the rollout's isolated E2B repository workspace."""
        result = self._require_sandbox().commands.run(
            command,
            cwd=WORKSPACE,
            timeout=self.config.command_timeout_seconds,
        )
        self.state.command_count += 1
        return {
            "exit_code": result.exit_code,
            "stdout": self._bounded(result.stdout),
            "stderr": self._bounded(result.stderr),
        }

    @vf.tool
    def read_file(self, path: str) -> str:
        """Read a UTF-8 file under /workspace from the isolated E2B sandbox."""
        return self._bounded(self._require_sandbox().files.read(self._path(path)))

    @vf.tool
    def write_file(self, path: str, content: str) -> str:
        """Write a UTF-8 file under /workspace in the isolated E2B sandbox."""
        target = self._path(path)
        self._require_sandbox().commands.run(f"mkdir -p {posixpath.dirname(target)}")
        self._require_sandbox().files.write(target, content)
        return f"wrote {path}"

    @vf.tool
    def submit(self) -> dict[str, Any]:
        """Run the hidden final-state checker and submit the current repository state."""
        if self._task is None:
            raise RuntimeError("task data is unavailable")
        sandbox = self._require_sandbox()
        result = sandbox.commands.run(
            self._task.check_command,
            cwd=WORKSPACE,
            timeout=self.config.command_timeout_seconds,
        )
        self.state.submitted = True
        self.state.score = float(result.exit_code == 0)
        self.state.checker_stdout = self._bounded(result.stdout)
        self.state.checker_stderr = self._bounded(result.stderr)
        for relative_path in self._task.files:
            try:
                content = sandbox.files.read(self._path(relative_path))
                self.state.artifacts[relative_path] = content.encode()
            except Exception:
                self.state.artifacts[relative_path] = None
        return {
            "passed": result.exit_code == 0,
            "exit_code": result.exit_code,
            "stdout": self.state.checker_stdout,
            "stderr": self.state.checker_stderr,
        }


if __name__ == "__main__":
    E2BRepoToolset.run()
