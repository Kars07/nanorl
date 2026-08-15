"""A small, concrete Verifiers v1 Runtime backed by an E2B microVM.

The pinned Verifiers RuntimeConfig union is intentionally closed and has no E2B
variant.  This project therefore uses this adapter only where it can pass a live
``Runtime`` object directly (not through the Verifiers CLI config parser).  It is
used by the Harbor execution probe and does not patch or impersonate a Prime,
Modal, or Docker runtime.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import shlex
from typing import ClassVar, Literal

from e2b import Sandbox
from pydantic_config import BaseConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import BaseRuntimeInfo, ProgramResult, Runtime, register


def _ensure_e2b_transport_compatibility() -> None:
    """Bridge the old pyqwest bundled in the pinned Prime environment."""
    import e2b.envd.client_sync as client_sync

    transport = client_sync.SyncHTTPTransport
    if "tls_include_system_certs" in inspect.signature(transport).parameters:
        return

    def compatible_transport(**kwargs):
        kwargs.pop("tls_include_system_certs", None)
        proxy = kwargs.pop("proxy", None)
        if proxy is not None:
            raise RuntimeError("the pinned pyqwest compatibility path forbids proxies")
        return transport(**kwargs)

    client_sync.SyncHTTPTransport = compatible_transport


class E2BConfig(BaseConfig):
    type: Literal["e2b"] = "e2b"
    template: str = "base"
    workdir: str = "/app"
    timeout_seconds: int = 900
    command_timeout_seconds: int = 180
    allow_internet_access: bool = True
    secure: bool = True
    owner: str = "task2-production-rl2"


class E2BRuntimeInfo(E2BConfig, BaseRuntimeInfo):
    pass


class E2BRuntime(Runtime):
    """Verifiers file/process primitives implemented with the E2B Python API."""

    is_local: ClassVar[bool] = False

    def __init__(
        self,
        config: E2BConfig,
        name: str | None = None,
        *,
        api_key: str | None = None,
    ) -> None:
        super().__init__(name)
        self.config = config
        self.info = E2BRuntimeInfo(**config.model_dump())
        self._api_key = api_key
        self._sandbox: Sandbox | None = None

    def _box(self) -> Sandbox:
        if self._sandbox is None:
            raise RuntimeError("E2B runtime has not been started")
        return self._sandbox

    async def start(self) -> None:
        _ensure_e2b_transport_compatibility()
        api_key = self._api_key or os.environ.get("E2B_API_KEY")
        key_file = os.environ.get("E2B_KEY_FILE")
        if not api_key and key_file:
            api_key = await asyncio.to_thread(
                lambda: open(key_file, encoding="utf-8").read().strip()
            )
        if not api_key:
            raise RuntimeError("E2B_API_KEY or E2B_KEY_FILE is required")

        self._sandbox = await asyncio.to_thread(
            Sandbox.create,
            self.config.template,
            api_key=api_key,
            timeout=self.config.timeout_seconds,
            secure=self.config.secure,
            allow_internet_access=self.config.allow_internet_access,
            metadata={"owner": self.config.owner, "runtime": self.name},
        )
        register(self)
        self.info.id = self._sandbox.sandbox_id
        # Harbor's verifier stages into root-level paths and evaluates in /app.
        # E2B's default user receives ownership before any untrusted action runs.
        bootstrap = (
            "sudo mkdir -p /app /tests /logs/verifier && "
            "sudo chown -R $(id -u):$(id -g) /app /tests /logs"
        )
        try:
            result = await self._run_at(
                ["bash", "-lc", bootstrap], {}, cwd="/home/user"
            )
            if result.exit_code:
                raise SandboxError(f"E2B bootstrap failed: {result.stderr[-500:]}")
        except BaseException:
            self.cleanup()
            raise

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        return await self._run_at(argv, env, cwd=self.config.workdir)

    async def _run_at(
        self, argv: list[str], env: dict[str, str], *, cwd: str
    ) -> ProgramResult:
        command = shlex.join(argv)

        def execute():
            return self._box().commands.run(
                command,
                cwd=cwd,
                envs=env,
                timeout=self.config.command_timeout_seconds,
            )

        try:
            result = await asyncio.to_thread(execute)
        except Exception as exc:  # provider errors become the runtime boundary error
            raise SandboxError(f"E2B command failed: {exc}") from exc
        return ProgramResult(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    async def _read(self, path: str) -> bytes:
        try:
            value = await asyncio.to_thread(self._box().files.read, path, format="bytes")
        except Exception as exc:
            raise SandboxError(f"read {path!r}: {exc}") from exc
        return bytes(value)

    async def write(self, path: str, data: bytes) -> None:
        try:
            await asyncio.to_thread(self._box().files.write, path, data)
        except Exception as exc:
            raise SandboxError(f"write {path!r}: {exc}") from exc

    def cleanup(self) -> None:
        sandbox, self._sandbox = self._sandbox, None
        if sandbox is not None:
            try:
                sandbox.kill()
            except Exception:
                pass
