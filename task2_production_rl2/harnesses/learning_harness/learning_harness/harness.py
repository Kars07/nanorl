from __future__ import annotations

from pathlib import Path

import verifiers.v1 as vf


PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


class LearningHarnessHarnessConfig(vf.HarnessConfig):
    """Configuration for the deliberately small, one-model-call harness."""


class LearningHarnessHarness(vf.Harness[LearningHarnessHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = False
    SUPPORTS_RESUME = False
    EXECUTES_CODE = False
    NEEDS_CONTAINER = False

    async def setup(self, runtime: vf.Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)

    async def launch(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: vf.TaskData,
    ) -> vf.ProgramResult:
        if mcp_urls:
            raise ValueError("learning-harness is intentionally tool-free")
        system_prompt, prompt = self.resolve_text_prompt(data)
        program = await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)
        args = [
            *program,
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--prompt={prompt or ''}",
        ]
        if system_prompt:
            args.append(f"--system-prompt={system_prompt}")
        return await runtime.run_program(args, self.config.resolved_env)
