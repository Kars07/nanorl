"""Execute the official Harbor hello-world task in E2B and grade it upstream."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import verifiers.v1 as vf
from openai import OpenAI
from verifiers.v1.graph import MessageNode
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTaskset
from verifiers.v1.types import AssistantMessage, Sampling, Usage, UserMessage

from task2_runtime import E2BConfig, E2BRuntime


def validate_minimal_dockerfile(task_dir: Path) -> dict[str, object]:
    """Accept only the exact image-independent subset this adapter implements."""
    path = task_dir / "environment" / "Dockerfile"
    logical = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    allowed = ["FROM ubuntu:24.04", "WORKDIR /app"]
    if logical != allowed:
        raise RuntimeError(
            "E2B Harbor adapter refuses Dockerfiles beyond the audited "
            f"FROM+WORKDIR subset; got {logical!r}"
        )
    return {"path": str(path), "instructions": logical, "emulated_workdir": "/app"}


def extract_command(text: str) -> str:
    tagged = re.search(r"<command>\s*(.*?)\s*</command>", text, re.DOTALL)
    if tagged:
        return tagged.group(1).strip()
    fenced = re.search(r"```(?:bash|sh)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    raise ValueError("model response did not contain <command> or a shell code fence")


def command_from_response(message) -> tuple[str, dict[str, object]]:
    """Prefer a schema-checked tool call; retain a text fallback for portability."""
    if message.tool_calls:
        call = message.tool_calls[0]
        if call.function.name != "execute":
            raise ValueError(f"unexpected tool call {call.function.name!r}")
        arguments = json.loads(call.function.arguments)
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("execute tool call omitted its non-empty command")
        return command.strip(), {
            "mode": "tool_call",
            "tool_call_id": call.id,
            "tool_name": call.function.name,
            "arguments": arguments,
        }
    text = message.content or ""
    return extract_command(text), {"mode": "text", "content": text}


async def execute(args: argparse.Namespace) -> dict[str, object]:
    strict_config = HarborConfig(dataset=args.dataset)
    strict_error = None
    try:
        next(iter(HarborTaskset(config=strict_config).head(1)))
    except Exception as exc:  # expected: upstream refuses Dockerfile builds
        strict_error = f"{type(exc).__name__}: {exc}"
    if strict_error is None:
        raise RuntimeError("expected strict Harbor load to reject its Dockerfile")

    taskset = HarborTaskset(
        config=HarborConfig(dataset=args.dataset, ignore_dockerfile=True)
    )
    task = next(iter(taskset.head(1)))
    dockerfile = validate_minimal_dockerfile(Path(task.data.task_dir))

    client = OpenAI(base_url=args.base_url, api_key="local-no-auth")
    messages = [
        {
            "role": "system",
            "content": (
                "You are a terminal agent in /app. Solve the user's task by calling "
                "the execute tool exactly once. Do not explain the command."
            ),
        },
        {"role": "user", "content": task.data.prompt_text},
    ]
    call_started = time.time()
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=args.model,
        messages=messages,
        temperature=0.0,
        max_tokens=256,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "execute",
                    "description": "Execute one shell command in /app.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "execute"}},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    call_ended = time.time()
    response_message = response.choices[0].message
    answer = response_message.content or ""
    command, action_request = command_from_response(response_message)

    runtime = E2BRuntime(
        E2BConfig(
            workdir=task.data.workdir or "/app",
            timeout_seconds=args.sandbox_timeout,
            command_timeout_seconds=args.command_timeout,
            allow_internet_access=True,
        )
    )
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig(model=args.model)),
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        nodes=[
            MessageNode(
                parent=None,
                message=UserMessage(content=task.data.prompt_text),
                sampled=False,
            ),
            MessageNode(
                parent=0,
                message=AssistantMessage(
                    content=answer or json.dumps(action_request, sort_keys=True)
                ),
                sampled=True,
            ),
        ],
        calls=[
            vf.ModelCall(
                node=1,
                model=args.model,
                sampling=Sampling(temperature=0.0, max_tokens=256),
                endpoint="/v1/chat/completions",
                finish_reason=response.choices[0].finish_reason,
                usage=(
                    Usage.from_openai(response.usage)
                    if response.usage
                    else None
                ),
                time=vf.TimeSpan(start=call_started, end=call_ended),
            )
        ],
        tools=[
            vf.Tool(
                name="execute",
                description="Execute one shell command in /app.",
                parameters={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                    "additionalProperties": False,
                },
                strict=True,
            )
        ],
    )
    trace.record_run(vf.EvalRunInfo(id="harbor-e2b-modal"))

    await runtime.start()
    action = None
    verifier_started = 0.0
    try:
        action = await runtime.run(["bash", "-lc", command], {})
        verifier_started = time.time()
        await task.score(trace, runtime)
        verifier_ended = time.time()
        trace.info.update(
            {
                "runtime": "e2b",
                "sandbox_id": runtime.info.id,
                "modal_hosted_driver": True,
                "dockerfile_adapter": dockerfile,
                "agent_command": command,
                "model_action_request": action_request,
                "agent_exit_code": action.exit_code,
                "agent_stdout": action.stdout,
                "agent_stderr": action.stderr,
                "tests_staged_after_agent": True,
                "verifier_seconds": verifier_ended - verifier_started,
            }
        )
        trace.stop("agent_completed")
        trace.ok = action.exit_code == 0
        trace.is_completed = True
    finally:
        await runtime.stop()

    return {
        "command": "uv run --frozen python scripts/run_harbor_e2b.py",
        "dataset": args.dataset,
        "taskset_class": type(taskset).__name__,
        "task_class": type(task).__name__,
        "model": args.model,
        "base_url": args.base_url,
        "strict_loader_error": strict_error,
        "dockerfile_adapter": dockerfile,
        "trace": trace.model_dump(mode="json"),
        "reward": trace.reward,
        "passed": trace.ok and trace.reward == 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="harbor/hello-world")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--sandbox-timeout", type=int, default=900)
    parser.add_argument("--command-timeout", type=int, default=180)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = asyncio.run(execute(args))
    payload = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
