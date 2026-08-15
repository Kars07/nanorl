from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from verifiers.v1.types import Usage

from scripts.run_harbor_e2b import command_from_response, validate_minimal_dockerfile


def test_harbor_action_comes_from_typed_execute_call() -> None:
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(
                    name="execute",
                    arguments=json.dumps(
                        {"command": "printf 'Hello, world!' > hello.txt"}
                    ),
                ),
            )
        ],
    )
    command, evidence = command_from_response(message)
    assert command == "printf 'Hello, world!' > hello.txt"
    assert evidence["mode"] == "tool_call"


def test_harbor_adapter_rejects_unimplemented_dockerfile(tmp_path: Path) -> None:
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nWORKDIR /app\nRUN apt-get update\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="refuses Dockerfiles"):
        validate_minimal_dockerfile(tmp_path)


def test_pinned_usage_schema_accepts_openai_accounting() -> None:
    source = SimpleNamespace(
        prompt_tokens=11,
        completion_tokens=7,
        prompt_tokens_details=None,
        completion_tokens_details=None,
    )
    usage = Usage.from_openai(source)
    assert usage is not None
    assert usage.prompt_tokens == 11
    assert usage.total_tokens == 18
