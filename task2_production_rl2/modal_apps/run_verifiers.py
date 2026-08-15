from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

VERIFIERS_REV = "7251c60934d2c42af85d42a1da3da62269b7957e"
REMOTE_ENV = "/opt/tiny_repo_repair_v1"
REMOTE_HARNESS = "/opt/learning_harness"

app = modal.App("task2-verifiers-v1")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv==0.11.21")
    .add_local_dir(
        "environments/tiny_repo_repair_v1",
        remote_path=REMOTE_ENV,
        copy=True,
    )
    .add_local_dir(
        "harnesses/learning_harness",
        remote_path=REMOTE_HARNESS,
        copy=True,
    )
    .add_local_file(
        "scripts/e2b_environment_smoke.py",
        remote_path="/opt/e2b_environment_smoke.py",
        copy=True,
    )
)


def _run(
    argv: list[str], timeout: int = 900, env: dict[str, str] | None = None
) -> dict[str, object]:
    result = subprocess.run(argv, text=True, capture_output=True, timeout=timeout, env=env)
    return {
        "command": " ".join(argv),
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _local_key(env_name: str, dotenv_name: str) -> str:
    key = os.environ.get(env_name)
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not key:
        for line in env_path.read_text().splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == dotenv_name:
                key = value.strip().strip("'\"")
                break
    if not key:
        raise RuntimeError(f".env must define {dotenv_name} or {env_name} must be set")
    return key


@app.function(image=image, cpu=2, memory=4096, timeout=1200)
def verifiers_command(
    mode: str = "dry-run", e2b_api_key: str = ""
) -> dict[str, object]:
    setup = _run(
        [
            "uv",
            "pip",
            "install",
            "--system",
            f"git+https://github.com/PrimeIntellect-ai/verifiers.git@{VERIFIERS_REV}",
            REMOTE_ENV,
            REMOTE_HARNESS,
        ]
    )
    if setup["exit_code"] != 0:
        return {"setup": setup}
    if mode == "dry-run":
        command = [
            "uv", "run", "--no-project", "eval", "tiny-repo-repair-v1",
            "--dry-run", "-n", "1", "--env.agent.harness.id", "null",
            "--env.agent.runtime.type", "subprocess",
            "--client.base-url", "http://127.0.0.1:8000/v1", "--no-push",
            "--client.api-key-var", "TASK2_LOCAL_INFERENCE_KEY",
        ]
    elif mode == "custom-harness-dry-run":
        command = [
            "uv", "run", "--no-project", "eval", "learning-harness",
            "--dry-run", "-n", "1", "--env.agent.harness.id", "learning-harness",
            "--env.agent.runtime.type", "subprocess",
            "--client.base-url", "http://127.0.0.1:8000/v1", "--no-push",
            "--client.api-key-var", "TASK2_LOCAL_INFERENCE_KEY",
        ]
    else:
        raise ValueError(
            "This CPU app only supports dry-run. Model evaluation is implemented "
            "in self_hosted_rollout.py and must target its Modal-owned inference server."
        )
    command_env = os.environ.copy()
    if e2b_api_key:
        key_path = Path("/tmp/task2-e2b-key")
        key_path.write_text(e2b_api_key)
        key_path.chmod(0o600)
        command_env["E2B_KEY_FILE"] = str(key_path)
    run = _run(command, env=command_env)
    artifacts: dict[str, str] = {}
    output_root = Path("outputs")
    if output_root.exists():
        for path in output_root.rglob("*"):
            if path.is_file() and path.stat().st_size <= 2_000_000:
                artifacts[str(path)] = path.read_text(errors="replace")
    return {"setup": setup, "run": run, "artifacts": artifacts}


@app.function(image=image, cpu=2, memory=4096, timeout=1200)
def e2b_environment_smoke(e2b_api_key: str) -> dict[str, object]:
    setup = _run(
        [
            "uv", "pip", "install", "--system",
            f"git+https://github.com/PrimeIntellect-ai/verifiers.git@{VERIFIERS_REV}",
            REMOTE_ENV,
        ]
    )
    if setup["exit_code"] != 0:
        return {"setup": setup}
    result = subprocess.run(
        ["uv", "run", "--no-project", "python", "/opt/e2b_environment_smoke.py"],
        env={**os.environ, "E2B_API_KEY": e2b_api_key},
        text=True,
        capture_output=True,
        timeout=600,
    )
    return {
        "setup_exit_code": setup["exit_code"],
        "command": "uv run --no-project python /opt/e2b_environment_smoke.py",
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.local_entrypoint()
def main(mode: str = "dry-run") -> None:
    if mode == "e2b-smoke":
        print(e2b_environment_smoke.remote(_local_key("E2B_API_KEY", "e2b_api")))
        return
    result = verifiers_command.remote(mode)
    print(result["setup"]["command"])
    print(result["setup"]["stdout"])
    print(result["setup"]["stderr"])
    if "run" in result:
        print(result["run"]["command"])
        print(result["run"]["stdout"])
        print(result["run"]["stderr"])
    if result.get("artifacts"):
        print("[artifacts]")
        for path, content in result["artifacts"].items():
            print(path)
            print(content)
    raise SystemExit(int(result.get("run", result["setup"])["exit_code"]))
