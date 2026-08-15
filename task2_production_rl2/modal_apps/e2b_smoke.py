from __future__ import annotations

import os
import json
from pathlib import Path

import modal

app = modal.App("task2-e2b-sandbox-smoke")
image = modal.Image.debian_slim(python_version="3.12").pip_install("e2b==2.35.0")


def _local_e2b_key() -> str:
    key = os.environ.get("E2B_API_KEY")
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not key:
        for line in env_path.read_text().splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "e2b_api":
                key = value.strip().strip("'\"")
                break
    if not key:
        raise RuntimeError(".env must define e2b_api or E2B_API_KEY must be set")
    return key


@app.function(image=image, timeout=300)
def smoke(e2b_api_key: str) -> dict[str, object]:
    from e2b import Sandbox

    sandbox = Sandbox.create(
        api_key=e2b_api_key,
        timeout=120,
        secure=True,
        allow_internet_access=False,
        metadata={"owner": "task2-production-rl2", "purpose": "smoke"},
    )
    try:
        sandbox.files.write("/workspace/probe.txt", "prime-rl -> modal -> e2b")
        result = sandbox.commands.run(
            "python -c \"from pathlib import Path; print(Path('/workspace/probe.txt').read_text())\"",
            timeout=30,
        )
        return {
            "sandbox_id": sandbox.sandbox_id,
            "exit_code": result.exit_code,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    finally:
        sandbox.kill()


@app.function(image=image, timeout=600)
def failure_lab(e2b_api_key: str) -> dict[str, object]:
    """Deliberately inject sandbox/verifier failures in one real E2B microVM."""
    from e2b import Sandbox

    sandbox = Sandbox.create(
        api_key=e2b_api_key,
        timeout=180,
        secure=True,
        allow_internet_access=False,
        metadata={"owner": "task2-production-rl2", "purpose": "failure-lab"},
    )
    try:
        def exit_code(command: str, timeout: int = 20) -> int:
            try:
                return sandbox.commands.run(
                    command, cwd="/workspace", timeout=timeout
                ).exit_code
            except Exception as exc:
                code = getattr(exc, "exit_code", None)
                if code is None and type(exc).__name__ == "TimeoutException":
                    return 124
                if code is None:
                    raise
                return int(code)

        sandbox.files.write("/workspace/solution.py", "def target(): return 1\n")
        public_exit = exit_code(
            "python -c \"from solution import target; assert target() == 1\"",
        )
        impossible_hidden_exit = exit_code(
            "python -c \"from solution import target; assert target() == 2\"",
        )
        hidden_absent_exit = exit_code(
            "test ! -e /workspace/hidden_check.py && test ! -e /hidden_check.py",
        )
        missing_dependency_exit = exit_code(
            "python -c \"import task2_dependency_that_does_not_exist\"",
        )
        network_exit = exit_code(
            "python -c \"import urllib.request; urllib.request.urlopen('https://example.com', timeout=3)\"",
            timeout=10,
        )
        timeout_type = ""
        try:
            sandbox.commands.run("sleep 5", cwd="/workspace", timeout=1)
        except Exception as exc:  # the injected timeout is the expected outcome
            timeout_type = type(exc).__name__
        checks = {
            "impossible_hidden_test": public_exit == 0 and impossible_hidden_exit != 0,
            "hidden_verifier_not_in_guest": hidden_absent_exit == 0,
            "network_denied": network_exit != 0,
            "runtime_timeout": bool(timeout_type),
            "missing_dependency": missing_dependency_exit != 0,
        }
        return {
            "sandbox_id": sandbox.sandbox_id,
            "sandbox": {"secure": True, "allow_internet_access": False},
            "checks": checks,
            "all_passed": all(checks.values()),
            "evidence": {
                "public_exit": public_exit,
                "impossible_hidden_exit": impossible_hidden_exit,
                "hidden_absent_exit": hidden_absent_exit,
                "network_exit": network_exit,
                "timeout_exception": timeout_type,
                "missing_dependency_exit": missing_dependency_exit,
            },
        }
    finally:
        sandbox.kill()


@app.local_entrypoint()
def main(mode: str = "smoke") -> None:
    if mode == "failure-lab":
        result = failure_lab.remote(_local_e2b_key())
        output = Path("artifacts/failure_injections/e2b.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        if not result["all_passed"]:
            raise SystemExit(1)
        return
    print(smoke.remote(_local_e2b_key()))
