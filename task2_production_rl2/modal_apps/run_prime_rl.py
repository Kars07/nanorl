from __future__ import annotations

import subprocess

import modal

PRIME_RL_REV = "8c1f196dd39699726ee8ff52f6ee2495c5fa38df"
PRIME_RL_DIR = "/opt/prime-rl"

app = modal.App("task2-self-managed-prime-rl")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv==0.11.21")
)
outputs = modal.Volume.from_name("task2-prime-rl-outputs", create_if_missing=True)


def _run(argv: list[str], *, cwd: str | None = None, timeout: int = 1800) -> dict[str, object]:
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    return {
        "command": " ".join(argv),
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.function(image=image, cpu=4, memory=8192, volumes={"/outputs": outputs}, timeout=2400)
def dry_run() -> dict[str, object]:
    clone = _run(
        [
            "git", "clone", "--filter=blob:none", "--no-checkout",
            "https://github.com/PrimeIntellect-ai/prime-rl.git", PRIME_RL_DIR,
        ]
    )
    if clone["exit_code"] != 0:
        return {"clone": clone}
    checkout = _run(["git", "checkout", PRIME_RL_REV], cwd=PRIME_RL_DIR)
    if checkout["exit_code"] != 0:
        return {"clone": clone, "checkout": checkout}
    submodules = _run(
        [
            "git", "-c", "url.https://github.com/.insteadOf=git@github.com:",
            "submodule", "update", "--init",
            "deps/pydantic-config", "deps/renderers", "deps/verifiers",
        ],
        cwd=PRIME_RL_DIR,
        timeout=1200,
    )
    if submodules["exit_code"] != 0:
        return {"clone": clone, "checkout": checkout, "submodules": submodules}
    install = _run(
        [
            "uv", "pip", "install", "--system",
            f"{PRIME_RL_DIR}/packages/prime-rl-configs",
            f"{PRIME_RL_DIR}/deps/verifiers/environments/reverse_text",
            "nvidia-ml-py", "psutil", "setproctitle", "loguru", "jinja2",
        ],
        timeout=1200,
    )
    if install["exit_code"] != 0:
        return {"clone": clone, "checkout": checkout, "submodules": submodules, "install": install}
    launcher = _run(
        ["uv", "pip", "install", "--system", "--no-deps", PRIME_RL_DIR],
        timeout=600,
    )
    if launcher["exit_code"] != 0:
        return {
            "clone": clone, "checkout": checkout, "submodules": submodules,
            "install": install, "launcher": launcher,
        }
    command = [
        "uv", "run", "--no-project",
        "rl", "@", f"{PRIME_RL_DIR}/examples/basic/reverse-text/rl.toml",
        "--dry-run", "--output-dir", "/outputs/official-reverse-text-dry-run",
        "--no-wandb",
    ]
    run = _run(command, cwd=PRIME_RL_DIR)
    outputs.commit()
    configs = _run(
        ["find", "/outputs/official-reverse-text-dry-run", "-maxdepth", "4", "-type", "f", "-print"]
    )
    return {
        "clone": clone, "checkout": checkout, "submodules": submodules,
        "install": install, "launcher": launcher, "run": run, "configs": configs,
    }


@app.local_entrypoint()
def main(mode: str = "dry-run") -> None:
    if mode != "dry-run":
        raise ValueError("Only dry-run is enabled until the official config resolves successfully")
    result = dry_run.remote()
    for phase in ("clone", "checkout", "submodules", "install", "launcher", "run", "configs"):
        if phase not in result:
            continue
        print(f"[{phase}] {result[phase]['command']}")
        print(result[phase]["stdout"])
        print(result[phase]["stderr"])
        if result[phase]["exit_code"] != 0:
            raise SystemExit(int(result[phase]["exit_code"]))
