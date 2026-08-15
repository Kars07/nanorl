# Modal deployment

Two different concepts must not be conflated:

1. A Verifiers Modal runtime would execute an agent program remotely.
2. This project wraps Prime-RL's normal local launcher in one Modal machine. Modal is not a
   Prime trainer backend.

`modal_apps/self_hosted_rollout.py` builds the pinned Prime checkout, initializes pinned
submodules, installs the committed lock with `uv --frozen`, and adds only the exact router
and flash-attn wheels declared upstream. HF/vLLM caches and outputs use named Modal volumes.
The official and adapted jobs request `A10:2`; the self-hosted evaluation requests one A10.

The inference/eval pair communicates only through `127.0.0.1`. The training launcher owns
inference, trainer, and orchestrator child processes on the same node. Logs and completed
checkpoints survive container termination in `task2-prime-rl-outputs`.

Each repo-repair rollout provisions an E2B microVM inside its MCP tool-server process.
Verifiers terminates that process with SIGTERM, so the tool server installs a signal handler
that kills the remote VM before exit. This fixed a measured leak that reached 20 concurrent
sandboxes. `scripts/cleanup_e2b.py` is an owner-scoped recovery tool and defaults to dry-run.

This follows Modal's documented single-node multi-GPU pattern: the function requests
`gpu="A10:2"` and launches Prime's normal subprocess entrypoint. Modal supplies the machine
and persistent volumes; it is not represented as an upstream Prime trainer backend. The
official Prime training docs likewise identify `uv run rl` as the combined local launcher,
`uv run inference` as the augmented vLLM server, and `--dry-run` as the typed config check.

The E2B lifecycle is explicit. Python sandbox timeouts are seconds, each sandbox carries
owner metadata, and normal/signal cleanup calls `kill()`. The final owner-scoped inventory
reported zero live `task2-production-rl2` sandboxes.
