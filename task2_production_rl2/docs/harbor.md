# Harbor on E2B inside Modal

`scripts/run_harbor_e2b.py` performs a real, deliberately narrow Harbor execution. The
Modal function `evaluate_harbor_e2b` starts pinned Prime-RL inference with
`Qwen/Qwen3-1.7B`, loads the official `harbor/hello-world` package with pinned Verifiers v1,
asks the self-hosted model for a schema-constrained `execute(command)` action, executes that
action in an E2B microVM, and calls upstream `HarborTask.score`. The score path stages the
package's real `tests/` only after the agent command and runs its real `/tests/test.sh`.

The successful artifact is `artifacts/harbor/harbor_e2b_result.json`: trace
`843c921fb2604df7bcfa4ea69ce6a282`, one model call, command
`echo 'Hello, world!' > /app/hello.txt`, no errors, and `solved=1.0`. The Modal summary and
runner log are beside it. `scripts/cleanup_e2b.py` reported zero owner-scoped live sandboxes
afterward.

## Honest image boundary

The official task declares a Dockerfile rather than a pullable image. The strict upstream
loader rejection is saved in the result. This project permits `ignore_dockerfile=True` only
after `validate_minimal_dockerfile` proves the file is exactly:

```dockerfile
FROM ubuntu:24.04
WORKDIR /app
```

The adapter realizes that workdir in E2B and rejects any additional Dockerfile instruction;
the rejection path is unit-tested. This is faithful for this task, not a general Harbor
Dockerfile builder. Richer Harbor images and separate-verifier images remain unsupported until
an E2B template/build pipeline can reproduce them exactly.

## Runtime ownership

`task2_runtime.E2BRuntime` implements the pinned Verifiers `Runtime` file and process
primitives directly with E2B SDK 2.35.0. It is passed as a live object because the pinned
`RuntimeConfig` discriminated union is closed and has no E2B member; framework source is not
patched and the adapter does not identify itself as Docker, Modal, or Prime. Modal hosts the
driver and Prime inference process; E2B hosts the untrusted task command and verifier process.
