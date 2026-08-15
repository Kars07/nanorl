# Harnesses

The pinned registry includes `null`, `bash`, `codex`, `claude-code`, `browser-use`, `pi`,
`rlm`, `hermes-agent`, `openclaw`, `kimi-code`, `pool`, and others. Capability is explicit:
`Harness.SUPPORTS_MCP`, `SUPPORTS_RESUME`, `EXECUTES_CODE`, and `NEEDS_CONTAINER` are read
before execution.

This environment uses `null` because the model needs only the four E2B MCP tools. It is a
real multi-turn chat/tool harness, supports MCP, and does not hand the policy a host shell.
That makes process ownership unambiguous: model actions reach only our MCP schema; shell
commands execute in E2B.

Heavy coding harnesses remain useful for larger policies, but they bring their own program,
installation, and context behavior. `bash` is appropriate for controlled terminal studies.
`browser-use` is appropriate only with a deterministic site snapshot and an explicit
network policy. A custom harness should be added only when these extension surfaces cannot
represent the desired context/continuation behavior.

## Executed comparison

`evaluate_harness_suite` served Qwen3-0.6B with Prime's `uv run --frozen inference` and
ran the same `learning-harness` task through three actual Verifiers CLI configurations:

| Harness | Runtime | Trace calls | CLI result | Reward |
| --- | --- | ---: | --- | ---: |
| `null` | trusted `subprocess` | 1 | exit 0, trace OK | 0 |
| built-in `bash` | trusted `subprocess` | 1 | exit 0, trace OK | 0 |
| `learning-harness` | trusted `subprocess` | 1 | exit 0, trace OK | 0 |

The saved traces and resolved configs are under `artifacts/evals/harness_suite/`. The
built-in bash run emitted Verifiers' own warning that subprocess executes on the local
system and is not isolation. These were trusted harness-shape probes only; untrusted repo
commands in the capstone went through the E2B MCP toolset.

The custom harness only contributes its system prompt and packaged runtime program. It
does not implement a second agent loop or inference client; Verifiers still owns model
interception and Trace construction.
