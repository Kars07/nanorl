# Tasksets and configuration scope

`TinyRepoRepairData` contains immutable task facts: prompt, seed files, category, and the
hidden command. `RepairState` contains mutable rollout facts: sandbox ID, submission flag,
score, checker output, command count, and collected artifacts. Mutable state is never kept
on `TaskData`.

`TinyRepoRepairConfig.num_tasks` is a load-time `TasksetConfig` field. E2B timeouts,
internet policy, and output limits live in `TinyRepoRepairTaskConfig`, because they govern
each task execution. Tests prove that requesting three tasks loads indices 0–2 only and
that no seed file contains the hidden command. The loader is a generator; no E2B sandbox
is provisioned until a selected task is set up by a worker.

The production-sized dataset has 150 deterministic repository instances across 24 audited
repair archetypes. Indices 0â€“99 are train, 100â€“124 validation, and 125â€“149 test. Tests
assert exact split sizes and pairwise disjoint IDs. The small executed two-step job still
uses the first 16 training instances as a systems smoke; dataset scale is not confused with
optimizer-step count or evidence of learned quality.

Five additional installed packages exercise terminal final state, deterministic browser
navigation, structured issue lookup, ordered long-horizon tool use, and proposer-solver
control flow. The proposer-solver package uses a real `vf.Env`: a `proposer` trace creates
the episode context, then `n=2` concurrent runs of the same `solver` role attempt that
proposal. This role naming follows Prime's hierarchical-GRPO contract, where solvers are
compared only against attempts on the same proposed problem.

Actual self-hosted evaluation artifacts live under `artifacts/evals/environment_suite/`.
The summary distinguishes registered tool schemas (`tools`) from calls emitted by the
policy (`called_tools`) so a loaded Toolset cannot be mistaken for tool use.

On the self-hosted Qwen3-1.7B run, terminal, structured lookup, and ordered workflow tasks
each received objective reward 1.0. The proposer-solver episode produced three traces—one
proposer and two concurrent `solver` attempts—and all three received 1.0; final metrics
recorded `episode_agents=3` and `solver_passes=2`. The first browser attempt is retained as
a useful failure: it navigated and submitted the semantically correct phrase, but exact
string scoring rejected it. The verifier now normalizes standalone numeric words/digits
and awards correctness only after a page was opened; the browser-only regression run is
stored separately. That actual regression trace made five model calls, searched, received
an explicit evidence-required rejection, opened `planets`, read the page, submitted again,
and earned reward 1.0 with `pages_visited=1`.
