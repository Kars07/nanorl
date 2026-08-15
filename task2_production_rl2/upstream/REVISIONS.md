# Pinned upstream revisions

Recorded 2026-08-13. Upstreams are not vendored into this learning workspace.

| repository | commit | commit date | remote | status |
|---|---|---|---|---|
| PrimeIntellect-ai/prime-rl | `8c1f196dd39699726ee8ff52f6ee2495c5fa38df` | 2026-08-12 | `https://github.com/PrimeIntellect-ai/prime-rl.git` | clean external checkout |
| PrimeIntellect-ai/verifiers (Prime-RL submodule) | `7251c60934d2c42af85d42a1da3da62269b7957e` | pinned by Prime-RL | submodule | clean |
| PrimeIntellect-ai/renderers | `2846a3dcd29318c1fc98de3498bab4190997af9e` | pinned by Prime-RL | submodule | clean |
| PrimeIntellect-ai/prime-envs | `b30aad36371a903f5350290fdbcf22525025624f` | pinned by Prime-RL | submodule | clean |
| PrimeIntellect-ai/pydantic-config | `4f5ae373582ceffdbf7e6bd1998c9ad568fcc1ad` | pinned by Prime-RL | submodule | clean |
| local cloned verifiers | `a298bcfe4a3a410b7287254d61a65947906c6a89` | 2026-08-08 | `https://github.com/PrimeIntellect-ai/verifiers.git` | dirty before Task 2; preserved |

Runtime baseline: Python 3.12.11, uv 0.11.21, Modal SDK 1.3.3, E2B SDK 2.35.0,
Windows host with RTX 4050 Laptop GPU (6 GB). Docker is not installed. Prime-RL's exact
PyTorch, vLLM, CUDA, Verifiers, and Renderers dependency resolution comes from the pinned
Prime-RL `uv.lock` inside the Modal image.

