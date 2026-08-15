# Weight synchronization

On a single multi-GPU node Prime defaults to NCCL weight broadcast. The trainer broadcasts
after an optimizer step; the orchestrator coordinates a policy version update and the
inference worker loads the new tensors through Prime's vLLM extension. Rollout transport is
different: it moves training records from orchestrator to trainer, not parameters.

The official log is direct evidence of successful synchronization: each completed step is
followed by an in-flight update, later rollouts carry newer policy versions, and training
reaches a stable step-20 weight export. Filesystem broadcast is the fallback when NCCL/NIXL
is not selected or supported; multi-node configurations must set hosts/world sizes and are
outside this single-node Modal wrapper.

The resumed step was also checked at tensor level. `inspect_checkpoint_fingerprints`
opened the actual HF safetensors exports for steps 20 and 21 and hashed the same tensors:

| Tensor | Step 20 SHA-256 prefix | Step 21 SHA-256 prefix | Changed |
| --- | --- | --- | --- |
| `model.embed_tokens.weight` | `87ac04d569af` | `537ad0a9eefc` | yes |
| `model.layers.0.self_attn.q_proj.weight` | `d79de426c76e` | `f1f7d8461fc0` | yes |
| `model.norm.weight` | `0fa78ae0d05f` | `0fa78ae0d05f` | no |

The unchanged final norm is recorded rather than hidden: one optimizer step changed the
embedding and attention projection but not every selected parameter. The full shapes,
means, standard deviations, and hashes are saved in
`artifacts/weights/official_reverse_text_weight_fingerprints.json`.
