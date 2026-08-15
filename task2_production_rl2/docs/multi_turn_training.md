# Multi-turn training — actual artifact

Trace `07ce2c5365bd4927bb5f7e756b41ded8` contains four model calls and one branch. Pinned
Prime `trace_to_samples` followed by `GRPOAlgorithm.finalize_group` produced one aligned
sample with 2,178 token IDs/logprobs/mask/advantages.

| field | actual value |
|---|---:|
| context tokens | 623 |
| trainable tokens | 1,555 |
| sampled spans | `[519,1031)`, `[1056,1254)`, `[1295,1807)`, `[1845,2178)` |
| context gaps | `[1031,1056)`, `[1254,1295)`, `[1807,1845)` |
| advantage range | 0.0–0.5 |
| nonzero advantage tokens | 1,555 |

The context gaps contain the observations/tool results needed by later calls; they are not
policy-gradient targets. The four assistant generations are trainable. Exact-extension
rendering allowed Prime to merge the calls into one sample. `probes/inspect_interleaving.py`
recomputes these spans directly from the saved mask and asserts all arrays align.

The full arrays are in `artifacts/samples/repo_repair_training_sample.json`; no synthetic
token IDs or logprobs are used.
