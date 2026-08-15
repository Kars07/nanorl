# Scaling concepts

| Technique | Solves | Memory/communication | Minimum useful test |
|---|---|---|---|
| FSDP | trainer parameter/optimizer sharding | less memory; collectives | 2 trainer GPUs |
| expert parallel | MoE expert placement | less expert memory; all-to-all | multi-GPU MoE |
| context parallel | long sequence activation split | less activation memory; ring comms | 2 GPUs, long context |
| activation checkpointing | activation memory | recomputation cost | 1 trainer GPU |
| CPU/optimizer offload | GPU memory pressure | PCIe/host RAM cost | constrained GPU |
| LM-head chunking | large vocab logits | extra kernels/launches | large-vocab model |
| inference replicas/DP | rollout throughput | duplicate weights | ≥2 inference GPUs |
| TP | model too large for one GPU | per-layer collectives | ≥2 GPUs |
| P/D disaggregation | independent prefill/decode scaling | KV transfer complexity | multi-node/large load |

This project's measured small-model bottleneck is E2B/tool latency, not inference capacity;
adding inference replicas would not fix sandbox provisioning time.
