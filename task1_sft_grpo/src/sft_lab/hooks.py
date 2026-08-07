"""Native PyTorch hooks for activation and gradient inspection."""

import math
from typing import Any, Dict

import torch
import torch.nn as nn


class ActivationTrackerHook:
    """Hook to capture activation statistics on target modules during forward pass."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles = []
        self.stats: Dict[str, Dict[str, float]] = {}

        self._attach_hooks()

    def _attach_hooks(self):
        # Identify key target modules in Qwen2 architecture
        target_names = []
        for name, module in self.model.named_modules():
            if "embed_tokens" in name:
                target_names.append((name, module))
            elif "layers.0.self_attn" in name or "layers.0.mlp" in name:
                target_names.append((name, module))
            elif "norm" in name and "layers" not in name:  # final norm
                target_names.append((name, module))
            elif name == "lm_head":
                target_names.append((name, module))

        # If specific names not found, take first embedding, first layer attn/mlp, last norm
        if not target_names:
            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:
                    target_names.append((name, module))
                    if len(target_names) >= 5:
                        break

        for name, module in target_names:
            handle = module.register_forward_hook(self._make_hook(name))
            self.handles.append(handle)

    def _make_hook(self, module_name: str):
        def hook(module, input_val, output_val):
            if isinstance(output_val, tuple):
                act = output_val[0]
            else:
                act = output_val

            if isinstance(act, torch.Tensor):
                act_float = act.detach().float()
                rms = float(torch.sqrt(torch.mean(act_float**2)).item())
                max_abs = float(torch.max(torch.abs(act_float)).item())
                nan_count = int(torch.isnan(act_float).sum().item())
                inf_count = int(torch.isinf(act_float).sum().item())

                self.stats[module_name] = {
                    "rms": rms,
                    "max_abs": max_abs,
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                    "shape": list(act.shape),
                }

        return hook

    def clear(self):
        self.stats.clear()

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def compute_gradient_stats(model: nn.Module) -> Dict[str, Any]:
    """Compute global and per-layer gradient norms and max values."""
    total_sq_norm = 0.0
    layer_grads = {}

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            g = param.grad.detach().float()
            g_norm = float(torch.norm(g, 2).item())
            g_max = float(torch.max(torch.abs(g)).item())
            total_sq_norm += g_norm**2
            layer_grads[name] = {"norm": g_norm, "max_abs": g_max}

    global_norm = float(math.sqrt(total_sq_norm))
    return {
        "global_grad_norm": global_norm,
        "per_parameter_grads": layer_grads,
    }
