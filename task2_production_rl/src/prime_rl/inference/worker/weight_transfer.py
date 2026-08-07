"""Weight transfer worker reloading state dict into live inference engine model."""

from typing import Dict

import torch
import torch.nn as nn


def load_weights_checkpoint_layerwise(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Reload model state dict layerwise in-place into live inference model."""
    device = next(model.parameters()).device
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state_dict:
                param.copy_(state_dict[name].to(device))
