"""Weight fingerprinting utility to verify parameter identity across processes."""

import hashlib
from typing import Any, Dict

import torch
import torch.nn as nn


def compute_weight_fingerprint(model: nn.Module, max_tensors: int = 10) -> Dict[str, Any]:
    """Compute lightweight L2 norm and SHA256 checksum fingerprint across selected parameters."""
    total_l2 = 0.0
    total_params = 0
    tensor_hashes = []

    for idx, (name, param) in enumerate(model.named_parameters()):
        if idx >= max_tensors:
            break
        p_detach = param.detach().float()
        l2_sq = float(torch.sum(p_detach**2).item())
        total_l2 += l2_sq
        total_params += param.numel()

        # Compute SHA256 digest of raw byte representation of first few floats
        sample_bytes = p_detach.flatten()[:100].cpu().numpy().tobytes()
        h = hashlib.sha256(sample_bytes).hexdigest()[:8]
        tensor_hashes.append(f"{name}:{h}")

    global_l2 = total_l2**0.5
    combined_hash_str = "|".join(tensor_hashes)
    digest = hashlib.sha256(combined_hash_str.encode("utf-8")).hexdigest()[:12]

    return {
        "global_l2": global_l2,
        "total_params": total_params,
        "digest": digest,
        "sample_tensor_hashes": tensor_hashes,
    }


def assert_weight_identity(fp1: Dict[str, Any], fp2: Dict[str, Any], tol: float = 1e-5):
    """Assert two weight fingerprints match within numerical tolerance."""
    assert fp1["digest"] == fp2["digest"], f"Weight digest mismatch: {fp1['digest']} vs {fp2['digest']}"
    diff = abs(fp1["global_l2"] - fp2["global_l2"])
    assert diff < tol, f"Weight L2 difference {diff} exceeds tolerance {tol}"
