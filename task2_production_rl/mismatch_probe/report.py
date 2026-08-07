"""Generate training/inference numerical mismatch probe report."""

import json
import os
from typing import Any, Dict


def generate_mismatch_report(probe_results: Dict[str, Any], weight_fp: Dict[str, Any]) -> str:
    """Save structured JSON and markdown reports for mismatch probe."""
    os.makedirs("artifacts/reports", exist_ok=True)
    json_path = "artifacts/reports/mismatch_report.json"
    md_path = "artifacts/reports/mismatch_report.md"

    combined = {
        "probe_results": probe_results,
        "weight_fingerprint": weight_fp,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    md_content = f"""# Training / Inference Numerical Mismatch Probe Report

## 1. Executive Summary
- **Max Absolute Logprob Disagreement**: `{probe_results["max_abs_diff"]:.6e}`
- **Mean Absolute Logprob Disagreement**: `{probe_results["mean_abs_diff"]:.6e}`
- **Worst Disagreement Token Index**: `{probe_results["worst_disagreement_index"]}` (Token: `{repr(probe_results["worst_disagreement_token"])}`)

## 2. Weight Fingerprint
- **Global Parameter L2 Norm**: `{weight_fp["global_l2"]:.6f}`
- **Parameter Element Count**: `{weight_fp["total_params"]:,}`
- **SHA256 Fingerprint Digest**: `{weight_fp["digest"]}`

## 3. Probe Token Sequence
- **Prompt Length**: `{probe_results["prompt_length"]}` tokens
- **Completion Length**: `{probe_results["completion_length"]}` tokens
- **Prompt Text**: `{repr(probe_results["prompt_text"])}`
- **Completion Text**: `{repr(probe_results["completion_text"])}`
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Mismatch report saved to {json_path} and {md_path}")
    return md_path
