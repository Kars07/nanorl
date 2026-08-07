"""Disk delta weight sync encoding for slime Megatron training ranks."""

import numpy as np


def overwrite_encode(new: np.ndarray, changed_mask: np.ndarray) -> np.ndarray:
    """The 'overwrite' delta: changed-position count (u4), positions (u4 each), then new values."""
    pos = np.flatnonzero(changed_mask).astype("<u4")
    return np.concatenate(
        [np.array([pos.size], "<u4").view(np.uint8), pos.view(np.uint8), new.view(np.uint8)[changed_mask]]
    )
