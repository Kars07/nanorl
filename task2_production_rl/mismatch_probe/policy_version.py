"""Policy versioning utility for tracking model parameter updates."""

import time
from typing import Any, Dict


class PolicyVersionTracker:
    """Tracks policy version IDs, update timestamps, and metadata."""

    def __init__(self, initial_version: int = 0):
        self._version = initial_version
        self._history = [
            {
                "version": initial_version,
                "timestamp": time.time(),
                "action": "init",
            }
        ]

    @property
    def version(self) -> int:
        return self._version

    def increment(self, metadata: Dict[str, Any] | None = None) -> int:
        self._version += 1
        record = {
            "version": self._version,
            "timestamp": time.time(),
            "action": "step",
        }
        if metadata:
            record["metadata"] = metadata
        self._history.append(record)
        return self._version

    def get_history(self) -> list[Dict[str, Any]]:
        return list(self._history)
