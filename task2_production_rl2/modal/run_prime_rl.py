"""Canonical layout entrypoint; implementation is kept out of `modal/` to avoid import shadowing."""

from modal_apps.self_hosted_rollout import app, main

__all__ = ["app", "main"]
