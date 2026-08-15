"""Canonical repository-layout exports for Modal-hosted Verifiers evaluation."""

from modal_apps.self_hosted_rollout import app, evaluate_on_self_hosted_prime_inference, evaluate_repo_checkpoint

__all__ = ["app", "evaluate_on_self_hosted_prime_inference", "evaluate_repo_checkpoint"]
