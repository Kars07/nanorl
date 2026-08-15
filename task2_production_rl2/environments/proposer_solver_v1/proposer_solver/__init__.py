"""Prime-RL hierarchical-GRPO compatibility export.

Pinned Prime validates proposer-solver environments by importing this historical
module name. The implementation remains in the versioned learning package.
"""

from proposer_solver_v1.taskset import (
    ProposerSolverEnv,
    ProposerSolverEnvConfig,
    ProposerSolverTaskset,
)

__all__ = ["ProposerSolverEnv", "ProposerSolverEnvConfig", "ProposerSolverTaskset"]
