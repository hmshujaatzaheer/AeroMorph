"""
Core components of the AeroMorph framework.

This module contains fundamental data structures, state representations,
and the Morphological Decision Space formalization.

Components:
    - MorphologicalDecisionSpace: Unified decision space 𝒟_M
    - MorphingConfigSpace: Configuration manifold ℳ
    - ExtendedRobotState: Robot state including morphology
    - PerceptionState: Multi-modal sensor fusion state
    - Type definitions: MorphingDecision, FeasibilityResult, etc.
"""

from aeromorph.core.types import (
    MorphAction,
    MorphingDecision,
    FeasibilityResult,
    SwarmSyncResult,
    EnergyBudget,
    MorphingConstraints,
    PerceptionFeatures,
    MissionPlan
)

from aeromorph.core.decision_space import (
    MorphologicalDecisionSpace,
    PerceptionSpace,
    SpatialSpace,
    EnergySpace,
    CoordinationSpace
)

from aeromorph.core.config_space import (
    MorphingConfigSpace,
    LatticeConfigSpace,
    MorphingBounds
)

from aeromorph.core.state import (
    Pose,
    Velocity,
    PerceptionState,
    ExtendedRobotState,
    SwarmState
)

__all__ = [
    # Types
    "MorphAction",
    "MorphingDecision",
    "FeasibilityResult",
    "SwarmSyncResult",
    "EnergyBudget",
    "MorphingConstraints",
    "PerceptionFeatures",
    "MissionPlan",
    # Decision Space
    "MorphologicalDecisionSpace",
    "PerceptionSpace",
    "SpatialSpace",
    "EnergySpace",
    "CoordinationSpace",
    # Configuration Space
    "MorphingConfigSpace",
    "LatticeConfigSpace",
    "MorphingBounds",
    # State
    "Pose",
    "Velocity",
    "PerceptionState",
    "ExtendedRobotState",
    "SwarmState",
]
