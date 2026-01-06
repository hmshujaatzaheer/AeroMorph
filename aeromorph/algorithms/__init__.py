"""
Algorithm implementations for AeroMorph framework.

This module contains the novel algorithms introduced in the AeroMorph
research proposal:
    - P2MA: Perception-to-Morphology-Action algorithm
    - SpatialFeasibilityChecker: Collision-free morphing verification
    - UnifiedMorphOptimizer: Joint trajectory-morphing optimization
"""

from aeromorph.algorithms.p2ma import (
    P2MAAlgorithm,
    P2MAConfig,
    MorphDecisionEngine,
    PerceptionFusion,
    MorphingUtilityFunction
)

from aeromorph.algorithms.spatial_feasibility import (
    SpatialFeasibilityChecker,
    CollisionGeometry,
    BoundingVolume
)

from aeromorph.algorithms.unified_optimizer import (
    UnifiedMorphOptimizer,
    OptimizationConfig,
    AerodynamicModel
)

__all__ = [
    # P2MA
    "P2MAAlgorithm",
    "P2MAConfig",
    "MorphDecisionEngine",
    "PerceptionFusion",
    "MorphingUtilityFunction",
    # Spatial Feasibility
    "SpatialFeasibilityChecker",
    "CollisionGeometry",
    "BoundingVolume",
    # Unified Optimizer
    "UnifiedMorphOptimizer",
    "OptimizationConfig",
    "AerodynamicModel",
]
