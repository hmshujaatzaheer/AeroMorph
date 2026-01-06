"""
AeroMorph: Unified Perception-Driven Morphological Adaptation Framework
========================================================================

A research framework for autonomous morphing decisions in aerial robots,
integrating perception, spatial feasibility, energy constraints, and
swarm coordination.

Main Components:
    - MorphDecisionEngine: Core decision-making engine
    - P2MAAlgorithm: Perception-to-Morphology-Action algorithm
    - SpatialFeasibilityChecker: Collision-free morphing verification
    - SwarmMorphProtocol: Multi-robot coordination protocol
    - EnergyAwarePlanner: Energy-constrained mission planning

Example:
    >>> from aeromorph import MorphDecisionEngine
    >>> engine = MorphDecisionEngine(config_space_dim=6)
    >>> decision = engine.decide(current_config, perception, trajectory)

Author: H M Shujaat Zaheer
Email: shujabis@gmail.com
License: MIT
"""

__version__ = "0.1.0"
__author__ = "H M Shujaat Zaheer"
__email__ = "shujabis@gmail.com"

from aeromorph.core.decision_space import MorphologicalDecisionSpace
from aeromorph.core.config_space import MorphingConfigSpace
from aeromorph.core.state import ExtendedRobotState, PerceptionState
from aeromorph.core.types import MorphingDecision, FeasibilityResult

from aeromorph.algorithms.p2ma import P2MAAlgorithm, MorphDecisionEngine
from aeromorph.algorithms.spatial_feasibility import SpatialFeasibilityChecker
from aeromorph.algorithms.unified_optimizer import UnifiedMorphOptimizer

from aeromorph.swarm.ea_smp import SwarmMorphProtocol
from aeromorph.planning.energy_aware import EnergyAwarePlanner

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Core components
    "MorphologicalDecisionSpace",
    "MorphingConfigSpace",
    "ExtendedRobotState",
    "PerceptionState",
    "MorphingDecision",
    "FeasibilityResult",
    # Algorithms
    "P2MAAlgorithm",
    "MorphDecisionEngine",
    "SpatialFeasibilityChecker",
    "UnifiedMorphOptimizer",
    # Planning & Swarm
    "SwarmMorphProtocol",
    "EnergyAwarePlanner",
]
