"""
Core type definitions for AeroMorph framework.

This module defines fundamental data types used throughout the framework
for morphing decisions, feasibility results, and robot states.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict, Any
import numpy as np


class MorphAction(Enum):
    """Actions that can be taken by the morphing decision engine."""
    HOLD = "hold"          # Maintain current configuration
    MORPH = "morph"        # Execute morphing to target
    PARTIAL = "partial"    # Partial morphing (constrained)
    ABORT = "abort"        # Abort planned morphing


@dataclass
class MorphingDecision:
    """
    Result of a morphing decision from the P2MA algorithm.
    
    Attributes:
        action: The decided action (HOLD, MORPH, PARTIAL, ABORT)
        target_config: Target morphing configuration if action is MORPH
        current_config: Current morphing configuration
        energy_cost: Estimated energy cost of morphing (Joules)
        utility: Computed utility value for the decision
        spatial_feasibility: Whether spatial feasibility was verified
        trigger_reason: Reason for the morphing trigger
        timestamp: Decision timestamp
    """
    action: MorphAction
    target_config: Optional[np.ndarray] = None
    current_config: Optional[np.ndarray] = None
    energy_cost: float = 0.0
    utility: float = 0.0
    spatial_feasibility: bool = True
    trigger_reason: str = ""
    timestamp: float = 0.0
    
    def __repr__(self) -> str:
        return (f"MorphingDecision(action={self.action.value}, "
                f"utility={self.utility:.3f}, cost={self.energy_cost:.2f}J)")


@dataclass
class FeasibilityResult:
    """
    Result of spatial feasibility verification.
    
    Attributes:
        feasible: Whether the morphing path is collision-free
        trajectory: Interpolated morphing trajectory if feasible
        collision_point: Location of collision if not feasible
        collision_type: Type of collision (obstacle/self)
        min_clearance: Minimum clearance along the path
        computation_time: Time taken for verification (ms)
    """
    feasible: bool
    trajectory: Optional[List[np.ndarray]] = None
    collision_point: Optional[np.ndarray] = None
    collision_type: Optional[str] = None
    min_clearance: float = float('inf')
    computation_time: float = 0.0
    
    def __repr__(self) -> str:
        status = "✓ Feasible" if self.feasible else "✗ Infeasible"
        return f"FeasibilityResult({status}, clearance={self.min_clearance:.3f}m)"


@dataclass
class SwarmSyncResult:
    """
    Result of swarm morphing coordination.
    
    Attributes:
        leader_id: ID of the elected leader robot
        morph_timestamp: Agreed timestamp for synchronized morphing
        participating: List of robot IDs participating in morph
        excluded: List of robot IDs excluded (feasibility failure)
        sync_error: Estimated synchronization error (ms)
        energy_balanced: Whether leader was elected based on energy
    """
    leader_id: int
    morph_timestamp: float
    participating: List[int] = field(default_factory=list)
    excluded: List[int] = field(default_factory=list)
    sync_error: float = 0.0
    energy_balanced: bool = True
    
    def __repr__(self) -> str:
        return (f"SwarmSyncResult(leader={self.leader_id}, "
                f"robots={len(self.participating)}, sync_err={self.sync_error:.1f}ms)")


@dataclass
class EnergyBudget:
    """
    Energy budget for morphing operations.
    
    Attributes:
        total_capacity: Total battery capacity (Joules)
        current_level: Current battery level (Joules)
        morphing_fraction: Fraction allocated for morphing (0-1)
        consumed_morphing: Energy already used for morphing
        reserved: Energy reserved for return-to-base
    """
    total_capacity: float
    current_level: float
    morphing_fraction: float = 0.15
    consumed_morphing: float = 0.0
    reserved: float = 0.0
    
    @property
    def morphing_budget(self) -> float:
        """Available budget for morphing operations."""
        return self.total_capacity * self.morphing_fraction - self.consumed_morphing
    
    @property
    def remaining_ratio(self) -> float:
        """Ratio of remaining to total energy."""
        return self.current_level / self.total_capacity
    
    def can_afford(self, morph_cost: float) -> bool:
        """Check if a morphing action can be afforded."""
        return (morph_cost <= self.morphing_budget and 
                morph_cost <= self.current_level - self.reserved)


@dataclass
class MorphingConstraints:
    """
    Physical constraints on morphing operations.
    
    Attributes:
        max_rate: Maximum morphing rate (units/second)
        min_config: Minimum configuration bounds
        max_config: Maximum configuration bounds
        forbidden_zones: Configurations to avoid
        actuator_limits: Per-actuator force/torque limits
    """
    max_rate: float = 1.0
    min_config: Optional[np.ndarray] = None
    max_config: Optional[np.ndarray] = None
    forbidden_zones: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    actuator_limits: Optional[np.ndarray] = None
    
    def is_valid(self, config: np.ndarray) -> bool:
        """Check if a configuration is within valid bounds."""
        if self.min_config is not None and np.any(config < self.min_config):
            return False
        if self.max_config is not None and np.any(config > self.max_config):
            return False
        for zone_min, zone_max in self.forbidden_zones:
            if np.all(config >= zone_min) and np.all(config <= zone_max):
                return False
        return True


@dataclass
class PerceptionFeatures:
    """
    Features extracted from perception for morphing decisions.
    
    Attributes:
        min_passage_width: Minimum width of passages ahead (meters)
        turbulence_index: Estimated wind turbulence (0-1)
        obstacle_density: Density of obstacles in FOV
        terrain_type: Classified terrain type
        landing_feasibility: Whether landing is possible
        wind_direction: Estimated wind direction vector
        wind_speed: Estimated wind speed (m/s)
    """
    min_passage_width: float = float('inf')
    turbulence_index: float = 0.0
    obstacle_density: float = 0.0
    terrain_type: str = "unknown"
    landing_feasibility: float = 0.0
    wind_direction: Optional[np.ndarray] = None
    wind_speed: float = 0.0


@dataclass
class MissionPlan:
    """
    Mission plan including morphing schedule.
    
    Attributes:
        trajectory: Planned trajectory waypoints
        morph_schedule: List of (time, config) morphing events
        total_morph_energy: Total energy allocated to morphing
        estimated_duration: Estimated mission duration (seconds)
        feasible: Whether the plan is feasible
        contingencies: Backup plans for different scenarios
    """
    trajectory: List[np.ndarray] = field(default_factory=list)
    morph_schedule: List[Tuple[float, np.ndarray]] = field(default_factory=list)
    total_morph_energy: float = 0.0
    estimated_duration: float = 0.0
    feasible: bool = True
    contingencies: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        status = "✓ Feasible" if self.feasible else "✗ Infeasible"
        return (f"MissionPlan({status}, morphs={len(self.morph_schedule)}, "
                f"energy={self.total_morph_energy:.1f}J)")
