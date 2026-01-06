"""
Morphological Decision Space (𝒟_M) Implementation.

This module implements the formal Morphological Decision Space that unifies
perception, spatial feasibility, energy constraints, and coordination into
a single decision framework.

Mathematical Formulation:
    𝒟_M = 𝒫 × 𝒮 × ℰ × 𝒞

Where:
    𝒫 - Perception Space (sensor fusion, environmental features)
    𝒮 - Spatial Space (collision-free morphing feasibility)
    ℰ - Energy Space (energy constraints and budgets)
    𝒞 - Coordination Space (swarm synchronization states)

Reference: AeroMorph PhD Research Proposal, Section 3.1
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class PerceptionSpace:
    """
    Perception Space 𝒫 - Environmental understanding from sensors.
    
    Represents the robot's understanding of its environment derived
    from multi-modal sensor fusion (LiDAR, Camera, IMU, etc.).
    
    Attributes:
        occupancy_grid: 3D voxel representation of obstacles
        wind_field: Estimated wind field in robot frame
        passage_widths: Width of passages along planned trajectory
        terrain_classification: Semantic terrain labels
        visibility: Sensor visibility/confidence scores
        timestamp: Time of last perception update
    """
    occupancy_grid: Optional[np.ndarray] = None  # (X, Y, Z) voxel grid
    wind_field: Optional[np.ndarray] = None       # (3,) wind vector or field
    passage_widths: Optional[np.ndarray] = None   # Passage widths ahead
    terrain_classification: Dict[str, float] = field(default_factory=dict)
    visibility: float = 1.0
    timestamp: float = 0.0
    
    def get_min_clearance(self, trajectory: np.ndarray) -> float:
        """Compute minimum clearance along a trajectory."""
        if self.passage_widths is None:
            return float('inf')
        return float(np.min(self.passage_widths))
    
    def get_turbulence_index(self) -> float:
        """Compute turbulence index from wind field."""
        if self.wind_field is None:
            return 0.0
        # Simple turbulence estimation based on wind variance
        return float(np.std(self.wind_field)) / 10.0  # Normalized


@dataclass
class SpatialSpace:
    """
    Spatial Space 𝒮 - Collision-free morphing configuration space.
    
    Represents the spatial constraints on morphing, including
    obstacle collisions and self-collisions during transformation.
    
    Attributes:
        robot_geometry: Current robot geometry representation
        collision_model: Collision checking model (bounding volumes)
        self_collision_pairs: Link pairs to check for self-collision
        safety_margin: Safety margin for collision checking (meters)
        morphing_workspace: Available workspace for morphing
    """
    robot_geometry: Optional[Any] = None  # Geometry representation
    collision_model: Optional[Any] = None  # Bounding volume hierarchy
    self_collision_pairs: List[tuple] = field(default_factory=list)
    safety_margin: float = 0.1
    morphing_workspace: Optional[np.ndarray] = None  # Workspace bounds
    
    def compute_bounding_ellipsoid(self, config: np.ndarray) -> np.ndarray:
        """
        Compute Löwner-John bounding ellipsoid for configuration.
        
        Args:
            config: Morphing configuration
            
        Returns:
            Ellipsoid parameters (center, axes, orientation)
        """
        # Placeholder: Returns identity ellipsoid
        # Real implementation would compute tight-fitting ellipsoid
        center = np.zeros(3)
        axes = np.ones(3)
        orientation = np.eye(3)
        return np.concatenate([center, axes, orientation.flatten()])


@dataclass
class EnergySpace:
    """
    Energy Space ℰ - Energy constraints and optimization.
    
    Represents the energy budget and constraints for morphing
    operations within a mission context.
    
    Attributes:
        battery_capacity: Total battery capacity (Joules)
        current_energy: Current battery level (Joules)
        morphing_budget: Energy allocated for morphing
        flight_power_model: Function mapping state to power consumption
        morph_energy_model: Function mapping config change to energy
        reserve_energy: Energy reserved for return-to-base
    """
    battery_capacity: float = 5000.0
    current_energy: float = 5000.0
    morphing_budget: float = 750.0  # 15% of capacity
    flight_power_model: Optional[Callable] = None
    morph_energy_model: Optional[Callable] = None
    reserve_energy: float = 500.0
    
    @property
    def available_morphing_energy(self) -> float:
        """Energy available for morphing operations."""
        return min(self.morphing_budget, 
                   self.current_energy - self.reserve_energy)
    
    def estimate_morph_cost(self, delta_config: np.ndarray) -> float:
        """
        Estimate energy cost of a morphing action.
        
        Args:
            delta_config: Configuration change vector
            
        Returns:
            Estimated energy cost in Joules
        """
        if self.morph_energy_model is not None:
            return self.morph_energy_model(delta_config)
        
        # Default model: linear cost based on config change magnitude
        # k_span * |Δspan| + k_camber * |Δcamber| + k_stiffness * |ΔK|
        k_span = 300.0      # J per unit span change
        k_camber = 500.0    # J per unit camber change
        k_base = 50.0       # Base energy cost
        
        return k_base + k_span * np.sum(np.abs(delta_config[:3])) + \
               k_camber * np.sum(np.abs(delta_config[3:]))


@dataclass
class CoordinationSpace:
    """
    Coordination Space 𝒞 - Swarm synchronization state.
    
    Represents the state of multi-robot coordination for
    synchronized morphing operations.
    
    Attributes:
        num_robots: Number of robots in swarm
        neighbor_states: States of neighboring robots
        communication_graph: Adjacency matrix of communication
        consensus_weights: Weights for consensus protocol
        leader_id: Current leader robot ID (if elected)
        sync_tolerance: Acceptable synchronization error (seconds)
    """
    num_robots: int = 1
    neighbor_states: Dict[int, np.ndarray] = field(default_factory=dict)
    communication_graph: Optional[np.ndarray] = None
    consensus_weights: Optional[np.ndarray] = None
    leader_id: Optional[int] = None
    sync_tolerance: float = 0.1  # 100ms
    
    def is_connected(self) -> bool:
        """Check if communication graph is connected."""
        if self.communication_graph is None or self.num_robots <= 1:
            return True
        
        # Check algebraic connectivity (Fiedler value > 0)
        laplacian = np.diag(np.sum(self.communication_graph, axis=1)) - \
                    self.communication_graph
        eigenvalues = np.linalg.eigvalsh(laplacian)
        return eigenvalues[1] > 1e-6  # Second smallest eigenvalue > 0
    
    def get_consensus_update(self, robot_id: int, 
                             current_config: np.ndarray,
                             target_config: np.ndarray) -> np.ndarray:
        """
        Compute consensus update for distributed morphing.
        
        α̇ᵢ(t) = Σⱼ∈𝒩ᵢ aᵢⱼ(αⱼ(t) - αᵢ(t)) + bᵢ(α_leader(t) - αᵢ(t))
        
        Args:
            robot_id: ID of the robot computing update
            current_config: Robot's current morphing configuration
            target_config: Target configuration from leader
            
        Returns:
            Configuration update vector
        """
        if self.consensus_weights is None:
            return target_config - current_config
        
        update = np.zeros_like(current_config)
        
        # Neighbor consensus term
        for neighbor_id, neighbor_config in self.neighbor_states.items():
            if neighbor_id != robot_id:
                weight = self.consensus_weights[robot_id, neighbor_id]
                update += weight * (neighbor_config - current_config)
        
        # Leader tracking term
        leader_weight = 0.5  # b_i
        update += leader_weight * (target_config - current_config)
        
        return update


class MorphologicalDecisionSpace:
    """
    Morphological Decision Space 𝒟_M = 𝒫 × 𝒮 × ℰ × 𝒞
    
    Unified decision space integrating all aspects of morphing decisions:
    perception, spatial feasibility, energy constraints, and coordination.
    
    This is the core abstraction of the AeroMorph framework, providing
    a formal foundation for autonomous morphing decision-making.
    
    Example:
        >>> decision_space = MorphologicalDecisionSpace(config_dim=6)
        >>> decision_space.update_perception(perception_state)
        >>> decision_space.update_energy(battery_level)
        >>> utility = decision_space.compute_utility(target_config)
    """
    
    def __init__(self, 
                 config_dim: int = 6,
                 num_robots: int = 1,
                 safety_margin: float = 0.1,
                 morphing_budget_fraction: float = 0.15):
        """
        Initialize the Morphological Decision Space.
        
        Args:
            config_dim: Dimension of morphing configuration space
            num_robots: Number of robots in swarm (1 for single robot)
            safety_margin: Safety margin for collision checking (meters)
            morphing_budget_fraction: Fraction of energy for morphing
        """
        self.config_dim = config_dim
        
        # Initialize subspaces
        self.perception = PerceptionSpace()
        self.spatial = SpatialSpace(safety_margin=safety_margin)
        self.energy = EnergySpace(
            morphing_budget=5000.0 * morphing_budget_fraction
        )
        self.coordination = CoordinationSpace(num_robots=num_robots)
        
        # Utility function weights
        self.weights = {
            'clearance': 1.0,
            'aerodynamic': 0.8,
            'energy': 0.5,
            'synchronization': 0.3
        }
        
        # Decision thresholds
        self.utility_threshold = 0.3
        self.energy_threshold = 0.1  # Min 10% energy for morphing
    
    def update_perception(self, 
                          occupancy_grid: Optional[np.ndarray] = None,
                          wind_field: Optional[np.ndarray] = None,
                          passage_widths: Optional[np.ndarray] = None,
                          timestamp: float = 0.0) -> None:
        """
        Update the perception space with new sensor data.
        
        Args:
            occupancy_grid: 3D voxel occupancy grid
            wind_field: Estimated wind field
            passage_widths: Passage widths along trajectory
            timestamp: Time of perception update
        """
        self.perception.occupancy_grid = occupancy_grid
        self.perception.wind_field = wind_field
        self.perception.passage_widths = passage_widths
        self.perception.timestamp = timestamp
    
    def update_energy(self, current_energy: float) -> None:
        """
        Update the energy space with current battery level.
        
        Args:
            current_energy: Current battery energy (Joules)
        """
        self.energy.current_energy = current_energy
    
    def update_coordination(self, 
                            neighbor_states: Dict[int, np.ndarray],
                            communication_graph: Optional[np.ndarray] = None) -> None:
        """
        Update the coordination space with swarm state.
        
        Args:
            neighbor_states: Dictionary of neighbor robot configurations
            communication_graph: Adjacency matrix of communication links
        """
        self.coordination.neighbor_states = neighbor_states
        if communication_graph is not None:
            self.coordination.communication_graph = communication_graph
    
    def compute_utility(self, 
                        current_config: np.ndarray,
                        target_config: np.ndarray,
                        trajectory: Optional[np.ndarray] = None) -> float:
        """
        Compute the utility of morphing from current to target configuration.
        
        U_morph(α, p_t) = w₁·f_clearance(α, d) + w₂·f_aero(α, w) - w₃·E_morph(α)
        
        Args:
            current_config: Current morphing configuration
            target_config: Target morphing configuration
            trajectory: Planned trajectory for context
            
        Returns:
            Utility value (higher is better)
        """
        delta_config = target_config - current_config
        
        # Clearance benefit (positive if target improves passage clearance)
        clearance_benefit = self._compute_clearance_benefit(
            current_config, target_config, trajectory
        )
        
        # Aerodynamic benefit (positive if target improves aero performance)
        aero_benefit = self._compute_aerodynamic_benefit(
            current_config, target_config
        )
        
        # Energy cost (always positive, subtracted from utility)
        energy_cost = self.energy.estimate_morph_cost(delta_config)
        normalized_cost = energy_cost / self.energy.battery_capacity
        
        # Synchronization cost (for swarm operations)
        sync_cost = self._compute_sync_cost(target_config)
        
        # Weighted utility
        utility = (
            self.weights['clearance'] * clearance_benefit +
            self.weights['aerodynamic'] * aero_benefit -
            self.weights['energy'] * normalized_cost -
            self.weights['synchronization'] * sync_cost
        )
        
        return utility
    
    def _compute_clearance_benefit(self,
                                   current_config: np.ndarray,
                                   target_config: np.ndarray,
                                   trajectory: Optional[np.ndarray]) -> float:
        """Compute benefit of morphing for passage clearance."""
        if self.perception.passage_widths is None:
            return 0.0
        
        min_clearance = float(np.min(self.perception.passage_widths))
        
        # Estimate robot width for current and target configs
        # Simplified: assume first config dimension is wingspan
        current_width = 0.5 + 0.3 * current_config[0]  # Base + variable
        target_width = 0.5 + 0.3 * target_config[0]
        
        # Benefit is positive if target is narrower and fits better
        current_margin = min_clearance - current_width
        target_margin = min_clearance - target_width
        
        if current_margin > 0 and target_margin > 0:
            return 0.0  # Both fit, no benefit
        elif target_margin > current_margin:
            return (target_margin - current_margin) / min_clearance
        else:
            return 0.0
    
    def _compute_aerodynamic_benefit(self,
                                     current_config: np.ndarray,
                                     target_config: np.ndarray) -> float:
        """Compute aerodynamic benefit of morphing."""
        if self.perception.wind_field is None:
            return 0.0
        
        turbulence = self.perception.get_turbulence_index()
        
        # In high turbulence, prefer stiffer configurations
        # Simplified: assume last config dimension is stiffness
        if turbulence > 0.5:
            stiffness_improvement = target_config[-1] - current_config[-1]
            return stiffness_improvement * turbulence
        else:
            # In low turbulence, prefer efficient (wider) configurations
            efficiency_improvement = target_config[0] - current_config[0]
            return efficiency_improvement * (1 - turbulence)
    
    def _compute_sync_cost(self, target_config: np.ndarray) -> float:
        """Compute synchronization cost for swarm morphing."""
        if self.coordination.num_robots <= 1:
            return 0.0
        
        # Cost based on deviation from neighbor configurations
        total_deviation = 0.0
        for neighbor_config in self.coordination.neighbor_states.values():
            deviation = np.linalg.norm(target_config - neighbor_config)
            total_deviation += deviation
        
        if len(self.coordination.neighbor_states) > 0:
            return total_deviation / len(self.coordination.neighbor_states)
        return 0.0
    
    def should_morph(self, 
                     current_config: np.ndarray,
                     target_config: np.ndarray,
                     trajectory: Optional[np.ndarray] = None) -> bool:
        """
        Determine if morphing should be triggered.
        
        Args:
            current_config: Current morphing configuration
            target_config: Proposed target configuration
            trajectory: Planned trajectory
            
        Returns:
            True if morphing is recommended
        """
        # Check energy constraint
        delta_config = target_config - current_config
        morph_cost = self.energy.estimate_morph_cost(delta_config)
        if morph_cost > self.energy.available_morphing_energy:
            return False
        
        # Check utility threshold
        utility = self.compute_utility(current_config, target_config, trajectory)
        if utility < self.utility_threshold:
            return False
        
        return True
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get the full decision space state as a vector.
        
        Returns:
            Flattened state vector for learning algorithms
        """
        state_components = []
        
        # Perception features
        if self.perception.passage_widths is not None:
            state_components.append(np.min(self.perception.passage_widths))
        else:
            state_components.append(float('inf'))
        
        state_components.append(self.perception.get_turbulence_index())
        state_components.append(self.perception.visibility)
        
        # Energy features
        state_components.append(self.energy.current_energy / self.energy.battery_capacity)
        state_components.append(self.energy.available_morphing_energy / self.energy.morphing_budget)
        
        # Coordination features
        state_components.append(float(self.coordination.is_connected()))
        state_components.append(float(self.coordination.num_robots))
        
        return np.array(state_components)
    
    def __repr__(self) -> str:
        return (f"MorphologicalDecisionSpace(dim={self.config_dim}, "
                f"robots={self.coordination.num_robots}, "
                f"energy={self.energy.current_energy:.0f}J)")
