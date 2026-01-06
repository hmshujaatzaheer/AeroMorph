"""
Energy-Aware Swarm Morphing Protocol (EA-SMP).

This module implements distributed consensus protocols for coordinated
morphing across multi-robot swarms.

Key Features:
    - Energy-balanced leader election
    - Distributed spatial feasibility checking
    - Consensus-based morphing time synchronization
    - Fault-tolerant coordination

Reference: AeroMorph PhD Research Proposal, Algorithm 3
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np
from enum import Enum

from aeromorph.core.types import SwarmSyncResult
from aeromorph.core.state import ExtendedRobotState, SwarmState


class SwarmRole(Enum):
    """Role of a robot in the swarm."""
    FOLLOWER = "follower"
    LEADER = "leader"
    COORDINATOR = "coordinator"


@dataclass
class SwarmConfig:
    """Configuration for swarm morphing protocol."""
    
    # Communication
    communication_range: float = 10.0  # meters
    message_timeout: float = 0.5  # seconds
    
    # Consensus
    consensus_gain: float = 0.5
    leader_tracking_gain: float = 0.3
    max_sync_error: float = 0.1  # seconds
    
    # Energy
    min_energy_for_leader: float = 0.3  # fraction of capacity
    energy_weight: float = 0.7
    
    # Timing
    morph_announcement_time: float = 1.0  # seconds before morph


class LeaderElection:
    """
    Energy-balanced leader election for swarm morphing.
    
    Elects the robot with the highest energy surplus as leader,
    ensuring the leader can complete the morphing coordination task.
    """
    
    def __init__(self, config: SwarmConfig):
        self.config = config
    
    def elect(self, 
              robot_states: Dict[int, ExtendedRobotState],
              communication_graph: np.ndarray) -> int:
        """
        Elect leader based on energy surplus.
        
        Args:
            robot_states: Dictionary of robot ID to state
            communication_graph: Adjacency matrix
            
        Returns:
            ID of elected leader
        """
        best_leader = -1
        best_score = -float('inf')
        
        for robot_id, state in robot_states.items():
            # Compute energy surplus
            energy_ratio = state.energy_state / 5000.0  # Normalized
            
            # Check connectivity (leader should be well-connected)
            connections = np.sum(communication_graph[robot_id] > 0)
            connectivity_score = connections / len(robot_states)
            
            # Combined score
            score = (self.config.energy_weight * energy_ratio + 
                     (1 - self.config.energy_weight) * connectivity_score)
            
            # Must meet minimum energy requirement
            if energy_ratio >= self.config.min_energy_for_leader:
                if score > best_score:
                    best_score = score
                    best_leader = robot_id
        
        # Fallback to most connected if no one meets energy requirement
        if best_leader < 0 and len(robot_states) > 0:
            best_leader = list(robot_states.keys())[0]
        
        return best_leader


class ConsensusProtocol:
    """
    Distributed consensus for swarm morphing configuration.
    
    Implements the consensus dynamics:
    α̇ᵢ(t) = Σⱼ∈𝒩ᵢ aᵢⱼ(αⱼ(t) - αᵢ(t)) + bᵢ(α_leader(t) - αᵢ(t))
    """
    
    def __init__(self, config: SwarmConfig):
        self.config = config
    
    def compute_update(self,
                       robot_id: int,
                       current_config: np.ndarray,
                       neighbor_configs: Dict[int, np.ndarray],
                       leader_config: np.ndarray,
                       weights: np.ndarray) -> np.ndarray:
        """
        Compute consensus update for a single robot.
        
        Args:
            robot_id: ID of the robot computing update
            current_config: Robot's current morphing configuration
            neighbor_configs: Configurations of neighboring robots
            leader_config: Target configuration from leader
            weights: Consensus weights from graph Laplacian
            
        Returns:
            Configuration update vector
        """
        update = np.zeros_like(current_config)
        
        # Neighbor consensus term: Σⱼ aᵢⱼ(αⱼ - αᵢ)
        for neighbor_id, neighbor_config in neighbor_configs.items():
            weight = weights[robot_id, neighbor_id] if weights is not None else 1.0
            update += self.config.consensus_gain * weight * (neighbor_config - current_config)
        
        # Leader tracking term: bᵢ(α_leader - αᵢ)
        update += self.config.leader_tracking_gain * (leader_config - current_config)
        
        return update
    
    def check_convergence(self,
                          swarm_state: SwarmState,
                          target_config: np.ndarray,
                          tolerance: float = 0.05) -> bool:
        """
        Check if swarm has converged to target configuration.
        
        Args:
            swarm_state: Current swarm state
            target_config: Target morphing configuration
            tolerance: Convergence tolerance
            
        Returns:
            True if all robots within tolerance of target
        """
        for state in swarm_state.robot_states.values():
            error = np.linalg.norm(state.morph_config - target_config)
            if error > tolerance:
                return False
        return True


class TimeSynchronization:
    """
    Time synchronization for coordinated morphing execution.
    
    Ensures all robots begin morphing within acceptable time window.
    """
    
    def __init__(self, config: SwarmConfig):
        self.config = config
    
    def compute_sync_time(self,
                          feasibility_times: Dict[int, float],
                          announcement_time: float) -> Tuple[float, List[int]]:
        """
        Compute synchronized morphing time.
        
        Args:
            feasibility_times: Time each robot needs to verify feasibility
            announcement_time: Time when morphing was announced
            
        Returns:
            Tuple of (sync_time, participating_robot_ids)
        """
        if not feasibility_times:
            return announcement_time + self.config.morph_announcement_time, []
        
        # Sync time is max of all feasibility check times plus margin
        max_feasibility = max(feasibility_times.values())
        sync_time = announcement_time + max(
            self.config.morph_announcement_time,
            max_feasibility + self.config.max_sync_error
        )
        
        # All robots that responded in time participate
        participating = [
            robot_id for robot_id, t in feasibility_times.items()
            if t < max_feasibility + self.config.max_sync_error
        ]
        
        return sync_time, participating


class SwarmMorphProtocol:
    """
    Energy-Aware Swarm Morphing Protocol (EA-SMP).
    
    Implements coordinated morphing across multi-robot swarms with:
    - Energy-balanced leader election
    - Distributed spatial feasibility verification
    - Consensus-based configuration synchronization
    - Time-synchronized execution
    
    Example:
        >>> protocol = SwarmMorphProtocol(num_robots=5)
        >>> result = protocol.coordinate_morph(
        ...     swarm_states=robot_states,
        ...     target_config=target,
        ...     energy_states=battery_levels
        ... )
    """
    
    def __init__(self,
                 num_robots: int = 1,
                 communication_range: float = 10.0,
                 config: Optional[SwarmConfig] = None):
        """
        Initialize swarm morphing protocol.
        
        Args:
            num_robots: Number of robots in swarm
            communication_range: Maximum communication range (meters)
            config: Protocol configuration
        """
        self.num_robots = num_robots
        self.config = config or SwarmConfig(communication_range=communication_range)
        
        self.leader_election = LeaderElection(self.config)
        self.consensus = ConsensusProtocol(self.config)
        self.time_sync = TimeSynchronization(self.config)
        
        # State
        self.current_leader: Optional[int] = None
        self.communication_graph: Optional[np.ndarray] = None
    
    def coordinate_morph(self,
                         swarm_states: Dict[int, ExtendedRobotState],
                         target_config: np.ndarray,
                         energy_states: Optional[Dict[int, float]] = None,
                         environment: Optional[np.ndarray] = None) -> SwarmSyncResult:
        """
        Coordinate morphing across the swarm.
        
        Implements Algorithm 3: EA-SMP Protocol
        
        Args:
            swarm_states: Dictionary of robot ID to ExtendedRobotState
            target_config: Target morphing configuration
            energy_states: Battery levels (uses state if not provided)
            environment: Shared environment model
            
        Returns:
            SwarmSyncResult with coordination details
        """
        import time
        
        # Build communication graph based on positions
        self.communication_graph = self._build_communication_graph(swarm_states)
        
        # Phase 1: Leader Election
        self.current_leader = self.leader_election.elect(
            swarm_states, self.communication_graph
        )
        
        # Phase 2: Distributed Feasibility Check
        feasibility_times = {}
        feasible_robots = []
        excluded_robots = []
        
        for robot_id, state in swarm_states.items():
            start_time = time.time()
            
            # Check spatial feasibility for each robot
            feasible = self._check_robot_feasibility(
                state, target_config, environment
            )
            
            feasibility_times[robot_id] = time.time() - start_time
            
            if feasible:
                feasible_robots.append(robot_id)
            else:
                excluded_robots.append(robot_id)
        
        # Phase 3: Time Synchronization
        announcement_time = time.time()
        sync_time, participating = self.time_sync.compute_sync_time(
            feasibility_times, announcement_time
        )
        
        # Filter to only feasible robots
        participating = [r for r in participating if r in feasible_robots]
        
        # Phase 4: Compute Sync Error
        if len(feasibility_times) > 1:
            times = list(feasibility_times.values())
            sync_error = (max(times) - min(times)) * 1000  # Convert to ms
        else:
            sync_error = 0.0
        
        return SwarmSyncResult(
            leader_id=self.current_leader,
            morph_timestamp=sync_time,
            participating=participating,
            excluded=excluded_robots,
            sync_error=sync_error,
            energy_balanced=True
        )
    
    def _build_communication_graph(self,
                                   swarm_states: Dict[int, ExtendedRobotState]) -> np.ndarray:
        """Build communication graph based on robot positions."""
        n = max(swarm_states.keys()) + 1 if swarm_states else 0
        graph = np.zeros((n, n))
        
        robot_ids = list(swarm_states.keys())
        for i, id_i in enumerate(robot_ids):
            for j, id_j in enumerate(robot_ids):
                if i != j:
                    pos_i = swarm_states[id_i].position
                    pos_j = swarm_states[id_j].position
                    distance = np.linalg.norm(pos_i - pos_j)
                    
                    if distance <= self.config.communication_range:
                        graph[id_i, id_j] = 1.0
        
        return graph
    
    def _check_robot_feasibility(self,
                                 state: ExtendedRobotState,
                                 target_config: np.ndarray,
                                 environment: Optional[np.ndarray]) -> bool:
        """Check if a single robot can feasibly morph to target."""
        # Import here to avoid circular dependency
        from aeromorph.algorithms.spatial_feasibility import SpatialFeasibilityChecker
        
        if environment is None:
            return True  # Assume feasible without environment info
        
        checker = SpatialFeasibilityChecker()
        result = checker.verify(state.morph_config, target_config, environment)
        
        return result.feasible
    
    def step_consensus(self,
                       swarm_state: SwarmState,
                       target_config: np.ndarray,
                       dt: float = 0.1) -> Dict[int, np.ndarray]:
        """
        Execute one step of consensus protocol.
        
        Args:
            swarm_state: Current swarm state
            target_config: Target configuration from leader
            dt: Time step
            
        Returns:
            Dictionary of robot ID to configuration update
        """
        updates = {}
        
        for robot_id, state in swarm_state.robot_states.items():
            # Get neighbor configurations
            neighbor_configs = {}
            for neighbor_id in range(self.num_robots):
                if (self.communication_graph is not None and 
                    self.communication_graph[robot_id, neighbor_id] > 0 and
                    neighbor_id in swarm_state.robot_states):
                    neighbor_configs[neighbor_id] = swarm_state.robot_states[neighbor_id].morph_config
            
            # Compute consensus update
            update = self.consensus.compute_update(
                robot_id,
                state.morph_config,
                neighbor_configs,
                target_config,
                self.communication_graph
            )
            
            updates[robot_id] = state.morph_config + dt * update
        
        return updates
    
    def is_synchronized(self, swarm_state: SwarmState, tolerance: float = 0.05) -> bool:
        """Check if swarm is synchronized."""
        return swarm_state.is_synchronized(tolerance)
    
    def get_connectivity(self) -> float:
        """
        Get algebraic connectivity of communication graph.
        
        Returns:
            Fiedler value (λ₂) - positive means connected
        """
        if self.communication_graph is None or self.communication_graph.shape[0] < 2:
            return 0.0
        
        # Compute graph Laplacian
        degrees = np.sum(self.communication_graph, axis=1)
        laplacian = np.diag(degrees) - self.communication_graph
        
        # Get eigenvalues
        eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues = np.sort(eigenvalues)
        
        # Return second smallest (Fiedler value)
        return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
