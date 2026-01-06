"""
Perception-to-Morphology-Action (P2MA) Algorithm Implementation.

This module implements the core P2MA algorithm that enables autonomous
morphing decisions based on environmental perception.

Algorithm Stages:
    1. Perception Fusion - Multi-modal sensor fusion via EKF
    2. Morphing Need Assessment - Utility computation for morphing
    3. Spatial Feasibility Verification - Collision-free path checking
    4. Energy-Constrained Optimization - Budget-aware morphing

Reference: AeroMorph PhD Research Proposal, Algorithm 1
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable
import numpy as np
from abc import ABC, abstractmethod

from aeromorph.core.types import (
    MorphAction,
    MorphingDecision,
    FeasibilityResult,
    PerceptionFeatures
)
from aeromorph.core.state import PerceptionState, ExtendedRobotState
from aeromorph.core.decision_space import MorphologicalDecisionSpace
from aeromorph.core.config_space import MorphingConfigSpace


@dataclass
class P2MAConfig:
    """Configuration for the P2MA algorithm."""
    
    # Utility thresholds
    utility_trigger_threshold: float = 0.3
    clearance_threshold: float = 1.0  # meters
    turbulence_threshold: float = 0.5
    
    # Utility weights
    weight_clearance: float = 1.0
    weight_aerodynamic: float = 0.8
    weight_energy: float = 0.5
    
    # Energy constraints
    energy_reserve_fraction: float = 0.1
    
    # Spatial feasibility
    safety_margin: float = 0.1  # meters
    interpolation_steps: int = 20
    
    # Timing
    max_decision_time: float = 0.2  # seconds


class PerceptionFusion:
    """
    Multi-modal sensor fusion using Extended Kalman Filter.
    
    Fuses LiDAR, camera, and IMU data to produce unified
    environmental representation for morphing decisions.
    """
    
    def __init__(self, 
                 lidar_weight: float = 0.4,
                 depth_weight: float = 0.3,
                 imu_weight: float = 0.3):
        """
        Initialize perception fusion.
        
        Args:
            lidar_weight: Weight for LiDAR observations
            depth_weight: Weight for depth camera observations
            imu_weight: Weight for IMU observations
        """
        self.weights = {
            'lidar': lidar_weight,
            'depth': depth_weight,
            'imu': imu_weight
        }
        
        # EKF state: [obstacle_dist, passage_width, turbulence, wind_x, wind_y, wind_z]
        self.state = np.zeros(6)
        self.covariance = np.eye(6) * 1.0
        
        # Process noise
        self.Q = np.eye(6) * 0.01
        
        # Observation noise
        self.R_lidar = np.eye(2) * 0.1
        self.R_depth = np.eye(2) * 0.2
        self.R_imu = np.eye(3) * 0.05
    
    def fuse(self, perception: PerceptionState) -> np.ndarray:
        """
        Fuse sensor data to produce unified perception state.
        
        Args:
            perception: Raw perception state from sensors
            
        Returns:
            Fused state vector [obstacle_dist, passage_width, turbulence,
                               wind_x, wind_y, wind_z]
        """
        # Prediction step
        self._predict()
        
        # Update with LiDAR
        if perception.has_lidar():
            obs_dist = perception.get_min_obstacle_distance()
            passage_width = perception.estimate_passage_width()
            self._update_lidar(np.array([obs_dist, passage_width]))
        
        # Update with depth camera
        if perception.has_depth():
            depth_features = self._extract_depth_features(perception.camera_depth)
            self._update_depth(depth_features)
        
        # Update with IMU (for turbulence and wind)
        if perception.imu_data is not None:
            self._update_imu(perception.imu_data)
        
        return self.state.copy()
    
    def _predict(self):
        """EKF prediction step with constant velocity model."""
        # State transition (identity - assume slow-changing environment)
        F = np.eye(6)
        
        # Predict state and covariance
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + self.Q
    
    def _update_lidar(self, observation: np.ndarray):
        """Update with LiDAR observation."""
        H = np.zeros((2, 6))
        H[0, 0] = 1  # obstacle distance
        H[1, 1] = 1  # passage width
        
        self._kalman_update(observation, H, self.R_lidar)
    
    def _update_depth(self, observation: np.ndarray):
        """Update with depth camera observation."""
        H = np.zeros((2, 6))
        H[0, 0] = 1  # obstacle distance
        H[1, 1] = 1  # passage width
        
        self._kalman_update(observation, H, self.R_depth)
    
    def _update_imu(self, imu_data: np.ndarray):
        """Update with IMU observation for turbulence estimation."""
        # Extract turbulence from acceleration variance
        acc = imu_data[:3]
        turbulence = np.std(np.abs(acc))
        
        # Simple wind estimation from gyro drift
        wind_estimate = imu_data[3:6] * 0.1  # Simplified
        
        observation = np.array([turbulence, *wind_estimate[:2]])
        
        H = np.zeros((3, 6))
        H[0, 2] = 1  # turbulence
        H[1, 3] = 1  # wind_x
        H[2, 4] = 1  # wind_y
        
        R = np.eye(3) * 0.1
        self._kalman_update(observation, H[:3, :], R)
    
    def _kalman_update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray):
        """Standard Kalman filter update."""
        # Innovation
        y = z - H @ self.state
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + R
        
        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state = self.state + K @ y
        self.covariance = (np.eye(6) - K @ H) @ self.covariance
    
    def _extract_depth_features(self, depth_image: np.ndarray) -> np.ndarray:
        """Extract obstacle distance and passage width from depth image."""
        # Simplified: use center region for obstacle distance
        h, w = depth_image.shape
        center_region = depth_image[h//3:2*h//3, w//3:2*w//3]
        
        obstacle_dist = float(np.percentile(center_region[center_region > 0], 10))
        
        # Estimate passage width from horizontal extent of close obstacles
        close_mask = depth_image < obstacle_dist * 1.5
        if np.any(close_mask):
            cols_with_obstacles = np.any(close_mask, axis=0)
            left = np.argmax(cols_with_obstacles)
            right = w - np.argmax(cols_with_obstacles[::-1])
            passage_width = (right - left) / w * 5.0  # Scale to meters
        else:
            passage_width = float('inf')
        
        return np.array([obstacle_dist, passage_width])
    
    def get_perception_features(self) -> PerceptionFeatures:
        """Convert fused state to perception features."""
        return PerceptionFeatures(
            min_passage_width=self.state[1],
            turbulence_index=min(1.0, self.state[2]),
            wind_direction=self.state[3:6],
            wind_speed=float(np.linalg.norm(self.state[3:6]))
        )


class MorphingUtilityFunction:
    """
    Computes utility of morphing based on perception and energy state.
    
    U_morph(α, p_t) = w₁·f_clearance(α, d) + w₂·f_aero(α, w) - w₃·E_morph(α)
    """
    
    def __init__(self, config: P2MAConfig, config_space: MorphingConfigSpace):
        """
        Initialize utility function.
        
        Args:
            config: P2MA configuration
            config_space: Morphing configuration space
        """
        self.config = config
        self.config_space = config_space
    
    def compute(self,
                current_config: np.ndarray,
                target_config: np.ndarray,
                perception: PerceptionFeatures,
                energy_available: float) -> float:
        """
        Compute morphing utility.
        
        Args:
            current_config: Current morphing configuration
            target_config: Proposed target configuration
            perception: Perception features
            energy_available: Available energy for morphing (Joules)
            
        Returns:
            Utility value (higher is better)
        """
        # Clearance benefit
        clearance_benefit = self._clearance_benefit(
            current_config, target_config, perception.min_passage_width
        )
        
        # Aerodynamic benefit
        aero_benefit = self._aerodynamic_benefit(
            current_config, target_config, perception
        )
        
        # Energy cost (normalized)
        energy_cost = self.config_space.compute_energy_cost(
            current_config, target_config
        )
        normalized_cost = energy_cost / max(energy_available, 1.0)
        
        # Weighted sum
        utility = (
            self.config.weight_clearance * clearance_benefit +
            self.config.weight_aerodynamic * aero_benefit -
            self.config.weight_energy * normalized_cost
        )
        
        return utility
    
    def _clearance_benefit(self,
                           current: np.ndarray,
                           target: np.ndarray,
                           passage_width: float) -> float:
        """Compute benefit for passage clearance."""
        if passage_width == float('inf'):
            return 0.0
        
        # Estimate robot width from wingspan config
        current_width = 0.5 + 1.0 * current[0]  # meters
        target_width = 0.5 + 1.0 * target[0]
        
        current_margin = passage_width - current_width
        target_margin = passage_width - target_width
        
        # Benefit if target fits better
        if target_margin > current_margin and target_margin > 0:
            return min(1.0, (target_margin - current_margin) / passage_width)
        elif current_margin < 0 and target_margin >= 0:
            return 1.0  # Critical: enables passage
        else:
            return 0.0
    
    def _aerodynamic_benefit(self,
                             current: np.ndarray,
                             target: np.ndarray,
                             perception: PerceptionFeatures) -> float:
        """Compute aerodynamic benefit based on conditions."""
        turbulence = perception.turbulence_index
        
        # Get aerodynamic properties
        current_aero = self.config_space.get_aerodynamic_properties(current)
        target_aero = self.config_space.get_aerodynamic_properties(target)
        
        if turbulence > self.config.turbulence_threshold:
            # In turbulence, prefer stability (higher stiffness)
            stiffness_gain = target[-1] - current[-1]
            return max(0, stiffness_gain * turbulence)
        else:
            # In calm conditions, prefer efficiency (higher L/D)
            ld_gain = target_aero['L_D_ratio'] - current_aero['L_D_ratio']
            return max(0, ld_gain / 50.0)  # Normalized


class P2MAAlgorithm:
    """
    Perception-to-Morphology-Action Algorithm.
    
    Implements the complete P2MA pipeline for autonomous morphing decisions:
    1. Perception Fusion (EKF-based multi-modal fusion)
    2. Morphing Need Assessment (utility computation)
    3. Spatial Feasibility Verification (collision checking)
    4. Energy-Constrained Optimization (budget management)
    
    Example:
        >>> p2ma = P2MAAlgorithm(config_dim=6)
        >>> decision = p2ma.decide(
        ...     current_config=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        ...     perception=perception_state,
        ...     energy_budget=500.0,
        ...     trajectory=planned_path
        ... )
    """
    
    def __init__(self,
                 config_dim: int = 6,
                 config: Optional[P2MAConfig] = None,
                 spatial_checker: Optional['SpatialFeasibilityChecker'] = None):
        """
        Initialize P2MA algorithm.
        
        Args:
            config_dim: Dimension of morphing configuration space
            config: Algorithm configuration
            spatial_checker: Spatial feasibility checker instance
        """
        self.config = config or P2MAConfig()
        self.config_space = MorphingConfigSpace(dim=config_dim)
        
        self.perception_fusion = PerceptionFusion()
        self.utility_function = MorphingUtilityFunction(
            self.config, self.config_space
        )
        
        # Import here to avoid circular dependency
        from aeromorph.algorithms.spatial_feasibility import SpatialFeasibilityChecker
        self.spatial_checker = spatial_checker or SpatialFeasibilityChecker(
            safety_margin=self.config.safety_margin,
            interpolation_steps=self.config.interpolation_steps
        )
        
        # Candidate configurations for different situations
        self.candidate_configs = {
            'narrow': self.config_space.get_narrow_config(),
            'efficient': self.config_space.get_efficient_config(),
            'stable': self.config_space.get_stable_config()
        }
    
    def decide(self,
               current_config: np.ndarray,
               perception: PerceptionState,
               energy_budget: float,
               trajectory: Optional[np.ndarray] = None,
               environment: Optional[np.ndarray] = None) -> MorphingDecision:
        """
        Make autonomous morphing decision.
        
        This is the main entry point implementing Algorithm 1 from the proposal.
        
        Args:
            current_config: Current morphing configuration α_curr
            perception: Current perception state from sensors
            energy_budget: Available energy for morphing (Joules)
            trajectory: Planned trajectory (optional)
            environment: Occupancy grid (optional)
            
        Returns:
            MorphingDecision with action and target configuration
        """
        import time
        start_time = time.time()
        
        # Stage 1: Perception Fusion
        fused_state = self.perception_fusion.fuse(perception)
        features = self.perception_fusion.get_perception_features()
        
        # Stage 2: Morphing Need Assessment
        target_config, utility, trigger_reason = self._assess_morphing_need(
            current_config, features, energy_budget
        )
        
        # Check if morphing is needed
        if utility < self.config.utility_trigger_threshold:
            return MorphingDecision(
                action=MorphAction.HOLD,
                current_config=current_config,
                utility=utility,
                trigger_reason="Utility below threshold"
            )
        
        # Stage 3: Spatial Feasibility Verification
        if environment is not None:
            feasibility = self.spatial_checker.verify(
                current_config, target_config, environment
            )
            
            if not feasibility.feasible:
                # Try to find partial morph
                partial_config = self._find_partial_morph(
                    current_config, target_config, environment
                )
                
                if partial_config is not None:
                    target_config = partial_config
                    trigger_reason = f"Partial morph: {trigger_reason}"
                else:
                    return MorphingDecision(
                        action=MorphAction.HOLD,
                        current_config=current_config,
                        utility=utility,
                        spatial_feasibility=False,
                        trigger_reason="Spatial infeasibility, no partial morph found"
                    )
        
        # Stage 4: Energy-Constrained Optimization
        energy_cost = self.config_space.compute_energy_cost(
            current_config, target_config
        )
        
        reserve = energy_budget * self.config.energy_reserve_fraction
        if energy_cost > energy_budget - reserve:
            # Find energy-constrained configuration
            target_config = self._energy_constrained_morph(
                current_config, target_config, energy_budget - reserve
            )
            energy_cost = self.config_space.compute_energy_cost(
                current_config, target_config
            )
            trigger_reason = f"Energy-constrained: {trigger_reason}"
        
        computation_time = time.time() - start_time
        
        return MorphingDecision(
            action=MorphAction.MORPH,
            target_config=target_config,
            current_config=current_config,
            energy_cost=energy_cost,
            utility=utility,
            spatial_feasibility=True,
            trigger_reason=trigger_reason,
            timestamp=computation_time
        )
    
    def _assess_morphing_need(self,
                              current_config: np.ndarray,
                              features: PerceptionFeatures,
                              energy_budget: float) -> Tuple[np.ndarray, float, str]:
        """
        Assess whether morphing is needed and select target configuration.
        
        Returns:
            Tuple of (target_config, utility, trigger_reason)
        """
        best_utility = -float('inf')
        best_config = current_config
        best_reason = "No improvement found"
        
        # Check each candidate configuration
        for name, candidate in self.candidate_configs.items():
            utility = self.utility_function.compute(
                current_config, candidate, features, energy_budget
            )
            
            if utility > best_utility:
                best_utility = utility
                best_config = candidate
                best_reason = f"{name.capitalize()} configuration selected"
        
        # Check clearance-based morphing
        if features.min_passage_width < self.config.clearance_threshold:
            narrow_config = self.candidate_configs['narrow']
            utility = self.utility_function.compute(
                current_config, narrow_config, features, energy_budget
            )
            if utility > best_utility:
                best_utility = utility
                best_config = narrow_config
                best_reason = f"Narrow passage detected ({features.min_passage_width:.2f}m)"
        
        # Check turbulence-based morphing
        if features.turbulence_index > self.config.turbulence_threshold:
            stable_config = self.candidate_configs['stable']
            utility = self.utility_function.compute(
                current_config, stable_config, features, energy_budget
            )
            if utility > best_utility:
                best_utility = utility
                best_config = stable_config
                best_reason = f"High turbulence detected ({features.turbulence_index:.2f})"
        
        return best_config, best_utility, best_reason
    
    def _find_partial_morph(self,
                            start: np.ndarray,
                            target: np.ndarray,
                            environment: np.ndarray) -> Optional[np.ndarray]:
        """Find largest feasible partial morph towards target."""
        # Binary search for feasible configuration
        low, high = 0.0, 1.0
        best_feasible = None
        
        for _ in range(5):  # Binary search iterations
            mid = (low + high) / 2
            partial = start + mid * (target - start)
            
            feasibility = self.spatial_checker.verify(start, partial, environment)
            
            if feasibility.feasible:
                best_feasible = partial
                low = mid
            else:
                high = mid
        
        return best_feasible
    
    def _energy_constrained_morph(self,
                                  start: np.ndarray,
                                  target: np.ndarray,
                                  max_energy: float) -> np.ndarray:
        """Find configuration change that fits within energy budget."""
        # Scale morph to fit energy budget
        full_cost = self.config_space.compute_energy_cost(start, target)
        
        if full_cost <= max_energy:
            return target
        
        # Scale down the morph
        scale = max_energy / full_cost * 0.9  # 90% to ensure margin
        return start + scale * (target - start)


class MorphDecisionEngine:
    """
    High-level morphing decision engine.
    
    Wraps P2MA algorithm with state management and provides
    a simplified interface for integration with robot controllers.
    
    Example:
        >>> engine = MorphDecisionEngine(config_space_dim=6)
        >>> decision = engine.decide(current_config, perception, trajectory)
    """
    
    def __init__(self,
                 config_space_dim: int = 6,
                 energy_budget: float = 1000.0,
                 **kwargs):
        """
        Initialize decision engine.
        
        Args:
            config_space_dim: Dimension of configuration space
            energy_budget: Initial energy budget (Joules)
            **kwargs: Additional arguments passed to P2MA
        """
        self.p2ma = P2MAAlgorithm(config_dim=config_space_dim, **kwargs)
        self.energy_budget = energy_budget
        self.decision_history: List[MorphingDecision] = []
    
    def decide(self,
               current_config: np.ndarray,
               perception: PerceptionState,
               trajectory: Optional[np.ndarray] = None,
               environment: Optional[np.ndarray] = None) -> MorphingDecision:
        """
        Make morphing decision using P2MA algorithm.
        
        Args:
            current_config: Current morphing configuration
            perception: Perception state from sensors
            trajectory: Planned trajectory
            environment: Occupancy grid
            
        Returns:
            MorphingDecision
        """
        decision = self.p2ma.decide(
            current_config=current_config,
            perception=perception,
            energy_budget=self.energy_budget,
            trajectory=trajectory,
            environment=environment
        )
        
        # Update energy budget if morphing
        if decision.action == MorphAction.MORPH:
            self.energy_budget -= decision.energy_cost
        
        # Record decision
        self.decision_history.append(decision)
        
        return decision
    
    def update_energy(self, energy: float):
        """Update available energy budget."""
        self.energy_budget = energy
    
    def get_decision_stats(self) -> dict:
        """Get statistics about decisions made."""
        if not self.decision_history:
            return {'total': 0, 'morphs': 0, 'holds': 0}
        
        morphs = sum(1 for d in self.decision_history 
                     if d.action == MorphAction.MORPH)
        holds = sum(1 for d in self.decision_history 
                    if d.action == MorphAction.HOLD)
        
        return {
            'total': len(self.decision_history),
            'morphs': morphs,
            'holds': holds,
            'morph_rate': morphs / len(self.decision_history),
            'total_energy_used': sum(d.energy_cost for d in self.decision_history)
        }
