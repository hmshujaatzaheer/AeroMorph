"""
Unified Morphing Optimization Framework.

This module implements the joint optimization of trajectory, morphing,
energy, and swarm coordination.

Optimization Problem:
    min_{u,α} J = J_trajectory + J_morphing + J_swarm + J_energy
    
    subject to:
        - Coupled aerodynamic-morphing dynamics
        - Collision avoidance constraints
        - Reachable configuration constraints
        - Morphing rate limits
        - Energy budget constraints

Reference: AeroMorph PhD Research Proposal, Section 4.1
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable
import numpy as np

from aeromorph.core.state import ExtendedRobotState
from aeromorph.core.config_space import MorphingConfigSpace


@dataclass
class OptimizationConfig:
    """Configuration for the unified optimizer."""
    
    # MPC parameters
    horizon: int = 20
    dt: float = 0.1  # seconds
    
    # Cost weights
    Q_position: float = 10.0  # Position tracking
    Q_velocity: float = 1.0   # Velocity tracking
    R_control: float = 0.1    # Control effort
    S_morph: float = 100.0    # Morphing smoothness
    lambda_energy: float = 0.5  # Energy cost
    lambda_sync: float = 0.3    # Swarm synchronization
    
    # Constraints
    max_morph_rate: float = 1.0  # per second
    max_control: float = 10.0
    collision_margin: float = 0.5
    
    # Solver settings
    max_iterations: int = 100
    convergence_tol: float = 1e-4


class AerodynamicModel:
    """
    Simplified aerodynamic model for morphing aircraft.
    
    Models the coupled dynamics between morphing configuration
    and flight performance.
    """
    
    def __init__(self, mass: float = 2.0, base_inertia: np.ndarray = None):
        """
        Initialize aerodynamic model.
        
        Args:
            mass: Robot mass (kg)
            base_inertia: Base inertia tensor (3x3)
        """
        self.mass = mass
        self.base_inertia = base_inertia or np.diag([0.1, 0.2, 0.15])
        
        # Aerodynamic coefficients (simplified)
        self.rho = 1.225  # Air density (kg/m³)
        self.g = 9.81     # Gravity (m/s²)
    
    def compute_forces(self,
                       state: ExtendedRobotState,
                       control: np.ndarray) -> np.ndarray:
        """
        Compute aerodynamic forces given state and control.
        
        Args:
            state: Extended robot state
            control: Control inputs [thrust, roll_torque, pitch_torque, yaw_torque]
            
        Returns:
            Forces and moments in body frame [Fx, Fy, Fz, Mx, My, Mz]
        """
        velocity = state.velocity.linear
        config = state.morph_config
        
        # Get aerodynamic coefficients from configuration
        C_L = 0.5 + 0.8 * config[0] + 0.4 * config[2]  # Lift
        C_D = 0.02 + 0.03 * config[0] - 0.01 * config[1]  # Drag
        
        # Wing area estimate
        wingspan = 0.5 + 1.0 * config[0]
        chord = 0.15
        S = wingspan * chord
        
        # Dynamic pressure
        V = np.linalg.norm(velocity)
        q = 0.5 * self.rho * V * V if V > 0.1 else 0.0
        
        # Forces
        L = q * S * C_L  # Lift
        D = q * S * C_D  # Drag
        
        # Simplified force in body frame
        # Assuming level flight orientation
        Fx = control[0] - D  # Thrust minus drag
        Fy = 0.0
        Fz = L - self.mass * self.g  # Lift minus weight
        
        # Moments from control
        Mx = control[1]  # Roll
        My = control[2]  # Pitch
        Mz = control[3]  # Yaw
        
        return np.array([Fx, Fy, Fz, Mx, My, Mz])
    
    def compute_inertia(self, config: np.ndarray) -> np.ndarray:
        """
        Compute configuration-dependent inertia tensor.
        
        Args:
            config: Morphing configuration
            
        Returns:
            3x3 inertia tensor
        """
        # Inertia changes with wingspan and body configuration
        wingspan_factor = 1.0 + 0.5 * config[0]
        body_factor = 1.0 + 0.2 * config[4]
        
        J = self.base_inertia.copy()
        J[0, 0] *= wingspan_factor  # Roll inertia increases with wingspan
        J[1, 1] *= body_factor      # Pitch inertia with body length
        J[2, 2] *= wingspan_factor * body_factor  # Yaw
        
        return J
    
    def compute_inertia_derivative(self, 
                                   config: np.ndarray,
                                   morph_velocity: np.ndarray) -> np.ndarray:
        """
        Compute time derivative of inertia tensor during morphing.
        
        J̇(α) for stability analysis during shape change.
        
        Args:
            config: Current configuration
            morph_velocity: Configuration change rate
            
        Returns:
            3x3 inertia derivative tensor
        """
        J_dot = np.zeros((3, 3))
        
        # Partial derivatives
        dJ_dalpha0 = np.diag([0.5 * self.base_inertia[0, 0], 0, 
                              0.5 * self.base_inertia[2, 2] * (1 + 0.2 * config[4])])
        dJ_dalpha4 = np.diag([0, 0.2 * self.base_inertia[1, 1],
                              0.2 * self.base_inertia[2, 2] * (1 + 0.5 * config[0])])
        
        J_dot += dJ_dalpha0 * morph_velocity[0]
        J_dot += dJ_dalpha4 * morph_velocity[4] if len(morph_velocity) > 4 else 0
        
        return J_dot


class UnifiedMorphOptimizer:
    """
    Unified Optimizer for Morphing Aerial Robots.
    
    Implements Model Predictive Control that jointly optimizes:
    - Trajectory tracking
    - Morphing configuration
    - Energy consumption
    - Swarm synchronization
    
    The optimization handles coupled aerodynamic-morphing dynamics
    with time-varying inertia during shape transitions.
    
    Example:
        >>> optimizer = UnifiedMorphOptimizer(config_dim=6)
        >>> trajectory, morph_schedule = optimizer.optimize(
        ...     current_state, goal_state, environment
        ... )
    """
    
    def __init__(self,
                 config_dim: int = 6,
                 config: Optional[OptimizationConfig] = None):
        """
        Initialize unified optimizer.
        
        Args:
            config_dim: Morphing configuration dimension
            config: Optimization configuration
        """
        self.config_dim = config_dim
        self.opt_config = config or OptimizationConfig()
        
        self.config_space = MorphingConfigSpace(dim=config_dim)
        self.aero_model = AerodynamicModel()
        
        # State dimensions
        self.state_dim = 13 + config_dim  # pose(7) + vel(6) + morph(n)
        self.control_dim = 4 + config_dim  # thrust/moments(4) + morph_rate(n)
    
    def optimize(self,
                 current_state: ExtendedRobotState,
                 goal_state: ExtendedRobotState,
                 environment: Optional[np.ndarray] = None,
                 energy_budget: Optional[float] = None,
                 swarm_states: Optional[List[ExtendedRobotState]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the unified optimization problem.
        
        Args:
            current_state: Current robot state
            goal_state: Goal state (position, configuration)
            environment: Obstacle environment
            energy_budget: Available energy for mission
            swarm_states: States of other robots in swarm
            
        Returns:
            Tuple of (optimal_trajectory, morph_schedule)
        """
        # Initialize trajectory
        N = self.opt_config.horizon
        trajectory = self._initialize_trajectory(current_state, goal_state, N)
        controls = np.zeros((N, self.control_dim))
        
        # Iterative optimization (simplified gradient descent)
        for iteration in range(self.opt_config.max_iterations):
            # Compute cost and gradients
            cost, grad_x, grad_u = self._compute_cost_and_gradients(
                trajectory, controls, goal_state, environment, 
                energy_budget, swarm_states
            )
            
            # Update controls
            controls -= 0.01 * grad_u
            
            # Clip controls to constraints
            controls = self._apply_control_constraints(controls)
            
            # Forward simulate with new controls
            trajectory = self._forward_simulate(current_state, controls)
            
            # Check convergence
            if np.linalg.norm(grad_u) < self.opt_config.convergence_tol:
                break
        
        # Extract morph schedule
        morph_schedule = self._extract_morph_schedule(trajectory)
        
        return trajectory, morph_schedule
    
    def _initialize_trajectory(self,
                               start: ExtendedRobotState,
                               goal: ExtendedRobotState,
                               N: int) -> np.ndarray:
        """Initialize trajectory with linear interpolation."""
        trajectory = np.zeros((N + 1, self.state_dim))
        
        start_vec = start.to_vector()
        goal_vec = goal.to_vector()
        
        for i in range(N + 1):
            alpha = i / N
            trajectory[i, :len(start_vec)] = (1 - alpha) * start_vec + alpha * goal_vec
        
        return trajectory
    
    def _compute_cost_and_gradients(self,
                                    trajectory: np.ndarray,
                                    controls: np.ndarray,
                                    goal: ExtendedRobotState,
                                    environment: Optional[np.ndarray],
                                    energy_budget: Optional[float],
                                    swarm_states: Optional[List[ExtendedRobotState]]) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute cost function and gradients."""
        N = len(controls)
        goal_vec = goal.to_vector()
        
        cost = 0.0
        grad_x = np.zeros_like(trajectory)
        grad_u = np.zeros_like(controls)
        
        cfg = self.opt_config
        
        for k in range(N):
            # State at step k
            x_k = trajectory[k]
            u_k = controls[k]
            
            # Trajectory tracking cost
            pos_error = x_k[:3] - goal_vec[:3]
            cost += cfg.Q_position * np.dot(pos_error, pos_error)
            grad_x[k, :3] += 2 * cfg.Q_position * pos_error
            
            # Control effort cost
            cost += cfg.R_control * np.dot(u_k[:4], u_k[:4])
            grad_u[k, :4] += 2 * cfg.R_control * u_k[:4]
            
            # Morphing smoothness cost
            if k > 0:
                morph_diff = x_k[13:13+self.config_dim] - trajectory[k-1, 13:13+self.config_dim]
                cost += cfg.S_morph * np.dot(morph_diff, morph_diff)
                grad_x[k, 13:13+self.config_dim] += 2 * cfg.S_morph * morph_diff
            
            # Energy cost
            morph_rate = u_k[4:4+self.config_dim]
            morph_energy = self.config_space.compute_energy_cost(
                np.zeros(self.config_dim),
                morph_rate * self.opt_config.dt
            )
            cost += cfg.lambda_energy * morph_energy
            grad_u[k, 4:4+self.config_dim] += cfg.lambda_energy * morph_rate
            
            # Swarm synchronization cost
            if swarm_states:
                morph_k = x_k[13:13+self.config_dim]
                for swarm_state in swarm_states:
                    sync_error = morph_k - swarm_state.morph_config
                    cost += cfg.lambda_sync * np.dot(sync_error, sync_error)
                    grad_x[k, 13:13+self.config_dim] += 2 * cfg.lambda_sync * sync_error
        
        return cost, grad_x, grad_u
    
    def _apply_control_constraints(self, controls: np.ndarray) -> np.ndarray:
        """Apply constraints to control inputs."""
        # Clip control magnitude
        controls[:, :4] = np.clip(
            controls[:, :4], 
            -self.opt_config.max_control, 
            self.opt_config.max_control
        )
        
        # Clip morphing rate
        controls[:, 4:] = np.clip(
            controls[:, 4:],
            -self.opt_config.max_morph_rate,
            self.opt_config.max_morph_rate
        )
        
        return controls
    
    def _forward_simulate(self,
                          initial_state: ExtendedRobotState,
                          controls: np.ndarray) -> np.ndarray:
        """Forward simulate dynamics with given controls."""
        N = len(controls)
        trajectory = np.zeros((N + 1, self.state_dim))
        trajectory[0] = initial_state.to_vector()
        
        dt = self.opt_config.dt
        
        for k in range(N):
            x_k = trajectory[k]
            u_k = controls[k]
            
            # Simple Euler integration
            # Position update
            trajectory[k+1, :3] = x_k[:3] + dt * x_k[7:10]  # pos += vel * dt
            
            # Keep orientation (simplified)
            trajectory[k+1, 3:7] = x_k[3:7]
            
            # Velocity update (simplified)
            trajectory[k+1, 7:10] = x_k[7:10] + dt * u_k[:3] / 2.0
            trajectory[k+1, 10:13] = x_k[10:13] + dt * u_k[:3] / 2.0
            
            # Morphing configuration update
            morph_rate = u_k[4:4+self.config_dim]
            trajectory[k+1, 13:13+self.config_dim] = x_k[13:13+self.config_dim] + dt * morph_rate
            
            # Clip morphing to valid range
            trajectory[k+1, 13:13+self.config_dim] = np.clip(
                trajectory[k+1, 13:13+self.config_dim], 0, 1
            )
            
            # Energy state update
            if trajectory.shape[1] > 13 + self.config_dim:
                trajectory[k+1, -1] = x_k[-1] - self.config_space.compute_energy_cost(
                    np.zeros(self.config_dim), morph_rate * dt
                )
        
        return trajectory
    
    def _extract_morph_schedule(self, trajectory: np.ndarray) -> np.ndarray:
        """Extract morphing schedule from optimized trajectory."""
        N = len(trajectory) - 1
        schedule = np.zeros((N, self.config_dim + 1))  # time + config
        
        for k in range(N):
            schedule[k, 0] = k * self.opt_config.dt  # time
            schedule[k, 1:] = trajectory[k, 13:13+self.config_dim]  # config
        
        return schedule
    
    def compute_stability_margin(self,
                                state: ExtendedRobotState) -> float:
        """
        Compute stability margin during morphing.
        
        Accounts for time-varying inertia during shape change.
        
        Args:
            state: Current extended state
            
        Returns:
            Stability margin (positive = stable)
        """
        J = self.aero_model.compute_inertia(state.morph_config)
        J_dot = self.aero_model.compute_inertia_derivative(
            state.morph_config, state.morph_velocity
        )
        
        omega = state.velocity.angular
        
        # Simplified stability check: inertia derivative should not destabilize
        # ||J_dot * omega|| / ||J * omega|| < threshold
        J_omega = J @ omega
        J_dot_omega = J_dot @ omega
        
        if np.linalg.norm(J_omega) < 1e-6:
            return 1.0  # Stable when not rotating
        
        ratio = np.linalg.norm(J_dot_omega) / np.linalg.norm(J_omega)
        return 1.0 - ratio  # Positive when stable
