"""
Robot State Representations for AeroMorph Framework.

This module defines the extended robot state that incorporates morphing
configuration alongside traditional pose and velocity states.

Extended State:
    x_ext = [x_pose ∈ SE(3), x_vel ∈ ℝ⁶, α ∈ ℳ, E_state ∈ ℝ⁺]ᵀ

Reference: AeroMorph PhD Research Proposal, Section 3.1
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np


@dataclass
class Pose:
    """
    Robot pose in SE(3).
    
    Attributes:
        position: 3D position [x, y, z] in world frame (meters)
        orientation: Quaternion [w, x, y, z] or Euler angles [roll, pitch, yaw]
        use_quaternion: Whether orientation is quaternion (True) or Euler (False)
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    use_quaternion: bool = True
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 homogeneous transformation matrix."""
        T = np.eye(4)
        T[:3, 3] = self.position
        
        if self.use_quaternion:
            T[:3, :3] = self._quat_to_rotation(self.orientation)
        else:
            T[:3, :3] = self._euler_to_rotation(self.orientation)
        
        return T
    
    def _quat_to_rotation(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
    
    def _euler_to_rotation(self, euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles to rotation matrix (ZYX convention)."""
        roll, pitch, yaw = euler
        
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])


@dataclass
class Velocity:
    """
    Robot velocity in ℝ⁶.
    
    Attributes:
        linear: Linear velocity [vx, vy, vz] in body frame (m/s)
        angular: Angular velocity [ωx, ωy, ωz] in body frame (rad/s)
    """
    linear: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def to_vector(self) -> np.ndarray:
        """Convert to 6D twist vector."""
        return np.concatenate([self.linear, self.angular])
    
    @property
    def speed(self) -> float:
        """Get linear speed magnitude."""
        return float(np.linalg.norm(self.linear))
    
    @property
    def rotation_rate(self) -> float:
        """Get angular rate magnitude."""
        return float(np.linalg.norm(self.angular))


@dataclass
class PerceptionState:
    """
    Perception state from multi-modal sensors.
    
    Encapsulates all sensor data needed for morphing decisions.
    
    Attributes:
        lidar_points: LiDAR point cloud (N, 3)
        camera_depth: Depth image (H, W)
        camera_rgb: RGB image (H, W, 3)
        imu_data: IMU measurements [ax, ay, az, wx, wy, wz]
        wind_estimate: Estimated wind vector in body frame
        gps_position: GPS position if available
        timestamp: Sensor timestamp
    """
    lidar_points: Optional[np.ndarray] = None
    camera_depth: Optional[np.ndarray] = None
    camera_rgb: Optional[np.ndarray] = None
    imu_data: Optional[np.ndarray] = None
    wind_estimate: Optional[np.ndarray] = None
    gps_position: Optional[np.ndarray] = None
    timestamp: float = 0.0
    
    def has_lidar(self) -> bool:
        """Check if LiDAR data is available."""
        return self.lidar_points is not None and len(self.lidar_points) > 0
    
    def has_depth(self) -> bool:
        """Check if depth image is available."""
        return self.camera_depth is not None
    
    def get_min_obstacle_distance(self, direction: np.ndarray = None) -> float:
        """
        Get minimum distance to obstacles.
        
        Args:
            direction: Optional direction to consider (forward if None)
            
        Returns:
            Minimum distance in meters, or inf if no obstacles detected
        """
        if not self.has_lidar():
            return float('inf')
        
        if direction is None:
            direction = np.array([1, 0, 0])  # Forward
        
        # Project points onto direction and find minimum positive
        distances = self.lidar_points @ direction
        positive_distances = distances[distances > 0]
        
        if len(positive_distances) == 0:
            return float('inf')
        
        return float(np.min(positive_distances))
    
    def estimate_passage_width(self, 
                               forward_distance: float = 5.0) -> float:
        """
        Estimate the width of the passage ahead.
        
        Args:
            forward_distance: How far ahead to look (meters)
            
        Returns:
            Estimated passage width in meters
        """
        if not self.has_lidar():
            return float('inf')
        
        # Get points within the forward cone
        forward_mask = (self.lidar_points[:, 0] > 0) & \
                       (self.lidar_points[:, 0] < forward_distance)
        forward_points = self.lidar_points[forward_mask]
        
        if len(forward_points) < 2:
            return float('inf')
        
        # Estimate width from lateral extent
        left_bound = np.min(forward_points[:, 1])
        right_bound = np.max(forward_points[:, 1])
        
        return float(right_bound - left_bound)


@dataclass
class ExtendedRobotState:
    """
    Extended Robot State incorporating morphing configuration.
    
    x_ext = [x_pose ∈ SE(3), x_vel ∈ ℝ⁶, α ∈ ℳ, E_state ∈ ℝ⁺]ᵀ
    
    This is the complete state representation for morphing aerial robots,
    combining traditional flight state with morphing configuration and
    energy state.
    
    Attributes:
        pose: Robot pose in SE(3)
        velocity: Robot velocity in ℝ⁶
        morph_config: Morphing configuration α ∈ ℳ
        morph_velocity: Rate of morphing configuration change
        energy_state: Current battery energy (Joules)
        perception: Current perception state
        timestamp: State timestamp
    
    Example:
        >>> state = ExtendedRobotState(
        ...     pose=Pose(position=np.array([0, 0, 10])),
        ...     morph_config=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        ...     energy_state=4500.0
        ... )
        >>> state_vector = state.to_vector()
    """
    pose: Pose = field(default_factory=Pose)
    velocity: Velocity = field(default_factory=Velocity)
    morph_config: np.ndarray = field(default_factory=lambda: np.ones(6) * 0.5)
    morph_velocity: np.ndarray = field(default_factory=lambda: np.zeros(6))
    energy_state: float = 5000.0
    perception: Optional[PerceptionState] = None
    timestamp: float = 0.0
    
    @property
    def position(self) -> np.ndarray:
        """Shortcut to get position."""
        return self.pose.position
    
    @property
    def config_dim(self) -> int:
        """Dimension of morphing configuration."""
        return len(self.morph_config)
    
    def to_vector(self) -> np.ndarray:
        """
        Convert to flat state vector.
        
        Returns:
            State vector: [position(3), orientation(4), linear_vel(3), 
                          angular_vel(3), morph_config(n), energy(1)]
        """
        return np.concatenate([
            self.pose.position,
            self.pose.orientation,
            self.velocity.linear,
            self.velocity.angular,
            self.morph_config,
            [self.energy_state]
        ])
    
    @classmethod
    def from_vector(cls, 
                    vector: np.ndarray, 
                    config_dim: int = 6) -> 'ExtendedRobotState':
        """
        Create state from flat vector.
        
        Args:
            vector: State vector
            config_dim: Dimension of morphing configuration
            
        Returns:
            ExtendedRobotState instance
        """
        idx = 0
        
        position = vector[idx:idx+3]; idx += 3
        orientation = vector[idx:idx+4]; idx += 4
        linear_vel = vector[idx:idx+3]; idx += 3
        angular_vel = vector[idx:idx+3]; idx += 3
        morph_config = vector[idx:idx+config_dim]; idx += config_dim
        energy = vector[idx]
        
        return cls(
            pose=Pose(position=position, orientation=orientation),
            velocity=Velocity(linear=linear_vel, angular=angular_vel),
            morph_config=morph_config,
            energy_state=energy
        )
    
    def copy(self) -> 'ExtendedRobotState':
        """Create a deep copy of the state."""
        return ExtendedRobotState(
            pose=Pose(
                position=self.pose.position.copy(),
                orientation=self.pose.orientation.copy(),
                use_quaternion=self.pose.use_quaternion
            ),
            velocity=Velocity(
                linear=self.velocity.linear.copy(),
                angular=self.velocity.angular.copy()
            ),
            morph_config=self.morph_config.copy(),
            morph_velocity=self.morph_velocity.copy(),
            energy_state=self.energy_state,
            perception=self.perception,
            timestamp=self.timestamp
        )
    
    def is_morphing(self, threshold: float = 0.01) -> bool:
        """Check if robot is currently morphing."""
        return np.linalg.norm(self.morph_velocity) > threshold
    
    def get_morphing_progress(self, 
                              target: np.ndarray,
                              start: Optional[np.ndarray] = None) -> float:
        """
        Get progress towards target morphing configuration.
        
        Args:
            target: Target configuration
            start: Starting configuration (uses current if None)
            
        Returns:
            Progress ratio (0.0 to 1.0)
        """
        if start is None:
            return 1.0  # Already at target if no start specified
        
        total_dist = np.linalg.norm(target - start)
        current_dist = np.linalg.norm(target - self.morph_config)
        
        if total_dist < 1e-6:
            return 1.0
        
        return float(1.0 - current_dist / total_dist)
    
    def __repr__(self) -> str:
        pos = self.pose.position
        return (f"ExtendedRobotState(pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}], "
                f"config={self.morph_config[:3].round(2)}..., "
                f"energy={self.energy_state:.0f}J)")


@dataclass  
class SwarmState:
    """
    State of a robot swarm for coordinated morphing.
    
    Attributes:
        robot_states: Dictionary of robot ID to ExtendedRobotState
        communication_graph: Adjacency matrix of communication links
        leader_id: Current swarm leader ID
        consensus_config: Current consensus morphing configuration
    """
    robot_states: Dict[int, ExtendedRobotState] = field(default_factory=dict)
    communication_graph: Optional[np.ndarray] = None
    leader_id: Optional[int] = None
    consensus_config: Optional[np.ndarray] = None
    
    @property
    def num_robots(self) -> int:
        """Number of robots in swarm."""
        return len(self.robot_states)
    
    def get_mean_config(self) -> np.ndarray:
        """Get mean morphing configuration across swarm."""
        if self.num_robots == 0:
            return np.array([])
        
        configs = [state.morph_config for state in self.robot_states.values()]
        return np.mean(configs, axis=0)
    
    def get_mean_energy(self) -> float:
        """Get mean energy level across swarm."""
        if self.num_robots == 0:
            return 0.0
        
        energies = [state.energy_state for state in self.robot_states.values()]
        return float(np.mean(energies))
    
    def get_config_variance(self) -> float:
        """Get variance in morphing configurations (for sync assessment)."""
        if self.num_robots <= 1:
            return 0.0
        
        configs = [state.morph_config for state in self.robot_states.values()]
        return float(np.mean(np.var(configs, axis=0)))
    
    def is_synchronized(self, tolerance: float = 0.05) -> bool:
        """Check if swarm configurations are synchronized."""
        return self.get_config_variance() < tolerance ** 2
