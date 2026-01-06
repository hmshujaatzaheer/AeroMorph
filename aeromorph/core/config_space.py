"""
Morphing Configuration Space Implementation.

This module implements the morphing configuration manifold ℳ that represents
all achievable physical shapes for a morphing aerial robot.

For lattice-based structures:
    α = (τ₁, τ₂, ..., τₙ, θ₁, θ₂, ..., θₘ) ∈ [0,1]ⁿ × [0,2π]ᵐ

Where:
    τᵢ - Topology indices (BCC to X-cube blend)
    θⱼ - Superposition rotation angles

Reference: AeroMorph PhD Research Proposal, Section 4.2
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class MorphingBounds:
    """Bounds on morphing configuration parameters."""
    min_bounds: np.ndarray
    max_bounds: np.ndarray
    
    def clip(self, config: np.ndarray) -> np.ndarray:
        """Clip configuration to valid bounds."""
        return np.clip(config, self.min_bounds, self.max_bounds)
    
    def is_valid(self, config: np.ndarray) -> bool:
        """Check if configuration is within bounds."""
        return np.all(config >= self.min_bounds) and np.all(config <= self.max_bounds)
    
    def sample_uniform(self) -> np.ndarray:
        """Sample uniformly from the configuration space."""
        return np.random.uniform(self.min_bounds, self.max_bounds)


class MorphingConfigSpace:
    """
    Morphing Configuration Manifold ℳ.
    
    Represents the space of all achievable morphing configurations
    for an aerial robot. Supports both continuous and discrete
    configuration representations.
    
    Configuration Dimensions (default 6-DOF):
        0: wingspan (normalized 0-1)
        1: wing_sweep (normalized 0-1)  
        2: wing_camber (normalized 0-1)
        3: tail_angle (normalized 0-1)
        4: body_length (normalized 0-1)
        5: stiffness (normalized 0-1)
    
    Example:
        >>> config_space = MorphingConfigSpace(dim=6)
        >>> config = config_space.sample()
        >>> aero_props = config_space.get_aerodynamic_properties(config)
    """
    
    # Default configuration dimension names
    DEFAULT_DIM_NAMES = [
        'wingspan', 'wing_sweep', 'wing_camber',
        'tail_angle', 'body_length', 'stiffness'
    ]
    
    def __init__(self,
                 dim: int = 6,
                 bounds: Optional[MorphingBounds] = None,
                 dim_names: Optional[List[str]] = None):
        """
        Initialize the morphing configuration space.
        
        Args:
            dim: Dimension of configuration space
            bounds: Optional bounds on configurations
            dim_names: Names for each dimension
        """
        self.dim = dim
        self.bounds = bounds or MorphingBounds(
            min_bounds=np.zeros(dim),
            max_bounds=np.ones(dim)
        )
        self.dim_names = dim_names or self.DEFAULT_DIM_NAMES[:dim]
        
        # Aerodynamic property mappings (simplified models)
        self._init_aero_models()
        
        # Energy cost coefficients
        self.energy_coefficients = np.array([
            300.0,   # wingspan change cost
            200.0,   # sweep change cost
            150.0,   # camber change cost
            100.0,   # tail change cost
            250.0,   # body length change cost
            50.0     # stiffness change cost
        ])[:dim]
    
    def _init_aero_models(self):
        """Initialize simplified aerodynamic property models."""
        # Lift coefficient mapping (simplified)
        # C_L increases with wingspan and camber
        self.lift_model = lambda c: 0.5 + 0.8 * c[0] + 0.4 * c[2]
        
        # Drag coefficient mapping (simplified)
        # C_D increases with wingspan, decreases with sweep
        self.drag_model = lambda c: 0.02 + 0.03 * c[0] - 0.01 * c[1]
        
        # Roll authority mapping
        # Higher with larger wingspan and sweep
        self.roll_model = lambda c: 0.3 + 0.5 * c[0] + 0.3 * c[1]
    
    def sample(self, n: int = 1) -> np.ndarray:
        """
        Sample random configurations from the space.
        
        Args:
            n: Number of samples
            
        Returns:
            Array of shape (n, dim) or (dim,) if n=1
        """
        samples = np.random.uniform(
            self.bounds.min_bounds,
            self.bounds.max_bounds,
            size=(n, self.dim)
        )
        return samples[0] if n == 1 else samples
    
    def interpolate(self,
                    start: np.ndarray,
                    end: np.ndarray,
                    num_steps: int = 10) -> np.ndarray:
        """
        Interpolate between two configurations.
        
        Args:
            start: Starting configuration
            end: Ending configuration
            num_steps: Number of interpolation steps
            
        Returns:
            Array of shape (num_steps, dim) with interpolated configs
        """
        t = np.linspace(0, 1, num_steps)[:, np.newaxis]
        trajectory = start + t * (end - start)
        return trajectory
    
    def get_aerodynamic_properties(self, config: np.ndarray) -> dict:
        """
        Compute aerodynamic properties for a configuration.
        
        Args:
            config: Morphing configuration
            
        Returns:
            Dictionary of aerodynamic properties
        """
        return {
            'C_L': self.lift_model(config),
            'C_D': self.drag_model(config),
            'L_D_ratio': self.lift_model(config) / max(self.drag_model(config), 0.001),
            'roll_authority': self.roll_model(config),
            'estimated_wingspan': 0.5 + 1.0 * config[0],  # meters
            'estimated_stiffness': 25 + 275 * config[-1]  # kPa (25-300 range)
        }
    
    def compute_energy_cost(self, 
                           start: np.ndarray, 
                           end: np.ndarray) -> float:
        """
        Compute energy cost of morphing between configurations.
        
        E_morph(Δα) = Σᵢ kᵢ × |Δαᵢ|
        
        Args:
            start: Starting configuration
            end: Target configuration
            
        Returns:
            Energy cost in Joules
        """
        delta = np.abs(end - start)
        return float(np.dot(self.energy_coefficients, delta))
    
    def compute_distance(self, 
                        config1: np.ndarray, 
                        config2: np.ndarray,
                        metric: str = 'euclidean') -> float:
        """
        Compute distance between two configurations.
        
        Args:
            config1: First configuration
            config2: Second configuration
            metric: Distance metric ('euclidean', 'manhattan', 'weighted')
            
        Returns:
            Distance value
        """
        if metric == 'euclidean':
            return float(np.linalg.norm(config1 - config2))
        elif metric == 'manhattan':
            return float(np.sum(np.abs(config1 - config2)))
        elif metric == 'weighted':
            # Weight by energy coefficients (more expensive changes = longer distance)
            weights = self.energy_coefficients / np.sum(self.energy_coefficients)
            return float(np.sum(weights * np.abs(config1 - config2)))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def get_narrow_config(self) -> np.ndarray:
        """Get configuration optimized for narrow passages."""
        config = np.zeros(self.dim)
        config[0] = 0.0   # Minimum wingspan
        config[1] = 0.8   # High sweep (fold back)
        config[-1] = 0.7  # Higher stiffness for stability
        return self.bounds.clip(config)
    
    def get_efficient_config(self) -> np.ndarray:
        """Get configuration optimized for energy-efficient flight."""
        config = np.ones(self.dim) * 0.5
        config[0] = 0.9   # Large wingspan
        config[1] = 0.3   # Moderate sweep
        config[2] = 0.6   # Good camber for lift
        config[-1] = 0.4  # Moderate stiffness
        return self.bounds.clip(config)
    
    def get_stable_config(self) -> np.ndarray:
        """Get configuration optimized for turbulent conditions."""
        config = np.ones(self.dim) * 0.5
        config[0] = 0.5   # Medium wingspan
        config[1] = 0.5   # Medium sweep
        config[-1] = 1.0  # Maximum stiffness
        return self.bounds.clip(config)
    
    def project_to_manifold(self, config: np.ndarray) -> np.ndarray:
        """
        Project a configuration onto the valid manifold.
        
        Args:
            config: Configuration (possibly invalid)
            
        Returns:
            Nearest valid configuration
        """
        return self.bounds.clip(config)
    
    def get_config_dict(self, config: np.ndarray) -> dict:
        """
        Convert configuration array to named dictionary.
        
        Args:
            config: Configuration array
            
        Returns:
            Dictionary with dimension names as keys
        """
        return {name: float(config[i]) for i, name in enumerate(self.dim_names)}
    
    def from_dict(self, config_dict: dict) -> np.ndarray:
        """
        Convert named dictionary to configuration array.
        
        Args:
            config_dict: Dictionary with dimension names as keys
            
        Returns:
            Configuration array
        """
        config = np.zeros(self.dim)
        for i, name in enumerate(self.dim_names):
            if name in config_dict:
                config[i] = config_dict[name]
        return config
    
    def __repr__(self) -> str:
        return f"MorphingConfigSpace(dim={self.dim}, dims={self.dim_names})"


class LatticeConfigSpace(MorphingConfigSpace):
    """
    Configuration space for lattice-based morphing structures.
    
    Extends the base configuration space with lattice-specific
    parameters following the programmable lattice methodology.
    
    Lattice Parameters:
        - Topology indices τᵢ ∈ [0,1] (BCC to X-cube blend)
        - Superposition angles θⱼ ∈ [0, 2π]
    """
    
    def __init__(self,
                 num_cells: int = 4,
                 num_rotation_axes: int = 2):
        """
        Initialize lattice configuration space.
        
        Args:
            num_cells: Number of lattice cells with variable topology
            num_rotation_axes: Number of superposition rotation axes
        """
        self.num_cells = num_cells
        self.num_rotation_axes = num_rotation_axes
        
        dim = num_cells + num_rotation_axes
        
        # Topology indices: [0, 1]
        # Rotation angles: [0, 2π]
        min_bounds = np.concatenate([
            np.zeros(num_cells),           # τ_min
            np.zeros(num_rotation_axes)    # θ_min
        ])
        max_bounds = np.concatenate([
            np.ones(num_cells),            # τ_max
            np.ones(num_rotation_axes) * 2 * np.pi  # θ_max
        ])
        
        bounds = MorphingBounds(min_bounds, max_bounds)
        
        dim_names = [f'topology_{i}' for i in range(num_cells)] + \
                    [f'rotation_{i}' for i in range(num_rotation_axes)]
        
        super().__init__(dim=dim, bounds=bounds, dim_names=dim_names)
    
    def get_topology_indices(self, config: np.ndarray) -> np.ndarray:
        """Extract topology indices from configuration."""
        return config[:self.num_cells]
    
    def get_rotation_angles(self, config: np.ndarray) -> np.ndarray:
        """Extract rotation angles from configuration."""
        return config[self.num_cells:]
    
    def estimate_stiffness(self, config: np.ndarray) -> float:
        """
        Estimate resulting stiffness from lattice configuration.
        
        Based on topology: BCC (τ=0) → 25 kPa, X-cube (τ=1) → 300 kPa
        
        Args:
            config: Lattice configuration
            
        Returns:
            Estimated Young's modulus in kPa
        """
        topology = self.get_topology_indices(config)
        mean_topology = np.mean(topology)
        
        # Linear interpolation between BCC and X-cube stiffness
        E_bcc = 25.0    # kPa
        E_xcube = 300.0  # kPa
        
        return E_bcc + mean_topology * (E_xcube - E_bcc)
    
    def estimate_num_configurations(self) -> int:
        """
        Estimate total number of discrete configurations.
        
        Based on CREATE Lab's claim of >1 million configurations
        from just two cell types with superposition.
        
        Returns:
            Estimated number of unique configurations
        """
        # Discretize topology (10 levels) and rotation (36 levels)
        topology_levels = 10
        rotation_levels = 36
        
        return (topology_levels ** self.num_cells) * \
               (rotation_levels ** self.num_rotation_axes)
