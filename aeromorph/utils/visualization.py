"""
Visualization utilities for AeroMorph framework.
"""

import numpy as np
from typing import Optional, List, Tuple


def plot_trajectory(trajectory: np.ndarray,
                    morph_points: Optional[List[int]] = None,
                    ax=None,
                    show: bool = True):
    """
    Plot 3D trajectory with morphing points highlighted.
    
    Args:
        trajectory: Waypoints (N, 3)
        morph_points: Indices where morphing occurs
        ax: Matplotlib 3D axes (created if None)
        show: Whether to show plot
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            'b-', linewidth=2, label='Trajectory')
    
    # Highlight morphing points
    if morph_points:
        morph_pos = trajectory[morph_points]
        ax.scatter(morph_pos[:, 0], morph_pos[:, 1], morph_pos[:, 2],
                   c='red', s=100, marker='*', label='Morph Events')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    ax.set_title('AeroMorph Mission Trajectory')
    
    if show:
        plt.show()
    
    return ax


def plot_morph_schedule(schedule: List[Tuple[float, np.ndarray]],
                        config_names: Optional[List[str]] = None,
                        show: bool = True):
    """
    Plot morphing schedule over time.
    
    Args:
        schedule: List of (time, config) tuples
        config_names: Names for configuration dimensions
        show: Whether to show plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    if not schedule:
        print("Empty schedule")
        return
    
    times = [t for t, _ in schedule]
    configs = np.array([c for _, c in schedule])
    
    if config_names is None:
        config_names = [f'dim_{i}' for i in range(configs.shape[1])]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i in range(min(configs.shape[1], len(config_names))):
        ax.step(times, configs[:, i], where='post', label=config_names[i])
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Configuration Value')
    ax.set_title('AeroMorph Morphing Schedule')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    if show:
        plt.show()
    
    return ax


def visualize_config(config: np.ndarray,
                     config_space=None,
                     show: bool = True):
    """
    Visualize a morphing configuration as robot shape.
    
    Args:
        config: Morphing configuration
        config_space: Configuration space (for bounds)
        show: Whether to show plot
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract configuration parameters
    wingspan = 0.5 + 1.0 * config[0]
    sweep = config[1] * 30  # degrees
    body_length = 0.3 + 0.2 * config[4] if len(config) > 4 else 0.4
    
    # Draw simplified robot shape (top view)
    # Body
    body = Polygon([
        [-body_length/2, -0.05],
        [body_length/2, -0.05],
        [body_length/2, 0.05],
        [-body_length/2, 0.05]
    ], facecolor='gray', edgecolor='black')
    ax.add_patch(body)
    
    # Wings
    sweep_rad = np.radians(sweep)
    wing_points_left = [
        [0, 0],
        [-wingspan/4 * np.sin(sweep_rad), -wingspan/2 * np.cos(sweep_rad)],
        [-wingspan/4 * np.sin(sweep_rad) - 0.1, -wingspan/2 * np.cos(sweep_rad)],
        [-0.1, 0]
    ]
    wing_points_right = [
        [0, 0],
        [-wingspan/4 * np.sin(sweep_rad), wingspan/2 * np.cos(sweep_rad)],
        [-wingspan/4 * np.sin(sweep_rad) - 0.1, wingspan/2 * np.cos(sweep_rad)],
        [-0.1, 0]
    ]
    
    left_wing = Polygon(wing_points_left, facecolor='lightblue', edgecolor='blue')
    right_wing = Polygon(wing_points_right, facecolor='lightblue', edgecolor='blue')
    ax.add_patch(left_wing)
    ax.add_patch(right_wing)
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_title(f'Configuration: wingspan={wingspan:.2f}m, sweep={sweep:.1f}°')
    ax.grid(True, alpha=0.3)
    
    if show:
        plt.show()
    
    return ax
