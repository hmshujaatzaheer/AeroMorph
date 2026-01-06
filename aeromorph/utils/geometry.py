"""
Geometric utility functions for AeroMorph framework.
"""

import numpy as np
from typing import Tuple


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles to quaternion.
    
    Args:
        euler: Euler angles [roll, pitch, yaw] in radians
        
    Returns:
        Quaternion [w, x, y, z]
    """
    roll, pitch, yaw = euler
    
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def transform_points(points: np.ndarray, 
                     rotation: np.ndarray, 
                     translation: np.ndarray) -> np.ndarray:
    """
    Transform points by rotation and translation.
    
    Args:
        points: Points to transform (N, 3)
        rotation: 3x3 rotation matrix
        translation: 3D translation vector
        
    Returns:
        Transformed points (N, 3)
    """
    return (rotation @ points.T).T + translation


def compute_aabb(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute axis-aligned bounding box.
    
    Args:
        points: Points (N, 3)
        
    Returns:
        Tuple of (min_corner, max_corner)
    """
    return np.min(points, axis=0), np.max(points, axis=0)


def point_in_ellipsoid(point: np.ndarray,
                       center: np.ndarray,
                       axes: np.ndarray) -> bool:
    """
    Check if point is inside ellipsoid.
    
    Args:
        point: 3D point
        center: Ellipsoid center
        axes: Ellipsoid semi-axes lengths
        
    Returns:
        True if point is inside
    """
    normalized = (point - center) / axes
    return np.sum(normalized ** 2) <= 1.0
