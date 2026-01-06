"""
Pytest configuration and fixtures for AeroMorph tests.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_config():
    """Sample morphing configuration for testing."""
    return np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])


@pytest.fixture
def sample_perception():
    """Sample perception state for testing."""
    from aeromorph.core.state import PerceptionState
    return PerceptionState(
        lidar_points=np.random.randn(100, 3) * 5,
        timestamp=0.0
    )


@pytest.fixture
def sample_robot_state():
    """Sample extended robot state for testing."""
    from aeromorph.core.state import ExtendedRobotState, Pose
    return ExtendedRobotState(
        pose=Pose(position=np.array([0.0, 0.0, 10.0])),
        morph_config=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        energy_state=4000.0
    )
