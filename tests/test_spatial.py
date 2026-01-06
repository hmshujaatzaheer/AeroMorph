"""
Tests for Spatial Feasibility Analysis.
"""

import pytest
import numpy as np
from aeromorph.algorithms.spatial_feasibility import SpatialFeasibilityChecker


class TestSpatialFeasibilityChecker:
    """Test suite for spatial feasibility checker."""
    
    def test_initialization(self):
        """Test checker initialization."""
        checker = SpatialFeasibilityChecker(safety_margin=0.1)
        assert checker.safety_margin == 0.1
        assert checker.interpolation_steps == 20
    
    def test_verify_empty_environment(self):
        """Test verification in empty environment."""
        checker = SpatialFeasibilityChecker()
        
        start = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        target = np.array([0.3, 0.7, 0.5, 0.5, 0.5, 0.5])
        environment = np.empty((0, 3))  # No obstacles
        
        result = checker.verify(start, target, environment)
        assert result.feasible
    
    def test_verify_with_obstacles(self):
        """Test verification with obstacles."""
        checker = SpatialFeasibilityChecker(safety_margin=0.5)
        
        start = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        target = np.array([0.3, 0.7, 0.5, 0.5, 0.5, 0.5])
        
        # Dense obstacles very close
        obstacles = np.random.randn(100, 3) * 0.1
        
        result = checker.verify(start, target, obstacles)
        assert isinstance(result.feasible, bool)
        assert result.computation_time >= 0
    
    def test_trajectory_generation(self):
        """Test trajectory interpolation."""
        checker = SpatialFeasibilityChecker(interpolation_steps=10)
        
        start = np.zeros(6)
        target = np.ones(6)
        
        trajectory = checker._interpolate_morph(start, target)
        assert len(trajectory) == 10
        assert np.allclose(trajectory[0], start)
        assert np.allclose(trajectory[-1], target)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
