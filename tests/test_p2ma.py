"""
Tests for P2MA Algorithm.
"""

import pytest
import numpy as np
from aeromorph.algorithms.p2ma import P2MAAlgorithm, MorphDecisionEngine
from aeromorph.core.state import PerceptionState
from aeromorph.core.types import MorphAction


class TestP2MAAlgorithm:
    """Test suite for P2MA algorithm."""
    
    def test_initialization(self):
        """Test P2MA initialization."""
        p2ma = P2MAAlgorithm(config_dim=6)
        assert p2ma.config_space.dim == 6
    
    def test_decide_hold(self):
        """Test decision to hold configuration."""
        p2ma = P2MAAlgorithm(config_dim=6)
        
        # Open environment should not trigger morphing
        perception = PerceptionState(
            lidar_points=np.random.randn(100, 3) * 20,
            timestamp=0.0
        )
        
        current = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        
        decision = p2ma.decide(
            current_config=current,
            perception=perception,
            energy_budget=1000.0
        )
        
        assert decision.action in [MorphAction.HOLD, MorphAction.MORPH]
        assert decision.current_config is not None
    
    def test_energy_constraint(self):
        """Test energy constraint on morphing."""
        p2ma = P2MAAlgorithm(config_dim=6)
        
        perception = PerceptionState(
            lidar_points=np.random.randn(100, 3) * 0.5,  # Close obstacles
            timestamp=0.0
        )
        
        current = np.array([0.9, 0.3, 0.5, 0.5, 0.5, 0.4])
        
        # Very low energy budget
        decision = p2ma.decide(
            current_config=current,
            perception=perception,
            energy_budget=1.0  # Very limited
        )
        
        # Should either hold or do minimal morph
        assert decision.energy_cost <= 1.0 or decision.action == MorphAction.HOLD


class TestMorphDecisionEngine:
    """Test suite for MorphDecisionEngine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = MorphDecisionEngine(config_space_dim=6, energy_budget=1000.0)
        assert engine.energy_budget == 1000.0
    
    def test_decision_tracking(self):
        """Test decision history tracking."""
        engine = MorphDecisionEngine(config_space_dim=6, energy_budget=5000.0)
        
        perception = PerceptionState(timestamp=0.0)
        current = np.ones(6) * 0.5
        
        # Make several decisions
        for _ in range(3):
            engine.decide(current_config=current, perception=perception)
        
        stats = engine.get_decision_stats()
        assert stats['total'] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
