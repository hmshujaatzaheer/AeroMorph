"""
Tests for Morphological Decision Space.
"""

import pytest
import numpy as np
from aeromorph.core.decision_space import MorphologicalDecisionSpace


class TestMorphologicalDecisionSpace:
    """Test suite for MorphologicalDecisionSpace."""
    
    def test_initialization(self):
        """Test decision space initialization."""
        ds = MorphologicalDecisionSpace(config_dim=6)
        assert ds.config_dim == 6
        assert ds.coordination.num_robots == 1
    
    def test_perception_update(self):
        """Test perception space update."""
        ds = MorphologicalDecisionSpace(config_dim=6)
        
        passage_widths = np.array([1.5, 2.0, 2.5])
        ds.update_perception(passage_widths=passage_widths, timestamp=1.0)
        
        assert ds.perception.timestamp == 1.0
        assert np.allclose(ds.perception.passage_widths, passage_widths)
    
    def test_energy_update(self):
        """Test energy space update."""
        ds = MorphologicalDecisionSpace(config_dim=6)
        
        ds.update_energy(3000.0)
        assert ds.energy.current_energy == 3000.0
    
    def test_utility_computation(self):
        """Test utility computation."""
        ds = MorphologicalDecisionSpace(config_dim=6)
        
        current = np.array([0.7, 0.3, 0.5, 0.5, 0.5, 0.4])
        target = np.array([0.3, 0.6, 0.5, 0.5, 0.5, 0.8])
        
        utility = ds.compute_utility(current, target)
        assert isinstance(utility, float)
    
    def test_should_morph(self):
        """Test morphing decision."""
        ds = MorphologicalDecisionSpace(config_dim=6)
        ds.update_energy(5000.0)
        
        current = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        target = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        
        # Same config should not trigger morph
        should = ds.should_morph(current, target)
        assert isinstance(should, bool)
    
    def test_state_vector(self):
        """Test state vector generation."""
        ds = MorphologicalDecisionSpace(config_dim=6)
        
        state = ds.get_state_vector()
        assert isinstance(state, np.ndarray)
        assert len(state) == 7  # perception(3) + energy(2) + coord(2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
