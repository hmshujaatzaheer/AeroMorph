"""
Tests for Swarm Morphing Protocol.
"""

import pytest
import numpy as np
from aeromorph.swarm.ea_smp import SwarmMorphProtocol, SwarmConfig
from aeromorph.core.state import ExtendedRobotState, Pose


class TestSwarmMorphProtocol:
    """Test suite for swarm morphing protocol."""
    
    def test_initialization(self):
        """Test protocol initialization."""
        protocol = SwarmMorphProtocol(num_robots=5)
        assert protocol.num_robots == 5
    
    def test_coordinate_single_robot(self):
        """Test coordination with single robot."""
        protocol = SwarmMorphProtocol(num_robots=1)
        
        states = {0: ExtendedRobotState(
            pose=Pose(position=np.zeros(3)),
            morph_config=np.ones(6) * 0.5,
            energy_state=4000.0
        )}
        
        target = np.ones(6) * 0.7
        
        result = protocol.coordinate_morph(
            swarm_states=states,
            target_config=target
        )
        
        assert result.leader_id == 0
        assert 0 in result.participating
    
    def test_leader_election(self):
        """Test energy-balanced leader election."""
        protocol = SwarmMorphProtocol(num_robots=3, communication_range=20.0)
        
        # Robot 1 has highest energy
        states = {
            0: ExtendedRobotState(
                pose=Pose(position=np.array([0, 0, 0])),
                morph_config=np.ones(6) * 0.5,
                energy_state=3000.0
            ),
            1: ExtendedRobotState(
                pose=Pose(position=np.array([5, 0, 0])),
                morph_config=np.ones(6) * 0.5,
                energy_state=5000.0  # Highest
            ),
            2: ExtendedRobotState(
                pose=Pose(position=np.array([10, 0, 0])),
                morph_config=np.ones(6) * 0.5,
                energy_state=4000.0
            ),
        }
        
        target = np.ones(6) * 0.7
        
        result = protocol.coordinate_morph(
            swarm_states=states,
            target_config=target
        )
        
        # Leader should be robot 1 (highest energy)
        assert result.leader_id == 1
        assert result.energy_balanced


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
